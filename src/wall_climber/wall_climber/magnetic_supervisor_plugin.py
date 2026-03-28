"""Magnetic adhesion supervisor plugin  +  pen trail drawing.

Runs on an invisible Supervisor Robot defined in wall_world.wbt.
Every simulation step it:
  1. Locates the wall_climber robot (by scanning root children)
  2. Reads its world-frame position
  3. Applies:
       - adhesion  force  (+Y  toward the whiteboard surface)
       - anti-gravity     (+Z  counteract gravity)
  4. Finds the pen_holder Solid inside the wall_climber
  5. When the marker tip is close to the whiteboard, drops small
     black dots to form a visible writing trail.

Because this controller has  supervisor TRUE  it can use
getPosition() / addForce() / importMFNodeFromString() on any
node in the scene — something the URDF-spawned wall_climber
(supervisor FALSE) cannot do itself.
"""

import json
import math
import rclpy
from geometry_msgs.msg import PointStamped, Pose2D
from std_msgs.msg import Bool, Float64, String


class MagneticSupervisorPlugin:
    _BOARD_FRAME_ID = 'board'
    _SAFE_UNAVAILABLE_GAP = 1.0

    def init(self, webots_node, properties):
        self._supervisor = webots_node.robot
        self._target = None          # Node handle for wall_climber
        self._step_count = 0
        self._search_every = 1       # search EVERY step until found

        # --- tunables (overridden from <plugin> children in xacro) ---
        self._target_name = str(properties.get('target_robot', 'wall_climber'))
        self._wall_y = float(properties.get('wall_y', '2.415'))
        self._max_dist = float(properties.get('activation_distance', '0.8'))
        self._adhesion = float(properties.get('adhesion_force', '50.0'))
        self._mass = float(properties.get('robot_mass', '2.5'))
        self._gravity = float(properties.get('gravity', '9.81'))
        self._ag_ratio = float(properties.get('anti_gravity_ratio', '1.05'))

        # --- pen trail state ---
        self._pen_node = None     # will hold the pen_holder Solid
        self._pen_search_at = 100      # first search after N steps (robot needs time to spawn)
        self._pen_search_interval = 50  # retry every N steps if not found
        self._root_children = None     # root.children MFNode
        self._last_pos = None     # (x, z) of last contact position
        self._pen_contact_latched = False
        self._tip_sphere_local_center = None
        self._tip_sphere_radius = None
        self._tip_geometry_ready = False
        self._tip_geometry_warned = False
        self._contact_engage_gap = float(
            properties.get('pen_contact_engage_gap', '0.0005')
        )
        self._contact_release_gap = float(
            properties.get('pen_contact_release_gap', '0.0012')
        )
        self._fallback_tip_sphere_local_center = None
        self._fallback_tip_sphere_radius = None

        # Single-mesh trail (IndexedFaceSet) — one node, unlimited quads
        self._trail_half_width = float(
            properties.get('trail_half_width', '0.010')
        )
        self._trail_round_segments = max(
            8, int(properties.get('trail_round_segments', '12'))
        )
        self._trail_max = int(properties.get('trail_max', '8000'))
        self._trail_min_spacing = float(
            properties.get('trail_min_spacing', '0.0004')
        )
        self._trail_segment_count = 0
        self._trail_mesh_ready = False
        self._trail_point_field = None   # MFVec3f handle
        self._trail_index_field = None   # MFInt32 handle
        self._trail_last_dir = None
        self._trail_last_round_pos = None

        # --- board geometry / writable-area parameters ---
        self._board_center_x = float(properties.get('board_center_x', '0.0'))
        self._board_center_z = float(properties.get('board_center_z', '1.8'))
        self._board_width = float(properties.get('board_width', '6.3'))
        self._board_height = float(properties.get('board_height', '2.8'))
        self._margin_left = float(properties.get('margin_left', '0.10'))
        self._margin_right = float(properties.get('margin_right', '0.10'))
        self._margin_top = float(properties.get('margin_top', '0.10'))
        self._margin_bottom = float(properties.get('margin_bottom', '0.10'))
        self._line_height = float(properties.get('line_height', '0.14'))

        self._board_left = self._board_center_x - self._board_width / 2.0
        self._board_top_z = self._board_center_z + self._board_height / 2.0

        self._writable_x_min = self._margin_left
        self._writable_x_max = self._board_width - self._margin_right
        self._writable_y_min = self._margin_top
        self._writable_y_max = self._board_height - self._margin_bottom

        if self._writable_x_min > self._writable_x_max:
            self._writable_x_max = self._writable_x_min
        if self._writable_y_min > self._writable_y_max:
            self._writable_y_max = self._writable_y_min

        self._board_pub_every = 2
        self._board_log_every = 150

        # --- ROS logging ---
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node('magnetic_supervisor')
        self._log = self._node.get_logger()
        self._fallback_tip_sphere_local_center = self._optional_vec3_property(
            properties,
            'pen_tip_center',
        )
        self._fallback_tip_sphere_radius = self._optional_float_property(
            properties,
            'pen_tip_radius',
        )
        self._robot_board_pub = self._node.create_publisher(
            Pose2D, '/wall_climber/robot_pose_board', 1
        )
        self._pen_board_pub = self._node.create_publisher(
            PointStamped, '/wall_climber/pen_pose_board', 1
        )
        self._board_info_pub = self._node.create_publisher(
            String, '/wall_climber/board_info', 1
        )
        self._pen_inside_pub = self._node.create_publisher(
            Bool, '/wall_climber/pen_inside_board', 1
        )
        self._pen_contact_pub = self._node.create_publisher(
            Bool, '/wall_climber/pen_contact', 1
        )
        self._pen_gap_pub = self._node.create_publisher(
            Float64, '/wall_climber/pen_gap', 1
        )

        # Default to True to support manual pen control (keyboard plugin)
        # stroke_executor will explicitly set this to False during non-drawing states
        self._drawing_active = True
        self._drawing_active_sub = self._node.create_subscription(
            Bool, '/wall_climber/drawing_active', self._drawing_active_cb, 1
        )

        self._board_info_json = json.dumps(
            {
                'width': self._board_width,
                'height': self._board_height,
                'margins': {
                    'left': self._margin_left,
                    'right': self._margin_right,
                    'top': self._margin_top,
                    'bottom': self._margin_bottom,
                },
                'writable_x_min': self._writable_x_min,
                'writable_x_max': self._writable_x_max,
                'writable_y_min': self._writable_y_min,
                'writable_y_max': self._writable_y_max,
                'line_height': self._line_height,
            },
            separators=(',', ':'),
        )

        self._log.info(
            f'Magnetic Supervisor ready  target={self._target_name}  '
            f'wall_y={self._wall_y}  F_adhesion={self._adhesion} N  '
            f'mass={self._mass} kg  ag_ratio={self._ag_ratio}'
        )
        self._log.info(
            'Board geometry: '
            f'center=({self._board_center_x:.3f}, {self._board_center_z:.3f})  '
            f'size=({self._board_width:.3f} x {self._board_height:.3f})'
        )
        self._log.info(
            'Writable limits: '
            f'x=[{self._writable_x_min:.3f}, {self._writable_x_max:.3f}]  '
            f'y=[{self._writable_y_min:.3f}, {self._writable_y_max:.3f}]  '
            f'line_height={self._line_height:.3f}'
        )
        self._log.info(
            'Pen contact hysteresis: '
            f'engage<= {self._contact_engage_gap:.4f} m, '
            f'release<= {self._contact_release_gap:.4f} m'
        )
        self._publish_board_info()

    def _optional_float_property(self, properties, key):
        if key not in properties:
            return None
        try:
            return float(properties.get(key))
        except Exception:
            self._log.warn(f'Invalid plugin property {key!r}; ignoring it.')
            return None

    def _optional_vec3_property(self, properties, prefix):
        keys = [f'{prefix}_x', f'{prefix}_y', f'{prefix}_z']
        if not all(key in properties for key in keys):
            return None
        values = []
        for key in keys:
            value = self._optional_float_property(properties, key)
            if value is None:
                return None
            values.append(value)
        return tuple(values)

    # ------------------------------------------------------------------
    #  Scene-tree helpers
    # ------------------------------------------------------------------
    def _find_target(self):
        """Walk the scene-tree root children to find *target_name*."""
        root = self._supervisor.getRoot()
        if root is None:
            return
        children = root.getField('children')
        if children is None:
            return
        for i in range(children.getCount()):
            child = children.getMFNode(i)
            if child is None:
                continue
            nf = child.getField('name')
            if nf is not None and nf.getSFString() == self._target_name:
                self._target = child
                self._log.info(
                    f'Found "{self._target_name}" (node id {child.getId()})'
                )
                return

    # ------------------------------------------------------------------
    #  Board-state helpers
    # ------------------------------------------------------------------
    def _world_to_board(self, world_x, world_z):
        """Map world (x,z) on the whiteboard to board coordinates.

        Board convention:
          origin = top-left, +x right, +y down.
        """
        board_x = world_x - self._board_left
        board_y = self._board_top_z - world_z
        return board_x, board_y

    def _compute_robot_theta(self):
        """Best-effort robot heading on the board plane.

        Uses local +X axis projected into world XZ plane. theta=0 means
        heading toward +board-x (world +X), which is horizontal writing.
        """
        if self._target is None:
            return 0.0

        try:
            orientation = self._target.getOrientation()
        except Exception:
            return 0.0

        if orientation is None or len(orientation) < 9:
            return 0.0

        # Webots orientation is a 3x3 matrix; local +X axis is best-effort
        # extracted as the first column in world coordinates.
        x_axis_world_x = float(orientation[0])
        x_axis_world_z = float(orientation[6])

        hx = x_axis_world_x      # board +x
        hy = -x_axis_world_z     # board +y (down)
        norm = math.sqrt(hx * hx + hy * hy)
        if norm < 1e-9:
            return 0.0

        return math.atan2(hy, hx)

    def _is_inside_writable(self, board_x, board_y):
        return (
            self._writable_x_min <= board_x <= self._writable_x_max
            and self._writable_y_min <= board_y <= self._writable_y_max
        )

    def _publish_robot_pose(self, board_x, board_y, theta):
        msg = Pose2D()
        msg.x = float(board_x)
        msg.y = float(board_y)
        msg.theta = float(theta)
        self._robot_board_pub.publish(msg)

    def _publish_pen_pose(self, board_x, board_y):
        msg = PointStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._BOARD_FRAME_ID
        msg.point.x = float(board_x)
        msg.point.y = float(board_y)
        msg.point.z = 0.0
        self._pen_board_pub.publish(msg)

    def _publish_pen_inside(self, inside):
        msg = Bool()
        msg.data = bool(inside)
        self._pen_inside_pub.publish(msg)

    def _publish_pen_contact(self, contact):
        msg = Bool()
        msg.data = bool(contact)
        self._pen_contact_pub.publish(msg)

    def _publish_pen_gap(self, gap):
        msg = Float64()
        msg.data = float(gap)
        self._pen_gap_pub.publish(msg)

    def _publish_board_info(self):
        msg = String()
        msg.data = self._board_info_json
        self._board_info_pub.publish(msg)

    def _drawing_active_cb(self, msg):
        self._drawing_active = bool(msg.data)

    def _log_board_state(self, robot_x, robot_y, robot_theta, pen_pos, inside):
        if self._step_count % self._board_log_every != 0:
            return

        if pen_pos is None:
            pen_text = 'pen=(n/a,n/a)'
        else:
            pen_text = f'pen=({pen_pos[0]:.3f},{pen_pos[1]:.3f})'

        self._log.info(
            f'[board] robot=({robot_x:.3f},{robot_y:.3f},{robot_theta:.3f})  '
            f'{pen_text}  inside={inside}'
        )

    # ------------------------------------------------------------------
    def _find_pen_holder(self):
        """Find the pen_holder Solid by locating the 'pen_mount_joint'
        motor in the scene tree.  urdf2webots renames all Solid nodes
        to solid0..solidN, but preserves motor/device names from the
        original URDF joint names.  So we walk the tree looking for a
        Joint node whose ``device`` list contains a motor named
        ``pen_mount_joint``, then return its ``endPoint`` Solid."""

        # Strategy 1: walk tree looking for joint with pen_mount_joint motor
        result = self._find_joint_endpoint(self._target, 'pen_mount_joint')
        if result is not None:
            self._log.info('Found pen_holder via pen_mount_joint motor search')
            return result

        # Strategy 2: getFromDef (unlikely but try)
        try:
            node = self._supervisor.getFromDef('pen_holder')
            if node is not None:
                self._log.info('Found pen_holder via getFromDef')
                return node
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    def _find_joint_endpoint(self, node, motor_name, depth=0):
        """Walk the scene tree to find a Joint whose ``device`` list
        contains a motor with the given name.  Return the joint's
        ``endPoint`` Solid node."""
        if node is None or depth > 40:
            return None

        # Check if this node has a 'device' field (Joint nodes do)
        try:
            dev_field = node.getField('device')
            if dev_field is not None:
                for i in range(dev_field.getCount()):
                    dev = dev_field.getMFNode(i)
                    if dev is None:
                        continue
                    nf = dev.getField('name')
                    if nf is not None:
                        dname = nf.getSFString()
                        if dname == motor_name:
                            # Found it — return the endPoint Solid
                            ep = node.getField('endPoint')
                            if ep is not None:
                                ep_node = ep.getSFNode()
                                self._log.info(
                                    f'  Found motor "{motor_name}" at '
                                    f'depth {depth}, endPoint id='
                                    f'{ep_node.getId()}'
                                )
                                return ep_node
        except Exception:
            pass

        # Recurse into 'children' (MFNode)
        try:
            ch = node.getField('children')
            if ch is not None:
                for i in range(ch.getCount()):
                    child = ch.getMFNode(i)
                    result = self._find_joint_endpoint(
                        child, motor_name, depth + 1
                    )
                    if result is not None:
                        return result
        except Exception:
            pass

        # Recurse into 'endPoint' (SFNode)
        try:
            ep = node.getField('endPoint')
            if ep is not None:
                ep_node = ep.getSFNode()
                result = self._find_joint_endpoint(
                    ep_node, motor_name, depth + 1
                )
                if result is not None:
                    return result
        except Exception:
            pass

        return None

    def _field(self, node, name):
        try:
            return node.getField(name)
        except Exception:
            return None

    def _field_sfvec3f(self, node, name):
        field = self._field(node, name)
        if field is None:
            return None
        try:
            value = field.getSFVec3f()
            return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return None

    def _field_sffloat(self, node, name):
        field = self._field(node, name)
        if field is None:
            return None
        try:
            return float(field.getSFFloat())
        except Exception:
            return None

    def _field_sfnode(self, node, name):
        field = self._field(node, name)
        if field is None:
            return None
        try:
            return field.getSFNode()
        except Exception:
            return None

    def _vec3_add(self, a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def _is_sphere_node(self, node):
        radius = self._field_sffloat(node, 'radius')
        if radius is None:
            return False
        return self._field(node, 'height') is None and self._field(node, 'size') is None

    def _find_tip_sphere_geometry(self, node, accumulated_translation=(0.0, 0.0, 0.0)):
        if node is None:
            return None

        local_translation = accumulated_translation
        translation = self._field_sfvec3f(node, 'translation')
        if translation is not None:
            local_translation = self._vec3_add(local_translation, translation)

        if self._is_sphere_node(node):
            radius = self._field_sffloat(node, 'radius')
            if radius is not None:
                return local_translation, radius

        geometry_node = self._field_sfnode(node, 'geometry')
        if geometry_node is not None:
            result = self._find_tip_sphere_geometry(geometry_node, local_translation)
            if result is not None:
                return result

        child_node = self._field_sfnode(node, 'child')
        if child_node is not None:
            result = self._find_tip_sphere_geometry(child_node, local_translation)
            if result is not None:
                return result

        children_field = self._field(node, 'children')
        if children_field is not None:
            try:
                for index in range(children_field.getCount()):
                    child = children_field.getMFNode(index)
                    result = self._find_tip_sphere_geometry(child, local_translation)
                    if result is not None:
                        return result
            except Exception:
                pass

        return None

    def _resolve_tip_geometry_from_collision(self):
        if self._pen_node is None:
            return None
        bounding_object = self._field_sfnode(self._pen_node, 'boundingObject')
        if bounding_object is None:
            return None
        return self._find_tip_sphere_geometry(bounding_object)

    def _resolve_tip_geometry(self):
        if self._tip_geometry_ready:
            return True

        resolved = self._resolve_tip_geometry_from_collision()
        source = 'pen_holder collision geometry'

        if resolved is None:
            if (
                self._fallback_tip_sphere_local_center is not None
                and self._fallback_tip_sphere_radius is not None
            ):
                resolved = (
                    self._fallback_tip_sphere_local_center,
                    self._fallback_tip_sphere_radius,
                )
                source = 'exact shared plugin properties'

        if resolved is not None:
            self._tip_sphere_local_center, self._tip_sphere_radius = resolved
            self._tip_geometry_ready = True
            self._tip_geometry_warned = False
            self._log.info(
                'Resolved pen-tip sphere from '
                f'{source}: center={self._tip_sphere_local_center}, '
                f'radius={self._tip_sphere_radius:.5f}'
            )
            return True

        if not self._tip_geometry_warned:
            self._log.error(
                'Exact pen-tip geometry is unavailable; forcing pen_contact=false '
                'and disabling trail drawing until exact geometry is available.'
            )
            self._tip_geometry_warned = True
        return False

    def _world_from_local(self, position, orientation, local_point):
        if orientation is None or len(orientation) < 9:
            return None
        return (
            float(position[0])
            + float(orientation[0]) * local_point[0]
            + float(orientation[1]) * local_point[1]
            + float(orientation[2]) * local_point[2],
            float(position[1])
            + float(orientation[3]) * local_point[0]
            + float(orientation[4]) * local_point[1]
            + float(orientation[5]) * local_point[2],
            float(position[2])
            + float(orientation[6]) * local_point[0]
            + float(orientation[7]) * local_point[1]
            + float(orientation[8]) * local_point[2],
        )

    # ------------------------------------------------------------------
    #  Trail drawing  (single IndexedFaceSet mesh)
    # ------------------------------------------------------------------
    def _init_trail_mesh(self):
        """Create ONE IndexedFaceSet node for all pen strokes.

        Instead of inserting thousands of individual Transform+Shape+Box
        nodes (which choke Webots' scene-tree traversal), we place a
        single mesh and append quad vertices/indices to it.
        """
        y = self._wall_y - 0.006
        node_str = (
            f'DEF TRAIL Transform {{ '
            f'translation 0 {y:.5f} 0 '
            f'children [ Shape {{ '
            f'appearance Appearance {{ material Material {{ '
            f'diffuseColor 0 0 0 emissiveColor 0.05 0.05 0.05 }} }} '
            f'geometry IndexedFaceSet {{ '
            f'coord Coordinate {{ point [ '
            f'-100 0 -100, -99.999 0 -100, -99.999 0 -99.999, -100 0 -99.999 ] }} '
            f'coordIndex [ 0 1 2 3 -1 ] '
            f'}} }} ] }}'
        )
        try:
            self._root_children.importMFNodeFromString(-1, node_str)
        except Exception as exc:
            self._log.error(f'Failed to create trail mesh: {exc}')
            return False

        # Navigate the scene tree to get field handles
        trail_node = self._supervisor.getFromDef('TRAIL')
        if trail_node is None:
            # Fallback: grab the last child we just added
            count = self._root_children.getCount()
            trail_node = self._root_children.getMFNode(count - 1)

        try:
            shape = trail_node.getField('children').getMFNode(0)
            ifs = shape.getField('geometry').getSFNode()
            coord = ifs.getField('coord').getSFNode()
            self._trail_point_field = coord.getField('point')
            self._trail_index_field = ifs.getField('coordIndex')
            self._trail_mesh_ready = True
            self._log.info(
                'Trail mesh (IndexedFaceSet) created — '
                'single-node rendering, zero scene-tree overhead'
            )
            return True
        except Exception as exc:
            self._log.error(f'Failed to get trail mesh fields: {exc}')
            return False

    def _add_trail_quad(self, x0, z0, x1, z1):
        """Append a quad (two triangles) from (x0,z0) to (x1,z1)."""
        if self._trail_segment_count >= self._trail_max:
            return
        dx = x1 - x0
        dz = z1 - z0
        length = math.sqrt(dx * dx + dz * dz)
        if length < 1e-6:
            return
        # Normalise direction
        dx /= length
        dz /= length
        # Perpendicular in XZ plane, scaled to half-width
        hw = self._trail_half_width
        px, pz = -dz * hw, dx * hw

        n = self._trail_point_field.getCount()
        # Four corners (y=0 in local frame; parent is at wall_y-0.006)
        self._trail_point_field.insertMFVec3f(-1, [x0 + px, 0, z0 + pz])
        self._trail_point_field.insertMFVec3f(-1, [x0 - px, 0, z0 - pz])
        self._trail_point_field.insertMFVec3f(-1, [x1 - px, 0, z1 - pz])
        self._trail_point_field.insertMFVec3f(-1, [x1 + px, 0, z1 + pz])
        # Face indices (quad + terminator)
        self._trail_index_field.insertMFInt32(-1, n)
        self._trail_index_field.insertMFInt32(-1, n + 1)
        self._trail_index_field.insertMFInt32(-1, n + 2)
        self._trail_index_field.insertMFInt32(-1, n + 3)
        self._trail_index_field.insertMFInt32(-1, -1)

        self._trail_segment_count += 1

    def _trail_round_guard_dist(self):
        return max(self._trail_min_spacing * 0.5, self._trail_half_width * 0.35)

    def _should_skip_round_feature(self, x, z):
        if self._trail_last_round_pos is None:
            return False
        lx, lz = self._trail_last_round_pos
        return math.hypot(x - lx, z - lz) < self._trail_round_guard_dist()

    def _add_trail_round_cap(self, x, z):
        """Append a filled round cap to hide raw quad starts/ends."""
        if self._trail_segment_count >= self._trail_max:
            return
        if self._should_skip_round_feature(x, z):
            return

        hw = self._trail_half_width
        segments = self._trail_round_segments
        n = self._trail_point_field.getCount()
        for i in range(segments):
            angle = (2.0 * math.pi * i) / segments
            self._trail_point_field.insertMFVec3f(
                -1,
                [x + math.cos(angle) * hw, 0, z + math.sin(angle) * hw],
            )
            self._trail_index_field.insertMFInt32(-1, n + i)
        self._trail_index_field.insertMFInt32(-1, -1)
        self._trail_segment_count += 1
        self._trail_last_round_pos = (x, z)

    def _add_trail_round_join(self, x, z):
        # Round joins fill small gaps/overlaps when segment direction changes.
        self._add_trail_round_cap(x, z)

    def _add_trail_dot(self, x, z):
        """Place a round first-contact cap instead of a square dot."""
        self._add_trail_round_cap(x, z)

    def _draw_line_to(self, x, z):
        """Draw from last contact to (x, z) as a continuous quad."""
        if not self._trail_mesh_ready:
            if not self._init_trail_mesh():
                return

        if self._last_pos is None:
            # Avoid stamping an initial blob; start the stroke after real motion.
            self._last_pos = (x, z)
            if hasattr(self, '_trail_last_dir'):
                self._trail_last_dir = None
            return

        lx, lz = self._last_pos
        dx = x - lx
        dz = z - lz
        dist = math.sqrt(dx * dx + dz * dz)

        if dist < self._trail_min_spacing:
            return

        dir_x = dx / dist
        dir_z = dz / dist
        if self._trail_last_dir is not None:
            prev_x, prev_z = self._trail_last_dir
            dot = max(-1.0, min(1.0, prev_x * dir_x + prev_z * dir_z))
            if dot < 0.995:
                self._add_trail_round_join(lx, lz)

        self._add_trail_quad(lx, lz, x, z)
        self._last_pos = (x, z)
        self._trail_last_dir = (dir_x, dir_z)

    # ------------------------------------------------------------------
    def step(self):
        rclpy.spin_once(self._node, timeout_sec=0)
        self._step_count += 1
        publish_due = (self._step_count % self._board_pub_every == 0)

        # --- wait for the URDF-spawned robot to appear ----------------
        if self._target is None:
            if self._step_count % self._search_every == 0:
                self._find_target()
            return

        # --- read world position of the target robot ------------------
        try:
            pos = self._target.getPosition()          # [x, y, z]
        except Exception:
            self._target = None                        # lost → re-search
            return
        if pos is None or len(pos) < 3:
            return

        robot_board_x, robot_board_y = self._world_to_board(pos[0], pos[2])
        robot_theta = self._compute_robot_theta()

        if publish_due:
            self._publish_robot_pose(robot_board_x, robot_board_y, robot_theta)
            self._publish_board_info()

        dist = abs(self._wall_y - pos[1])

        if dist > self._max_dist:
            if self._step_count % 300 == 0:
                self._log.warn(
                    f'Too far from wall  y={pos[1]:.3f}  d={dist:.3f}  '
                    'adhesion inactive'
                )
            if publish_due:
                self._publish_pen_inside(False)
            self._log_board_state(
                robot_board_x, robot_board_y, robot_theta, None, False
            )
            return

        # --- compute forces (world frame, ENU) -----------------------
        #   +Y  → toward the whiteboard surface  (adhesion)
        #   +Z  → upward, opposing gravity        (anti-gravity)
        fy = self._adhesion
        fz = self._mass * self._gravity * self._ag_ratio
        force = [0.0, fy, fz]

        try:
            self._target.addForce(force, False)        # world frame
        except Exception as exc:
            if self._step_count % 300 == 0:
                self._log.error(f'addForce failed: {exc}')

        # --- pen trail drawing ----------------------------------------
        # Periodically search for pen_holder until found
        if self._pen_node is None:
            if self._step_count >= self._pen_search_at:
                self._log.info(
                    f'Searching for pen_holder (step {self._step_count})...'
                )
                self._pen_node = self._find_pen_holder()
                if self._pen_node is not None:
                    self._root_children = (
                        self._supervisor.getRoot().getField('children')
                    )
                    self._resolve_tip_geometry()
                    self._log.info(
                        f'pen_holder found (id {self._pen_node.getId()}) '
                        '— trail drawing ENABLED'
                    )
                else:
                    self._log.warn(
                        f'pen_holder NOT found yet (step {self._step_count}), '
                        f'will retry in {self._pen_search_interval} steps'
                    )
                # Schedule next search attempt
                self._pen_search_at = (
                    self._step_count + self._pen_search_interval
                )
            if publish_due:
                self._publish_pen_inside(False)
            self._log_board_state(
                robot_board_x, robot_board_y, robot_theta, None, False
            )
            return  # skip trail this step regardless

        try:
            pen_pos = self._pen_node.getPosition()     # [x, y, z] world
        except Exception:
            self._log.warn('Lost pen_holder node, will re-search')
            self._pen_node = None
            self._pen_search_at = self._step_count + 10
            if publish_due:
                self._publish_pen_inside(False)
            self._log_board_state(
                robot_board_x, robot_board_y, robot_theta, None, False
            )
            return
        if pen_pos is None or len(pen_pos) < 3:
            if publish_due:
                self._publish_pen_inside(False)
            self._log_board_state(
                robot_board_x, robot_board_y, robot_theta, None, False
            )
            return

        tip_geometry_ready = self._resolve_tip_geometry()
        pen_orientation = None
        try:
            pen_orientation = self._pen_node.getOrientation()
        except Exception:
            pen_orientation = None

        tip_center_world = None
        gap = self._SAFE_UNAVAILABLE_GAP
        contact = False
        wall_front_y = self._wall_y - 0.005

        if tip_geometry_ready and pen_orientation is not None:
            tip_center_world = self._world_from_local(
                pen_pos,
                pen_orientation,
                self._tip_sphere_local_center,
            )
            if tip_center_world is not None:
                tip_surface_y = tip_center_world[1] + self._tip_sphere_radius
                gap = wall_front_y - tip_surface_y
                if self._pen_contact_latched:
                    contact = gap <= self._contact_release_gap
                else:
                    contact = gap <= self._contact_engage_gap
        else:
            self._pen_contact_latched = False

        effective_pen_world = tip_center_world if tip_center_world is not None else pen_pos
        pen_board_x, pen_board_y = self._world_to_board(
            effective_pen_world[0],
            effective_pen_world[2],
        )
        pen_inside = self._is_inside_writable(pen_board_x, pen_board_y)
        if publish_due:
            self._publish_pen_pose(pen_board_x, pen_board_y)
            self._publish_pen_inside(pen_inside)
        self._log_board_state(
            robot_board_x,
            robot_board_y,
            robot_theta,
            (pen_board_x, pen_board_y),
            pen_inside,
        )

        self._pen_contact_latched = contact
        self._publish_pen_contact(contact)
        self._publish_pen_gap(gap)

        if self._step_count % 200 == 0:
            if tip_center_world is not None:
                self._log.info(
                    f'[trail] pen_holder=({pen_pos[0]:.3f}, {pen_pos[1]:.3f}, '
                    f'{pen_pos[2]:.3f})  tip_center=({tip_center_world[0]:.3f}, '
                    f'{tip_center_world[1]:.3f}, {tip_center_world[2]:.3f})  '
                    f'radius={self._tip_sphere_radius:.4f}  gap={gap:.4f}  '
                    f'wall_front={wall_front_y:.3f}  segs={self._trail_segment_count}'
                )
            else:
                self._log.info(
                    '[trail] exact tip geometry unavailable; '
                    f'gap forced to {gap:.3f} and drawing disabled'
                )

        trail_drawing_enabled = (
            self._trail_segment_count < self._trail_max and tip_geometry_ready
        )

        # Keep trail continuity while the pen is still touching the board but
        # drawing is temporarily paused, for example during corner settling.
        if contact:
            if self._drawing_active and trail_drawing_enabled:
                x, z = effective_pen_world[0], effective_pen_world[2]
                self._draw_line_to(x, z)
        else:
            # Only a real lift-off should cap and break the current stroke.
            if trail_drawing_enabled and self._last_pos is not None:
                self._add_trail_round_cap(self._last_pos[0], self._last_pos[1])
            self._last_pos = None
            self._trail_last_dir = None
            self._trail_last_round_pos = None
