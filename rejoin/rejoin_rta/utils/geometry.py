import math
import numpy as np
from scipy.spatial.transform import Rotation as R


class RelativePoint:
    def __init__(self, ref, cartesian_offset=None, track_orientation=False):

        self._cartesian_offset = cartesian_offset
        self.track_orientation = track_orientation

        self._center = np.array([0,0,0], dtype=np.float64)

        # save reference object
        self.ref = ref
        # register self to reference object dependency list
        self.ref.register_dependent_obj(self)

        self.update()

    def reset(self):
        self.update()

    def update(self):
        
        if self.track_orientation:
            rot_mat = self.get_ref_rot_mat()
        else:
            rot_mat = np.eye(3)

        rotated_offset = np.matmul(rot_mat, self._cartesian_offset)

        self._center = self.get_ref_center() + rotated_offset

    def step(self, *args):
        self.update()

    def get_ref_rot_mat(self):
        raise NotImplemented

    def get_ref_center(self):
        raise NotImplemented

class RelativePoint2D(RelativePoint):    
    def __init__(self, ref, track_orientation=False, x_offset=None, y_offset=None, r_offset=None, theta_offset=None, aspect_angle=None):
        # check that both x_offset and y_offset are used at the same time if used
        assert (x_offset is None) == (y_offset is None), "if either x_offset or y_offset is used, both x_offset and y_offset must be used"

        # check that only r_offset, theta_offset or x_offset, y_offset are specified
        assert ((r_offset is not None) or (theta_offset is not None) or (aspect_angle is not None)) != ((x_offset is not None) or (y_offset is not None)), \
            "user either polar or cartesian relative position definiton, not both"

        # check that only theta_offset or aspect_angle is used
        assert (( theta_offset is None ) or ( aspect_angle is None )), "specify either theta_offset or aspect_angle, not both"


        # convert aspect angle to theta
        if aspect_angle is not None:
            theta_offset = (math.pi - aspect_angle)

        # convert polar to x,y offset
        if (x_offset is None) and (y_offset is None):
            x_offset = r_offset * math.cos(theta_offset)
            y_offset = r_offset * math.sin(theta_offset)

        # default values if nothing is specified
        if x_offset is None:
            x_offset = 0
        if y_offset is None:
            y_offset = 0

        cartesian_offset = np.array([x_offset, y_offset, 0], dtype=np.float64)

        self.track_orientation = track_orientation
        super(RelativePoint2D, self).__init__(ref, cartesian_offset=cartesian_offset, track_orientation=track_orientation)

    def _generate_info(self):
        info = {
            'x': self.x,
            'y': self.y,
        }

        return info

    def get_ref_rot_mat(self):
        orientation_angle = self.ref.orientation

        rot_mat = R.from_euler('Z', orientation_angle).as_matrix()

        return rot_mat

    def get_ref_center(self):
        return np.array([ self.ref.x, self.ref.y, 0], dtype=np.float64)

    @property
    def x(self):
        return self._center[0]

    @property
    def y(self):
        return self._center[1]

    @property
    def position(self) -> np.ndarray:
        return self._center[0:2]

class RelativeCircle2D(RelativePoint2D):
    def __init__(self, ref, radius=None, **kwargs):
        assert radius is not None, "Please specify a radius"

        self._radius = radius
        super(RelativeCircle2D, self).__init__(ref, **kwargs)

    def _generate_info(self):
        info = super(RelativeCircle2D, self)._generate_info()
        info['radius'] = self.radius
        return info

    def contains(self, point_coords):
        center_distance = math.sqrt((point_coords[0]-self.x)**2 + (point_coords[1]-self.y)**2)

        return self._radius >= center_distance

    @property
    def radius(self):
        return self._radius

if __name__=='__main__':

    class test_ref:
        def __init__(self, x, y, orientation):
            self.x = x
            self.y = y
            self.orientation = orientation

    ref = test_ref(10, 345, 0)
    rp = RelativeCircle2D(ref, radius=150, track_orientation=True, r_offset=100, aspect_angle=(60*math.pi/180))

    print("x={}, y={}".format(rp.x, rp.y))

    ref.orientation = math.pi
    rp.update()
    print("x={}, y={}".format(rp.x, rp.y))

    ref.orientation = math.pi/2
    ref.x = 75
    ref.y=13
    rp.update()
    print("x={}, y={}".format(rp.x, rp.y))

    ref.orientation = 14
    rp.update()
    print("x={}, y={}".format(rp.x, rp.y))

def distance2d(a,b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)