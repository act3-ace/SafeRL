import abc
import math
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
from rejoin_rta import BaseEnvObj

class BaseGeometery(BaseEnvObj):

    @property
    @abc.abstractmethod
    def position(self):
        ...

    @position.setter
    @abc.abstractmethod
    def position(self, value):
        ...

    # need to redefine orientation property to add a setter. Is it possible to avoid doing this?
    @property   
    @abc.abstractmethod
    def orientation(self) -> Rotation:
        ...
   
    @orientation.setter
    @abc.abstractmethod
    def orientation(self, value):
        ...

class Point(BaseGeometery):
    
    
    def __init__(self, x=0, y=0):
        self._center = np.array( [x ,y ,0 ] , dtype=np.float64)

    @property
    def x(self):
        return self._center[0]

    @property
    def y(self):
        return self._center[1]

    @property
    def z(self):
        return self._center[2]

    @property
    def position(self):
        return copy.deepcopy(self._center)

    @position.setter
    def position(self, value):
        assert isinstance(value, np.ndarray) and value.shape == (3,) , "Position must be set in a numpy ndarray with shape=(3,)"
        self._center = copy.deepcopy(value)

    @property
    def orientation(self):
        # always return a no rotation quaternion as points do not have an orientation
        return R.from_quat([0, 0, 0, 1])
    
    @orientation.setter
    def orientation(self, value):
        # simply pass as points do not have an orientation
        pass

class RelativeGeometry(BaseGeometery):
    
    
    def __init__(self, 
        ref, 
        track_orientation=False, 
        x_offset=None, 
        y_offset=None, 
        z_offset=None, 
        r_offset=None, 
        theta_offset=None, 
        aspect_angle=None, 
        **kwargs):

        # check that both x_offset and y_offset are used at the same time if used
        assert (x_offset is None) == (y_offset is None) == (z_offset is None), \
            "if either x_offset or y_offset is used, both x_offset and y_offset must be used"

        # check that only r_offset, theta_offset or x/y/z offset are specified
        assert ((r_offset is not None) or (theta_offset is not None) or (aspect_angle is not None)) != ((x_offset is not None) or (y_offset is not None) or (z_offset is None) ), \
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
        if z_offset is None:
            z_offset = 0

        self.ref = ref
        self.track_orientation = track_orientation

        self._cartesian_offset = np.array([x_offset, y_offset, z_offset], dtype=np.float64)

        super().__init__(**kwargs)

        self.update()

    def update(self):
        ref_orientation = self.ref.orientation

        offset = ref_orientation.apply(self._cartesian_offset)

        self.position = self.ref.position + offset

        if self.track_orientation:
            self.orientation = self.ref.orientation


class RelativePoint(abc.ABC):
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

    @abc.abstractmethod
    def get_ref_rot_mat(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def get_ref_center(self) -> np.ndarray:
        ...

class RelativePoint2d(RelativePoint):    
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
        super(RelativePoint2d, self).__init__(ref, cartesian_offset=cartesian_offset, track_orientation=track_orientation)

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
    def position2d(self) -> np.ndarray:
        return self._center[0:2]

    @property
    def position(self) -> np.ndarray:
        return self._center[0:3]

class RelativeCircle2d(RelativePoint2d):
    def __init__(self, ref, radius=None, **kwargs):
        assert radius is not None, "Please specify a radius"

        self._radius = radius
        super(RelativeCircle2d, self).__init__(ref, **kwargs)

    def _generate_info(self):
        info = super(RelativeCircle2d, self)._generate_info()
        info['radius'] = self.radius
        return info

    def contains(self, other):
        if type(other) == list or type(other) == tuple:
            center_distance = math.sqrt((other[0]-self.x)**2 + (other[1]-self.y)**2)
        else:
            center_distance = np.linalg.norm(self.position2d - other.position2d)

        return self._radius >= center_distance

    @property
    def radius(self):
        return self._radius

class RelativePoint3d(RelativePoint):
    def __init__(self, ref, x_offset=0, y_offset=0, z_offset=0, **kwargs):

        cartesian_offset = np.array([x_offset, y_offset, z_offset], dtype=np.float64)
        super(RelativePoint3d, self).__init__(ref, cartesian_offset=cartesian_offset, **kwargs)

    def _generate_info(self):
        info = {
            'x': self.x,
            'y': self.y,
            'z': self.z
        }

        return info

    # TODO implement
    def get_ref_rot_mat(self):
        raise NotImplementedError

    def get_ref_center(self):
        return self.ref.position

    @property
    def x(self):
        return self._center[0]

    @property
    def y(self):
        return self._center[1]

    @property
    def z(self):
        return self._center[2]
    
    @property
    def position(self):
        return self._center


class RelativeCylinder(RelativePoint3d):
    def __init__(self, ref, x_offset=0, y_offset=0, z_offset=0, radius=None, height=None, **kwargs):

        self._radius = radius
        self._height = height
        
        super(RelativeCylinder, self).__init__(ref,**kwargs)

    def _generate_info(self):
        info = super(RelativeCylinder, self)._generate_info()
        info['radius'] = self.radius
        info['height'] = self.height
        return info

    def contains(self, other):
        if type(other) == list or type(other) == tuple:
            radial_distance = math.sqrt((other[0]-self.x)**2 + (other[1]-self.y)**2)
            axial_distance = abs(self.z - other[2])
        else:
            radial_distance = np.linalg.norm(self.position[0:2] - other.position[0:2])
            axial_distance = abs(self.position[2] - other.position[2])

        is_contained = ( radial_distance <= self._radius ) and ( axial_distance <= (self._height / 2) )
        return is_contained

    @property
    def radius(self):
        return self._radius

    @property
    def height(self):
        return self._height

if __name__=='__main__':

    class test_ref:
        def __init__(self, x, y, orientation):
            self.x = x
            self.y = y
            self.orientation = orientation

    ref = test_ref(10, 345, 0)
    rp = RelativeCircle2d(ref, radius=150, track_orientation=True, r_offset=100, aspect_angle=(60*math.pi/180))

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

def distance(a,b):
    distance = np.linalg.norm( a.position - b.position )
    return distance