import abc
import math
import numpy as np
import copy
from scipy.spatial.transform import Rotation
from saferl.environment.models.platforms import BaseEnvObj

POINT_CONTAINS_DISTANCE = 1e-10


def geo_from_config(cls, config):
    assert issubclass(cls, BaseGeometry) or issubclass(cls, RelativeGeometry)
    obj = cls(**config)
    return obj


class BaseGeometry(BaseEnvObj):

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

    @property
    def velocity(self):
        return np.array([0, 0, 0], dtype=np.float64)

    @abc.abstractmethod
    def contains(self, other):
        ...

    @abc.abstractmethod
    def _generate_info(self):
        ...


class Point(BaseGeometry):

    def __init__(self, x=0, y=0, z=0):
        self._center = np.array([x, y, z], dtype=np.float64)

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
        assert isinstance(value, np.ndarray) and value.shape == (
        3,), "Position must be set in a numpy ndarray with shape=(3,)"
        self._center = copy.deepcopy(value)

    @property
    def orientation(self):
        # always return a no rotation quaternion as points do not have an orientation
        return Rotation.from_quat([0, 0, 0, 1])

    @orientation.setter
    def orientation(self, value):
        # simply pass as points do not have an orientation
        pass

    def contains(self, other):
        distance = np.linalg.norm(self.position - other.position)
        is_contained = distance < POINT_CONTAINS_DISTANCE
        return is_contained

    def _generate_info(self):
        info = {
            'x': self.x,
            'y': self.y,
            'z': self.z,
        }

        return info


class Circle(Point):

    def __init__(self, x=0, y=0, z=0, radius=1):
        self.radius = radius

        super().__init__(x=x, y=y, z=z)

    def contains(self, other):
        radial_distance = np.linalg.norm(self.position[0:2] - other.position[0:2])
        is_contained = radial_distance <= self.radius
        return is_contained

    def _generate_info(self):
        info = super()._generate_info()
        info['radius'] = self.radius

        return info


class Sphere(Circle):

    def contains(self, other):
        distance = np.linalg.norm(self.position - other.position)
        is_contained = distance <= self.radius
        return is_contained


class Cylinder(Circle):

    def __init__(self, x=0, y=0, z=0, radius=1, height=1):
        self.height = height

        super().__init__(x=x, y=y, z=z, radius=radius)

    def contains(self, other):
        radial_distance = np.linalg.norm(self.position[0:2] - other.position[0:2])
        axial_distance = abs(self.position[2] - other.position[2])

        is_contained = (radial_distance <= self.radius) and (axial_distance <= (self.height / 2))
        return is_contained

    def _generate_info(self):
        info = super()._generate_info()
        info['height'] = self.height

        return info


class RelativeGeometry(BaseEnvObj):

    def __init__(self,
                 ref,
                 shape,
                 track_orientation=False,
                 x_offset=None,
                 y_offset=None,
                 z_offset=None,
                 r_offset=None,
                 theta_offset=None,
                 aspect_angle=None,
                 euler_decomp_axis=None,
                 init=None,
                 **kwargs):

        # check that both x_offset and y_offset are used at the same time if used
        assert (x_offset is None) == (y_offset is None), \
            "if either x_offset or y_offset is used, both x_offset and y_offset must be used"

        # check that only r_offset, theta_offset or x/y/z offset are specified
        assert ((r_offset is not None) or (theta_offset is not None) or (aspect_angle is not None)) != (
                    (x_offset is not None) or (y_offset is not None)), \
            "user either polar or x/y relative position definiton, not both"

        # check that only theta_offset or aspect_angle is used
        assert ((theta_offset is None) or (
                    aspect_angle is None)), "specify either theta_offset or aspect_angle, not both"

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
        self.euler_decomp_axis = euler_decomp_axis

        self._cartesian_offset = np.array([x_offset, y_offset, z_offset], dtype=np.float64)

        self.shape = shape

        if init is None:
            self.init_dict = {}
        else:
            self.init_dict = init

        self.ref.register_dependent_obj(self)
        self.update()

    def update(self):
        ref_orientation = self.ref.orientation

        if self.euler_decomp_axis == 'z':
            raise NotImplementedError
        elif self.euler_decomp_axis is not None:
            raise ValueError("Invalid euler_decomp_axis {}".format(self.euler_decomp_axis))

        offset = ref_orientation.apply(self._cartesian_offset)

        self.shape.position = self.ref.position + offset

        if self.track_orientation:
            self.shape.orientation = self.ref.orientation

    def step(self, *args, **kwargs):
        self.update()

    def reset(self):
        self.update()

    def _generate_info(self):
        return self.shape._generate_info()

    @property
    def velocity(self):
        return self.ref.velocity

    @property
    def x(self):
        return self.shape.x

    @property
    def y(self):
        return self.shape.y

    @property
    def z(self):
        return self.shape.z

    @property
    def position(self):
        return self.shape.position

    @property
    def orientation(self) -> Rotation:
        return self.shape.orientation

    def contains(self, other):
        return self.shape.contains(other)


class RelativePoint(RelativeGeometry):
    def __init__(self,
                 ref,
                 track_orientation=False,
                 x_offset=None,
                 y_offset=None,
                 z_offset=None,
                 r_offset=None,
                 theta_offset=None,
                 aspect_angle=None,
                 init=None,
                 **kwargs):
        shape = Point(**kwargs)

        super().__init__(
            ref,
            shape,
            track_orientation=track_orientation,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
            r_offset=r_offset,
            theta_offset=theta_offset,
            aspect_angle=aspect_angle,
            init=init,
        )


class RelativeCircle(RelativeGeometry):

    def __init__(self,
                 ref,
                 track_orientation=False,
                 x_offset=None,
                 y_offset=None,
                 z_offset=None,
                 r_offset=None,
                 theta_offset=None,
                 aspect_angle=None,
                 init=None,
                 **kwargs):
        shape = Circle(**kwargs)

        super().__init__(
            ref,
            shape,
            track_orientation=track_orientation,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
            r_offset=r_offset,
            theta_offset=theta_offset,
            aspect_angle=aspect_angle,
            init=init
        )

    @property
    def radius(self):
        return self.shape.radius


class RelativeSphere(RelativeGeometry):

    def __init__(self,
                 ref,
                 track_orientation=False,
                 x_offset=None,
                 y_offset=None,
                 z_offset=None,
                 r_offset=None,
                 theta_offset=None,
                 aspect_angle=None,
                 init=None,
                 **kwargs):
        shape = Sphere(**kwargs)

        super().__init__(
            ref,
            shape,
            track_orientation=track_orientation,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
            r_offset=r_offset,
            theta_offset=theta_offset,
            aspect_angle=aspect_angle,
            init=init
        )

    @property
    def radius(self):
        return self.shape.radius


class RelativeCylinder(RelativeGeometry):
    def __init__(self,
                 ref,
                 track_orientation=False,
                 x_offset=None,
                 y_offset=None,
                 z_offset=None,
                 r_offset=None,
                 theta_offset=None,
                 aspect_angle=None,
                 init=None,
                 **kwargs):
        shape = Cylinder(**kwargs)

        super().__init__(
            ref,
            shape,
            track_orientation=track_orientation,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
            r_offset=r_offset,
            theta_offset=theta_offset,
            aspect_angle=aspect_angle,
            init=init
        )

    @property
    def radius(self):
        return self.shape.radius

    @property
    def height(self):
        return self.shape.height


def distance(a, b):
    return np.linalg.norm(a.position - b.position)
