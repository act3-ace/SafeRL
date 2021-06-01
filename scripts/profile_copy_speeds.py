import os
import sys
import cProfile
import pstats
import numpy as np
import copy

from scipy.spatial.transform import Rotation as R

#manual copy
def copy_array(arr):
    new = np.zeros(arr.shape)
    for pos in range(arr.shape[0]):
        new[pos] = arr[0]
    return new



x = np.ones((3,5))


# shallow copy
#cProfile.run('copy.copy(x)','shallow_copy_profile')

#deepcopy
#cProfile.run('copy.deepcopy(x)','deepcopy_profile')

#numpy.copy
#cProfile.run('np.copy(x)','numpy_copy_profile')

#ndarray.copy
#cProfile.run('x.copy()','ndarray_copy_profile')

#manual copy
#result,is_ok = copy_array(x)

#cProfile.run('copy_array(x)','manual_copy_profile')

#print(result)
#print(is_ok)

#y = x.__deepcopy__()
#print(y)
# profile Array creation


#ndarray creation
#cProfile.run('np.array([[1,19,3,4,5],[1,2,3,4,5],[4,5,6,7,8],[1,2,3,4,5],[3,4,5,6,7]])','np.ndarray_creation_profile')


#cProfile.run('R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])','quaternion_rotation_profile')

#r = R.from_quat([0,0,np.sin(np.pi/4),np.cos(np.pi/4)])
#r.as_quat()


#cProfile.run('copy.deepcopy(r.as_quat())','quaternion_deepcopy_profile')
#cProfile.run('r.as_quat()','quaternion_new_instance')

#cProfile.run('np.arange(100)','np.arange_profile')

#cProfile.run('np.linspace(1.,4.,6)','np.linspace_profile')

#cProfile.run('np.zeros((100,100))','np.zeros_profile')

#cProfile.run('np.ones((100,100))', 'np.ones_profile')

#cProfile.run('np.empty((100,100))','np.empty_profile')

# benchmark scipy rotation speeds 


#r = R.from_quat([0,0,np.sin(np.pi/4),np.cos(np.pi/4)])

#v = [1,2,3]
#result = r.apply(v)
#print(result)

#cProfile.run('r.apply(v)','apply_rotation_profile')


#r1 = R.from_euler('z',90,degrees=True)
#r2 = R.from_rotvec([np.pi/4,0,0])

v = [1,2,3]

#cProfile.run('r1.apply(v)','r1_rotation_profile')

#cProfile.run('r2.apply(v)','r2_rotation_profile')

#r3 = R.from_quat([np.sin(np.pi/4),np.cos(np.pi/4),0,0])
#cProfile.run('r3.apply(v)','r3_rot_prof')
r4 = R.from_quat([-0.35812921, -0.15686761,  0.58684856, -0.70904498])
#cProfile.run('R.from_quat([-0.35812921, -0.15686761,  0.58684856, -0.70904498])','rotation_creation_profile')

r5 = R.from_euler('ZYX', [290, 40, 25], degrees=True)

cProfile.run("R.from_euler('ZYX', [290, 40, 25], degrees=True)",'euler_rot_profile')





