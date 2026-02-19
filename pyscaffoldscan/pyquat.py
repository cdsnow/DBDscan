from numpy import array, dot, negative, pi, zeros, sqrt, degrees, trace, arccos, sign, cos, sin, cross, radians, ndarray
from numpy.random import randn
import numpy
from functools import reduce

try:
    import colorama
    F_YELLOW = colorama.Fore.YELLOW
    F_RESET = colorama.Fore.RESET
except:
    F_YELLOW = ''
    F_RESET = ''

xtal_quaternions = { \
    'norot':(1, 0, 0, 0),
    'rot180x':(0, 1, 0, 0),
    'rot180y':(0, 0, 1, 0),
    'rot180z':(0, 0, 0, 1),
    'rot120+x+y+z':(0.5, 0.5, 0.5, 0.5),
    'rot120+x+y-z':(0.5, 0.5, 0.5, -0.5),
    'rot120+x-y+z':(0.5, 0.5, -0.5, 0.5),
    'rot120-x+y+z':(0.5, -0.5, 0.5, 0.5),
    'rot120+x-y-z':(0.5, 0.5, -0.5, -0.5),
    'rot120-x+y-z':(0.5, -0.5, 0.5, -0.5),
    'rot120-x-y+z':(0.5, -0.5, -0.5, 0.5),
    'rot120-x-y-z':(0.5, -0.5, -0.5, -0.5),
    'rot180+sqrt3x+y':(0.0, 0.8660254037844386, 0.5, 0.0),
    'rot180+sqrt3x-y':(0.0, 0.8660254037844386, -0.5, 0.0),
    'rot180+sqrt3y+x':(0.0, 0.5, 0.8660254037844386, 0.0),
    'rot180+sqrt3y-x':(0.0, -0.5, 0.8660254037844386, 0.0),
    'rot60z':(0.8660254037844386, 0.0, 0.0, 0.5),
    'rot60-z':(0.8660254037844386, 0.0, 0.0, -0.5),
    'rot120z':(0.5, 0.0, 0.0, 0.8660254037844386),
    'rot120-z':(0.5, 0.0, 0.0, -0.8660254037844386),
    'rot180x+y':(0.0, 0.7071067811865476, 0.7071067811865476, 0.0),
    'rot180x-y':(0.0, 0.7071067811865476, -0.7071067811865476, 0.0),
    'rot180x+z':(0.0, 0.7071067811865476, 0.0, 0.7071067811865476),
    'rot180x-z':(0.0, 0.7071067811865476, 0.0, -0.7071067811865476),
    'rot180y+z':(0.0, 0.0, 0.7071067811865476, 0.7071067811865476),
    'rot180y-z':(0.0, 0.0, 0.7071067811865476, -0.7071067811865476),
    'rot90x':(0.7071067811865476, 0.7071067811865476, 0.0, 0.0),
    'rot90-x':(0.7071067811865476, -0.7071067811865476, 0.0, 0.0),
    'rot90y':(0.7071067811865476, 0.0, 0.7071067811865476, 0.0),
    'rot90-y':(0.7071067811865476, 0.0, -0.7071067811865476, 0.0),
    'rot90z':(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
    'rot90-z':(0.7071067811865476, 0.0, 0.0, -0.7071067811865476),
}

qkey2qname = { \
 (0, 100000, 0, 0): 'rot180x',
 (0, 0, 100000, 0): 'rot180y',
 (0, 0, 0, 100000): 'rot180z',
 (0, 0, 70711, -70711): 'rot180y-z',
 (0, 0, 70711,  70711): 'rot180y+z',
 (0, 70711, -70711, 0): 'rot180x-y',
 (0, 70711,  70711, 0): 'rot180x+y',
 (0, 70711, 0, -70711): 'rot180x-z',
 (0, 70711, 0,  70711): 'rot180x+z',
 (0, -50000,  86603, 0): 'rot180+sqrt3y-x',
 (0,  50000,  86603, 0): 'rot180+sqrt3y+x',
 (0,  86603, -50000, 0): 'rot180+sqrt3x-y',
 (0,  86603,  50000, 0): 'rot180+sqrt3x+y',
 (50000, 0, 0, -86603): 'rot120-z',
 (50000, 0, 0,  86603): 'rot120z',
 (50000, -50000, -50000, -50000): 'rot120-x-y-z',
 (50000, -50000, -50000,  50000): 'rot120-x-y+z',
 (50000, -50000,  50000, -50000): 'rot120-x+y-z',
 (50000, -50000,  50000,  50000): 'rot120-x+y+z',
 (50000,  50000, -50000, -50000): 'rot120+x-y-z',
 (50000,  50000, -50000,  50000): 'rot120+x-y+z',
 (50000,  50000,  50000, -50000): 'rot120+x+y-z',
 (50000,  50000,  50000,  50000): 'rot120+x+y+z',
 (70711,  70711, 0, 0): 'rot90x',
 (70711, -70711, 0, 0): 'rot90-x',
 (70711, 0,  70711, 0): 'rot90y',
 (70711, 0, -70711, 0): 'rot90-y',
 (70711, 0, 0,  70711): 'rot90z',
 (70711, 0, 0, -70711): 'rot90-z',
 (86603, 0, 0,  50000): 'rot60z',
 (86603, 0, 0, -50000): 'rot60-z',
 (100000, 0, 0, 0): 'norot'}

qkey2axisname = { \
 (0, 100000, 0, 0): 'x',
 (0, 0, 100000, 0): 'y',
 (0, 0, 0, 100000): 'z',
 (0, 0, 70711, -70711): 'y-z',
 (0, 0, 70711,  70711): 'y+z',
 (0, 70711, -70711, 0): 'x-y',
 (0, 70711,  70711, 0): 'x+y',
 (0, 70711, 0, -70711): 'x-z',
 (0, 70711, 0,  70711): 'x+z',
 (0, -50000,  86603, 0): 'sqrt3y-x',
 (0,  50000,  86603, 0): 'sqrt3y+x',
 (0,  86603, -50000, 0): 'sqrt3x-y',
 (0,  86603,  50000, 0): 'sqrt3x+y',
 (50000, 0, 0, -86603): '-z',
 (50000, 0, 0,  86603): 'z',
 (50000, -50000, -50000, -50000): '-x-y-z',
 (50000, -50000, -50000,  50000): '-x-y+z',
 (50000, -50000,  50000, -50000): '-x+y-z',
 (50000, -50000,  50000,  50000): '-x+y+z',
 (50000,  50000, -50000, -50000): '+x-y-z',
 (50000,  50000, -50000,  50000): '+x-y+z',
 (50000,  50000,  50000, -50000): '+x+y-z',
 (50000,  50000,  50000,  50000): '+x+y+z',
 (70711,  70711, 0, 0): 'x',
 (70711, -70711, 0, 0): '-x',
 (70711, 0,  70711, 0): 'y',
 (70711, 0, -70711, 0): '-y',
 (70711, 0, 0,  70711): 'z',
 (70711, 0, 0, -70711): '-z',
 (86603, 0, 0,  50000): 'z',
 (86603, 0, 0, -50000): '-z',
 (100000, 0, 0, 0): 'norot'}


def Norm(v):
    """Get the norm of a vector"""
    return dot(v, v)**0.5 

def RotMatTheta(R):
    """How many degrees does the provided rotation matrix rotate by"""
    return degrees(arccos(0.5*(trace(R) - 1))) 

def PerpUnitVec(va, vb, verbose=False):
    """Return a unit vector perpendicular to the provided vectors"""
    if not isinstance(va, ndarray): va = array(va)
    if not isinstance(vb, ndarray): vb = array(vb)
    perpaxis = cross(va, vb)
    if verbose: print('Cross product', perpaxis)
    if max(abs(perpaxis)) < 0.0001: 
        testaxes = list(map(array, [[1,0,0],[0,1,0],[0,0,1]]))
        for testaxis in testaxes:
            if dot(testaxis, va) < 0.999: break 
        if verbose: print('Picking an arbitrary perpendicular axis, based on',testaxis)
        perpaxis = testaxis - dot(testaxis,va) * va
    perpaxis = perpaxis / Norm(perpaxis)
    assert dot(perpaxis, va) < 0.001
    assert dot(perpaxis, vb) < 0.001
    if verbose: print('Normalized perpendicular axis = ',perpaxis)
    return perpaxis

def VecVecQuat(va, vb, verbose=False):
    """Given two vectors, return the quaternion necessary to align va with vb""" 
    if not isinstance(va, ndarray): va = array(va,dtype='float')
    else: va = va.astype('float')
    if not isinstance(vb, ndarray): vb = array(vb,dtype='float')
    else: vb = vb.astype('float')
    if verbose: print('Working on',va,'with',vb)
    va = va / Norm(va)
    vb = vb / Norm(vb)
    if verbose: print('Normalized',va,'and',vb)
    if verbose: print('dot product of va,vb =',dot(va,vb))

    perpaxis = PerpUnitVec(va, vb, verbose=verbose)

    tmp = dot(va,vb)
    if tmp < -1.0: tmp = -1.0
    if tmp > 1.0: tmp = 1.0
    theta = -degrees(arccos(tmp))
    if verbose: print('theta =', theta)
    Q = Quat((theta,perpaxis))
    trial_va = dot(va, Q.R())
    if verbose: 
        print('Quaternion',Q)
        print(Q.AsThetaAxis())
        print('rotates the normalized va',va)
        print('to va',trial_va)
        print('which compares with the normalized vb',vb)
        print('with a dot product of',dot(trial_va,vb))
    assert dot(trial_va, vb) > 0.9990
    return Quat((theta,perpaxis))

class Quat:
    "Quaternion tools for moving proteins about"
  
    def __init__(self, arg):
        """Convert various inputs into a unit quaternion"""
        if isinstance(arg, str) and arg in xtal_quaternions: ## A request for a named quaternion
            self.q = xtal_quaternions[arg]
        elif len(arg)==4: ## A quaternion
            self.q = array(arg)
        elif isinstance(arg,ndarray) and arg.shape == (3,3):  ## A rotation matrix
            rotmat = arg.round(9)
            self.from_rotmat(rotmat)
        elif len(arg)==2 and len(arg[1])==3:  ## A rotation theta, and axis
            self.from_theta_axis(arg)
        elif arg == 'random':  ## Request for a random unit quaternion 
            self.q = randn(4)
        else:
            assert False, 'Unrecognized input with which to create a quaternion'

        self.Normalize()
        self.Standardize()

        ## Check if the resulting quaternion is a named crystallographic quaternion
        self.name = 'unassigned'
        self.qkey = tuple(int(round(x)) for x in 100000*self.q) 
        try: self.name = qkey2qname[self.qkey]
        except KeyError: pass ## No matching crystallographic quaternion
        self.axis = 'unassigned'
        try: self.axis = qkey2axisname[self.qkey]
        except KeyError: pass ## No matching crystallographic quaternion
        self.shape = (1,4)

    def __repr__(self):
        return 'Quat([%r, %r, %r, %r])'% tuple(self.q)

    def __str__(self):
        reprstr = F_YELLOW + 'Quat([' + F_RESET
        reprstr += '%5.2f, %5.2f, %5.2f, %5.2f' % tuple(self.q) 
        reprstr += F_YELLOW + '])' + F_RESET
        return reprstr

    def __len__(self):
        return len(self.q)

    def __getitem__(self,index):
        return self.q[index]
 
    def __mul__(self, other):
        assert isinstance(other,Quat), 'Not sure how to multiple a quaternion by anything other than a quaternion'
        return self.TimesQ(other)

    def __pow__(self, other):
        assert isinstance(other,int)
        if other == 0: return Quat((1,0,0,0))
        elif other < 0:
            return reduce(lambda x,y: x*y, [self.I()]*-other)
        else:
            return reduce(lambda x,y: x*y, [self]*other)

    def __eq__(self, other, epsilon=0.999):
        ## Note: having an __eq__ function makes Quat objects non-hashable
        if self.name == 'unassigned':
            if isinstance(other, self.__class__):
                return self.RelativeQ(other)[0] > epsilon
        else: return self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def Clone(self):
        """Copy the quaternion"""
        return Quat(self.q)

    def Normalize(self):
        """Normalize to produce a unit quaternion"""
        n = dot(self.q, self.q)**0.5
        self.q = self.q / n
  
    def Standardize(self):
        """Standardize signs for the quaternion terms"""
        if self.q[0]<0:  self.q = negative(self.q)
        ## Get rid of any negative zero (they return False to < 0 queries)
        if self.q[0]==0: self.q[0]=0 
        if self.q[1]==0: self.q[1]=0 
        if self.q[2]==0: self.q[2]=0 
        if self.q[3]==0: self.q[3]=0 
      
    def from_rotmat(self, R):
        """Convert a 3x3 rotation matrix into a quaternion"""
        q0 = 0.25 * (R[0][0] + R[1][1] + R[2][2] + 1)
        q1 = 0.25 * (R[0][0] - R[1][1] - R[2][2] + 1)
        q2 = 0.25 * (-R[0][0] + R[1][1] - R[2][2] + 1)
        q3 = 0.25 * (-R[0][0] - R[1][1] + R[2][2] + 1)
        if q0 < 0: q0 = 0
        if q1 < 0: q1 = 0
        if q2 < 0: q2 = 0
        if q3 < 0: q3 = 0
        q0, q1, q2, q3 = sqrt(q0), sqrt(q1), sqrt(q2), sqrt(q3)
        if q0 >= q1 and q0 >= q2 and q0 >= q3:
            q1 = q1 * sign(R[2][1] - R[1][2])
            q2 = q2 * sign(R[0][2] - R[2][0])
            q3 = q3 * sign(R[1][0] - R[0][1])
        elif q1 >= q0 and q1 >= q2 and q1 >= q3:
            q0 = q0 * sign(R[2][1] - R[1][2])
            q2 = q2 * sign(R[1][0] + R[0][1])
            q3 = q3 * sign(R[0][2] + R[2][0])
        elif q2 >= q0 and q2 >= q1 and q2 >= q3:
            q0 = q0 * sign(R[0][2] - R[2][0])
            q1 = q1 * sign(R[1][0] + R[0][1])
            q3 = q3 * sign(R[2][1] + R[1][2])
        elif q3 >= q0 and q3 >= q1 and q3 >= q2:
            q0 = q0 * sign(R[1][0] - R[0][1])
            q1 = q1 * sign(R[2][0] + R[0][2])
            q2 = q2 * sign(R[2][1] + R[1][2])
        else: assert False
      
        self.q = array([q0,q1,q2,q3])

    def from_theta_axis(self, theta_axis):
        """Convert a rotation in degrees and an axis into a unit quaternion"""
        theta, v = theta_axis
        theta = radians(theta)
        w = cos(theta/2)
        unitv = v / Norm(v)
        unitv = unitv * sin(theta/2)
        self.q = array([w,unitv[0],unitv[1],unitv[2]])
        self.q = self.q / Norm(self.q)

    def Axis(self, verbose=True):
        """Return the quaternion axis as an array"""
        if self.q[0] == 1 or self.name=='norot': 
            return array([0,0,0])
        theta = 2*arccos(self.q[0])
        axis = self.q[1:].copy()
        axis = axis / sin(theta/2)
        return array(axis)

    def AsThetaAxis(self, verbose=True):
        """Describe the quaternion in terms of degree rotation about an axis"""
        if self.q[0] == 1: 
            if verbose: print('No rotation')
            return(0, (0,0,0))
        theta = 2*arccos(self.q[0])
        axis = self.q[1:].copy()
        axis = axis / sin(theta/2)
        if axis[0] == 0 and axis[1] == 0 and axis[2] == 1: axis = 'z'
        elif axis[0] == 1 and axis[1] == 0 and axis[2] == 0: axis = 'x'
        elif axis[0] == 0 and axis[1] == 1 and axis[2] == 0: axis = 'y'
        elif axis[0] == 0 and axis[1] == 0 and axis[2] == -1: axis = '-z'
        elif axis[0] == -1 and axis[1] == 0 and axis[2] == 0: axis = '-x'
        elif axis[0] == 0 and axis[1] == -1 and axis[2] == 0: axis = '-y'
        if verbose: print('%g degree rotation about the %s axis' % (degrees(theta), str(axis)))
        return (degrees(theta), axis)

    def R(self):
        """Produce a 3x3 rotation matrix corresponding to the quaternion"""
        a, b, c, d = self.q
        R = zeros([3,3])
        R[0][0] = a*a + b*b - c*c - d*d
        R[0][1] = 2*(b*c - a*d)
        R[0][2] = 2*(b*d + a*c)
        R[1][0] = 2*(b*c + a*d)
        R[1][1] = a*a - b*b + c*c - d*d
        R[1][2] = 2*(c*d - a*b)
        R[2][0] = 2*(b*d - a*c)
        R[2][1] = 2*(c*d + a*b)
        R[2][2] = a*a - b*b - c*c + d*d
        return R

    def __call__(self, coords):
        if hasattr(coords, 'Rotate'):
            coords.Rotate(self)
            return 
        assert coords.shape[1] == 3, 'Supplied argument does not look like a numpy array with 3D coordinates ' + str(coords)
        ## Transpose to get the right handedness
        return numpy.dot(coords, self.R().T)
    
    def Theta(self, otherquat):
        """Calculate the relative rotation theta to another quaternion in degrees.
        Note: this serves as a distance metric.
        """
        dw = self.q[0]*otherquat.q[0] + dot(self.q[1:], otherquat.q[1:])
        if abs(dw - 1.0) < 0.000001: return 0.0
        theta = 2*arccos(dw)
        if theta > pi: theta = abs(theta - 2*pi)
        return degrees(theta)

    def TimesQ(self, qB):
        """Calculate the quaternion that results from quaternion multiplication"""
        w = self.q[0]*qB.q[0] - dot(self.q[1:], qB.q[1:])
        v = self.q[0]*qB.q[1:] + qB.q[0]*self.q[1:] + cross(self.q[1:], qB.q[1:])
        return Quat([w,v[0],v[1],v[2]])

    def I(self):
        """Get the inverse of the unit quaternion"""
        X = Quat([self.q[0], -self.q[1], -self.q[2], -self.q[3]])
        return X

    def RelativeQ(self, qB):
        """Calculate the relative quaternion to get to the provided quaternion"""
        qA = self.I()
        return qA.TimesQ(qB)

    def VecVecDiff(self, va, vb):
        """Given two vectors, return the quaternion necessary to align va with vb""" 
        perp = cross(va, vb)
        d = dot(va, vb)
        w = (dot(va,va)*(dot(vb,vb)))**0.5 + d
        return Quat([w,perp[0],perp[1],perp[2]])


if __name__ == '__main__':
    for k, v in xtal_quaternions.items():
        q = Quat(v)
        print('%20s  =>'%q.name, q)       

