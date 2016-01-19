import numpy as np
from .motion import RigidMotion

__all__ = ['Body', 'TransformedBody', 'Pitching', 'Heaving',
           'cylinder', 'flat_plate', 'naca_airfoil', 'joukowski_foil',
           'van_de_vooren_foil', 'karman_trefftz_foil']

class Body(object):
    """Base class for representing bodies
    """
    def __init__(self, points):
        """Create a body with nodes at the given points

        Parameters
        ----------
        points : 2d array, shape (n,2)
            Array of points defining the boundary of the body
            For a closed body, the boundary curve should be positively oriented
            (counter-clockwise around outside of body), starting from trailing
            edge
        """
        self._time = 0
        self._points = np.array(points, dtype="float64")

    @property
    def time(self):
        """The time used to specify the body's motion"""
        return self._time

    @time.setter
    def time(self, value):
        self._time = value

    def get_points(self, body_frame=False):
        return self._points

    def get_body(self):
        """Return the Body object in the body-fixed frame"""
        return self

    def get_motion(self):
        """Return the transformation from the body-fixed to inertial frame"""
        return None

def cylinder(radius, num_points):
    """Return a circular Body with the given radius and number of points"""
    th = np.linspace(0, 2 * np.pi, num_points)
    points = radius * np.array([np.cos(th), np.sin(th)]).T
    return Body(points)

def flat_plate(num_points):
    """Return a flat plate Body with the given number of points.
    
    In body coordinates the plate runs from (0,0) to (1,0)."""
    x = np.linspace(1, 0, num_points) # from 1 to 0 so trailing edge at index 0
    y = np.zeros_like(x)
    return Body(np.array([x, y]).T)

def joukowski_foil(xcenter=-.1, ycenter=.1, a=1, num_points=32):
    """Return a Joukowski foil Body.
    
    The foil has its trailing edge at (2a,0).  The foil has a total of
    num_points along the boundary.  Refer to chapter 4 of [1]_ for details.

    Parameters
    ----------
    xcenter, ycenter : float
        (xcenter,ycenter) is the center of the Joukowski preimage circle.
        xcenter should be negative and small; its magnitude determines
        the bluffness of the foil.  ycenter should be small; it
        determines the magnitude of the camber (positive gives upward
        camber, and negative gives downward camber).

    a : float
        radius of the Joukowski preimage circle

    num_points : int
        number of points along the boundary

    References
    ----------
    .. [1] Acheson, D. J., "Elementary Fluid Dynamics", Oxford, 1990.
    """

    t = np.linspace(0,2*np.pi,num_points)
    r = np.sqrt((a-xcenter)**2+ycenter**2)
    chi = xcenter + r*np.cos(t)
    eta = ycenter + r*np.sin(t)
    mag2 = chi*chi + eta*eta
    x = chi*(1+a**2/mag2)
    y = eta*(1-a**2/mag2)
    return Body(np.array([x,y]).T)

def karman_trefftz_foil(xcenter=-.1, ycenter=0, a=.1, angle_deg=10, num_points=32):
    """Return a Karman-Trefftz foil Body.
    
    The Karman-Trefftz foil is a modified version of the Joukowski
    foil but with a nonzero interior angle --- rather than a cusp ---  at the 
    trailing edge.  Refer to [1]_ for details.
    
    Parameters
    ----------
    xcenter, ycenter, a : float.
        The same as in joukowski_foil().

    angle_deg : float
        The interior angle, in degrees, at the trailing edge.

    num_points : int
        Number of points along the boundary

    See Also
    --------
    joukowski_foil()

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Joukowsky_transform
    """

    angle_rad = angle_deg*np.pi/180
    n = 2-angle_rad/np.pi
    t = np.linspace(0,2*np.pi,num_points)
    ctr = xcenter + 1j*ycenter
    r = np.linalg.norm(ctr-a)
    zeta = ctr+r*np.exp(1j*t)
    mag2 = np.linalg.norm(zeta)
    z = n*((1+1/zeta)**n+(1-1/zeta)**n)/((1+1/zeta)**n-(1-1/zeta)**n)
    x = [w.real for w in z]
    y = [w.imag for w in z]
    return Body(np.array([x,y]).T)

def van_de_vooren_foil(semichord=1.0, thickness=0.15, angle_deg=5,
num_points=32):
    """Return a van de Vooren foil Body.

    Refer to section 6.6 of [1]_

    Parameters
    ----------
    semichord : float
        half the chord c, so c=2*semichord

    thickness : float
        vertical thickness as a fraction (0 < thickness < 1) of the semichord

    angle_deg : float
        interior angle, in degrees, at the trailing edge

    num_points : int
        number of points along the boundary

    References
    ----------
    .. [1] Katz, Joseph and Plotkin, Allen, "Low-Speed Aerodynamics", 2nd Ed.,
       Cambridge University Press, 2001.
    """

    k = 2-(angle_deg*np.pi/180)
    a = 2*semichord*((1+thickness)**(k-1))*2**(-k)
    t = np.linspace(0,2*np.pi,num_points)
    num = (a*(np.cos(t)-1)+1j*a*np.sin(t))**k
    den = (a*(np.cos(t)-thickness)+1j*a*np.sin(t))**(k-1)
    z = (num/den)+semichord
    x = [w.real for w in z]
    y = [w.imag for w in z]
    return Body(np.array([x,y]).T)

def naca_airfoil(code, num_points=20, te_clamp=False, uniform=False, chord=1):
    """Returns a NACA 4-digit series foil.

    Refer to [1]_ for a detailed description and formulas.

    Parameters:
    -----------
    code_str : string
        The 4-digit NACA code describing the foil.  If the first digit is M, 
        the second digit is P, and the last two digits are TH, then:
        M = maximum camber, as a percentage of the chord
        P = distance of max camber from leading edge in tens of percent of chord
        TH = thickenss of the airfoil as a percentage of chord
    
        For example, 2315 means a cambered airfoil whose max camber is 2% of the
        chord, located 30% chordwise from the leading edge, and whose thickness
        is 15% of the chord.
    
    num_points : int
        The number of points along each surface of the foil. There will be 
        2*num_points-1 points returned, with num_points along the edge and 
        num_points-1 along the bottom edge.
    
    te_clamp : boolean
        Trailing edge clamp.  When false, the trailing edge has a slight 
        nonzero thickness.  When true, the foil adjusted so that the trailing 
        edge has exactly zero thickness.
     
    uniform : boolean
        Distribution of points along chord.  When false, points are uniformly 
        distributd along the chord.  When true, points are distributed more 
        densly near leading edge (where the curvature is large).
    
    chord : double
        Length of chord
    
    Returns:
    --------
    A two-column matrix q, whose first column encodes the x coordinates and
    whose second column encodes the y coordinates of the points on the foil.
    The rows are ordered so that the points traverse the foil in the
    counterclockwise fashion from the trailing edge to the leading edge and
    back again.
    
    References:
    -----------
    .. [1] https://en.wikipedia.org/wiki/NACA_airfoil
    
    Example:
    --------
    q = naca_foil('2315',50)
 """
    code_str = "%04d" % int(code)
    if len(code_str) != 4:
        raise ValueError("NACA designation is more than 4 digits")
    m = .01*int(code_str[0])
    p = .1*int(code_str[1])
    thick = .01*int(code_str[2:4])
    if (uniform is True):
        x = np.linspace(0,chord,num_points)
    else:
        t = np.linspace(0,0.5*np.pi,num_points)
        x = chord*(1-np.cos(t))

    # Construct the thickness line, yt
    xc = x/chord
    coefs = [-.1015, .2843, -.3516, -.1260, 0]
    sqrt_coef = .2969
    if (te_clamp is True):
        coefs[0] = -.1036
    yt = 5*thick*chord*(sqrt_coef*np.sqrt(xc)+np.polyval(coefs,xc))

    # Avoid unnecessary work (and avoid divide-by-zero) when no camber
    if p:
        # Construct the camber line, yc
        front = np.where(x <= p*chord)
        back = np.where(x > p*chord)
        yc = np.zeros_like(x)
        yc[front] = (m/p**2)*x[front]*(2*p-xc[front])
        yc[back] = (m/(1-p)**2)*(chord-x[back])*(1+xc[back]-2*p)

        # Combine thickness and camber line to produce the NACA airfoil.
        # The usual formulas for the camber adjustment involve sin(arctan(.)) 
        # and cos(arctan(.)), but to avoid branch cuts in the arctangent 
        # function, I use instead the equivalent Cartesian formulations:
        # sin(arctan(x)) = x/sqrt(1+x^2)
        # cos(arctan(x)) = 1/sqrt(1+x^2)
        dycdx = np.zeros_like(x)
        dycdx[front] = (2*m/p**2)*(p-xc[front])
        dycdx[back] = (2*m/(1-p)**2)*(p-xc[back])
        z = np.sqrt(1+dycdx**2)
        xU = x - yt*dycdx/z # X coordinates along upper surface
        xL = x + yt*dycdx/z # X coordinates along lower surface
        yU = yc + yt/z      # Y coordinates along upper surface
        yL = yc - yt/z      # Y coordinates along lower surface
        x = np.hstack([xU[-1:0:-1],xL])
        y = np.hstack([yU[-1:0:-1],yL])
    else:
        x = np.hstack([x[-1:0:-1],x])
        y = np.hstack([yt[-1:0:-1],-yt])
    # Note: The [-1:0:-1] indexing above accomplishes two things: (1) it
    # reorders the vectors so the foil is described counterclockwise from the
    # trailing edge to the leading edge and back again, and (2) it omits the
    # repeated entry at the leading edge that would otherwise be present from
    # concatenating two copies of x.  It is important not to have two identical
    # successive entries, or else the matrix of influence coefficients in the
    # panel (boundary element) method becomes ill conditioned.

    return Body(np.array([x,y]).T)

class TransformedBody(object):
    """Base class for rigid (Euclidean) transformations of existing bodies
    """
    def __init__(self, body, angle=0, displacement=(0,0)):
        """angles are clockwise, in degrees"""
        self._parent = body
        self._body = body.get_body()
        self._motion = RigidMotion(-angle * np.pi / 180, displacement)

    def get_body(self):
        return self._body

    def get_motion(self):
        self._update()
        return self._motion.compose(self._parent.get_motion())

    def set_motion(self, value):
        self._motion = value

    @property
    def time(self):
        return self._body.time

    @time.setter
    def time(self, value):
        self._body.time = value

    def _update(self):
        # update body motion: subclasses override this
        pass

    def get_points(self, body_frame=False):
        q = self._body.get_points()
        if body_frame:
            return q
        return self.get_motion().map_position(q)


class Pitching(TransformedBody):
    """Sinusoidal pitching for an existing body
    """
    def __init__(self, body, amplitude, frequency, phase=0.):
        """amplitude and phase given in degrees"""
        super(Pitching, self).__init__(body)
        self._amplitude = amplitude * np.pi / 180
        self._frequency = frequency
        self._phase = phase * np.pi / 180

    def _update(self):
        theta = self._frequency * self.time + self._phase
        alpha = self._amplitude * np.sin(theta)
        alphadot = self._amplitude * self._frequency * np.cos(theta)
        self.set_motion(RigidMotion(-alpha, (0,0), -alphadot, (0,0)))


class Heaving(TransformedBody):
    """Sinusoidal heaving for an existing body
    """
    def __init__(self, body, displacement, frequency, phase=0.):
        super(Heaving, self).__init__(body)
        self._displacement = np.array(displacement, dtype="float64")
        self._frequency = frequency
        self._phase = phase * np.pi / 180

    def _update(self):
        theta = self._frequency * self.time + self._phase
        x = self._displacement * np.sin(theta)
        xdot = self._displacement * self._frequency * np.cos(theta)
        self.set_motion(RigidMotion(0, x, 0, xdot))
