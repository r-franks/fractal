import numpy as np

def triangle_grid(n_gridlines=21, n_points=100001):
    """Returns a cartesian grid inside a triangle
    
    Parameters
    ----------
    n_gridlines: int
        number of horizontal and vertical grid lines
    
    n_points: int
        number points defining each grid line
        
    Returns
    -------
    X_hor: 2d numpy array
        Coords of horizontal lines on grid
    X_vert: 2d numpy array
        Coords of vertical lines on grid
    """  
    nx, ny = (21, 100001)
    x = np.linspace(0, 1, n_gridlines)
    y = np.linspace(0, 1, n_points)
    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.reshape(-1,1), yv.reshape(-1,1)

    where = (yv <= 0.5 - np.abs(xv-0.5)).flatten()
    xv = xv[where,:]
    yv = yv[where,:]

    yv = 0.5*np.tan(np.radians(60))*yv/yv.max()

    X_vert = np.concatenate([xv, yv], axis=1)
    
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_gridlines)
    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.reshape(-1,1), yv.reshape(-1,1)

    where = (yv <= 0.5 - np.abs(xv-0.5)).flatten()
    xv = xv[where,:]
    yv = yv[where,:]

    yv = 0.5*np.tan(np.radians(60))*yv/yv.max()

    X_hor = np.concatenate([xv, yv], axis=1)
    
    return X_hor, X_vert

def shift_iso_triangle(triangle):
    """Shifts isosceles triangle
    with symmetry across y-axis so
    topmost point is at (0,0)
    
    Parameters
    ----------
    triangle: ?x2 numpy array
        Coordinates describing points
        in an isosceles triangle

    Returns
    -------
    shifted triangle: ?x2 numpy array
        Coordinates describing shifted
        points
    """

    x_min, x_max = triangle[:,0].min(), triangle[:,0].max()
    y_min, y_max = triangle[:,1].min(), triangle[:,1].max()
    height = y_max - y_min
    base = x_max - x_min
    mid = (x_max + x_min)/2

    x_shift = triangle[:, 0] - mid
    y_shift = triangle[:, 1] - y_max
    shifted_triangle = np.concatenate([x_shift.reshape(-1,1), y_shift.reshape(-1,1)], axis=1)
    return shifted_triangle

def stretch_iso_triangle(triangle):
    """Stretches isosceles triangle
    with symmetry across y-axis so
    so base=2*pi*height
    
    Parameters
    ----------
    triangle: ?x2 numpy array
        Coordinates describing points
        in an isosceles triangle

    Returns
    -------
    stretched_triangle: ?x2 numpy array
        Coordinates describing stretched
        points
    """
    
    x_min, x_max = triangle[:,0].min(), triangle[:,0].max()
    y_min, y_max = triangle[:,1].min(), triangle[:,1].max()
    height = y_max - y_min
    base = x_max - x_min
    
    xnew = (2*np.pi*height/base)*triangle[:, 0]
    stretched_triangle = np.concatenate([xnew.reshape(-1,1), triangle[:,1].reshape(-1,1)], axis=1)
    return stretched_triangle

# warp triangle
def warp_iso_triangle(triangle, warpage=1, custom_map=None):
    """Warps isosceles triangle
    with symmetry across y-axis
    into a circle
    
    Parameters
    ----------
    triangle: ?x2 numpy array
        Coordinates describing points
        in an isosceles triangle
    custom_map: numpy function (r, theta -> r', theta')
        Specifies option transformations

    Returns
    -------
    warped_triangle: ?x2 numpy array
        Coordinates describing
        warped points
    """
    triangle = triangle[triangle[:,1] != 0]
    
    r = -triangle[:,1]
    theta = -np.pi/2 + warpage*triangle[:,0]/triangle[:,1]
    
    if custom_map:
        r, theta = custom_map(r, theta)

    xnew = r*np.cos(theta)
    ynew = r*np.sin(theta)
    warped_triangle = np.concatenate([xnew.reshape(-1,1), ynew.reshape(-1,1)],axis=1)
    return warped_triangle

def unfold_iso_triangle(triangle, warpage=1, custom_map=None):
    """Warps isosceles triangle
    with symmetry across y-axis
    into a circle
    
    Parameters
    ----------
    triangle: ?x2 numpy array
        Coordinates describing points
        in an isosceles triangle
    custom_map: numpy function (r, theta -> r', theta')
        Specifies option transformations

    Returns
    -------
    warped_triangle: ?x2 numpy array
        Coordinates describing
        warped points
    """
    
    shifted_triangle = shift_iso_triangle(triangle)
    stretched_triangle = stretch_iso_triangle(shifted_triangle)
    warped_triangle = warp_iso_triangle(stretched_triangle, warpage=warpage, custom_map=custom_map)
    
    return warped_triangle