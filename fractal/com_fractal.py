import numpy as np
from fractal.geometry_fractal import reg_poly

def com_tris_from_tri(tri):
    """Calculates center of mass given vertices
    of a triangle, returns three new triangles
    with the center of mass as one of the 
    vertices
    
    Parameters
    ----------
    tri: 3x2 numpy array
        Three 2d vertices describing a triangle
 
    Returns
    -------
    all_tri: (3x2)x3 numpy array
        Three 3x2 numpy arrays representing triangles
    """
    
    # calculate center of mass
    com = tri.sum(axis=0)/3
    
    # create triangles with one vertex replaced by center of mass
    tri1 = np.concatenate([tri[0], tri[2], com]).reshape(3,2,1)
    tri2 = np.concatenate([tri[1], tri[0], com]).reshape(3,2,1)
    tri3 = np.concatenate([tri[2], tri[1], com]).reshape(3,2,1)

    all_tri = np.concatenate([tri1, tri2, tri3], axis=2)
    # combine triangles into array
    return all_tri

def com_tris_from_array(tris):
    """Calculates a center of mass for vertices
    of each triangle, returns three new triangles
    for each triangle provided -- produced by
    replacing one of each triangle's vertices
    with the location of its original center
    of mass
    
    Parameters
    ----------
    tri: (3x2)x(n) numpy array
        n 3x2 numpy arrays representing triangles
 
    Returns
    -------
    new_tris: (3x2)x(3n) numpy array
        3*n 3x2 numpy arrays representing triangles
    """

    tris_coms = tris.sum(axis=(0))/3

    new_tris = []
    for com in range(tris_coms.shape[1]):
        new_tri1 = np.concatenate([tris[0,:,com], tris[2,:,com], tris_coms[:,com]]).reshape(3,2,1)
        new_tri2 = np.concatenate([tris[1,:,com], tris[0,:,com], tris_coms[:,com]]).reshape(3,2,1)
        new_tri3 = np.concatenate([tris[2,:,com], tris[1,:,com], tris_coms[:,com]]).reshape(3,2,1)
        new_tris.append(np.concatenate([new_tri1, new_tri2, new_tri3], axis=2))

    new_tris = np.concatenate(new_tris, axis=2)
    return new_tris

def com_from_triangle(triangle, iters=1):
    """Takes a triangle and an iteration number
    to produces a center of mass fractal
    
    Parameters
    ----------
    tri: 3x2 numpy array
        Three 2d vertices describing a triangle
    iters: int
        Number of iteractions
 
    Returns
    -------
    fractal: (3**(iters+1))x2 numpy array
        3**(iters+1) 2d vertices associated with fractal
    """
 
    com_tris = com_tris_from_tri(triangle)

    for i in range(iters):
        com_tris = com_tris_from_array(com_tris)
    fractal = np.concatenate([com_tris[0,0,:].reshape(-1,1), com_tris[0,1,:].reshape(-1,1)], axis=1)
    return fractal

def com_fractal(iters=1):
    """Produces a center of mass fractal
    
    Parameters
    ----------
    iters: int
        Number of iteractions
 
    Returns
    -------
    fractal: (3**(iters+1))x2 numpy array
        3**(iters+1) 2d vertices associated with fractal
    """

    c,v = reg_poly(3, np.pi/2)
    fractal = com_from_triangle(v, iters=iters)
    return fractal
