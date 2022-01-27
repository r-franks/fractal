import numpy as np

def reg_poly(n, orientation):
    """Returns coordinates of a regular polygon
    
    Parameters
    ----------
    orientation: float
        The angle/orientation of the polygon
    
    n: positive integer
        number of sides
        
    Returns
    -------
    center: 1d numpy array of two elements
        center of the polygon
    vertices: list of 1d numpy arrays of two elements
        vertices of the polygon, going counterclockwise
    """
    # calculate n angles evenly spaces between zero and 2pi and then add the orientation 
    angs = orientation + np.linspace(start=0, stop=2*np.pi*(n-1)/n, num=n) 
    
    # define the center of the polygon as the origin
    center = np.array([0,0]).reshape(1,-1)
    
    # calculate x-y coords where each angle meets the unit circle
    vertices = np.concatenate( [np.cos(angs).reshape(-1,1), np.sin(angs).reshape(-1,1)], axis = 1 )
    return center, vertices

def iterate_shape(center, vertices, iters, scaling=0.5, rotation=0):
    """Replaces vertices of shape with scaled and 
    rotated copies of the original shape. Can repeat
    process many times.
    
    Parameters
    ----------
    center: 1x2 numpy array
        The center of the original shape
    vertices: nx2 numpy array
        shape points
    iters: int
        number of times to repeat process
    scaling: float
        how much point-replacing copies
        of shape get scaled down from prev
        iteration
    rotation: float
        how much point-replacing copies
        of shape get rotated relative to
        prev iteration
    
    Yields
    -------
    all_coords: nx2 numpy array
        array of new points in shape generated
        by applied the transformations
    """
    
    # initialize all_coords with the center
    all_coords = center.reshape(1,-1)
    # count number of vertices
    n_vertices = vertices.shape[0]

    # calculate vars for rotation matrix
    if rotation != 0:
        cos = np.cos(rotation)
        sin = np.sin(rotation)
            
    # for each iteration...
    for i in range(iters):
        # define an array to contain the next iteration of coordinates
        new_coords = np.ndarray(shape = (all_coords.shape[0]*n_vertices, 2))
        
        # create counter to record where in the new_coords array to input coords
        counter = 0
        # rescale the vertices
        vertices = vertices * scaling
        # rotate the vertices if necessary
        if rotation != 0:
            vertices = np.array( [cos*vertices[:,0] - sin*vertices[:,1],
                                  sin*vertices[:,0] + cos*vertices[:,1]] ).T
            
        # for every coordinate in all_coords, 
        # record coords of a shape with vertices centered at that coordinate
        for coord in all_coords:
            for vertex in vertices:
                new_coords[counter] = coord + vertex
                counter += 1
        
        # update all_coords with new_coords
        all_coords = new_coords
        yield all_coords