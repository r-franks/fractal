import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from fractal.animation import animate
from tqdm import tqdm

# Appollonian window via chaos game
# Method from http://lagrange.math.siu.edu/Kocik/apollo/achaos/chaos-explain.htm

def from_2d(coords):
    # Converts (?, 2) array to two (?,) arrays
    return coords[:,0], coords[:,1]

def to_2d(x, y):
    # Converts two (?,) arrays to a (?, 2) array
    return np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1)

def f1(coords):   
    x, y = from_2d(coords)
    return to_2d(x, -y)

def f2(coords):    
    x, y = from_2d(coords)
    denom = 16*x**2 + 16*y**2 - 8*y + 1
    x_new = x / denom
    y_new = (4*x**2 + 4*y**2 - y) / denom
    return to_2d(x_new, y_new)

def f3(coords):
    x, y = from_2d(coords)
    denom = (x-1)**2 + (y-1)**2
    x_new = (x**2 + y**2 - x - 2*y + 1) / denom  
    y_new = (x**2 + y**2 - 2*x - y + 1) / denom
    return to_2d(x_new, y_new)

def f4(coords):
    x, y = from_2d(coords)
    denom = (x+1)**2 + (y-1)**2
    x_new = -(x**2 + y**2 + x - 2*y + 1) / denom
    y_new = (x**2 + y**2 + 2*x - y + 1) / denom
    return to_2d(x_new, y_new)

def chaos_map(coords):
    # apply chaos game iteration to all 1x2 arrays
    # in coords (a ?x2 array)
    
    n = coords.shape[0]
    new_coords = np.empty(coords.shape, dtype=coords.dtype)
    
    # randimly assign each coord a chaos game func
    f_choice = np.random.randint(1, 5, size=n)
    f1_ind = (f_choice == 1)
    f2_ind = (f_choice == 2)
    f3_ind = (f_choice == 3)
    f4_ind = (f_choice == 4)
    
    # apply appropriate chaos game funcs
    new_coords[f1_ind,:] = f1(coords[f1_ind,:])
    new_coords[f2_ind,:] = f2(coords[f2_ind,:])
    new_coords[f3_ind,:] = f3(coords[f3_ind,:])
    new_coords[f4_ind,:] = f4(coords[f4_ind,:])
    
    return new_coords

def chaos_map_iter(iters=10, n_inputs_reused=True, dtype="float32", seed=1234, prog_bar=True):
    """Generates Appollonian Gasket via Chaos Game
    
    Parameters
    ----------
    iters: int
        Number of game iterations
    n_inputs_reused: int
        how many prev inputs to reuse
        - higher n_inputs_reused makes chaos game
          worse but more parallelized
        - n_inputs_reused < 0 means all inputs are
          reused
    seed: int
        Seed for replicability
    dtype: str ("float16"/"float32"/"float64")
        Precision of chaos game
        
    Returns
    -------
    x: (2**iters, 2) numpy array if reuse_input==True
       (iters+1, 2) numpy array if reuse_input==False
        Coords of points on Appollonian Gasket
    """
    
    np.random.seed(1234)

    x = np.random.uniform(size=2).reshape(-1,2).astype(dtype)

    x_list = []
    x_list.append(x)
    
    if prog_bar:
        iter_range = tqdm(range(iters))
    else:
        iter_range = range(iters)

    for i in iter_range:
        if n_inputs_reused < 0:
            x_new = chaos_map(np.concatenate(x_list))
            x_list.extend(list(x_new.reshape(-1,1,2)))
        elif n_inputs_reused == 0:
            x_list.append(chaos_map(x_list[-1]))
        elif n_inputs_reused > 0:
            total_inputs = len(x_list)
            start_index = max(total_inputs - 1 - n_inputs_reused, 0)
            x_new = chaos_map(np.concatenate(x_list[start_index:total_inputs]))
            x_list.extend(list(x_new.reshape(-1,1,2)))
    
    return np.concatenate(x_list)