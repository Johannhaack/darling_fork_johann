# How do I deal with the extraction of the translation vectors, can I make that stable? I have some problems with the chi phy when the y are vectors

# What do we do with the masks, I talked with Ari and he has a good system against overlaps, if we do it like this we need to pass all the data which might make ti difficutl for Ram if we have a lot of grains, this also solves the problem with the dark_scan which we might not have


'''
    Here the question is what should be the input to this should we make some 3D ourself outside of darling or should there be a 3D Darling functionality? Passing a list of dset is not what we want as our Ram gets shitty => There needs to be a utility function that saves that np array of the 3D, this could be for mask, mosa, moment, gaussian, could be in properties => This process cant be optimesed as the fitting is using cpu parallisation already with numba

    Also how do we deal with the mask as it switches and is arbitrary between different experiments- especially as we dont want do reload everthing from the properties file => 

'''

import numpy as np
import scipy.ndimage 
import pyvista as pv
import IPython
import colorsys

import darling

def _generate_distinct_colors(n):
    """This is made to generate differentiable colours for the point clouds"""
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = [colorsys.hsv_to_rgb(h, 0.8, 0.9) for h in hues]
    return colors

def create_local_coordinate_system(size, pixel_size, aspect):
    
    x_size, y_size, z_size = size
    pixel_size_x = pixel_size[0] * aspect
    pixel_size_y = pixel_size[1]
    pixel_size_z = pixel_size[2] # Check with Adam if this is correct

    # Coordinate center (image center, not grain center)
    center_x = x_size // 2
    center_y = y_size // 2
    center_z = z_size // 2

    # Create coordinate grids
    x_coords = (np.arange(x_size) - center_x) * pixel_size_x
    y_coords = (np.arange(y_size) - center_y) * pixel_size_y
    z_coords = (np.arange(z_size) - center_z) * pixel_size_z  # center z-values around center slice


    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # Stack into (x, y, z, 3)
    grain_volume_coordinates = np.stack((X, Y, Z), axis=-1)    

    return grain_volume_coordinates

def coordinate_translation(translation_vector, grain_volume_coordinates):
    # translation_vector: (3,) -> [dx, dy, dz]
    # grain_volume_coordinates: (x, y, z, 3) where the last dimension are the coordinates
    
    # Apply the translation to the last dimension
    grain_volume_coordinates_translated = grain_volume_coordinates + np.array(translation_vector)
    
    return grain_volume_coordinates_translated

def coordinate_rotation(angle_vector, grain_volume_coordinates):
    
    # grain_volume_coordinates: shape (x, y, z, 3)
    
    # angle_vector: [mu, omega, chi, phi] in radians ... i am passing them in radians
    
    mu, omega, chi, phi = angle_vector

    # Rotation matrices from the image
    Rotation_mu = np.array([
        [np.cos(mu),  0, np.sin(mu)],
        [0,           1, 0],
        [-np.sin(mu), 0, np.cos(mu)]
    ])

    Rotation_omega = np.array([
        [np.cos(omega), np.sin(omega), 0],
        [-np.sin(omega), np.cos(omega), 0],
        [0,              0,             1]
    ])

    Rotation_chi = np.array([
        [1, 0,           0],
        [0, np.cos(chi), np.sin(chi)],
        [0, -np.sin(chi), np.cos(chi)]
    ])

    Rotation_phi = np.array([
        [np.cos(phi),  0, -np.sin(phi)],
        [0,            1, 0],
        [np.sin(phi),  0, np.cos(phi)]
    ])

    # Total rotation matrix: R = M * Omega * X * Phi
    R = Rotation_mu @ Rotation_omega @ Rotation_chi @ Rotation_phi

    # Reshape to apply rotation to last axis => The matrix multiplication can only take 2D Arguments
    
    flat_coords = grain_volume_coordinates.reshape(-1, 3)
    
    rotated_coords = flat_coords @ R.T  # Apply rotation
    
    grain_volume_coordinates_rotated = rotated_coords.reshape(grain_volume_coordinates.shape)
    
    return grain_volume_coordinates_rotated

def overlapping_mask_assignment():

    pass

def get_grain_points_list(global_coordinates, mask):
    """This function takes a 3D mask [x,y,z] and the global coordinates system [x,y,z,3] and returns a list of points.
        2D expects an input of z=1
    """
    #check if the x,y,z dimesnion is the same
    if global_coordinates.shape[0] != mask.shape[0]:
        raise ValueError("The x dimension of the global coordinates and the mask do not match")
    if global_coordinates.shape[1] != mask.shape[1]:
        raise ValueError("The y dimension of the global coordinates and the mask do not match")
    if global_coordinates.ndim ==3 or mask.ndim ==2:
        raise ValueError("2D Visualization is possible, when an 1 dimensional z dimension is added")
    if global_coordinates.shape[2] != mask.shape[2]:
        raise ValueError("The z dimension of the global coordinates and the mask do not match")
    
    return global_coordinates[mask]

def plot_n_grains_volume(point_list, grain_names = None, point_density=0.05):
    #I dont like this, would be greatr if they are bundled automatically, so the system or not mixed by accident, so you just pass the points here, and we do another function that get the points list
    """This function takes a list of list of 3D points, each inner list corresponds to a grain, and plots them in 3D using pyvista

    Args:
        point_list (list): List of list of 3D points
        grain_names (list): Corresponding Grain Names of the 3D points clouds, must correspind to the order of the list
        point_density (float): There is a point limit where pyvista works, with the desnity we can choose how much of that max point number we want
    
    """
    if IPython.get_ipython() is not None:
        pv.start_xvfb()  # Needed for headless Jupyter
        pv.set_jupyter_backend('trame') #trame

        notebook = True
    else:
        notebook = False

    print(type(point_list))
    if grain_names is not None and len(point_list) == len(grain_names):
        raise ValueError("Number of points cloud must match the grains names")  

    #module to reduce points
    max_points = 500000
    point_per_grain = max_points / len(point_list)*point_density
    sampled_points =[]

    for cloud in point_list:
        print(cloud.shape[0])
        if cloud.shape[0] < point_per_grain:
            point_per_grain = cloud.shape[0]

        idx = np.random.choice(cloud.shape[0], size=int(point_per_grain), replace=False)
        points_sampled = cloud[idx]
        sampled_points.append(points_sampled)


    colors = _generate_distinct_colors(len(sampled_points))

    plotter = pv.Plotter(notebook=notebook)


    legend_entries = []

    if grain_names is None:
        grain_names = []
        for i in range(len(sampled_points)):
            grain_names.append(f"Grain {i}")

    for i, (points, color) in enumerate(zip(sampled_points, colors)):
        cloud = pv.PolyData(points)
        plotter.add_mesh(cloud, color=color, render_points_as_spheres=True, point_size=3, label=grain_names[i])
        legend_entries.append([grain_names[i], color])

    plotter.add_legend(legend_entries)

    plotter.show() 


def get_translation_vector(reader, scan_ids):


    #This is only for uz as we expect ux and  the rest to be the same for the scans

    ux_values_list = []
    uy_values_list = []
    uz_values_list = []

    for scan_id in scan_ids:
        motor_values_scan = reader.config(scan_id)["motor_values"]
        ux_values_list.append(motor_values_scan["ux"])
        uy_values_list.append(motor_values_scan["uy"])
        uz_values_list.append(motor_values_scan["uz"])

    ux = np.mean(np.array(ux_values_list))
    uy = np.mean(np.array(uy_values_list))
    uz = np.mean(np.array(uz_values_list))
    #Here we are swiching y and x as the naming on the detector has first axis as y, the transformations are made on x being the first axis though
    translation_vector = np.array([uy*1000, ux*1000, uz*1000])

    return translation_vector

def get_rotation_angles(reader, scan_ids):

    motor_values = reader.config(scan_ids[0])["motor_values"]

    #Lets get phi and chi question is how we get it but it should be a list
    chi = np.mean(np.array(motor_values["chi"]))
    phi = np.mean(np.array(motor_values["phi"]))

    #For now they should stay constant in the scans
    omega = np.mean(np.array(motor_values["omega"]))
    mu = np.mean(np.array(motor_values["mu"]))
    rotation_vector = np.array([np.radians(mu), np.radians(omega), np.radians(chi), np.radians(phi)])

    return rotation_vector

def get_aspect_ratio(reader, scan_ids):

    #The aspect should stay the same for now
    motor_values = reader.config(scan_ids[0])["motor_values"]

    tth = motor_values["obpitch"]
    aspect = 1/np.tan(np.radians(tth))

    return aspect

def transform_to_global_coordinate_system(reader, detector_size, pixel_size, scan_ids, translation_vector, rotation_vector, aspect):

    """The pixel size needs to be in microns"""
    if scan_ids is None:
        #look for the scan ids in the reader
        scan_ids = darling.metadata.get_ordered_scan_ids(reader.abs_path_to_h5_file)

        if len(scan_ids) == 0:
            Exception("No scan ids provided, please provide them still need to implement automatic search for it")    
        

    #The aspect should stay the same for now
    if aspect is None:
        aspect = get_aspect_ratio(reader, scan_ids)
    if translation_vector is None:
        translation_vector = get_translation_vector(reader, scan_ids)
    if rotation_vector is None:
        rotation_vector = get_rotation_angles(reader, scan_ids)

    size_x, size_y = detector_size  #We have data as y,x,z but the y is on 1 axis in that by Detector convention so we switch it to x as first axis only consequence is the change of the translation x and y 
    size_z = len(scan_ids)

    size = (size_x, size_y, size_z)

    #Create the local coordinate system
    initial_coord_system = create_local_coordinate_system(size, pixel_size, aspect)

    rotated_coord_system = coordinate_rotation(rotation_vector, initial_coord_system)
    #Apply the translation
    final_coord_system = coordinate_translation(translation_vector, rotated_coord_system)
    

    return final_coord_system






    



    






