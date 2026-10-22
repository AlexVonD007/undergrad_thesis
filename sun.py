import numpy as np
from rot import rot_2d, cos_dir, rot3d, unitary_sun_reflection
from utilidades import v2l, d2r, xyz2thetaphi
import pandas as pd
from pvlib import solarposition
from pvlib import location
from datetime import date, timedelta
from scipy.spatial import Delaunay
# from serach_optim import robust_ray_intersection

def sun(vectors, time, steps):
    phi_step = (time / 12) * np.pi / steps  
    rotated_vectors=np.zeros_like(vectors)
 
    step_vectors = {}                          

    for step in range(steps+1):
        # Calcula los ángulos de rotación para el paso actual
        phi = step * phi_step

        # Calcula y almacena los vectores rotados para el paso actual
        rotated_vectors = [rot_2d(vec, -phi) for vec in vectors]
        step_vectors[f"step_{step}"] = np.array(rotated_vectors) # Guarda los vectores en el diccionario con una clave específica para cada paso

    return step_vectors

def sun2(pos, time, steps, radius):
    phi=(time/12) * np.pi / steps 
    rot_vec=np.zeros_like(pos)
    step_vec={}
    for step in range(steps+1):
        phi_step=step * phi
        for j in range(len(pos)):
            # rot_vec[j]=[pos[j, 0], pos[j, 1], radius*np.sin(phi_step)]
            rot_vec[j] = [0, radius * np.cos(-phi_step) , radius * np.sin(-phi_step)]
            rot_vec[j] += pos[j]
            rot_vec[j]=np.linalg.norm(rot_vec[j])
        step_vec[f"step_{step}"] = np.array(rot_vec)

    return step_vec

def sun3(pos, time, steps, azimut, radius):    

    if isinstance(azimut, list):
        pass
    else:
        azimut=d2r(azimut)
        azi=(azimut/steps)
    vec=np.zeros_like(pos)
    vec_a=np.zeros_like(pos)
    phi=(time/12) * np.pi / steps 
    phi= (time * np.pi)/(12*steps)
    

    z=pos[:,2]
    z2=z.max()
    z=z.min()    
    x=pos[:,0]
    x=x.min()
    y=pos[:,1]
    y2=y.max()
    y=y.min()
    

    step_vec={}
    for step in range(steps +1):
        if isinstance(azimut, list):
            azimut_step=d2r(azimut[step])
        else:
            azimut_step=  step*azi
        phi_step=step * phi

        ra=np.cos(phi_step)
        rb=np.sin(phi_step)

        za=np.cos(azimut_step)
        zb=np.sin(azimut_step)
        # print(za, zb, r2d(azimut_step), r2d(azi))
        for i in range(len(pos)):
            if step==0:
                vec[i] = [ 3*radius*(1+za), (-y2-radius)*(ra+zb), 0.5*z + 0.5*z2 + z2*rb]
            else:
                # if z==0:
                #     z=radius
                # if round(y, 0)==0:
                #     y=-y2
                vec[i] = [ 3*radius*(1+za) ,(-y2-radius)*(ra+zb), z2*(1+rb)]
        # print(vec[2,2], vec[2,1],y2,y )
        step_vec[f"step_{step}"] = np.array(vec)
    return step_vec 

def sun_reflection(sun_vec, vec_surface):
    """
    Relfexxión de los vectores solares sobre una superficie
    
    sun_vec: Diccionario con los vectores solares por paso
    
    vec_surface: Vectores de la superficie sobre la que se refleja
    
    Retorna un diccionario con los vectores reflejados por paso"""
    out={}
    out_vec=np.zeros_like(vec_surface)
    for i in range(len(sun_vec)):
        vec=sun_vec[f"step_{i}"]        
        for j in range(len(vec_surface)):
            degree=np.nan_to_num(cos_dir(vec[j], vec_surface[j]))
            out_vec[j]=v2l(rot3d(vec[j], (degree)))
            # out_vec[j]=reflect_vector(vec[j], vec_surface[j])
        out[f"step_{i}"]=np.array(out_vec)

        # Comparar vectores para verificar si son iguales o distintos
    for i in range(1, len(out)):
        if np.array_equal(out[f"step_{i}"], out[f"step_{i-1}"]):
            print(f"Warning: Vectors at step {i} are identical to step {i-1}")

    return out

def intensity_per_hour(intensity_value_per_hour, matrix):
    intensity_h={}
    for i in range(len(intensity_value_per_hour)):
        intensity_h[f"step_{i}"] = np.full_like(matrix, intensity_value_per_hour[i])
    return intensity_h


def normal_distribution(matrix, intensity_list):
    
    x,y = np.shape(matrix)

    x_center, y_center = x //2, y//2

    sun_intensity={}
    for i in range(len(intensity_list)):
        normal_matrix=np.zeros_like(matrix)
        for j in range(x):
            for k in range(y):
                distance = np.sqrt((j - x_center)**2 + (k - y_center)**2)
                normal_matrix[j, k] = intensity_list[i] / (1 + distance)
        sun_intensity[f"step_{i}"]=normal_matrix
    return sun_intensity

def final_point(vector, origin):
    point = np.zeros_like(origin)
    for i in range(len(vector)):
        point[i] = [vector[i, 0] + origin[i, 0], vector[i, 1] + origin[i, 1], vector[i, 2] + origin[i, 2]]
    return point

def point_in_surface(x_s, y_s, z_s, points,intensity, tol=1e-3):
    x_min, x_max = x_s.min(), x_s.max()
    y_min, y_max = y_s.min(), y_s.max()
    z_plane = z_s.min()

    xp, yp, zp = points

    if (x_min - tol <= xp <= x_max + tol) and \
       (y_min - tol <= yp <= y_max + tol) and \
       abs(zp - z_plane) <= tol:
        dentro = intensity
    else:
        dentro = 0

    return dentro


def ray_count(reflected_vec, origin_points, intensity_list):
    # Reescalamiento
    rescale=np.abs((origin_points[:,2].min() - origin_points[:, 2] ) / reflected_vec[:,2])
    rescaled_vec=reflected_vec *rescale[:, np.newaxis]
    final_points=final_point(rescaled_vec, origin_points)
    rays=np.zeros(len(final_points))
    for i in range(len(final_points)):
        rays[i]=point_in_surface(origin_points[:,0], origin_points[:, 1], origin_points[:, 2], final_points[i], intensity_list[i])
    return rays

def daterange(start_date: date, end_date: date):
    days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + timedelta(n)

def solar_position(latitude: float, longitude: float, altitdue: float, times: pd.DatetimeIndex, start_date: date, end_date: date):
    """
    Creación de un df que tiene los valores de posición solar e intensidad por hora para un rango de tienpo dado.
    
    latitude: Latitud del lugar (float)
    
    longitude: Longitud del lugar (float)
    
    altitdue: Altitud del lugar (float)
    
    times: Tiempos de la simulación (pd.DatetimeIndex)
    
    start_date: Fecha de inicio del rango (date)
    
    end_date: Fecha de fin del rango (date)
    
    Retorna un DataFrame con las columnas: 'Date', 'time', 'azimuth', 'apparent_elevation', 'irradiance'"""
    sol_pos = solarposition.get_solarposition(times, latitude, longitude, altitdue)
    sol_pos = sol_pos.loc[sol_pos['apparent_elevation'] > 0, :]
    sol_pos_copy = sol_pos.copy()
    solpos_organized = pd.DataFrame(columns=['Date','time', 'azimuth', 'apparent_elevation'])

    for single_date in daterange(start_date, end_date):
        temp_df = sol_pos_copy.copy()
        temp_df=temp_df.loc[f'{single_date}']

        # print(f'{single_date}')
        for j in range(len(temp_df)):
            time = temp_df.index[j]
            temp_df2 = temp_df.copy()
            temp_df2 = temp_df2.loc[temp_df.index[j]]

            azimuth = temp_df2['azimuth']
            elevation = temp_df2['apparent_elevation']
            new_row = [single_date.strftime('%Y-%m-%d'), time.strftime('%H:%M:%S'), azimuth, elevation]
            solpos_organized.loc[len(solpos_organized)] = new_row
    return solpos_organized

def solar_dome(sun_df: pd.DataFrame, n: int, angular_aperture: float):
    """
    Generación de un domo solar a partir de un DataFrame con la posición solar y la irradiancia.
    
    sun_df: DataFrame con las columnas 'azimuth', 'apparent_elevation' e 'irradiance'.
    
    n: Número de puntos en la malla del domo.
    
    angular_aperture: Apertura angular en grados para la distribución de la irradiancia.
    
    Retorna las mallas de azimut, elevación, la irradiancia distribuida en el domo y las coordenadas X, Y , Z del domo."""
    copy = sun_df.copy()
    az = np.deg2rad(copy['azimuth'].values)
    el = np.deg2rad(copy['apparent_elevation'].values)
    irr = copy['irradiance'].values

    x_s = np.cos(el) * np.sin(az)
    y_s = np.cos(el) * np.cos(az)
    z_s = np.sin(el)
    sun_dirs = np.vstack([x_s, y_s, z_s]).T

    phi = np.linspace(0, 2*np.pi, n)
    ele = np.linspace(0, np.pi/2, n)
    phi_grid, ele_grid = np.meshgrid(phi, ele)

    x = np.cos(ele_grid) * np.sin(phi_grid)
    y = np.cos(ele_grid) * np.cos(phi_grid)
    z = np.sin(ele_grid)
    mesh_dirs = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    sigma = np.deg2rad(angular_aperture)
    irr_map = np.zeros(mesh_dirs.shape[0])
    for i in range(len(irr)):
        dot_prod = mesh_dirs @ sun_dirs[i]  # cos(theta)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        ang_dist = np.arccos(dot_prod)  # theta

        weight = np.exp(-(ang_dist**2) / (2 * sigma**2))
        irr_map += (irr[i] / np.max(irr)) * weight

    irr_map = irr_map.reshape(ele_grid.shape)



    return phi_grid, ele_grid, irr_map, x, y, z



def sun_pos(times: pd.DatetimeIndex, lat: float, long: float, alt: float):
    """
    Obtiene la posición solar para un conjunto de tiempos y coordenadas geográficas.
    
    times: Serie de tiempo (pd.DatetimeIndex).
    
    lat: Latitud (float).
    
    long: Longitud (float).
    
    alt: Altitud (float).
    
    Returns: DataFrame con la posición solar (azimuth y elevación aparente) filtrada por elevación aparente positiva."""
    solpos = solarposition.get_solarposition(times, latitude=lat, longitude=long, altitude=alt)
    solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]
    return solpos


def get_irradiance(times: pd.DatetimeIndex, lat: float, long: float, alt: float, tz = 'America/Bogota', profile ='dni'):
    """
    Obtiene la irradiancia solar directa para un conjunto de tiempos y coordenadas geográficas.
    
    times: Serie de tiempo (pd.DatetimeIndex).
    
    lat: Latitud (float).
    
    long: Longitud (float).
    
    alt: Altitud (float).
    
    tz: Zona horaria (str, por defecto 'America/Bogota').
    
    Returns: DataFrame con la irradiancia solar directa (DNI) para los tiempos especificados."""

    site_location =  location.Location(lat, long, tz=tz, altitude=alt)

    # Obtener datos de cielo despejado
    clearsky = site_location.get_clearsky(times)
    # Posición solar
    solar_position = site_location.get_solarposition(times)
    
    # Filtrar solo cuando el Sol está sobre el horizonte (elevación aparente > 0)
    mask = solar_position['apparent_elevation'] > 0
    filtered_times = times[mask]

    # Aplicar el modelo solo en esos tiempos
    clearsky_day = clearsky.loc[filtered_times]
    
    return pd.DataFrame({profile.upper(): clearsky_day[profile]})


def better_sun_df(sun_df: pd.DataFrame, start_date: date, end_date: date, irrandiance_df: pd.DataFrame, profile = 'DNI'):
    """
    Organiza el DataFrame de posición solar y el DataFrame de irradiancia en un formato más manejable.
    
    sun_df: DataFrame con la posición solar (azimuth y elevación aparente).
    
    start_date: Fecha de inicio (date).
    
    end_date: Fecha de fin (date).
    
    irrandiance_df: DataFrame con la irradiancia normal direct (DNI).
    
    Returns: DataFrame organizado con columnas 'Date', 'time', 'azimuth', 'apparent_elevation' e 'irradiance'."""


    solpos_organized = pd.DataFrame(columns=['Date','time', 'azimuth', 'apparent_elevation', 'irradiance'])
    sun_copy = sun_df.copy()

    for single_date in daterange(start_date, end_date):
        temp_df = sun_copy.copy()
        temp_df=temp_df.loc[f'{single_date}']
        irr_copy = irrandiance_df.copy()
        irr_copy = irr_copy.loc[f'{single_date}']

        # print(f'{single_date}')
        for j in range(len(temp_df)):
            time = temp_df.index[j]
            temp_df2 = temp_df.copy()
            temp_df2 = temp_df2.loc[temp_df.index[j]]

            irr_copy2 = irr_copy.copy()
            irr_copy2 = irr_copy2.loc[irr_copy.index[j]]

            azimuth = temp_df2['azimuth']
            elevation =  temp_df2['apparent_elevation']
            irradiance_value = irr_copy2[profile]
            new_row = [single_date.strftime('%Y-%m-%d'), time.strftime('%H:%M:%S'), azimuth, elevation, irradiance_value]
            solpos_organized.loc[len(solpos_organized)] = new_row
        # print(elevation)
    return solpos_organized

def sky_discretization(df : pd.DataFrame, n_bins_elevation : int, n_bins_azimtuh : int, irr_col_name= None):
    """
    Discretiza el cielo en parches para facilitar el análisis.
    
    df: DataFrame con los datos de la posición solar y su irradiancia (DNI).
    
    n_bins_elevation: Número de divisiones para la elevación aparente (int).
    
    n_bins_azimuth: Número de divisiones para el azimut (int).
    
    irr_col_name: Nombre de la columna de irradiancia (str, opcional).
    
    Returns: DataFrame con los parches discretizados y la irradiancia total por parche.
    """

    if irr_col_name is None:
        irr_col_name = 'irradiance'
    df_bins = df.copy()
    df_bins['apparent_elevation_bins'] = pd.cut(df['apparent_elevation'], bins= n_bins_elevation)
    df_bins['azimuth_bins'] = pd.cut(df['azimuth'], bins= n_bins_azimtuh)
    irr_bins = df_bins.groupby(['apparent_elevation_bins', 'azimuth_bins'], observed=True).agg({
        f'{irr_col_name}' : 'sum'
    }).reset_index()
    irr_bins = irr_bins.rename(columns={
        f'{irr_col_name}' : 'irr_bin_total'
    })
    df_bins = df_bins.merge(irr_bins, on =['apparent_elevation_bins', 'azimuth_bins'], how='left')
    df_bins['apparent_elevation'] =  (df_bins[f'{irr_col_name}'] / df_bins['irr_bin_total']) * df_bins['apparent_elevation']
    df_bins['azimuth'] =  (df_bins[f'{irr_col_name}'] / df_bins['irr_bin_total']) * df_bins['azimuth']
    df_bins = df_bins.groupby(['apparent_elevation_bins', 'azimuth_bins'], observed=True).agg({
        f'irr_bin_total' : 'first',
        'apparent_elevation' : 'sum',
        'azimuth' : 'sum'
    }).reset_index()
    return df_bins


def generar_puntos_divisiones(c: list, d: float, n_div: int):
    """
    Genera los puntos de una estructura con extremos y subdivisiones en las aristas.
    
    c: Coordenads XYZ del punto original (numpy array)

    d: distancia maxima del centor a los extremos (float)

    n_div: Numero de divisiones a generar (int)

    Retorna un arrya con los puntos paralelos.
    """

    x, y, z = c

    # --- Extremos base ---
    extremos = np.array([
        [x-d, y+d, z-d],
        [x+d, y-d, z+d],
        [x+d, y+d, z-d],
        [x-d, y-d, z+d]
    ])

    # --- Definir aristas a subdividir ---
    edges = [
        (0,2), (0,3), (1,2), (1,3)
    ]

    puntos = []
    # Añadimos los extremos
    for e in extremos:
        puntos.append(e)

    # Subdivisiones en las aristas
    for i,j in edges:
        p1, p2 = extremos[i], extremos[j]
        for k in range(1, n_div):  # evita repetir extremos
            alpha = k/(n_div)      # fracción
            p = (1-alpha)*p1 + alpha*p2
            puntos.append(p)




    return np.array(puntos)

def parallel_vecs(parallel_points : np.ndarray, sun_vecs: np.ndarray):
    """
    Dada una lista de puntos paralelos y un vector solar, genera vectores paralelos para cada punto.
    
    parallel_points: puntos paralelos (Arraylike)

    sun_vecs: vector solar (Arraylike)
    """
    sun_vecs_pro = []
    for i in range(len(parallel_points)):
        sun_vecs_temp = []
        for j in range(len(parallel_points[i])):
            sun_vecs_temp.append(1 * sun_vecs[i])
        sun_vecs_pro.append(sun_vecs_temp)
    
    return np.array(sun_vecs_pro)


# def self_reflection(surface_vector : np.ndarray, init_surface_idx : int, 
#                     vector : np.ndarray, origin : np.ndarray, tri_idx : int, triangles : Delaunay, points_ : np.ndarray,baricentric_points : np.ndarray, n_reflections = 3, is_vector= False):
    
#     """
#     Evalua la capacidad de un vector de tener autorefelxiones sobre la superficie

#     surface_vector: Lista con todos los vectores normales de la superficie (Arraylike)

#     init_surface_idx: Indice del vector normal de la superficie (int)

#     vector: Vector sobre el cual se evalua la autoreflexion (Arraylike)

#     origin: Punto de origen del vector sobre el cual se evalua la autoreflexion (Arraylike)

#     tri_idx: Indice del triguno inicial sobre el cual se empieza la busqueda (int)

#     triangles: elementos de malla de la superficie (Delaunay)

#     points_: Conjunto de puntos XYZ de la superficie (Arraylike)

#     baricentric_points: Conjunto de los centroides donde se originan los vectores normales (Arraylike)

#     n_reflections: Numero maximo de autorefleciones a tener, por defecto 3 (int)

#     is_vector: si el vector es de la forma (3,N) es false (valor por defecto), si es de la forma (3,) se debe de cambiar a True  (Bool)


#     Retorna: el vector autoreflejado, un boleano que indica si hubo autoreflexion, el indice del elemento de origen del vector autoreflejado y  el numero de autoreflexiones presentes


#     """
#     ref_count = 0
#     reflection = 0

    
#     # print(np.shape(vector))
#     if is_vector == False:
#         vector = np.squeeze(vector)

#     # print(np.shape(points_3DS))
#     # print(points_[161])
#     while ref_count <= n_reflections:
#         if ref_count == 0:
#             hit_idx, _ = robust_ray_intersection(origin, vector, triangles, points_, tri_idx)
#             if hit_idx != -1:
#                 reflection = 1
#                 reflected_vec = unitary_sun_reflection(vector,surface_vector[init_surface_idx])
#             else:
#                 ref_count = reflection + 1
#                 reflected_vec = vector
#         else:
#             new_origin = baricentric_points[hit_idx]
#             # new_sup_vec = [0 - baricentric_points[hit_idx, 0], 0 - baricentric_points[hit_idx, 1], 0 - baricentric_points[hit_idx, 2]]
#             new_sup_vec = surface_vector[hit_idx]
#             new_sup_vec = np.array(new_sup_vec)
#             hit_idx, _ = robust_ray_intersection(new_origin, reflected_vec, triangles, points_, tri_idx)
#             reflected_vec = unitary_sun_reflection(new_sup_vec, reflected_vec)
#         # print(hit_idx)
#         ref_count += 1


#     return reflected_vec, reflection, hit_idx, ref_count
