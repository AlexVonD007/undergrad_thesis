import numpy as np
import pandas as pd
import scipy as sp
from datetime import date, timedelta
from scipy.spatial import Delaunay
 


def daterange(start_date: date, end_date: date):
    days = int((end_date - start_date).days)
    for n in range(days):
        yield start_date + timedelta(n)


def d2r(ang: float):
    """
    Conveierte grafos a radianes.

    ang: Angulo en grados (float).

    Returns: Angulo en radianes (float).
    """
    rad= (ang/180)*np.pi
    return rad


def angle(v1,v2):
    dot = np.dot(v1,v2)
    mag1 =np.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    mag2 =np.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    phi=np.arccos(dot/(mag1*mag2))
    return phi


def r2d(rad: float):
    """
    Convierte radianes a grados.
    
    rad: Angulo en radianes (float).
    
    Returns: Angulo en grados (float)."""
    deg = (180/np.pi)*rad
    return deg


def mag(vec: np.ndarray):
    """
    Calcula la magnitud de un vector tridimensional.
    
    vec: Vector tridimensional (array-like o lista con 3 elementos).
    
    Returns: Magnitud del vector (float)."""
    m=np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    return m


def v2l(v: np.ndarray):
    """
    Convierte un vector tridimensional a una lista.
    
    v: Vector (Arraylike).
    
    Returns: Lista con los componentes del vector (list)."""
    return (v.T).tolist()


def test_creation(df):
    columns=df.columns
    c_l=list(columns)
    c_l.pop(0)
    new_columns=pd.Index(c_l)
    for i in new_columns:
        if i == new_columns[0]:
            new_df = df.groupby(columns[0])[i].apply(tuple).reset_index()
        else:
            x = df.groupby(columns[0])[i].apply(tuple).reset_index()
            new_df[i] = x[i]
    
    azi=new_df['Azimut (°)']
    iradiance=new_df['Irradiancia (W/m^2)']
    return azi, iradiance, new_df




def normal_tri(points: np.ndarray, triangles: Delaunay, test=False):
    """
    Calcula el vector normal de un triángulo dado su índice y los puntos de un mesh.

    points: Puntos del mesh (array-like con forma (n, 3) donde n es el número de puntos).

    triangles: Objeto de triangulación que contiene los triángulos (Delunay).

    Returns: Vector normal del triángulo (array de 3 elementos).
    """
    # idx = triangles.simplices[index]
    # p0, p1, p2 = points[idx]

    # A = p1 - p0
    # B = p2 - p0

    # # Ax = A[0]
    # # Ay = A[1]
    # # Az = A[2]
    # # Bx = B[0]
    # # By = B[1]
    # # Bz = B[2]
    

    # # Nx = Ay * Bz - Az * By
    # # Ny = Az * Bx - Ax * Bz
    # # Nz = Ax * By - Ay * Bx
    # # N = [-Nx, -Ny, -Nz]
    # N = np.cross(A, B, axis=0)
    # # nomr = np.linalg.norm(N)
    # # N = N / nomr

    v0 = points[triangles.simplices[:,0]]
    v1 = points[triangles.simplices[:,1]]
    v2 = points[triangles.simplices[:,2]]
    N = np.cross(v1-v0, v2 -v0)
    return N 
  

def create_division(first_parameter, second_parameter, mesh_poitns_2D, num_sections =4):
    middle_point_1 = (first_parameter.min() + first_parameter.max()) / (num_sections / 2)
    middle_point_2 = (second_parameter.min() + second_parameter.max()) / (num_sections / 2)

        #Usar lines space para discretizar

    segments = [
        mesh_poitns_2D[(first_parameter.ravel() <= middle_point_1) & (second_parameter.ravel() <= middle_point_2)],
        mesh_poitns_2D[(first_parameter.ravel() <= middle_point_1) & (second_parameter.ravel() >= middle_point_2)],
        mesh_poitns_2D[(first_parameter.ravel() >= middle_point_1) & (second_parameter.ravel() <= middle_point_2)],
        mesh_poitns_2D[(first_parameter.ravel() >= middle_point_1) & (second_parameter.ravel() >= middle_point_2)]
    ]

    return segments


def tri_in_segment(tri, sections):
    triangles =[]

    for i in range(len(sections)):
        tri_petenence = tri.find_simplex(sections[i])
        triangles.append(tri_petenence)

    return triangles


def line_tri_intersection(origin, vec, tri_index, triangles, points, epsilon=1e-10, debug=False, intersection = False):
    """
    Intersección rayo-triángulo optimizada
    """
    idx = triangles.simplices[tri_index]
    v0, v1, v2 = points[idx]
    
    # Vectores de aristas
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    h = np.cross(vec, edge2)
    # h = -h
    a = np.dot(edge1, h)
    
    if abs(a) < epsilon:
        return float('inf'), 0, 0
    
    f = 1.0 / a
    s = origin - v0
    u = f * np.dot(s, h)
    
    if u < 0.0 or u > 1.0:
        return float('inf'), u, 0
    
    q = np.cross(s, edge1)
    v = f * np.dot(vec, q)
    
    if v < 0.0 or u + v > 1.0:
        # if debug == True:
        #     # print('se para aqui', u, v)
        return float('inf'), u, v
    
    t = f * np.dot(edge2, q)
    if intersection==True:
        t = abs(t)
    
    if t > epsilon:
        if debug == True:
            print(f'Se intersect aen {tri_index}')
        return t, u, v
    else:
        if debug == True:
            print(f'Se intersect aen {tri_index}')
        return t, u, v


def xyz2thetaphi(point: list, a=1, b=1, c=1):
    """
    Convertir un punto en el dominio cartesiano al dominio polar elipsoidal,

    poin: punto a convertir en formato [x, y, z] (Array-like)

    a, b, c: semiejes del elipsoide en las direcciones x, y, z respectivamente.
    """
    x, y, z = point

    
    theta = np.arccos(z / (c + 1e-10))
    phi = np.arctan2(y / (b+ 1e-10), x / (a+ 1e-10))
    
    return theta, phi


def triangle_area(triangles, tri_idx, points, total_area = True):
    """
    Calcula el vector normal de un triángulo dado su índice y los puntos de un mesh.

    index: Índice del triángulo (int).

    points: Puntos del mesh (array-like con forma (n, 3) donde n es el número de puntos).

    triangles: Objeto de triangulación que contiene los triángulos (Delunay).

    total_area: Si es True, calcula el área total de todos los triángulos. Si es False, calcula el área del triángulo especificado por tri_idx (Bool). 

    Returns: Vector normal del triángulo (array de 3 elementos).
    """
    a_t = 0
    if total_area == False:
        A, B, C = points[triangles.simplices[tri_idx]]
        AB = B - A
        AC = C - A
        a_tri = np.cross(AC, AB)
        return 0.5 * np.linalg.norm(a_tri)
    else:
        for i in range(len(triangles.simplices)):
            A, B, C = points[triangles.simplices[i]]
            AB = B - A
            AC = C - A
            a_tri = np.cross(AC, AB)
            a_t += 0.5 * np.linalg.norm(a_tri)
        return a_t
    
def local_func2d(points_2D : np.ndarray):
    """
    Función para generar las funciones locales que me permiten aproximar cualquier función

    points_2D: Puntos en 2D en el dominio de mallado (array-like con forma (n, 2) donde n es el número de puntos)

    Returns: Lista de funciones locales (list of arrays)
    """
    hats = []

    n = len(points_2D)

    for i in range(n):
        hat = np.zeros(n)
        hat[i] = 1
        hats.append(hat)

    return hats

def local_interpolation_2d(coeffs: np.ndarray, points_2D: np.ndarray, x_points: np.ndarray, y_points: np.ndarray):
    """ 
    Función para realizar la interpolaCIÓN 2D a lo largo de todo el domino de mallado usando un conjunto de funciones locales y sus coeficientes.

    coeffs: Coeficientes para la interpolación (array-like con forma (n,) donde n es el número de puntos)

    points_2D: Puntos en 2D en el dominio de mallado (array-like con forma (n, 2) donde n es el número de puntos)

    x_points: Puntos en la dirección x para los que se quiere interpolar (array-like)

    y_points: Puntos en la dirección y para los que se quiere interpolar (array-like)

    Returns: V
    """
    z = np.zeros_like(x_points)
    hat_functions = local_func2d(points_2D)

    for i, hat in enumerate(hat_functions):
        interpolator = sp.interpolate.LinearNDInterpolator(points_2D, hat)
        z += coeffs[i] * interpolator(x_points, y_points)
    
    return z
    
def tmy_df(tmy_df, lat, lon, var_order=None, preview_cols=False):

    df = tmy_df.copy()

    time_zone = -5
    long_std = 15 * time_zone   # para UTC-5 es -75 grados
    time_off = 0.5              # centro de la hora

    phi_rad = np.deg2rad(lat)

    hour_year = np.arange(len(df))
    hour_local = hour_year % 24
    N_day = np.floor(hour_year / 24) + 1

    B = 2 * np.pi * (N_day - 1) / 365

    E = 229.2 * (
        0.000075
        + 0.001868 * np.cos(B)
        - 0.032077 * np.sin(B)
        - 0.014615 * np.cos(2 * B)
        - 0.04089 * np.sin(2 * B)
    )

    time_solar = hour_local + (4 * (lon - long_std) + E) / 60 + time_off

    omega_rad = np.deg2rad(15 * (time_solar - 12))

    delta_rad = np.deg2rad(23.45 * np.sin(2 * np.pi * (284 + N_day) / 365))

    cos_theta_z = (
        np.cos(phi_rad) * np.cos(delta_rad) * np.cos(omega_rad)
        + np.sin(phi_rad) * np.sin(delta_rad)
    )

    cos_theta_z = np.clip(cos_theta_z, -1, 1)

    theta_z_rad = np.arccos(cos_theta_z)
    theta_z_deg = np.rad2deg(theta_z_rad)

    elevation_deg = 90 - theta_z_deg

    denom = np.sin(theta_z_rad) * np.cos(phi_rad)
    arg = (cos_theta_z * np.sin(phi_rad) - np.sin(delta_rad)) / denom
    arg = np.clip(arg, -1, 1)

    gamma_s_rad = np.sign(omega_rad) * np.abs(np.arccos(arg))
    gamma_s_deg = np.rad2deg(gamma_s_rad)

    df['hour(local)'] = hour_local
    df['hour(solar)'] = time_solar
    df['apparent_elevation'] = theta_z_deg 
    df['zenith'] = theta_z_deg
    df['azimuth'] = gamma_s_deg

    df = df[df['Clearsky DHI'] != 0]

    if preview_cols:
        return df.columns.tolist()

    if var_order is None:
        var_order = [
            'Year',
            'Month',
            'Day',
            'hour(local)',
            'hour(solar)',
            'apparent_elevation',
            'azimuth',
            'zenith',
            'Clearsky DNI',
            'DNI',
            'Clearsky GHI',
            'GHI'
        ]

    return df[var_order]