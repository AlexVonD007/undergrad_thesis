import numpy as np
from utilidades import d2r, r2d, angle, mag, v2l

def rotate_vector_around_axis(v, axis, theta):

    if np.dot(axis,[-1,0,0])==0 or np.dot(axis,[-1,0,0])==-1:
        print("Son paralelos")
        
    # Normalizar el eje
    axis = axis / np.linalg.norm(axis)

    
    
    # Componentes de la fórmula de Rodrigues
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Producto cruzado
    k_cross_v = np.cross(axis, v)
    
    # Producto punto
    k_dot_v = np.dot(axis, v)
    
    # Fórmula de Rodrigues
    v_rot = v * cos_theta + k_cross_v * sin_theta + axis * k_dot_v * (1 - cos_theta)
    
    return v_rot


def rot_2d(vec, angle):
    v_rot_2d=[vec[0], vec[1]*np.cos(angle) + vec[2]*np.sin(angle), vec[1]*np.sin(angle) + vec[2]*np.cos(angle)]
    return v_rot_2d

def cos_dir(v: np.ndarray, axis: np.ndarray):
    """Calcula los cosenos directores de un vector v con respecto a un eje dado.

    v: vector 3D (array)

    axis: eje de referencia (array)

    Retorna: tupla con los cosenos directores (alpha, beta y gamma)
    """
    m_a = mag(axis)
    m_v=mag(v)
    alpha=np.arccos((v[0]*axis[0]) / (m_v*m_a) )
    beta = np.arccos((v[1]*axis[1]) / (m_v*m_a) )
    gamma = np.arccos((v[2]*axis[2]) / (m_v*m_a) )
    # alpha=np.arccos((v[0]) / (m_v) )
    # beta = np.arccos((v[1]) / (m_v) )
    # gamma = np.arccos((v[2]) / (m_v))
    return alpha, beta , gamma 

def rx(theta):
    rot_x=np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    return rot_x

def ry(theta):
    rot_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    return rot_y
def rz(theta):
    rot_z=np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    return rot_z

def rot3d(v: np.ndarray,t: np.ndarray):
    """"
    Rotacion 3D de un objeto a partir de la matriz de rotacion de rodrigez y los angulos a rotar en cada eje x,y,z

    v: vector a rotar (array)

    t: angulos a rotar para cada eje (array)
    
    Retorna: vector rotado (array)
    """
    R=rx(t[0])@ry(t[1])@rz(t[2])@v
    return R

def reflect_vector(incident_vec, normal_vec):
    # Normalizar los vectores
    incident_vec = incident_vec / mag(incident_vec)
    normal_vec = normal_vec / mag(normal_vec)
    
    # Calcular los cosenos directores
    alpha, beta, gamma = cos_dir(incident_vec, normal_vec)
    
    # Crear matrices de rotación
    R_alpha = np.array([[1, 0, 0],
                        [0, np.cos(alpha), -np.sin(alpha)],
                        [0, np.sin(alpha), np.cos(alpha)]])
    
    R_beta = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])
    
    R_gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                        [np.sin(gamma), np.cos(gamma), 0],
                        [0, 0, 1]])
    
    # Aplicar las rotaciones para reflejar el vector
    reflected_vec = R_alpha @ R_beta @ R_gamma @ incident_vec
    
    return reflected_vec

def reflect_vectors(incident_vectors, normal_vectors):
    reflected_vectors = np.zeros_like(incident_vectors)
    for i in range(len(incident_vectors)):
        reflected_vectors[i] = reflect_vector(incident_vectors[i], normal_vectors[i])
    return reflected_vectors



def unitary_sun_reflection(sun_vec: np.ndarray, sup_vec: np.ndarray, test=False):
    """
    Reflexión de un vector sobre un eje de superficie.
    
    sun_vec: Vector solar unitario (numpy array)
    
    sup_vec: Vector de superficie unitario (numpy array)
    
    Retorna un vector reflejado unitario."""
    sun_vec = sun_vec / np.linalg.norm(sun_vec)
    sup_vec = sup_vec / np.linalg.norm(sup_vec)

    if test == True:
        degree = np.nan_to_num(cos_dir(sun_vec, sup_vec))
        out = v2l(rot3d(sun_vec, degree))
    else:
        out = sun_vec - 2 * (np.dot(sup_vec,sun_vec) ) * sup_vec
    return out