import numpy as np
from scipy.spatial import Delaunay
from utilidades import line_tri_intersection
from sun import unitary_sun_reflection
import sympy as sp


def select_best_interval(intersections, tol=1e-10):
    """
    Selecciona la mejor intersección de una lista de candidatos.

    Parameters
    ----------
    intersections : list
        Lista de tuplas (tri_idx, t, u, v).

    tol : float
        Tolerancia numérica para las condiciones de intersección.

    Returns
    -------
    tri_idx, t, u, v
    """

    valid = []

    for tri_idx, t, u, v in intersections:

        # Condiciones de Möller-Trumbore
        valid_t = t >= -tol
        valid_u = u >= -tol
        valid_v = v >= -tol
        valid_sum = (u + v) <= 1 + tol

        if valid_t and valid_u and valid_v and valid_sum:
            valid.append((tri_idx, t, u, v))

    # Si existen intersecciones válidas,
    # seleccionar la más cercana al origen
    if valid:
        return min(valid, key=lambda x: x[1])

    # Si ninguna es válida, seleccionar la que tenga
    # menor violación del dominio
    best = min(
        intersections,
        key=lambda x: max(
            0.0,
            -x[1],       # t < 0
            -x[2],       # u < 0
            -x[3],       # v < 0
            x[2] + x[3] - 1  # u+v > 1
        )
    )

    return best

def search_algorit(
    origin: np.ndarray,
    vec: np.ndarray,
    tri_idx: list,
    triangles: Delaunay,
    points: np.ndarray
):
    """
    Algoritmo de búsqueda para la intersección de un rayo
    con una malla triangular.

    Parámetros
    ----------
    origin : np.ndarray
        Origen del rayo.

    vec : np.ndarray
        Vector dirección del rayo.

    tri_idx : list
        Índices de los triángulos iniciales a evaluar.

    triangles : Delaunay
        Malla triangular de scipy.spatial.

    points : np.ndarray
        Coordenadas de los puntos de la malla.

    Retorna
    -------
    optimal_tri : list
        Triángulos explorados/seleccionados durante la búsqueda.

    hit_idx : int
        Índice del triángulo donde se encontró la intersección.
        Retorna -1 si no se encontró.
    """

    explored_tri = []
    optimal_tri = []

    # ---------------------------------------------------------
    # 1. Evaluar los triángulos iniciales
    # ---------------------------------------------------------

    init_intersections = []

    for i in tri_idx:

        t_i, u_i, v_i = line_tri_intersection(
            origin,
            vec,
            i,
            triangles,
            points
        )

        init_intersections.append(
            (i, t_i, u_i, v_i)
        )

    # ---------------------------------------------------------
    # 2. Seleccionar el mejor candidato inicial
    # ---------------------------------------------------------

    if not init_intersections:
        return [], -1

    best_idx, best_t, best_u, best_v = \
        select_best_interval(init_intersections)

    idx = best_idx

    optimal_tri.append(idx)

    # ---------------------------------------------------------
    # 3. Comenzar búsqueda por vecinos
    # ---------------------------------------------------------

    hit_idx = -1

    while True:

        # -----------------------------------------------------
        # Verificar si el triángulo actual es una intersección
        # válida
        # -----------------------------------------------------

        t, u, v = line_tri_intersection(
            origin,
            vec,
            idx,
            triangles,
            points
        )

        # Condiciones de Möller-Trumbore
        if (
            t >= 0
            and u >= 0
            and v >= 0
            and u + v <= 1
        ):
            hit_idx = idx
            break

        # -----------------------------------------------------
        # Evitar recorrer indefinidamente toda la malla
        # -----------------------------------------------------

        if len(optimal_tri) >= len(triangles.simplices):
            hit_idx = -1
            break

        # -----------------------------------------------------
        # Obtener vecinos del triángulo actual
        # -----------------------------------------------------

        neigh = triangles.neighbors[idx]

        explored_tri.append(idx)

        intervals = []

        # -----------------------------------------------------
        # Evaluar vecinos
        # -----------------------------------------------------

        for i in neigh:

            # -1 significa que no existe vecino
            if i < 0:
                continue

            # Evitar volver a evaluar triángulos ya seleccionados
            if i in optimal_tri:
                continue

            t_i, u_i, v_i = line_tri_intersection(
                origin,
                vec,
                i,
                triangles,
                points
            )

            intervals.append(
                (i, t_i, u_i, v_i)
            )

        # -----------------------------------------------------
        # Si no existen vecinos disponibles, hacer búsqueda
        # global de respaldo
        # -----------------------------------------------------

        if not intervals:

            unexplored = [
                i for i in range(len(triangles.simplices))
                if i not in explored_tri
                and i not in optimal_tri
            ]

            if not unexplored:
                hit_idx = -1
                break

            new_idx = np.random.choice(unexplored)

            t_n, u_n, v_n = line_tri_intersection(
                origin,
                vec,
                new_idx,
                triangles,
                points
            )

            intervals.append(
                (new_idx, t_n, u_n, v_n)
            )

        # -----------------------------------------------------
        # Seleccionar el mejor vecino
        # -----------------------------------------------------

        best_idx, best_t, best_u, best_v = \
            select_best_interval(intervals)

        optimal_tri.append(best_idx)

        idx = best_idx

    return  hit_idx, optimal_tri

def robust_ray_intersection(origin, direction, triangles, points, initial_triangles=None, epsilon=1e-20, eval=50, exclution=None, debug=False, intersection = False):
    """
    Búsqueda robusta de intersecciones que prioriza triángulos iniciales
    y excluye triángulos específicos.

    Returns: hit_tri, best_t
    """
    direction_norm = direction / np.linalg.norm(direction)
    
    best_t = float('inf')
    best_tri_idx = -1
    
    if exclution is not None:
        if isinstance(exclution, int):
            exclution = [exclution]
    
    if initial_triangles is not None:
        for tri_idx in initial_triangles:
            if exclution is not None and tri_idx in exclution:
                continue
                
            t, u, v = line_tri_intersection(origin, direction_norm, tri_idx, triangles, points, epsilon, intersection=intersection)
            if t < best_t and t != float('inf'):
                best_t = t
                best_tri_idx = tri_idx
                if best_t <= 1.0:
                    return best_tri_idx, best_t
    

    if best_tri_idx == -1:
        priorities = []
        for i in range(len(triangles.simplices)):
            if exclution is not None and i in exclution:
                continue
                
            centroid = np.mean(points[triangles.simplices[i]], axis=0)

            to_centroid = centroid - origin

            projection = np.dot(to_centroid, direction_norm)

            perpendicular_dist = np.linalg.norm(to_centroid - projection * direction_norm)
            
            if projection >= 0:  # Solo considerar en la dirección del rayo
                priority = projection + 1.0/(perpendicular_dist + 0.001)  # Evitar división por cero
                priorities.append((i, priority, projection))
        
        if priorities:
            # Ordenar por prioridad (mayor primero)
            priorities.sort(key=lambda x: x[1], reverse=True)
            if debug == True:
                print(priorities)
            # Probar los N más prometedores (limitado por eval)
            for tri_idx, priority, projection in priorities[:eval]:
                t, u, v = line_tri_intersection(origin, direction_norm, tri_idx, triangles, points, epsilon, intersection=intersection)
                if t < best_t and t != float('inf'):
                    best_t = t
                    best_tri_idx = tri_idx
                    if best_t <= 1.0:
                        break
        
    
    return best_tri_idx, best_t

def taylor_pro(p: np.ndarray, p0: float, t: np.ndarray, t0: float, n_coefs: int , ec_x: sp.Function, ec_y: sp.Function, ec_z:sp.Function, coefs=None):
    """
    Calcular la expansión de Taylor de una función en un punto dado.
    
    p: variable 1 puntos generadores de superficie (array)
    
    p0: punto donde se evalua la función (float)
    
    t: variable 2 puntos generadores de superficie (array)
    
    t0: punto donde se evalua la función (float)
    
    n_coefs: número de coeficientes a calcular (int)
    
    ec_x: función de la coordenada x (sympy function)

    ec_y: función de la coordenada y (sympy function)

    ec_z: función de la coordenada z (sympy function)
    
    coefs: coeficientes de la expansión de Taylor (list, opcional)
    
    Retorna: coordenadas (xt, yt, zt) de la expansión de Taylor en el punto dado.
    
    Recomendación si se usa un array de NxM puntos genradores de superficie p0 y t0 deben de corresponder a p tal que p0 = p[N/n_coefs, M/n_coefs] y t0 = t[N/n_coefs, M/n_coefs]"""

    phi = sp.Symbol('x')
    theta = sp.Symbol('y')


    x_k = [p, t]
    a_k = [p0, t0]
    dk1 = x_k[0] - a_k[0]
    dk2 = x_k[1] - a_k[1]

    k = 1

    if coefs is None:
        pass
    else:
        if len(coefs) < n_coefs:
            print('Son necersairos más coeficientes')
            k = 0
    xt = 0
    yt = 0
    zt = 0
    for i in range(n_coefs+1):
        for j in range(n_coefs+1 - i):
            if k == 0:
                break
            else:
                if coefs is None:
                    partial_x = sp.diff(ec_x,phi, i, theta, j )
                    partial_eval = partial_x.subs({phi: p0, theta: t0})
                    term_x = (1/(sp.factorial(j)*sp.factorial(i))) * partial_eval * (dk1**i) * (dk2**j)
                    xt += term_x

                    partial_y = sp.diff(ec_y,phi, i, theta, j )
                    partial_eval_y = partial_y.subs({phi: p0, theta: t0})
                    term_y = (1/(sp.factorial(j)*sp.factorial(i))) * partial_eval_y * (dk1**i) * (dk2**j)
                    yt += term_y

                    partial_z = sp.diff(ec_z,phi, i, theta, j )
                    partial_eval_z = partial_z.subs({phi: p0, theta: t0})
                    term_z = (1/(sp.factorial(j)*sp.factorial(i))) * partial_eval_z * (dk1**i) * (dk2**j)
                    zt += term_z
                else:
                    if i == 0 and j == 0:
                        partial_x = sp.diff(ec_x,phi, i, theta, j )
                        partial_eval = partial_x.subs({phi: p0, theta: t0})
                        term_x = (1/(sp.factorial(j)*sp.factorial(i))) * partial_eval * (dk1**i) * (dk2**j)
                        xt += term_x

                        partial_y = sp.diff(ec_y,phi, i, theta, j )
                        partial_eval_y = partial_y.subs({phi: p0, theta: t0})
                        term_y = (1/(sp.factorial(j)*sp.factorial(i))) * partial_eval_y * (dk1**i) * (dk2**j)
                        yt += term_y

                        partial_z = sp.diff(ec_z,phi, i, theta, j )
                        partial_eval_z = partial_z.subs({phi: p0, theta: t0})
                        term_z = (1/(sp.factorial(j)*sp.factorial(i))) * partial_eval_z * (dk1**i) * (dk2**j)
                        zt += term_z
                    else:
                        partial_x = sp.diff(ec_x, phi, i, theta, j)
                        partial_eval = partial_x.subs({phi: p0, theta: t0})
                        term_x = (1/coefs[i-1]) * partial_eval * (dk1**i) * (dk2**j)
                        xt += term_x

                        partial_y = sp.diff(ec_y, phi, i, theta, j)
                        partial_eval_y = partial_y.subs({phi: p0, theta: t0})
                        term_y = (1/coefs[i-1]) * partial_eval_y * (dk1**i) * (dk2**j)
                        yt += term_y

                        partial_z = sp.diff(ec_z, phi, i, theta, j)
                        partial_eval_z = partial_z.subs({phi: p0, theta: t0})
                        term_z = (1/coefs[i-1]) * partial_eval_z * (dk1**i) * (dk2**j)
                        zt += term_z
                    
    return xt, yt, zt


def find_target_triangle(origin, current_tri_idx, triangles, baricentric_points, 
                        min_distance=0.3, max_distance=1.5):
    """
    Encuentra un triángulo objetivo adecuado para apuntar.
    """
    # Calcular distancias a todos los centroides
    distances = []
    for i, centroid in enumerate(baricentric_points):
        if i != current_tri_idx:
            dist = np.linalg.norm(centroid - origin)
            if min_distance <= dist <= max_distance:
                # Calcular ángulo con respecto a la normal (opcional)
                distances.append((i, dist, centroid))
    
    if not distances:
        # Si no hay triángulos en el rango, buscar el más cercano
        for i, centroid in enumerate(baricentric_points):
            if i != current_tri_idx:
                dist = np.linalg.norm(centroid - origin)
                distances.append((i, dist, centroid))
    
    # Ordenar por distancia
    distances.sort(key=lambda x: x[1])
    
    # Seleccionar el triángulo objetivo
    if distances:
        return distances[0][0], distances[0][2]  # Índice y centroide
    else:
        # Si no hay otros triángulos, usar uno aleatorio
        other_tri = np.random.choice([i for i in range(len(baricentric_points)) if i != current_tri_idx])
        return other_tri, baricentric_points[other_tri]

def self_reflection_optimized(surface_vector, init_surface_idx, vector, origin, 
                              tri_idx, triangles, points_, baricentric_points, 
                              self_reflections_number=3, is_vector=False,
                              ensure_reflection=True):
    """
    Algoritmo de autoreflexión que puede optimizar el vector para asegurar reflexiones.
    
    Args:
        ensure_reflection: Si True, modifica el vector para asegurar al menos una reflexión
    """
    
    reflection_history = []
    ref_count = 0
    reflection = 0
    current_tri_idx = tri_idx
    current_vector = vector.copy()
    current_origin = origin.copy()
    
    # Si se solicita asegurar reflexión, calcular un vector optimizado
    if ensure_reflection:
        # Calcular un vector que apunte a otro triángulo
        target_tri_idx, target_centroid = find_target_triangle(
            current_origin, current_tri_idx, triangles, baricentric_points
        )
        
        # Calcular vector hacia el objetivo
        direction_to_target = target_centroid - current_origin
        current_vector = direction_to_target / np.linalg.norm(direction_to_target)
        
        # Ajustar la magnitud (mantener la magnitud original si se desea)
        current_vector = current_vector * np.linalg.norm(vector)
        
        # print(f"Vector optimizado para reflexión:")
        # print(f"  Triángulo objetivo: {target_tri_idx}")
        # print(f"  Dirección: {current_vector}")
        # print(f"  Distancia al objetivo: {np.linalg.norm(direction_to_target):.3f}")
    
    # Guardar vector inicial (puede ser el original o el optimizado)
    reflection_history.append({
        'type': 'initial',
        'origin': current_origin.copy(),
        'vector': current_vector.copy(),
        'magnitude': np.linalg.norm(current_vector),
        'triangle_idx': current_tri_idx,
        'reflection_number': 0,
        'excluded_triangles': [current_tri_idx],
        'is_optimized': ensure_reflection
    })

    t = 0
    
    while ref_count < self_reflections_number:
        
        # Encontrar intersección excluyendo el triángulo actual
        hit_idx, t = robust_ray_intersection(
            origin=current_origin,
            direction=current_vector,
            triangles=triangles,
            points=points_,
            initial_triangles=None,
            epsilon=1e-10,
            eval=50,
            exclution=[current_tri_idx]  # Excluir el triángulo actual
        )
        
        # Si no hay intersección, terminar
        if hit_idx == -1:
            break
        if t==0:
            print(ref_count, hit_idx)
        t=1
        # Calcular punto de intersección
        intersection_point = current_origin + current_vector * t
        # current_vector = current_vector / np.linalg.norm(current_vector)
        s_v = surface_vector[hit_idx] / np.linalg.norm(surface_vector[hit_idx])

        # Calcular vector reflejado usando la normal del triángulo intersectado
        reflected_vec = unitary_sun_reflection(
            -1*current_vector,
            s_v
        )
        
        # Actualizar contadores
        ref_count += 1
        reflection = 1
        
        # Guardar información de esta reflexión
        reflection_history.append({
            'type': 'reflection',
            'origin': current_origin.copy(),
            'vector': reflected_vec.copy(),
            'magnitude': np.linalg.norm(reflected_vec),
            'triangle_idx': hit_idx,
            'intersection_point': intersection_point,
            'reflection_number': ref_count,
            'surface_normal': surface_vector[hit_idx].copy(),
            'excluded_triangles': [hit_idx],  # Para la próxima iteración
            'distance': t
        })
        
        # Actualizar para la siguiente iteración
        current_vector = reflected_vec.copy()
        current_origin = intersection_point.copy()  # Usar punto de intersección
        current_tri_idx = hit_idx
    
    # Si no hubo reflexiones, el vector final es el original
    if reflection == 0:
        reflected_vec = vector if not ensure_reflection else current_vector
        ref_count = 0
    
    return reflected_vec, reflection, current_tri_idx, ref_count, reflection_history
