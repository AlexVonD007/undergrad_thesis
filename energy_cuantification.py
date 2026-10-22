import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from utilidades import triangle_area
from serach_optim import robust_ray_intersection, self_reflection_optimized
from sun import unitary_sun_reflection 
from joblib import Parallel, delayed
from classes import Transmitance


def potency_due2sun(irr_list, sun_positions,dome_vectors, surface_normal, surface_centers, triangles, surface_points, obstacle_mesh= None, obstacle_points= None, eval = 50, debug = False, time = None, Return_for_plotting = False, x_plot = None, y_plot = None, z_plot = None, x_p = None, y_p = None, z_p = None, x_sp = None, y_sp = None, z_sp = None, n_iter = None):
    pt = 0 
    # sun_positions[:,2] = sun_positions[:, 2] + surface_points[:,2].min()
    t_plot = []
    if time is not None:
        irr_plot = np.zeros((len(triangles.simplices), len(time[:n_iter])))
    for i in range(len(triangles.simplices)):
        # if debug == True:
        #     i = np.random.randint(0, len(triangles.simplices))
        
        sun_vectors = sun_positions - surface_centers[i]
        sun_vectors = np.array(sun_vectors)

        pk = 0
        # if debug == True:
        #     fig = plt.figure(figsize=(12, 10))

        #     # Subplot izquierdo
        #     ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        #     ax1.set_title('Contribuciones cupula')
        #     ax1.set_xlabel('X')
        #     ax1.set_ylabel('Y')
        #     ax1.set_zlabel('Z')

        #     # Subplot derecho
        #     ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        #     ax2.set_title(f'Figura panel y elemento {i}')
        #     ax2.set_xlabel('X')
        #     ax2.set_ylabel('Y')
        #     ax2.set_zlabel('Z')
        #     xp, yp, zp = [], [], []
        #     xpn, ypn, zpn = [], [], []
        if n_iter is None:
            eval = len(sun_vectors)
        else:
            eval = n_iter
        for j in range(eval):
        # for j in range(11):

            # j = 3552
            sun_vec = sun_vectors[j]
            sun_vec = sun_vec / np.linalg.norm(sun_vec)
            # sun_vec = np.array([ 1/np.sqrt(2), 0, 1/np.sqrt(2)])
            if time is not None:
                if len(t_plot) < eval:
                    t_plot.append(time[j])
            # print(sun_vec)
            if obstacle_mesh is None and obstacle_points is  None:
                hit = -1
            else:
                hit, _ = robust_ray_intersection(surface_centers[i], sun_vec, obstacle_mesh, obstacle_points, eval=eval, intersection = True)
                # print(hit)
            # print(hit)
            # if debug == True:
            #     break
            # print(f'Producto punto en la superficie {i} con el vector solar es de {np.rad2deg(np.arccos(np.dot(sun_vectors[j]/np.linalg.norm(sun_vectors[j]), surface_normal[i]/np.linalg.norm(surface_normal[i]))))}')
            if hit==-1:

                sun_vec = dome_vectors[j] / np.linalg.norm(dome_vectors[j])
                # sun_vec = np.array([ 1/np.sqrt(2), 0, 1/np.sqrt(2)])
                normal_vec = surface_normal[i] / np.linalg.norm(surface_normal[i])
                # print(np.dot(-sun_vec, normal_vec))
                
                pk += irr_list[j] * np.clip(-sun_vec @ normal_vec,0, 1) * triangle_area(triangles, i, surface_points, total_area=False)
                # print(round(irr_list[j] * np.clip(np.dot(-sun_vec, normal_vec),0, 1)* triangle_area(triangles, i, surface_points, total_area=False), 2), (triangle_area(triangles, i, surface_points, total_area=False)), (np.clip(np.dot(-sun_vec, normal_vec)/(np.linalg.norm(normal_vec)*np.linalg.norm(sun_vec)),0, 1)), np.rad2deg(np.clip(np.dot(-sun_vec, normal_vec),0, 1)) ,irr_list[j])
                # print( np.clip(np.dot(-sun_vec, normal_vec),0, 1), np.rad2deg(np.arccos(np.dot(-sun_vec, normal_vec))) ,irr_list[j])
                # pk += irr_list[j] * np.clip(np.dot(sun_vec, normal_vec),0, 1) * triangle_area(triangles, i, surface_points, total_area=False)
                if Return_for_plotting == True:
                    irr_plot[i, j] = irr_list[j] * np.clip(np.dot(-sun_vec, normal_vec),0, 1) * triangle_area(triangles, i, surface_points, total_area=False)
                # if debug == True:
                #     xp.append(x_plot[j]*1e-11)
                #     yp.append(y_plot[j]*1e-11)
                #     zp.append(z_plot[j]*1e-11)
                # if z[j]*1e-11 <= 2 and x[j]*1e-11 < 0:
                #     print(j)
                # if debug== True:
                #     print(triangle_area(triangles, i, surface_points, total_area=False))
                #     break
                # print(j)

            else:
                if Return_for_plotting == True:
                    irr_plot[i, j] = 0
            #     # ax.scatter(x[j]*1e-11, y[j]*1e-11, z[j]*1e-11, color='g', s=5)
            #     if debug == True:
            #         xpn.append(x_plot[j]*1e-11)
            #         ypn.append(y_plot[j]*1e-11)
            #         zpn.append(z_plot[j]*1e-11)
            #     continue
        # if debug == True:
        #     break
       
        pt+= pk
        
        # print(i)
    #     if debug== True:
    #         print(f'Producto punto en la superficie {i} con el vector solar es de {np.rad2deg(np.arccos(np.dot(sun_vectors[j]/np.linalg.norm(sun_vectors[j]), surface_normal[i]/np.linalg.norm(surface_normal[i]))))}')
    #         xp = np.array(xp)
    #         yp = np.array(yp)
    #         zp = np.array(zp)

    #         xpn = np.array(xpn)
    #         ypn = np.array(ypn)
    #         zpn = np.array(zpn)
    #         ax1.plot_surface(x_p,y_p,z_p, color = 'r', alpha=0.5)
    #         ax1.plot_surface(x_sp, y_sp, z_sp, color='purple', edgecolor='k', alpha=0.5)
    #         ax1.scatter(xp, yp, zp, color='y', s=5)
    #         # ax.scatter(xpn, ypn, zpn, color='gray', s=5)
    #         ax1.scatter3D(surface_centers[i,0], surface_centers[i,1], surface_centers[i,2], color='r', s=10)
    #         sun_vec = sun_vec / np.linalg.norm(sun_vec)
    #         # ax.scatter3D(x[e]*1e-11, y[e]*1e-11, z[e]*1e-11, color='green', s=100)
    #         # ax.quiver(
    #         #         surface_centers[i,0], surface_centers[i,1], surface_centers[i,2] ,
    #         #         sun_vec[0], sun_vec[1], sun_vec[2],
    #         #         color='black', alpha=1, arrow_length_ratio=0.1, length=1e0)
    #         ax1.scatter(xpn, ypn, zpn, color='gray', s=5)
    #         #ax2.plot_trisurf(points_3DP[:, 0], points_3DP[:, 1], points_3DP[:, 2], triangles=tri_p.simplices, cmap='viridis', edgecolor='k', alpha=0.5)
    #         ax2.plot_trisurf(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], triangles=triangles.simplices, cmap='viridis', edgecolor='k', alpha=0.5)
    #         ax2.scatter3D(surface_centers[i,0], surface_centers[i,1], surface_centers[i,2], color='black', s=20)
    #         # ax.set_xlim3d([-0.1, 0.1])
    #         # ax.set_ylim3d([-0.6, 0.6])
    #         # ax.set_zlim3d([0.46, 0.49])
    #         plt.tight_layout()
    #         plt.show()

    #         break

    #     # if percentage % 10 == 0 or i == 0:
    #     #     print(f'Progreso {percentage}%')
    if Return_for_plotting == True:
        return pt, t_plot, irr_plot
    return pt

def potency_due2concentrator( irr_list, sun_positions,dome_vectors, surface_normal, surface_centers, triangles, surface_points, concenrtrator_tri, concentrator_points, concentrator_center,concentrator_normal, reflectivity = 0.97, debug = False, time=None, return_plot = False, PLANE_CONCENTRATOR = False, x_plot = None, y_plot = None, z_plot= None, n_iter = None):
    pt = 0
    c_i = []
    c2 = 0
    a = 0
    are = 0
    are_i =0
    t_plot = []
    if time is not None:
        irr_plot = np.zeros((len(concenrtrator_tri.simplices), len(time[:n_iter])))

    for i in range(len(concenrtrator_tri.simplices)):
        # if debug == True:
        #     i = np.random.randint(0, len(concenrtrator_tri.simplices))

        #     print(i)
        sun_vecs = sun_positions - concentrator_center[i]
        sun_vecs = np.array(sun_vecs)
        c=0

        vi = []
        vin = []
        xp, yp, zp = [], [], []
        xpn, ypn, zpn = [], [], []
        v_v = []
        v_nv = []
        pdc = 0
        # print(f' El pdc actual es {pdc}')
        # print('----------------------------------------------------------------------------------')
        # sun_vec = np.array([ 1/np.sqrt(2), 0, 1/np.sqrt(2)])
        # sun_vec = sun_vec / np.linalg.norm(sun_vec)
        # r_v = unitary_sun_reflection(-sun_vec, concentrator_normal[i]/np.linalg.norm(concentrator_normal[i]))
        # normal_vec = concentrator_normal[i]/np.linalg.norm(concentrator_normal[i])
        # r= 1
        # hit, _ = robust_ray_intersection(concentrator_center[i], r_v, triangles, surface_points)
        # if hit !=-1:
        #     pdc +=  np.clip(np.dot(r_v, normal_vec),0, 1) * (reflectivity**r) *triangle_area(concenrtrator_tri, i, concentrator_points, total_area=False)
        #     c2+=1
        if n_iter is not None:
            eval = n_iter
        else:
            eval = len(sun_vecs)

        for j in range(eval):
        # for j in range(11):

            sun_vec = sun_vecs[j]

            if time is not None:
                if len(t_plot) < eval:
                    t_plot.append(time[j])
      
            # hit, _  = robust_ray_intersection(concentrator_center[i], sun_vec, concenrtrator_tri,concentrator_points, exclution=[i] )
            if np.dot(sun_vec/np.linalg.norm(sun_vec), concentrator_normal[i]/np.linalg.norm(concentrator_normal[i])) > 0:
                hit, _  = robust_ray_intersection(concentrator_center[i], sun_vec/np.linalg.norm(sun_vec), concenrtrator_tri,concentrator_points, exclution=[i], intersection=True, eval=5)
                # if i <6 and hit !=-1:
                #     print('error')
                # if debug == True:
                #     c +=1
                    
                #     vi.append(sun_vec)
                #     xp.append(x_plot[j]*1e-11)
                #     yp.append(y_plot[j]*1e-11)
                #     zp.append(z_plot[j]*1e-11)
                if hit ==-1:
                    n_v = concentrator_normal[i] / np.linalg.norm(concentrator_normal[i])
                    c_v = -dome_vectors[j] / np.linalg.norm(dome_vectors[j])
                    r_v = unitary_sun_reflection(c_v, n_v)
                    r_v = r_v / np.linalg.norm(r_v)
                    r_v = -r_v
                    # r_v = r_v*1e6
                    hit_pane, _ = robust_ray_intersection(concentrator_center[i], r_v, triangles, surface_points, exclution=[i], eval = 5)
                    # print(i)
                    # print(hit_pane, i, np.rad2deg(np.arccos(np.dot(c_v, n_v))) , np.rad2deg(np.arccos(np.dot(r_v, n_v))), np.rad2deg(np.arccos(np.dot(c_v, r_v))))
                    if hit_pane !=-1:
                        c2+=1
                        r = 1
                        normal_vec = surface_normal[hit_pane] / np.linalg.norm(surface_normal[hit_pane])
                        vec = -r_v / np.linalg.norm(r_v)
                        if debug == True:
                            v_v.append(r_v)
                        # print(f'Irradiancia {irr[j]}, \n PDC = {pdc}')
                        # print(np.dot(vec, normal_vec), irr_list[j], triangle_area(concenrtrator_tri, i, concentrator_points, total_area=False))
                        # print(irr_list[j] *np.clip(np.dot(vec, normal_vec),0, 1), triangle_area(concenrtrator_tri, i, concentrator_points, total_area=False))
                        if PLANE_CONCENTRATOR == True:
                            pdc += irr_list[j] * np.clip(np.dot(vec, normal_vec),0, 1) * (reflectivity**r) *triangle_area(concenrtrator_tri, i, concentrator_points, total_area=False)
                        if time is not None:
                            irr_plot[i, j] = irr_list[j] * np.clip(np.dot(vec, normal_vec),0, 1)  *triangle_area(concenrtrator_tri, i, concentrator_points, total_area=False)
                        # if c2 == 1:
                        #     print(f'Contribución del vector {j} al elmento del concentrador {i}, con area{triangle_area(concenrtrator_tri, i, concentrator_points, total_area=False)} m2 es de : {pdc}')
                    else:
                        # if debug == True:
                        #     v_nv.append(r_v)
                        if PLANE_CONCENTRATOR == True:
                            continue
                        else:

                            r_vf, r_c, hit_idx, r, _ = self_reflection_optimized(concentrator_normal, hit, r_v, concentrator_center[hit], hit, concenrtrator_tri, concentrator_points, concentrator_center, ensure_reflection=False)
                            r = r+1
                            # print(r)
                            # r_vf = r_vf*1e6
                            # r_vf = r_vf - concentrator_center[hit_idx]

                            # print(r_v-r_vf)
                            # if debug == True:
                            #     v_v.append(r_vf)
                            hit_pane, _ = robust_ray_intersection(concentrator_center[hit_idx], r_vf, triangles, surface_points, intersection=True, eval=5)
                            # print(hit_pane)
                            if hit_pane !=-1:
                                # print(hit_pane)

                                # print('hay impacto sobre el panel')
                                normal_vec = surface_normal[hit_pane] / np.linalg.norm(surface_normal[hit_pane])
                                vec = -r_vf / np.linalg.norm(r_vf)
                        #     # print(np.dot(vec, normal_vec))
                                pdc += irr_list[j] * np.clip(np.dot(vec, normal_vec),0, 1) * (reflectivity**r) *triangle_area(concenrtrator_tri, i, concentrator_points, total_area=False)
                        #     # print(f'Contribución del vector {j} al elmento del concentrador {i}, con area{triangle_area(concenrtrator_tri, i, concentrator_points, total_area=False)} m2 es de : {pdc}')
                        #     # print(pdc)

                    # print(f'el vector {j} no se impacta sobbre el concentrador')
                else:
                    continue
                #     # if debug == True:
                #     #     xpn.append(x_plot[j]*1e-11)
                #     #     ypn.append(y_plot[j]*1e-11)
                #     #     zpn.append(z_plot[j]*1e-11)
                #     # print(f'el vector {j} se impacta sobbre el concentrador')
                #     continue
                

            # print(np.dot(sun_vec/np.linalg.norm(sun_vec), concentrator_normal[i]/np.linalg.norm(concentrator_normal[i])))
            # else:
            #     # if debug == True:
            #     #     vin.append(sun_vec)
            #     #     xpn.append(x_plot[j]*1e-11)
            #     #     ypn.append(y_plot[j]*1e-11)
            #     #     zpn.append(z_plot[j]*1e-11)
            #     continue
        
        # pt += pdc* (1/len(sun_vecs))
        pt += pdc * (1/eval+1) 
        # are += are_i * (1/12**2)
        # pt += pdc
        # print(f'La potencia del elemento {i} es {pt:.2f}')
    #     if debug == True:
    #         vi = np.array(vi)
    #         vin = np.array(vin)
    #         v_v = np.array(v_v)
    #         v_nv = np.array(v_nv)
    #         v_nv = v_nv / np.linalg.norm(v_nv)

    #         # print(v_v.shape)
    #         v_v = v_v / np.linalg.norm(v_v)
    #         # print(f' anglulo normal-reflejado {np.rad2deg(np.arccos(np.dot(r_v/np.linalg.norm(r_v), concentrator_normal[i]/np.linalg.norm(concentrator_normal[i]))))}')

    #         # print(f' anglulo normal-incidente {np.rad2deg(np.arccos(np.dot(sun_vec/np.linalg.norm(sun_vec), concentrator_normal[i]/np.linalg.norm(concentrator_normal[i]))))}')
    #         # print(f'Inciden {c} vectores sobre la superficie de los {len(sun_vecs)}, es decir que {len(sun_vecs) -c} no son validos')
    #         print(f'Impactan {len(v_v)} vectores sobre la superficie de los {len(v_v) + len(v_nv)}')

    #         fig = plt.figure(figsize=(12, 10))
    #         # Subplot izquierdo
    #         ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    #         ax1.set_title('Contribuciones cupula')
    #         ax1.set_xlabel('X')
    #         ax1.set_ylabel('Y')
    #         ax1.set_zlabel('Z')

    #         # Subplot derecho
    #         ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    #         ax2.set_title(f'Figura panel y elemento {i}')
    #         ax2.set_xlabel('X')
    #         ax2.set_ylabel('Y')
    #         ax2.set_zlabel('Z')
    #         ax1.plot_trisurf(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], triangles=triangles.simplices, cmap='viridis', edgecolor='k', alpha=0.5)
    #         ax1.plot_trisurf(concentrator_points[:, 0], concentrator_points[:, 1], concentrator_points[:, 2], triangles=concenrtrator_tri.simplices, cmap='viridis', edgecolor='k', alpha=0.5)
    #         # # ax1.quiver(
    #         # #     concentrator_center[i, 0], concentrator_center[i, 1], concentrator_center[i, 2],
    #         # #     vi[:, 0], vi[:, 1], vi[:,2], 
    #         # #     color='yellow', alpha=1, arrow_length_ratio=1e-7, length=1e-13
    #         # # )
    #         # # ax1.quiver(
    #         # #     concentrator_center[i, 0], concentrator_center[i, 1], concentrator_center[i, 2],
    #         # #     vin[:, 0], vin[:, 1], vin[:,2], 
    #         # #     color='black', alpha=1, arrow_length_ratio=1e-7, length=1e-13
    #         # # )
    #         # ax1.scatter(xpn, ypn, zpn, color='gray', s=5)
    #         # ax1.scatter(xp, yp, zp, color='y', s=5)

    #         ax2.plot_trisurf(concentrator_points[:, 0], concentrator_points[:, 1], concentrator_points[:, 2], triangles=concenrtrator_tri.simplices, cmap='viridis', edgecolor='k', alpha=0.5)
    #         ax2.plot_trisurf(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], triangles=triangles.simplices, cmap='viridis', edgecolor='k', alpha=0.5)
    #         ax2.scatter3D(concenrtrator_tri[i,0], concenrtrator_tri[i,1], concenrtrator_tri[i,2], color='black', s=50)
    #         # ax2.quiver(
    #         #     concentrator_center[i, 0], concentrator_center[i, 1], concentrator_center[i, 2],
    #         #     r_v[0], r_v[1], r_v[2], 
    #         #     color='yellow', alpha=1, arrow_length_ratio=1e-2, length=0.1
    #         # )
    #         # ax2.quiver(
    #         #     concentrator_center[i, 0], concentrator_center[i, 1], concentrator_center[i, 2],
    #         #     sun_vec[0], sun_vec[1], sun_vec[2], 
    #         #     color='red', alpha=0.5, arrow_length_ratio=1, length=0.1
    #         # )
    #         ax2.quiver(
    #             concentrator_center[i, 0], concentrator_center[i, 1], concentrator_center[i, 2],
    #             v_v[:, 0], v_v[:, 1], v_v[:, 2], 
    #             color='yellow', alpha=1, arrow_length_ratio=1e-7, length=1
    #         )
    #         ax2.quiver(
    #             concentrator_center[i, 0], concentrator_center[i, 1], concentrator_center[i, 2],
    #             v_nv[:, 0], v_nv[:, 1], v_nv[:, 2], 
    #             color='red', alpha=0.5, arrow_length_ratio=1e-7, length=1
    #         )
    #         plt.title(f'Vectores que tienen contribuciones sobre el elemento {i}')
    #         plt.show()
    #         a+=1
    #         # # if i == 10:
    #         break  
    if return_plot == True:
        return pt, t_plot, irr_plot
    return pt

def process_triangle_sun(i, irr_list, sun_positions, dome_vectors,
                         surface_normal, surface_centers, triangles,
                         surface_points, obstacle_mesh, obstacle_points,
                         max_rays, n_iter, return_plotting = False):
    """
    max_rays: número de rayos a evaluar por triángulo (antes llamado eval)
    n_iter: número de iteraciones (opcional)
    """
    pk = 0
    
    sun_vectors = sun_positions - surface_centers[i]
    sun_vectors = np.array(sun_vectors)
    
    if n_iter is None:
        n_iter = len(sun_vectors)
    else:
        n_iter = min(n_iter, len(sun_vectors))
    
    if return_plotting == True:
            irr_plot = np.zeros((len(triangles.simplices), n_iter))
    
    for j in range(n_iter):
        sun_vec = sun_vectors[j]
        norm = np.linalg.norm(sun_vec)
        sun_vec = sun_vec / norm
        
        if obstacle_mesh is None and obstacle_points is None:
            hit = -1
        else:
            hit, _ = robust_ray_intersection(surface_centers[i], sun_vec,
                                             obstacle_mesh, obstacle_points,
                                             eval=max_rays, intersection=True)
        if hit == -1:
            dome_vec = dome_vectors[j]
            norm_dome = np.linalg.norm(dome_vec)
            dome_vec = dome_vec / norm_dome
            normal_vec = surface_normal[i]
            norm_n = np.linalg.norm(normal_vec)
            normal_vec = normal_vec / norm_n
            area = triangle_area(triangles, i, surface_points, total_area=False)
            pk += irr_list[j] * np.clip(-dome_vec @ normal_vec, 0, 1) * area
            if return_plotting == True:
                irr_plot[i, j] = irr_list[j] * np.clip(-dome_vec @ normal_vec, 0, 1) * area
    if return_plotting:
        return irr_plot
    else:
        return pk

def potency_due2sun_parallel(irr_list, sun_positions, dome_vectors,
                           surface_normal, surface_centers, triangles,
                           surface_points, obstacle_mesh=None,
                           obstacle_points=None, eval=50, n_iter=None,
                           n_jobs=-1,time = None, return_plotting = False):
    """
    Versión paralelizada con joblib. Se pasan los objetos Delaunay directamente.
    """
    n_triangles = len(triangles.simplices)
    if return_plotting == False:
        resultados = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_triangle_sun)(
                i, irr_list, sun_positions, dome_vectors,
                surface_normal, surface_centers, triangles,
                surface_points, obstacle_mesh, obstacle_points,
                eval, n_iter
            ) for i in range(n_triangles)
        )
        return sum(resultados)
    else:
        irr_time = Parallel(n_jobs=n_jobs, backend='loky')(
                    delayed(process_triangle_sun)(
                        i, irr_list, sun_positions, dome_vectors,
                        surface_normal, surface_centers, triangles,
                        surface_points, obstacle_mesh, obstacle_points,
                        eval, n_iter, return_plotting = True
                    ) for i in range(n_triangles)
                )
        time_plot = np.array(time[:n_iter])
        irr_time = np.asarray(irr_time)
        irr_time = irr_time.sum(axis=0)
        irr_time = irr_time.sum(axis=0)
        return irr_time, time_plot

def process_triangle_concentrator(i,
                                  irr_list, sun_positions, dome_vectors,
                                  concentrator_mesh, concentrator_points,
                                  concentrator_centers, concentrator_normals,
                                  panel_mesh, panel_points, panel_normals,
                                  reflectivity, eval_steps, n_iter,
                                  plane_concentrator=False, return_plotting = False):
    """
    Worker para un triángulo del concentrador.
    Recibe objetos Delaunay directamente (NO arrays de simplices).
    """
    debuggin_info = []
    sun_vecs = sun_positions - concentrator_centers[i]
    if n_iter is None:
        n_rays = len(sun_vecs)
    else:
        n_rays = n_iter
    if return_plotting == True:
            irr_plot = np.zeros((len(concentrator_mesh.simplices), n_rays))
    pdc = 0.0
    impacted_rays = 0
    
    for j in range(n_rays):
        sun_vec = sun_vecs[j]
        norm_s = np.linalg.norm(sun_vec)
     
        sun_dir = sun_vec / norm_s
        
        n_conc = concentrator_normals[i]
        norm_nc = np.linalg.norm(n_conc)
     
        n_conc = n_conc / norm_nc
        
        if np.dot(sun_dir, n_conc) > 0:
            hit, _ = robust_ray_intersection(concentrator_centers[i], sun_vec,
                                             concentrator_mesh, concentrator_points,
                                             exclution=[i], intersection=True, eval=eval_steps)
            if hit == -1:
                c_v = -dome_vectors[j]
                norm_cv = np.linalg.norm(c_v)
         
                c_v = c_v / norm_cv
                
                r_v = unitary_sun_reflection(c_v, n_conc)
                r_v = r_v / np.linalg.norm(r_v)
                r_v = -r_v
                
                hit_panel, _ = robust_ray_intersection(concentrator_centers[i], r_v,
                                                       panel_mesh, panel_points,
                                                       intersection=True, eval=eval_steps)
                if hit_panel != -1:
                    r = 1
                    normal_panel = panel_normals[hit_panel]
                    norm_np = np.linalg.norm(normal_panel)
                    normal_panel = normal_panel / norm_np
                    rv = -r_v / np.linalg.norm(r_v)
                    area_conc = triangle_area(concentrator_mesh, i, concentrator_points, total_area=False)
                    a_p = triangle_area(panel_mesh, hit_panel, panel_points, total_area=False)
                    contrib = irr_list[j]  * (reflectivity ** r) * area_conc *np.dot(sun_dir, n_conc)  * np.clip(np.dot(rv, normal_panel), 0, 1)
                    # contrib = irr_list[j] * (reflectivity**r) * a_p *np.clip(np.dot(rv, normal_panel), 0, 1)
                    # debuggin_info.append((np.dot(rv, normal_panel), rv.copy(), normal_panel.copy(), area_conc.copy(), irr_list[j], r))
                    # impacted_rays += 1
                    if return_plotting:
                        irr_plot[i,j] = contrib
                    pdc += contrib
                else:
                        # Reflexiones múltiples
                        r_vf, r_c, hit_idx, r, _ = self_reflection_optimized(
                            concentrator_normals, hit, r_v, concentrator_centers[hit], hit,
                            concentrator_mesh, concentrator_points, concentrator_centers,
                            ensure_reflection=False
                        )
                        r = r + 1
                        hit_panel2, _ = robust_ray_intersection(concentrator_centers[hit_idx], r_vf,
                                                                panel_mesh, panel_points,
                                                                intersection=True, eval=eval_steps)
                        if hit_panel2 != -1:
                            normal_panel = panel_normals[hit_panel2]
                            norm_np = np.linalg.norm(normal_panel)
                            # if norm_np > 0:
                            normal_panel = normal_panel / norm_np
                            r_vf = -r_vf / np.linalg.norm(r_vf)
                            area_conc = triangle_area(concentrator_mesh, i, concentrator_points, total_area=False)
                            a_p = triangle_area(panel_mesh, hit_panel, panel_points, total_area=False)
                            # impacted_rays += 1
                            # pdc += irr_list[j]  * (reflectivity ** r) * a_p * np.clip(np.dot(r_vf, normal_panel), 0, 1)
                            if return_plotting:
                                irr_plot[i,j] = irr_list[j]  * (reflectivity ** r) * area_conc * np.clip(np.dot(r_vf, concentrator_normals[hit_idx]), 0, 1)  * np.clip(np.dot(rv, normal_panel), 0, 1)
                            pdc += irr_list[j]  * (reflectivity ** r) * area_conc * np.clip(np.dot(r_vf, concentrator_normals[hit_idx]), 0, 1)  * np.clip(np.dot(rv, normal_panel), 0, 1)
    #                         debuggin_info.append((np.dot(r_vf, normal_panel), r_vf.copy(), normal_panel.copy(), area_conc.copy(), irr_list[j], r))
    # debuggin_info.append((impacted_rays, n_rays, pdc))
    # if impacted_rays == 0:
    #     impacted_rays = 1
    if return_plotting:
        return irr_plot
    else:
        return pdc 

def process_triangle_concentrator2(i,
                                  irr_list, sun_positions, dome_vectors,
                                  concentrator_mesh, concentrator_points,
                                  concentrator_centers, concentrator_normals,
                                  panel_mesh, panel_points, panel_normals,
                                  reflectivity, eval_steps, n_iter,
                                  plane_concentrator=False, return_plotting = False):
    """
    Worker para un triángulo del concentrador.
    Recibe objetos Delaunay directamente (NO arrays de simplices).
    """
    debuggin_info = []
    sun_vecs = sun_positions - concentrator_centers[i]
    if n_iter is None:
        n_rays = len(sun_vecs)
    else:
        n_rays = n_iter
    if return_plotting == True:
            irr_plot = np.zeros((len(concentrator_mesh.simplices), n_rays))
    pdc = 0.0
    impacted_rays = 0
    
    for j in range(n_rays):
        sun_vec = sun_vecs[j]
        norm_s = np.linalg.norm(sun_vec)
     
        sun_dir = sun_vec / norm_s
        
        n_conc = concentrator_normals[i]
        norm_nc = np.linalg.norm(n_conc)
     
        n_conc = n_conc / norm_nc
        
        if np.dot(sun_dir, n_conc) > 0:
            hit, _ = robust_ray_intersection(concentrator_centers[i], sun_vec,
                                             concentrator_mesh, concentrator_points,
                                             exclution=[i], intersection=True, eval=eval_steps)
            if hit == -1:
                c_v = -dome_vectors[j]
                norm_cv = np.linalg.norm(c_v)
         
                c_v = c_v / norm_cv
                
                r_v = unitary_sun_reflection(c_v, n_conc)
                r_v = r_v / np.linalg.norm(r_v)
                r_v = -r_v
                
                hit_panel, _ = robust_ray_intersection(concentrator_centers[i], r_v,
                                                       panel_mesh, panel_points,
                                                       intersection=True, eval=eval_steps)
                if hit_panel != -1:
                    r = 1
                    normal_panel = panel_normals[hit_panel]
                    norm_np = np.linalg.norm(normal_panel)
                    normal_panel = normal_panel / norm_np
                    rv = -r_v / np.linalg.norm(r_v)
                    area_conc = triangle_area(concentrator_mesh, i, concentrator_points, total_area=False)
                    a_p = triangle_area(panel_mesh, hit_panel, panel_points, total_area=False)
                    # contrib = irr_list[j]  * (reflectivity ** r) * area_conc *np.dot(sun_dir, n_conc)  * np.clip(np.dot(rv, normal_panel), 0, 1)
                    contrib = irr_list[j] * (reflectivity**r) * a_p *np.clip(np.dot(rv, normal_panel), 0, 1)
                    # debuggin_info.append((np.dot(rv, normal_panel), rv.copy(), normal_panel.copy(), area_conc.copy(), irr_list[j], r))
                    # impacted_rays += 1
                    if return_plotting:
                        irr_plot[i,j] = contrib
                    pdc += contrib
                else:
                        # Reflexiones múltiples
                        r_vf, r_c, hit_idx, r, _ = self_reflection_optimized(
                            concentrator_normals, hit, r_v, concentrator_centers[hit], hit,
                            concentrator_mesh, concentrator_points, concentrator_centers,
                            ensure_reflection=False
                        )
                        r = r + 1
                        hit_panel2, _ = robust_ray_intersection(concentrator_centers[hit_idx], r_vf,
                                                                panel_mesh, panel_points,
                                                                intersection=True, eval=eval_steps)
                        if hit_panel2 != -1:
                            normal_panel = panel_normals[hit_panel2]
                            norm_np = np.linalg.norm(normal_panel)
                            # if norm_np > 0:
                            normal_panel = normal_panel / norm_np
                            r_vf = -r_vf / np.linalg.norm(r_vf)
                            area_conc = triangle_area(concentrator_mesh, i, concentrator_points, total_area=False)
                            a_p = triangle_area(panel_mesh, hit_panel, panel_points, total_area=False)
                            # impacted_rays += 1
                            pdc += irr_list[j]  * (reflectivity ** r) * a_p * np.clip(np.dot(r_vf, normal_panel), 0, 1)
                            if return_plotting:
                                irr_plot[i,j] = irr_list[j]  * (reflectivity ** r) * a_p * np.clip(np.dot(r_vf, normal_panel), 0, 1)
                            # pdc += irr_list[j]  * (reflectivity ** r) * area_conc * np.clip(np.dot(r_vf, concentrator_normals[hit_idx]), 0, 1)  * np.clip(np.dot(rv, normal_panel), 0, 1)
    #                         debuggin_info.append((np.dot(r_vf, normal_panel), r_vf.copy(), normal_panel.copy(), area_conc.copy(), irr_list[j], r))
    # debuggin_info.append((impacted_rays, n_rays, pdc))
    # if impacted_rays == 0:
    #     impacted_rays = 1
    if return_plotting:
        return irr_plot
    else:
        return pdc 


def potency_due2concentrator_parallel(
    irr_list,
    sun_positions,
    dome_vectors,
    concentrator_mesh,
    concentrator_centers,
    concentrator_normals,
    concentrator_points,
    panel_mesh,
    panel_points,
    panel_normals,
    reflectivity=0.97,
    eval_steps=5,
    n_iter=None,
    plane_concentrator=False,
    n_jobs=-1,time = None, return_plotting= False
):
    """
    Versión paralela que pasa los objetos Delaunay directamente a los workers.
    """
    n_triangles = len(concentrator_mesh.simplices)
    
    # # Nota: concentrator_points y panel_points son arrays de puntos (necesarios para triangle_area)
    if return_plotting == False:
        resultados = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_triangle_concentrator2)(
                i,
                irr_list, sun_positions, dome_vectors,
                concentrator_mesh, concentrator_points,
                concentrator_centers, concentrator_normals,
                panel_mesh, panel_points, panel_normals,
                reflectivity, eval_steps, n_iter,
                plane_concentrator
            ) for i in range(n_triangles)
        )
        return sum(resultados)
    else:
        irr_plot = Parallel(n_jobs=n_jobs, backend='loky')(
                    delayed(process_triangle_concentrator2)(
                        i,
                        irr_list, sun_positions, dome_vectors,
                        concentrator_mesh, concentrator_points,
                        concentrator_centers, concentrator_normals,
                        panel_mesh, panel_points, panel_normals,
                        reflectivity, eval_steps, n_iter,
                        plane_concentrator, return_plotting=True
                    ) for i in range(n_triangles)
                )
        time_plot = np.array(time[:n_iter])
        irr_plot = np.asarray(irr_plot)
        irr_plot = irr_plot.sum(axis=0)
        irr_plot = irr_plot.sum(axis=0)
        return irr_plot, time_plot

    

def process_triangle_sun_se(i, irr_list, sun_position, dome_vector, surface_normal, surface_centers, triangles, surface_points, obstacle_mesh = None, obstacle_points = None, n_iter = None, retur_plotting = False, glass_index = 1):
    """
    Worker para el calculo de un triángulo de la persiana solar
    """
    pk = 0
    if n_iter is None:
        n_iter = len(dome_vector)
    irr_plot = np.zeros((len(triangles.simplices), n_iter))
    sun_vectors = sun_position - surface_centers[i]
    sun_vectors = np.array(sun_vectors)
    element_normal = surface_normal[i] / np.linalg.norm(surface_normal[i])
    for j in range(n_iter):
        sun_vec = sun_vectors[j]
        norm_sun = np.linalg.norm(sun_vec)
        sun_vec = sun_vec / norm_sun
        if np.dot(sun_vec, element_normal) > 0:
            if obstacle_mesh is None and obstacle_points is  None:
                hit = -1
            else:
                hit, _ = robust_ray_intersection(surface_centers[i], sun_vec, obstacle_mesh, obstacle_points,intersection=True)

            if hit == -1:
                sun_vec = dome_vector[j] / np.linalg.norm(dome_vector[j])
                pk += irr_list[j] * np.clip(-sun_vec @ element_normal, 0, 1) * triangle_area(triangles, i , surface_points, total_area=False) * Transmitance(1, glass_index, np.arccos(np.clip(-sun_vec @ element_normal, 0, 1))).transmitance
                if retur_plotting == True:
                    irr_plot[i, j] = irr_list[j] * np.clip(-sun_vec @ element_normal, 0, 1) * triangle_area(triangles, i , surface_points, total_area=False) * Transmitance(1, glass_index, np.arccos(np.clip(-sun_vec @ element_normal, 0, 1))).transmitance
    if retur_plotting == True:
        return pk, irr_plot
    return pk

def potency_due2sun_se_parallel(irr_list, sun_position, dome_vector, surface_normal, surface_centers, triangles, surface_points, obstacle_mesh = None, obstacle_points = None, n_iter = None, time = None, retur_plotting = False, n_jobs = -1):
    """
    Paralelización del calculo de la potencia incidente sobre una persiana solar
    """
    n_triangles = len(triangles.simplices)

    if n_iter is None:
        n_iter = len(dome_vector)

    if retur_plotting == True:
        results = Parallel(n_jobs=n_jobs, backend='loky', prefer='threads')(
            delayed(process_triangle_sun_se)(
                i, 
                irr_list, 
                sun_position, 
                dome_vector,
                surface_normal, 
                surface_centers, 
                triangles, 
                surface_points, 
                obstacle_mesh, 
                obstacle_points, 
                n_iter, 
                retur_plotting 
            ) for i in range(n_triangles)
        )
        p_total, irr_time = zip(*results)   
        p_total = np.asarray(p_total)
        irr_time = np.asarray(irr_time)
        irr_time = irr_time.sum(axis=0)
        irr_time = irr_time.sum(axis=0)
        time_plot = np.array(time[:n_iter])
        return sum(p_total), irr_time, time_plot
    else:
        p_total = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process_triangle_sun_se)(
                i, 
                irr_list, 
                sun_position, 
                dome_vector,
                surface_normal, 
                surface_centers, 
                triangles, 
                surface_points, 
                obstacle_mesh, 
                obstacle_points, 
                n_iter, 
                retur_plotting 
            ) for i in range(n_triangles)
        )
        return sum(p_total)
    