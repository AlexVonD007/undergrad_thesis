import numpy as np
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.IGESControl import IGESControl_Writer
from datetime import datetime

class Reflectivity:
    def __init__(self, n_i, n_t, theta_i):
        self.n1 = n_i
        self.n2 = n_t
        self.theta_i = theta_i 

        cos_theta_t = np.sqrt(1 - (self.n1/self.n2)**2 * np.sin(self.theta_i)**2)
        self.theta_t = np.arcsin((self.n1/self.n2)*np.sin(self.theta_i)) 
        self.r_t =  (self.n1*np.cos(self.theta_i) - self.n2*cos_theta_t)/(self.n1*np.cos(self.theta_i) + self.n2*cos_theta_t)
        self.r_p =  (self.n2*np.cos(self.theta_i) - self.n1*cos_theta_t)/(self.n1*cos_theta_t + self.n2*np.cos(self.theta_i))
        self.reflectance = (self.r_t*self.r_t.conj() + self.r_p*self.r_p.conj())/2
        self.reflectance = self.reflectance.real

class Transmitance:
    def __init__(self, n_i, n_t, theta_i):
        self.n1 = n_i
        self.n2 = n_t
        self.theta_i = theta_i 
        self.theta_t = np.arcsin((self.n1/self.n2)*np.sin(self.theta_i))
        self.t_t = (2*self.n1*np.cos(self.theta_i))/(self.n1*np.cos(self.theta_i) + self.n2*np.cos(self.theta_t))
        self.t_p = (2*self.n1*np.cos(self.theta_i))/(self.n1*np.cos(self.theta_t) + self.n2*np.cos(self.theta_i))
        self.transmitance = (self.t_p**2 + self.t_t**2)*0.5


class Point_cloud2iges:
    def __init__(self, X, Y, Z, unit = 'mm'):
        conversion = {
            'm': 1e3,
            'mm': 1,
            'um': 1e-3,
            'µm': 1e-6,
            'nm': 1e-6
        }

        if unit not in conversion:
            raise ValueError(
                f"Unit '{unit}' not valid. "
                f"Use: {list(conversion.keys())}"
            )

        factor = conversion[unit]

        self.X = X * factor
        self.Y = Z * factor
        self.Z = Y * factor

        ny, nx = self.Z.shape
        
        self.points = TColgp_Array2OfPnt(
            1, nx,
            1, ny
        )
    
        for j in range(ny):
    
            for i in range(nx):
    
                self.points.SetValue(
                    i + 1,
                    j + 1,
                    gp_Pnt(
                        float(self.X[j, i]),
                        float(self.Y[j, i]),
                        float(self.Z[j, i])
                    )
                )
    
    
        self.surface_builder = GeomAPI_PointsToBSplineSurface(
            self.points
        )
    
        self.surface = self.surface_builder.Surface()
        
        self.face = BRepBuilderAPI_MakeFace(
            self.surface,
            1e-6
        ).Face()

    def save(self, filename):
        self.writer = IGESControl_Writer()
        self.writer.AddShape(self.face)
        if '.iges' not in filename:
            filename = filename + '.iges'

        self.writer.Write(filename)
        print(f'File generated: {filename}')

class TraceProSolarSource:

    def __init__(
        self,
        directions,
        radius: float = 1000.0,
        N: int = 50,
        esc: float = 1000.0,
        irradiance: float = 406.05,     # W/m^2
        wavelength_um: float = 0.5461,
        up_global=(0.0, 1.0, 0.0),
        swap_yz: bool = False,
    ):
        directions = np.atleast_2d(np.asarray(directions, dtype=float))

        if swap_yz:
            directions = self._swap_yz(directions)

        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        self.directions = directions / norms   # (M, 3), cada fila normalizada, ejes de TracePro
        M = self.directions.shape[0]

        self.radius = radius
        self.N = N
        self.esc = esc
        self.wavelength_um = wavelength_um

        up = np.array(up_global, dtype=float)
        self.up_global_default = up / np.linalg.norm(up)

        # --- Irradiancia y potencia total por dirección ---
        irr = np.atleast_1d(np.asarray(irradiance, dtype=float))
        if irr.size == 1:
            irr = np.full(M, irr[0])
        elif irr.size != M:
            raise ValueError(
                f"'irradiance' debe ser un número, o una lista de longitud "
                f"igual al número de direcciones ({M}); recibí {irr.size} valores."
            )
        self.irradiance = irr                                          # (M,) W/m^2

        area = np.pi * (radius / 1000.0) ** 2                          # área del disco, m^2
        self.P_total = self.irradiance * area                          # (M,) W, uno por dirección
        self._local_coords = self._build_ring_pattern()                # (n_rays, 2)
        self.n_rays = self._local_coords.shape[0]
        self.P_ray = self.P_total / self.n_rays                        # (M,) W por rayo, por dirección

    # ------------------------------------------------------------------
    @staticmethod
    def _swap_yz(vectors: np.ndarray) -> np.ndarray:
        """
        Intercambia los ejes Y y Z de uno o varios vectores (X, Y, Z) -> (X, Z, Y).
        Útil cuando los datos de entrada vienen de un sistema de coordenadas
        Z-up (común en geodesia, CAD o modelos solares) y deben convertirse
        al sistema Y-up que usa TracePro.
        """
        swapped = vectors.copy()
        swapped[:, [1, 2]] = swapped[:, [2, 1]]
        return swapped

    # ------------------------------------------------------------------
    def _build_ring_pattern(self) -> np.ndarray:
        """
        Calcula las coordenadas locales (a, b) del patrón de anillos concéntricos.
        Es independiente de la dirección, por eso se calcula una sola vez.

        El centro (radio 0) cuenta como el primer "anillo". Los N-1 anillos
        restantes quedan equiespaciados radialmente en pasos de radius/N,
        cubriendo desde radius/N hasta (N-1)*radius/N, sin solaparse.
        """
        coords = [(0.0, 0.0)]  # punto central
        ring_radii = self.radius * np.arange(1, self.N) / self.N
        for i, r_i in enumerate(ring_radii, start=1):
            n_points = 6 * i   # más puntos en anillos externos -> densidad ~constante
            theta = 2 * np.pi * np.arange(n_points) / n_points
            coords.extend(zip(r_i * np.cos(theta), r_i * np.sin(theta)))
        return np.array(coords)

    # ------------------------------------------------------------------
    def _basis_for_direction(self, direction: np.ndarray):
        """
        Calcula la base ortonormal (u, v) perpendicular a 'direction',
        usando el mismo vector ascendente que TracePro. Si la dirección
        queda casi paralela a la referencia (caso degenerado), cambia
        automáticamente a un eje de referencia distinto.
        """
        up_ref = self.up_global_default
        if abs(np.dot(direction, up_ref)) > 0.99:
            up_ref = np.array([0.0, 0.0, 1.0])

        u_vec = np.cross(up_ref, direction)
        u_vec /= np.linalg.norm(u_vec)
        v_vec = np.cross(direction, u_vec)
        return u_vec, v_vec

    # ------------------------------------------------------------------
    def rays_for_direction(self, index: int = 0):
        """Devuelve (positions, direction, P_ray) para la dirección `index`,
        usando la potencia total propia de esa dirección (self.P_total[index])."""
        direction = self.directions[index]
        u_vec, v_vec = self._basis_for_direction(direction)
        center = -direction * self.esc

        a = self._local_coords[:, 0][:, None]
        b = self._local_coords[:, 1][:, None]
        positions = center + a * u_vec + b * v_vec   # (n_rays, 3)

        return positions, direction, self.P_ray[index]

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt(v: float) -> str:
        """Formatea un número como TracePro (estilo Fortran):
        mantisa con 15 decimales y exponente de 3 dígitos, p.ej. -5.500000000000000e-001
        """
        s = f"{v:.15e}"
        mantissa, exp = s.split("e")
        exp_int = int(exp)
        sign = "+" if exp_int >= 0 else "-"
        return f"{mantissa}e{sign}{abs(exp_int):03d}"

    # ------------------------------------------------------------------
    def _rows_for_indices(self, indices, weights=None):
        """
        Arma las filas (positions, direction, flujo) para una lista de índices
        de dirección. Cada dirección usa su propia potencia total
        (self.P_total[index], calculada a partir de su irradiancia individual).

        weights: opcional, multiplicador extra por dirección (por defecto 1,
                 es decir, se respeta la potencia real de cada una tal cual).
                 Útil, por ejemplo, para escalar por duración si combinas
                 direcciones que no representan el mismo intervalo de tiempo.
        """
        indices = list(indices)
        M = len(indices)

        if weights is None:
            weights = np.ones(M)
        else:
            weights = np.asarray(weights, dtype=float)

        all_positions, all_dirs, all_flux = [], [], []
        for w, idx in zip(weights, indices):
            positions, direction, P_ray = self.rays_for_direction(idx)
            all_positions.append(positions)
            all_dirs.append(np.tile(direction, (self.n_rays, 1)))
            all_flux.append(np.full(self.n_rays, w * P_ray))

        return (
            np.vstack(all_positions),
            np.vstack(all_dirs),
            np.concatenate(all_flux),
        )

    # ------------------------------------------------------------------
    def _write_file(self, path: str, positions, dirs, flux):
        """Escribe el encabezado + filas de datos de TracePro a 'path'."""
        with open(path, "w") as f:
            f.write(f"!! Source file: {path}\n")
            f.write("Version:1.1\n")
            f.write("TracePro Release: 7 1 0\n")
            f.write("Data for Block 1 Surface 1\n")
            f.write("Linear Units in mm\n")
            f.write(f"Data generated at {datetime.now().strftime('%H:%M:%S %B %d, %Y')}\n")
            f.write(f"Wavelength {self.wavelength_um}\n")

            for i in range(positions.shape[0]):
                row = " ".join(
                    self._fmt(v) for v in [
                        positions[i, 0], positions[i, 1], positions[i, 2],
                        dirs[i, 0], dirs[i, 1], dirs[i, 2],
                        flux[i],
                    ]
                )
                f.write(row + "\n")

    # ------------------------------------------------------------------
    def save(self, path: str, index: int = 0):
        """
        Guarda un archivo fuente de TracePro (File Source) para la dirección
        `index` de self.directions.
        """
        positions, direction, P_ray = self.rays_for_direction(index)
        dirs = np.tile(direction, (self.n_rays, 1))
        flux = np.full(self.n_rays, P_ray)

        self._write_file(path, positions, dirs, flux)
        print(f"Archivo '{path}' generado con {self.n_rays} rayos (dirección #{index}: {direction}).")

    # ------------------------------------------------------------------
    def save_all(self, prefix: str = "solar_source"):
        """Guarda un archivo .txt por cada dirección en self.directions."""
        rutas = []
        for i in range(len(self.directions)):
            path = f"{prefix}_{i:03d}.txt"
            self.save(path, index=i)
            rutas.append(path)
        return rutas

    # ------------------------------------------------------------------
    def save_combined(self, path: str, indices=None, weights=None):
        """
        Guarda UN SOLO archivo con los rayos de varias direcciones combinados
        (cada rayo conserva su propia dirección individual; el formato de
        TracePro lo permite sin problema, ya que la dirección se especifica
        por rayo, no de forma global).

        indices : lista de índices de self.directions a incluir (por defecto,
                   todas).
        weights : multiplicador extra de potencia por dirección incluida
                   (por defecto 1, se respeta la potencia real de cada una).
        """
        if indices is None:
            indices = range(len(self.directions))

        positions, dirs, flux = self._rows_for_indices(indices, weights)
        self._write_file(path, positions, dirs, flux)

        print(
            f"Archivo '{path}' generado con {positions.shape[0]} rayos "
            f"({len(list(indices))} direcciones combinadas)."
        )