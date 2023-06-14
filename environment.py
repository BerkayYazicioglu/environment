import numpy as np
from scipy.spatial import ConvexHull
from ordered_set import OrderedSet
import noise
import json
from utils.bresenham import line

""" octaves -- specifies the number of passes for generating fBm noise,
    defaults to 1 (simple noise).
    
    persistence -- specifies the amplitude of each successive octave relative
    to the one below it. Defaults to 0.5 (each higher octave's amplitude
    is halved). Note the amplitude of the first pass is always 1.0.
    
    lacunarity -- specifies the frequency of each successive octave relative
    to the one below it, similar to persistence. Defaults to 2.0.
    
    repeatx, repeaty, repeatz -- specifies the interval along each axis when 
    the noise values repeat. This can be used as the tile size for creating 
    tileable textures
    
    base -- specifies a fixed offset for the input coordinates. Useful for
    generating different noise textures with the same repeat interval
 """


class Environment:
    def __init__(
        self,
        dimensions=(1000, 1000),
        shape=(50, 50),
        scale=100.0,
        octaves=3,
        persistence=0.5,
        lacunarity=2.0,
        filename=None,
        load=False,
    ):
        self.removedPoints = []
        self.selectedPoints = OrderedSet()
        self.selectedArea = []
        self.cache = np.zeros(shape)
        self.dimensions = dimensions
        self.shape = shape
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.data = dict(
            terrain=np.zeros(shape),
            destruction=np.zeros(shape),
            population=np.zeros(shape),
            roughness=np.zeros(shape),
        )
        self.filename = filename
        self.mode = "terrain"
        if load:
            with open(f"data/{self.filename}.json", "r") as fp:
                data = json.load(fp)
                self.dimensions = data["size"]
                self.shape = data["grid"]
                self.data = dict(
                    terrain=np.array(data["terrain"]),
                    destruction=np.array(data["destruction"]),
                    population=np.array(data["population"]),
                    roughness=np.array(data["roughness"]),
                )
            self.x_values = np.linspace(0, self.dimensions[0], self.shape[0])
            self.y_values = np.linspace(0, self.dimensions[1], self.shape[1])
            self.x, self.y = np.meshgrid(self.x_values, self.y_values)
        else:
            self.x_values = np.linspace(0, dimensions[0], shape[0])
            self.y_values = np.linspace(0, dimensions[1], shape[1])
            self.x, self.y = np.meshgrid(self.x_values, self.y_values)
            self.generate_perlin_noise(
                self.x, self.y, scale, octaves, persistence, lacunarity
            )
            if filename is not None:
                self.save()

    def save(self):
        data = dict(
            size=self.dimensions,
            grid=self.shape,
            terrain=self.data["terrain"].tolist(),
            destruction=self.data["destruction"].tolist(),
            population=self.data["population"].tolist(),
            roughness=self.data["roughness"].tolist(),
        )
        with open(f"data/{self.filename}.json", "w") as fp:
            json.dump(data, fp)

    def find_index(self, x_pos, y_pos):
        x_diff = np.abs(np.subtract.outer(x_pos, self.x_values))
        y_diff = np.abs(np.subtract.outer(y_pos, self.y_values))
        return np.column_stack((np.argmin(x_diff, axis=1), np.argmin(y_diff, axis=1)))

    def generate_perlin_noise(self, x, y, scale, octaves, persistence, lacunarity):
        x = x.ravel()
        y = y.ravel()
        idx = self.find_index(x, y)
        for i in idx:
            self.data["terrain"][i[0]][i[1]] += noise.pnoise2(
                self.x_values[i[0]] / scale,
                self.y_values[i[1]] / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=0,
            )

    def apply_value(self, value):
        self.cache[self.selectedArea[:, 0], self.selectedArea[:, 1]] += value

    def find_selected_area(self):
        def point_inside_polygon(point, vertices):
            x, y = point
            n = len(vertices)
            inside = False
            p1x, p1y = vertices[0]
            for i in range(n + 1):
                p2x, p2y = vertices[i % n]
                if (y > min(p1y, p2y)) and (y <= max(p1y, p2y)):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            return inside

        A = np.array(list(self.selectedPoints))
        if len(self.selectedPoints) <= 1:
            self.selectedArea = self.find_index(np.array(A[:, 0]), np.array(A[:, 1]))
        elif len(self.selectedPoints) == 2:
            idx = self.find_index(np.array(A[:, 0]), np.array(A[:, 1]))
            self.selectedArea = np.array(
                line(idx[0, 0], idx[0, 1], idx[1, 0], idx[1, 1])
            )
        else:
            A = np.array(list(self.selectedPoints))[:, :2]
            hull = ConvexHull(A)
            area_vertices = self.find_index(
                A[hull.vertices][:, 0], A[hull.vertices][:, 1]
            )
            grid = self.find_index(self.x.flatten(), self.y.flatten())
            is_inside = np.array(
                [point_inside_polygon(point, area_vertices) for point in grid]
            )
            self.selectedArea = np.column_stack((grid[:, 0], grid[:, 1]))[is_inside]
