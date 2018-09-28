import numpy as np

class Container:

    def __init__(self, data, spacing):
        self.data = data
        self.spacing = spacing
        self.pixel_dimension = np.array(data.shape)
        self.world_dimensions = self.pixel_dimension * spacing
        self.origin = - 0.5 *((self.pixel_dimension - 1) * spacing)

    def pixel_to_world(self, pixel):
        world = (pixel * self.spacing) + self.origin
        return world

    def world_to_pixel(self, world):
        pixel = (world - self.origin)/self.spacing
        return pixel

    def interpolation_2d(self, world_coordinates):
        world_coordinates = np.array(world_coordinates)
        assert world_coordinates.shape == (2,)
        pixel_coordinates = self.world_to_pixel(world_coordinates)
        # handle outside pixels
        if np.min(pixel_coordinates) < 0 or not np.all(np.greater(self.pixel_dimension, pixel_coordinates)):
            return 0
        x1, y1 = np.floor(pixel_coordinates).astype('int')
        x2, y2 = np.ceil(pixel_coordinates).astype('int')
        f1, f2, f3, f4 = self.data[x1, y1],  self.data[x2, y1], self.data[x2, y2], self.data[x1, y2]
        i1 = np.interp(world_coordinates[0], [x1, x2], [f1, f2])
        i2 = np.interp(world_coordinates[0], [x1, x2], [f3, f4])
        res = np.interp(world_coordinates[1], [y1, y2], [i1, i2])
        return res


if __name__ == '__main__':
    a = Container(data=np.eye(2),dimension=2,sizePerDimension=np.array([2,2]),spacingPerDimension=np.array([1,1]))
    print('{} \n {}'.format(a.data,a.interpolation_2d(a.pixel_to_world(np.array([0.75,0])))))
   # print(a.pixel_to_world(np.array([0.5,0.5])))
    #print(a.origin)
    print()
    print(a.interpolation_nd(a.pixel_to_world(np.array([0.75,0]))))