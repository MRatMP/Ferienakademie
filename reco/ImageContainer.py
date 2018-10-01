import numpy as np
import matplotlib.pyplot as plt
class Container:

    def __init__(self, data, spacing):
        self.data = self.preprocess(data)
        self.spacing = spacing
        self.pixel_dimension = np.array(self.data.shape)
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
        if np.min(pixel_coordinates) < 0 or np.any(np.greater_equal(np.ceil(pixel_coordinates), self.pixel_dimension)):
            return 0
        x1, y1 = np.floor(pixel_coordinates).astype('int')
        x2, y2 = np.ceil(pixel_coordinates).astype('int')
        #if x1 > self.pixel_dimension[0] or x2 >self.pixel_dimension[0] or y1 > self.pixel_dimension[1] or y2 >
        f1, f2, f3, f4 = self.data[x1, y1],  self.data[x2, y1], self.data[x2, y2], self.data[x1, y2]
        i1 = np.interp(world_coordinates[0], [x1, x2], [f1, f2])
        i2 = np.interp(world_coordinates[0], [x1, x2], [f3, f4])
        res = np.interp(world_coordinates[1], [y1, y2], [i1, i2])
        return res

    def inverse_interpolate(self):


    def show(self):
        plt.imshow(self.data)
        plt.show()

    def preprocess(self, data):
        if len(data.shape) == 3:
            return data[:,:,0]
        return data




if __name__ == '__main__':
    a = Container(data=np.eye(2),spacing=0.1)
    print('{}        {}'.format(a.data,a.interpolation_2d(a.pixel_to_world(np.array([0.75,0])))))
   # print(a.pixel_to_world(np.array([0.5,0.5])))
    #print(a.origin)
