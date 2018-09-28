import numpy as np

class Container:

    def __init__(self,**kwargs):
        self.data = kwargs.get('data',np.array([]))
        self.dimension = kwargs.get('dimension',0)
        self.sizePerDimension = kwargs.get('sizePerDimension',np.array([]))             # in pixels
        self.spacingPerDimension = kwargs.get('spacingPerDimension',np.array([]))
        self.origin = - 0.5 *((self.sizePerDimension - 1) * self.spacingPerDimension)

    def pixel_to_world(self, pixel):
        world = self.origin + pixel * self.spacingPerDimension
        return world

    def world_to_pixel(self, world):
        pixel = world / self.spacingPerDimension - self.origin
        return pixel


    def set_spacing(self, new_spacing):
        self.spacingPerDimension = new_spacing

    def interpolation_2d(self, world_coordinates):
        pixel_point = self.world_to_pixel(world_coordinates)
        interpolation_origin = pixel_point.astype(np.int)

        zo = interpolation_origin + np.array([0,1])
        oz = interpolation_origin + np.array([1,0])
        oo = interpolation_origin + np.array([1,1])
        zz = interpolation_origin

        x1_value = (oz[0] - pixel_point[0]) * self.data[tuple(zz)] + (pixel_point[0] - zz[0]) * self.data[tuple(oz)]
        x2_value = (oo[0] - pixel_point[0]) * self.data[tuple(zo)] + (pixel_point[1] - zo[0]) * self.data[tuple(oo)]

        x1 = oz
        x1[0] -= pixel_point[0]
        x2 = oo
        x2[0] -= pixel_point[0]


        value = x1_value * (x2[1] - pixel_point[1]) + x2_value * (pixel_point[1] - x1[1])

        return value

    def interpolation_2d_asdf(self, world_coordinates):
        world_coordinates = np.array(world_coordinates)
        assert world_coordinates.shape == (2,)
        pixel_coordinates = self.world_to_pixel(world_coordinates)
        x1, y1 = np.floor(pixel_coordinates).astype('int')
        x2, y2 = np.ceil(pixel_coordinates).astype('int')
        f1, f2, f3, f4 = self.data[x1, y1],  self.data[x2, y1], self.data[x2, y2], self.data[x1, y2]
        i1 = np.interp(world_coordinates[0], [x1, x2], [f1, f2])
        i2 = np.interp(world_coordinates[0], [x1, x2], [f3, f4])
        res = np.interp(world_coordinates[1], [y1, y2], [i1, i2])
        return res


    def interpolation_nd(self,world_coordinates):
        #interpolationSpace = np.zeros(tuple(np.ones(self.dimension)*2))
        pixel_point = self.world_to_pixel(world_coordinates)
        interpolationOriginIndex = pixel_point.astype(np.int)
        l = []
        for i in range(2**self.dimension):
            l.append(self.data[tuple(np.array([int(d) for d in bin(i)[2:].zfill(self.dimension)])+ interpolationOriginIndex)])

        interpolationSpace = np.array(l).reshape(tuple(np.ones(self.dimension,dtype=int)*2))


        #def recursive_step(hypercube,dim):
        return interpolationSpace


if __name__ == '__main__':
    a = Container(data=np.eye(2),dimension=2,sizePerDimension=np.array([2,2]),spacingPerDimension=np.array([1,1]))
    print('{} \n {}'.format(a.data,a.interpolation_2d(a.pixel_to_world(np.array([0.75,0])))))
   # print(a.pixel_to_world(np.array([0.5,0.5])))
    #print(a.origin)
    print()
    print(a.interpolation_nd(a.pixel_to_world(np.array([0.75,0]))))