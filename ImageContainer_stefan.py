import numpy as np
from numba import jit,float32,int8

class Container:

    def __init__(self,**kwargs):
        self.data = kwargs.get('data',np.array([]))
        self.dimension = kwargs.get('dimension',self.data.ndim)
        self.sizePerDimension = kwargs.get('sizePerDimension',np.array(self.data.shape))
        self.spacingPerDimension = kwargs.get('spacingPerDimension',np.ones_like(self.sizePerDimension))
        self.origin = - 0.5 *((self.sizePerDimension - 1) * self.spacingPerDimension)

    def pixel_to_world(self, pixel):
        world = self.origin + pixel * self.spacingPerDimension
        return world

    def line_to_world(self,point):
        direction = -self.origin / np.sqrt(self.origin.dot(self.origin.T))
        world = self.origin + direction * self.spacingPerDimension * point
        return world

    def world_to_line(self,world):
        point = np.sqrt((world - self.origin).dot((world - self.origin).T))/self.spacingPerDimension
        return point

    def world_to_pixel(self,world):
        pixel = world / self.spacingPerDimension - self.origin
        return pixel

    def set_origin(self,new_origin):
        self.origin = new_origin

    def set_spacing(self,new_spacing):
        pass

    @jit(nogil=True)
    def interpolation_2d(self, world_coordinates_array):
        pixel_array = self.world_to_pixel(world_coordinates_array)
        value = np.zeros(world_coordinates_array.shape[0])
        for i in range(world_coordinates_array.shape[0]):
            pixel_point = pixel_array[i,:]
            if pixel_point[0] >= self.sizePerDimension[0]-1 or pixel_point[1] >= self.sizePerDimension[1]-1 or pixel_point[0] < 0 or pixel_point[1] < 0 :
                value[i] = 0
            else:
                interpolation_origin = pixel_point.astype(np.int)
                zo = interpolation_origin + np.array([0,1])
                oz = interpolation_origin + np.array([1,0])
                oo = interpolation_origin + np.array([1,1])
                zz = interpolation_origin

                x1_value = (oz[0] - pixel_point[0]) * self.data[zz[0],zz[1]] + (pixel_point[0] - zz[0]) * self.data[oz[0],oz[1]]
                x2_value = (oo[0] - pixel_point[0]) * self.data[zo[0],zo[1]] + (pixel_point[0] - zo[0]) * self.data[oo[0],oo[1]]

                x1 = oz
                x1[0] -= pixel_point[0]
                x2 = oo
                x2[0] -= pixel_point[0]

                value[i] = x1_value * (x2[1] - pixel_point[1]) + x2_value * (pixel_point[1] - x1[1])

        return value

    def interpolation_1d(self,line_coordinate):
        point = self.world_to_line(line_coordinate)

        lower = np.floor(point).astype(np.int)
        higher = lower + 1

        if point >= self.sizePerDimension[0]-1 or point < 0 :
            value = 0
        else:
            value = self.data[lower] * (higher - point) + self.data[higher] * (point - lower)

        return value

    def interpolation_nd(self,world_coordinates):
        #interpolationSpace = np.zeros(tuple(np.ones(self.dimension)*2))
        pixel_point = self.world_to_pixel(world_coordinates)
        interpolationOriginIndex = pixel_point.astype(np.int)
        l = []
        for i in range(2**self.dimension):
            l.append(self.data[tuple(np.array([int(d) for d in bin(i)[2:].zfill(self.dimension)])+ interpolationOriginIndex)])

        interpolationSpace = np.array(l).reshape(tuple(np.ones(self.dimension,dtype=int)*2))

        return interpolationSpace
















if __name__ == '__main__':
    a = Container(data=np.arange(5))
    print(a.interpolation_1d(a.line_to_world(2.43)))
    #print(a.origin)

    im = np.zeros((20,20))
    x = np.linspace(-10,10,20)
    XX,YY = np.meshgrid(x,x)
    im[XX**2 + YY**2 < 5**2] = 1
    c = Container(data=im)

    #for i in range(30):
        #print(c.interpolation_2d(c.pixel_to_world(np.array([10.3,i*0.74 ]))))