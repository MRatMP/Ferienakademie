import ImageContainer
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import pyconrad as pyc
import sys
from joblib import Parallel, delayed

class sinogramme:
    def __init__(self, x_thetas, y_integral, data):
        self.x_axis = x_thetas
        self.y_axis = y_integral
        self.sgram = data

    def show(self):
        plt.imsave('sinogramme.jpeg', self.sgram.T)
        pyc.imshow(self.sgram.T, 'sinogramme')



class Projector:

    def __init__(self, volume, detector_resolution, phi1=0, phi2=np.pi, number_projections=180, sampling=500):
        self.volume = volume
        self.phi1 = phi1
        self.phi2 = phi2
        self.no_pro = number_projections
        self.res = detector_resolution
        self.sampling = sampling
        self.world_detectorsize = self.calc_detectorwidth()


    def create_sinogramme(self):
        # create an array holding all theatas
        thetas = self.calc_angles()

        # compute sampling points along rotated basis (thetas, s, sampling points for integral)
        # interpolate data for every sampling point along the rays
        #ray_integral_samples = self.calc_sampling_points(thetas)

        #parralel
        the = thetas.reshape(9,20).tolist()
        ray_integral_samples = Parallel(n_jobs=-2, verbose=10)(delayed(self.calc_sampling_points)(t) for t in the)


        #ray_integral_samples = integrate_lines(thetas, detectorpoints)
        ray_integral_samples = np.array(ray_integral_samples).reshape((self.no_pro, self.res, self.sampling))

        # sum over sampling points to emulate integral
        sinog = np.sum(ray_integral_samples, axis=2)
        sinog = sinog/np.max(sinog)

        return sinogramme(data=sinog, x_thetas=thetas, y_integral=self.res)

    def calc_angles(self):
        return np.arange(self.phi1, self.phi2, (self.phi2-self.phi1)/self.no_pro)

    def calc_detectorwidth(self):
        # in theory the detector should be at least as large as the diagonal picture
        return np.sqrt(np.sum(np.power(self.volume.world_dimensions, 2)))

    def calc_sampling_points(self, thetas):
        # shape = (thetas, detector_points(ray_no), number of points to integrate, x, y) 5 dimensional
        # values 3 dimensional
        #points = np.zeros((self.no_pro, self.res, self.sampling, 2))
        #print('computing startingpoints', end='')
        values = np.zeros((len(thetas), self.res, self.sampling))
        for i, t in enumerate(thetas):
            #m = np.sin(t)/(np.cos(t)+1e-10)         # y/x
            ct = np.cos(t)
            st = np.sin(t)
            #if(i%20 == 0):
            #    print(int(i/20), end='')
            #    sys.stdout.flush()
            for s in range(self.res):
                r = -(self.world_detectorsize/2)+(s*((self.world_detectorsize)/self.res))
                x_detector = ct * r
                y_detector = st * r
                ct2 = np.cos(t+(np.pi/2))
                st2 = np.sin(t+(np.pi/2))
                for no in range(self.sampling):
                    r_integ = -(self.world_detectorsize / 2) + (no * ((self.world_detectorsize) / self.sampling))
                    x_integ = x_detector + ct2*r_integ
                    y_integ = y_detector + st2*r_integ
                    #points[t, s, no, 0], points[t, s, no, 1] = x_integ, y_integ
                    values[i, s, no] = self.volume.interpolation_2d((x_integ, y_integ))
        return values



if __name__ == "__main__":
    pyc.setup_pyconrad(max_ram='2G')
    _ = pyc.ClassGetter('edu.stanford.rsl.tutorial.phantoms')
    shep_logan = _.SheppLogan(512, False).as_numpy()
    container = ImageContainer.Container(data=shep_logan, spacing=1)
    test_proj = Projector(container, detector_resolution=512, sampling=700)
    sino = test_proj.create_sinogramme()
    sino.show()