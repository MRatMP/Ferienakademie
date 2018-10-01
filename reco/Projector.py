import reco.imgContainer as imgContainer
import numpy as np
import matplotlib.pyplot as plt
import pyconrad as pyc
from joblib import Parallel, delayed

class Projector:

    def __init__(self, volume, detector_resolution, phi1=0, phi2=np.pi, number_projections=180, sampling=500):
        # exist upon initialization
        self.volume = volume
        self.res = detector_resolution              # s
        self.phi1, self.phi2 = phi1, phi2
        self.no_angles = number_projections
        self.thetas = self.calc_angles()
        self.sampling = sampling
        self.world_detectorsize = self.calc_detectorwidth()
        # exist after foreward_projection
        self.sino = None
        # exist after backwardprojection
        self.reco = None


    def backwardproject(self):
        assert self.sino is not None
        unitvectors = np.zeros((len(self.thetas), 2))
        for i, t in enumerate(self.thetas):
            u = np.array((np.cos(t), np.sin(t)))
            u /= np.sqrt(np.sum(np.power(u, 2)))
            unitvectors[i, :] = u
        #unitvectors.reshape(9, 20, 2)
        back_proj_t = Parallel(n_jobs=-2, verbose=10)(delayed(self.smear_back)(u) for u in unitvectors)
        self.reco = np.sum(back_proj_t, axis=2)

    '''
    calculate the backprojection per detector_position on the whole image
    '''
    def smear_back(self, u):
        reco = np.zeros_like(self.volume.data)
        for x, y, p in np.ndenumerate(reco):
            reco[x, y] = 1
        return reco


    def forwardproject(self):
        assert self.volume.data
        # create an array holding all theatas

        thetas = self.thetas

        # compute sampling points along rotated basis (thetas, s, sampling points for integral)
        # interpolate data for every sampling point along the rays
        #ray_integral_samples = self.calc_sampling_points(thetas)

        #parralel
        the = thetas.reshape(9,20).tolist()
        ray_integral_samples = Parallel(n_jobs=-2, verbose=10)(delayed(self.calc_sampling_points)(t) for t in the)


        #ray_integral_samples = integrate_lines(thetas, detectorpoints)
        ray_integral_samples = np.array(ray_integral_samples).reshape((self.no_angles, self.res, self.sampling))

        # sum over sampling points to emulate integral
        sinog = np.sum(ray_integral_samples, axis=2)
        sinog = sinog/np.max(sinog)

        self.sino = sinog

    def calc_angles(self):
        return np.arange(self.phi1, self.phi2, (self.phi2-self.phi1) / self.no_angles)

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

    def load_image(self, name):
        self.sino = plt.imread(name)
        if len(self.sino.shape) == 3:
            self.sino = self.sino[:, :, 0]

    def show_sino(self):
        assert self.sino is not None



if __name__ == "__main__":
    pyc.setup_pyconrad(max_ram='2G')
    _ = pyc.ClassGetter('edu.stanford.rsl.tutorial.phantoms')
    shep_logan = _.SheppLogan(256, False).as_numpy()
    container = imgContainer.Container(data=shep_logan, spacing=1)
    test_proj = Projector(container, detector_resolution=1.5*256, sampling=1.5*256)
    test_proj.forwardproject()
    #test_proj.load_image('../sinogramme.jpeg')
    #test_proj.backwardproject()