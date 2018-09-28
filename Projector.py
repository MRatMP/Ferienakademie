import ImageContainer
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

class sinogramme:
    def __init__(self, x_thetas, y_integral, data):
        self.x_axis = x_thetas
        self.y_axis = y_integral
        self.sgram = data

    def show(self):
        plt.imshow(self.sgram)
        plt.axis('on')
        plt.show()


class Projector:

    def __init__(self, volume, detector_resolution, phi1=0, phi2=np.pi, number_projections=180, sampling=500):
        self.volume = volume
        self.phi1 = phi1
        self.phi2 = phi2
        self.no_pro = number_projections
        self.res = detector_resolution
        self.integral_sampling_rate = sampling
        self.world_detectorsize = self.calc_detectorwidth()

    def create_sinogramme(self):
        # create an array holding all theatas
        thetas = self.calc_angles()
        # compute sampling points along rotated basis (thetas, s, sampling points for integral)
        sampling_points = np.zeros((self.no_pro, self.res, self.integral_sampling_rate))

        # sum over sampling points to emulate integral
        sino = np.sum(sampling_points, axis=2)
        return sinogramme(sino, x_thetas=thetas, y_integral=self.res)

    def calc_angles(self):
        return np.arange(self.phi1, self.phi2, (self.phi2-self.phi1)/self.no_pro)

    def calc_detectorwidth(self):
        # in theory the detector should be at least as large as the diagonal picture
        return np.sqrt(np.sum(np.power(self.volume.world_dimensions, 2)))


if __name__ == "__main__":
    shep_logan = img.imread('phantom.jpeg')
    container = ImageContainer.Container(data=shep_logan)
    test_proj = Projector(container,500)
    test_proj.create_sinogramme()