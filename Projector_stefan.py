import numpy as np
import matplotlib.pyplot as plt
from ImageContainer import Container
from numba import jit
from joblib import Parallel, delayed


class Projector:

    def __init__(self,volume : Container,numOfAngles,detectorSize,lineSpacing,detectorSpacing):
        self.volume = volume
        self.angleList = np.linspace(0,np.pi,numOfAngles)
        self.sinogram = np.zeros((detectorSize,numOfAngles))
        self.lineSpacing = lineSpacing
        self.detectorSpacing = detectorSpacing
        self.numOfLines = detectorSize
        self.numOfLinePoints = np.max(volume.sizePerDimension)

    def getOrthogonalVector2d(self,vector):
        return np.array([vector[1],-vector[0]])

    def getRotationMatrix2d(self,theta):
        return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

    #@jit
    def getListOfLineOriginsPerAngle(self,angle):
        lines = np.zeros((self.numOfLines,2))
        detector_vector = self.getRotationMatrix2d(angle).dot(np.array([1,0]))
        detector_origin = - detector_vector * self.detectorSpacing * (-0.5 + self.numOfLines / 2)

        for i in range(self.numOfLines):
            lines[i] = (-0.5* self.lineSpacing * (self.numOfLinePoints - 1) * self.getOrthogonalVector2d(detector_vector)+detector_origin + self.detectorSpacing * detector_vector*i)

        return lines
    #@jit
    def getLinePointArrayPerAngle(self,angle):
        linepointArray = np.zeros((self.numOfLines,self.numOfLinePoints))
        detector_vector = self.getRotationMatrix2d(angle).dot(np.array([1, 0]))
        lineDirection = self.getOrthogonalVector2d(detector_vector) # evtl negative
        origins = self.getListOfLineOriginsPerAngle(angle)
        for j in range(self.numOfLinePoints):
            linepointArray[:,j] = self.volume.interpolation_2d(origins+lineDirection*self.lineSpacing*j)
        return linepointArray



    #@jit(nogil=True)
    def forwardproject(self):
        numthreads = 4
        numOfAngles = len(self.angleList)
        #splittedSinogramList = [self.sinogram[:,0:int(numOfAngles/4)],self.sinogram[:,int(numOfAngles/4):int(numOfAngles/2)],self.sinogram[:,int(numOfAngles/2):3*int(numOfAngles/4)],self.sinogram[:,3*int(numOfAngles/4):numOfAngles]]
        def project_angle(i):
            linePointArray = self.getLinePointArrayPerAngle(self.angleList[i])
            self.sinogram[:,i] = np.sum(linePointArray,axis=1)
            print('Forward Angle: ', i)
        Parallel(n_jobs=4,require='sharedmem')(delayed(project_angle)(i) for i in range(numOfAngles))

    def backwardproject(self):
        self.volume.data *= 0
        for angleIndex in range(len(self.angleList)):
            print('Backward Angle: ',angleIndex)
            detectorLine = Container(data=self.sinogram[:,angleIndex])
            detectorLine.set_origin(-0.5 * self.detectorSpacing * (self.numOfLines - 1) * self.getRotationMatrix2d(self.angleList[angleIndex]).dot(np.array([1,0])))
            for i in range(self.volume.sizePerDimension[0]):
                for j in range(self.volume.sizePerDimension[1]):
                    detectorNormal = -detectorLine.origin / np.sqrt(detectorLine.origin.dot(detectorLine.origin.T))
                    self.volume.data[i,j] += detectorLine.interpolation_1d(detectorNormal.dot(self.volume.pixel_to_world([i,j]).T) * detectorNormal)


    def filter_sinogram(self):
        filter_kernel = np.abs(np.fft.fftfreq(n=self.sinogram.shape[0],d=1/self.sinogram.shape[0]))
        self.sinogram = np.real(np.fft.ifft(np.fft.fft(self.sinogram,axis=0) * filter_kernel[:,np.newaxis],axis=0))


    def show_sinogram(self):
        plt.figure()
        plt.imshow(self.sinogram,cmap='gray')
        plt.colorbar()

    def plot_detector(self):
        detector = self.getListOfLineOriginsPerAngle(np.pi / 4)
        x = [l[0] for l in detector]
        y = [l[1] for l in detector]
        plt.figure()
        plt.scatter(x,y)
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')


    def plot_linepoints(self):
        x = []
        y = []
        angle = np.pi / 4
        linepointArray = np.zeros((self.numOfLines, self.numOfLinePoints))
        detector_vector = self.getRotationMatrix2d(angle).dot(np.array([1, 0]))
        lineDirection = self.getOrthogonalVector2d(detector_vector)  # evtl negative
        origins = self.getListOfLineOriginsPerAngle(angle)
        for i in range(self.numOfLinePoints):
            for j in range(self.numOfLinePoints):
                x.append((origins[i, :] + lineDirection * self.lineSpacing * j)[0])
                y.append((origins[i, :] + lineDirection * self.lineSpacing * j)[1])
        plt.figure()
        plt.scatter(x,y)
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')


    def show_volume(self):
        plt.figure()
        plt.imshow(self.volume.data,cmap='gray')
        plt.colorbar()



if __name__ == '__main__':
    im = np.zeros((20,20))
    hip = plt.imread('phantom7.png')
    print(hip.shape)
    x = np.linspace(-10,10,20)
    XX,YY = np.meshgrid(x,x)
    im[XX**2 + YY**2 < 5**2] = 1
    #projector = Projector(Container(data=im.copy()),50,20,1,1)

    projector = Projector(Container(data=hip[::2,::2]), 180, int(1.5*128), 1, 1)

    projector.forwardproject()
    print('finished forward')
    projector.filter_sinogram()
    projector.backwardproject()
    #projector.plot_detector()
    #projector.plot_linepoints()
    projector.show_sinogram()
    #projector.plot_linepoints()
    #projector.plot_detector()

    projector.show_volume()


    plt.show()
