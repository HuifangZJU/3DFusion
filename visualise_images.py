import sys
import h5py
import numpy as np
import mayavi.mlab as mlab
import cv2
import os
import argparse
from scipy.interpolate import RectBivariateSpline
from PIL import Image
parser = argparse.ArgumentParser(description="Visualise recorded data frames")
parser.add_argument("--file", default="/mnt/tank/datasets/infralidar/dataset/col5.hdf5", required=False, help="Storage file")
args = parser.parse_args()
# MEAN_COLOR=[147.5924007222222, 143.86317471597224, 141.377818628125]
MEAN_COLOR=[148, 144, 141]
class Transform():
    def __init__(self, x,y,z, yaw,roll,pitch):
        self.x,self.y,self.z, self.yaw,self.roll,self.pitch = x,y,z, yaw,roll,pitch
        self.setMatrix()

    def setMatrix(self):
        # Transformation matrix
        self.matrix = np.matrix(np.identity(4))
        cy = np.cos(np.radians(self.yaw))
        sy = np.sin(np.radians(self.yaw))
        cr = np.cos(np.radians(self.roll))
        sr = np.sin(np.radians(self.roll))
        cp = np.cos(np.radians(self.pitch))
        sp = np.sin(np.radians(self.pitch))
        self.matrix[0, 3] = self.x
        self.matrix[1, 3] = self.y
        self.matrix[2, 3] = self.z
        self.matrix[0, 0] =  (cp * cy)
        self.matrix[0, 1] =  (cy * sp * sr - sy * cr)
        self.matrix[0, 2] = - (cy * sp * cr + sy * sr)
        self.matrix[1, 0] =  (sy * cp)
        self.matrix[1, 1] =  (sy * sp * sr + cy * cr)
        self.matrix[1, 2] =  (cy * sr - sy * sp * cr)
        self.matrix[2, 0] =  (sp)
        self.matrix[2, 1] = - (cp * sr)
        self.matrix[2, 2] =  (cp * cr)
        return self.matrix

    def transform_points(self, points, inverse=False):
        """
        Given a 4x4 transformation matrix, transform an array of 3D points.
        Expected point foramt: [[X0,Y0,Z0],..[Xn,Yn,Zn]]
        """

        matrix = self.matrix if not inverse else np.linalg.inv(self.matrix)

        # Needed foramt: [[X0,..Xn],[Z0,..Zn],[Z0,..Zn]]. So let's transpose
        # the point matrix.
        points = points.transpose()
        # Add 0s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[0,..0]]
        points = np.append(points, np.ones((1, points.shape[1])), axis=0)
        # Point transformation
        points = matrix * points
        # Return all but last row
        return points[0:3].transpose()


class Camera:
    def __init__(self, intrinsicMat, extrinsicMat, index):
        self.intrinsic = intrinsicMat
        self.extrinsic = extrinsicMat

        self.extTransform = Transform(0,0,0,0,0,0)
        self.extTransform.matrix = self.extrinsic
        self.id = index

    def toCamera(self, points, inverse=False):
        '''Transforms points to camera reference (3D). Points format: [[X0,..Xn],[Y0,..Yn],[Z0,..Zn]]'''
        points = self.extTransform.transform_points(points, inverse)
        return points

    def toImagePlane(self, points):
        '''Given points in world reference, transform them to camera reference and then to image plane'''

        ptsCamera = self.toCamera(points).transpose()
        ptsImagePlane = np.dot(self.intrinsic, ptsCamera).transpose()
        #Remove points that are not in front of the camera
        idx = np.array(ptsImagePlane[:,2] > 0).squeeze().reshape(-1)
        ptsImagePlane = ptsImagePlane[idx]

        #Normalize to Z
        ptsImagePlane[:, 0] /= ptsImagePlane[:, 2]
        ptsImagePlane[:, 1] /= ptsImagePlane[:, 2]

        return ptsImagePlane[:,0:2]
 
def getBBpts(repr):
    '''Given a representation get 8 BB edge points in world reference. Out shape [8,3]'''

    x,y,z = repr[0],repr[1],repr[2]
    pitch,roll,yaw = repr[3],repr[4],repr[5]
    trActor = Transform(x,y,z,yaw,pitch,roll)

    #BB Extensions (half length for each dimension)
    ex, ey, ez = repr[6],repr[7],repr[8]

    #Get 8 coordinates
    bbox_pts = np.array([
    [  ex,   ey,   ez],
    [- ex,   ey,   ez],
    [- ex, - ey,   ez],
    [  ex, - ey,   ez],
    [  ex,   ey, - ez],
    [- ex,   ey, - ez],
    [- ex, - ey, - ez],
    [  ex, - ey, - ez]
    ])

    #Transform the bbox points from actor ref frame to global reference
    bbox_pts = trActor.transform_points(bbox_pts)
    return bbox_pts

def drawPoints(im, pts, color):
    '''Draw points in an image'''
    for pt in pts:
        center = (int(pt[0,0]), int(pt[0,1]))
        cv2.circle(im, center, 3, color, -1)
def drawbbbox(im,umax, vmax, umin, vmin, color):
    '''Draw lines in a image'''

    cv2.line(im, (umax, vmax), (umax, vmin), color)
    cv2.line(im, (umax, vmax), (umin, vmax), color)
    cv2.line(im, (umin, vmin), (umin, vmax), color)
    cv2.line(im, (umin, vmin), (umax, vmin), color)

def savevehicletofile(sliceNum, cropims,camNums,vehicleIDs, coordinates, centers):
    all_vehicle = np.unique(vehicleIDs)
    repeated_observed_vehicle = []
    for v in all_vehicle:
        num = np.sum(vehicleIDs == v)
        if num > 1:
            repeated_observed_vehicle.append(v)
    for cropim, camNum, vehicleID, coordinate, center in zip(cropims, camNums, vehicleIDs, coordinates,centers):
        if vehicleID in repeated_observed_vehicle:
            path = "/home/shaoche/code/coop-3dod-infra/images/car_images_train/" + str(sliceNum) + "/" + str(vehicleID) + "/"
            if not os.path.exists(path):
                os.makedirs(path)
            imgpath = path + str(camNum) + "_" + str(coordinate[0]) + "_" + str(coordinate[1]) \
                      + "_" + str(coordinate[2])+ "_" + str(coordinate[3]) + ".jpeg"
            #cropim.save(imgpath)
            if not os.path.exists(path+'info.txt'):
                file = open(path+'info.txt', 'w', encoding='utf-8')
                # for value in coordinate:
                #     file.write(str(value) + ' ')
                for value in center:
                    file.write(str(value) + ' ')
                file.write('\n')
                file.close()








def getCropimg(frame,umax, vmax, umin, vmin):
    cropim=[]
    if umax - umin < 20 or vmax - vmin < 20:
        return cropim
    #cropframe = np.ones((300,400,3),dtype=np.uint8)
    #cropframe[:, :, 0] = MEAN_COLOR[0]
    #cropframe[:, :, 1] = MEAN_COLOR[1]
    #cropframe[:, :, 2] = MEAN_COLOR[2]
    #cropframe[vmin:vmax, umin:umax, :] = frame[vmin:vmax, umin:umax, :]
    cropframe = frame[vmin:vmax, umin:umax, :]
    try:
        temp = Image.fromarray(cropframe)
        cropim.append(temp)
    except ValueError:
        print(f"Current slice number is: {sliceNum}")
        print(vmin)
        print(vmax)
        print(umin)
        print(umax)
        sys.exit(0)
    return cropim

#Load file
fname = args.file 
f = h5py.File(fname, 'r')

#Load Cameras
#cameras_ = list(f.get('cameras'))
cameras_ = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5']
#cameras_ = [ 'cam1', 'cam3', 'cam5', 'cam7']
cameras  = []
dcameras = []
for cam in cameras_:
    intrinsic = np.array(f.get(f'cameras/{cam}/intrinsic'))
    extrinsic = np.array(f.get(f'cameras/{cam}/extrinsic'))
    if cam[0] == 'd':
        dcameras.append(Camera(intrinsic, extrinsic, int(cam[4])))
    else:
        cameras.append(Camera(intrinsic, extrinsic, int(cam[3])))
#Show slices
slices = f['slices']
dslices = f['dslices']
sliceNum = 0
print('total slice number : %d' % slices.shape[0])
while (sliceNum < slices.shape[0]):
    if sliceNum % 10 == 0:
        print(f"Slice {sliceNum}")

    #Normal frame visualisation

    #Get actors
    actorsRepr = np.array(f.get(f'objects/{sliceNum}'))
    bbpts = []
    pts = []
    for r in actorsRepr:
        if r[-1] == 0.0:
            bbpts.append(getBBpts(r))
            pts.append([r[0], r[1], r[2]])

    #bbpts = [getBBpts(r) for r in actorsRepr]
    colors = []
    vehicleIDperFrame=[]
    for i in range(len(bbpts)):
        color = 255 * np.random.rand(3)
        color = tuple(color.tolist())
        colors.append(color)
        vehicleIDperFrame.append(i)
    cropims = []
    camNums = []
    vehicleIDs = []
    coordinates = []
    centers = []
    for cam in cameras:
        #Get corresponding frame
        camNum = cam.id
        frame = np.array(slices[sliceNum, camNum])

        #Draw all object points on the frame

        for objBBpts, center, color, vehicleID in zip(bbpts, pts, colors,vehicleIDperFrame):
            ptsImagePlane = cam.toImagePlane(objBBpts)
            if len(ptsImagePlane) ==0 :
                continue
            umax = int(max(ptsImagePlane[:, 0])[0,0])
            umin = int(min(ptsImagePlane[:, 0])[0,0])
            vmax = int(max(ptsImagePlane[:, 1])[0,0])
            vmin = int(min(ptsImagePlane[:, 1])[0,0])
            if umax < -50 or umax > 450:
                continue
            if umin < -50 or umin > 450:
                continue
            if vmax < -50 or vmax > 350:
                continue
            if vmin < -50 or vmin > 350:
                continue
            umax = max(1,umax)
            umax = min(400,umax)
            umin = max(1, umin)
            umin = min(400, umin)
            vmax = max(1, vmax)
            vmax = min(300, vmax)
            vmin = max(1, vmin)
            vmin = min(300, vmin)
            # drawPoints(frame, ptsImagePlane,color)
            drawbbbox(frame, umax, vmax, umin, vmin, color)
            cropim = getCropimg(frame,umax, vmax, umin, vmin)
            coordinate = [umax, vmax, umin, vmin]

            if len(cropim) >0 :
                cropims.append(cropim[0])
                camNums.append(camNum)
                vehicleIDs.append(vehicleID)
                coordinates.append(coordinate)
                centers.append(center)
        im = Image.fromarray(frame)
        # Show frame
        cv2.imshow(str(camNum), frame)

    #savevehicletofile(sliceNum, cropims, camNums, vehicleIDs, coordinates, centers)

    try:
       cv2.waitKey(0)
    except KeyboardInterrupt:
       sys.exit(0)
    sliceNum += 1
    # inp = input()
    # if inp == ' ':
    #    sliceNum += 1
    # else:
    #    sliceNum = int(inp)
