from __future__ import division

import glob
import sys
import cv2
import time
import numpy as np
import os
import shutil
class HandKeypointDetector():
    def __init__(self,output_folder,show_debug=False):
        self.show_debug = show_debug
        file_dir = os.path.abspath(os.path.dirname(__file__))
        self.protoFile = file_dir+"\\hand\\pose_deploy.prototxt"
        self.weightsFile = file_dir+"\\hand\\pose_iter_102000.caffemodel"
        self.nPoints = 22
	
        self.data_out = output_folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        else:
            shutil.rmtree(output_folder)
            time.sleep(1)
            os.mkdir(output_folder)
        self.keypoints = np.zeros((2*(self.nPoints -1),3))
        self.resize_factor = 0.5
        self.rearrange_finger_indices = np.array([0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17])
        self.min_number_of_points = 8
        self.confidence_for_roi = 0.05
        self.roi_expansion = 0.09
        self.POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)
    def storeKeyPoints(self,prefix):
        cv2.imwrite(self.data_out + '\\' + prefix + self.output_file_name + '.png', self.debug_image)
        np.savez(self.data_out + '\\{}.npz'.format(self.output_file_name), num_hands=1, kp_coord_uv=self.keypoints[:, 0:2],
                 kp_visible=self.keypoints[:, 2], )

    def detectKeyPoints(self,data_folder):
        self.debug_image = None
        self.keypoints = np.zeros((2*(self.nPoints -1),3))
        self.output_file_name = ''
        single_imag = False

        try:
            if os.path.isdir(data_folder):
                files = glob.glob(data_folder + '\*.png')
                bb=None
            else:
                files = [data_folder]
                single_imag = False
        except:
            single_imag = True
            files = [0]
        for f in range(0,len(files),1):
            if single_imag:
                frame = data_folder.detach().cpu().numpy()
            else:
                frame = cv2.imread(files[f])
                import re
                self.output_file_name =  re.split('[\\\ .]', files[f])[-2] + '_skeleton'

            frame = cv2.resize(frame,None,fx=self.resize_factor,fy=self.resize_factor)
            # Select ROI

            # frame=frame[int(frame.shape[0] / 2):, :, :]
            # r = cv2.selectROI(frame)
            #
            # # Crop image
            # frame = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

            # frameCopy = np.copy(frame)
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            aspect_ratio = frameWidth/frameHeight

            threshold = 0.1

            t = time.time()
            # input image dimensions for the network
            inHeight = 368
            inWidth = int(((aspect_ratio*inHeight)*8)//8)
            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

            self.net.setInput(inpBlob)

            output = self.net.forward()
            if self.show_debug:
                print("time taken by network : {:.3f}".format(time.time() - t))

            # Empty list to store the detected keypoints
            points = []
            # points_probs = []
            for i in range(self.nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]
                probMap = cv2.resize(probMap, (frameWidth, frameHeight))

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                if prob > threshold :
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, int(255*prob)), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(frame, "{}".format(self.rearrange_finger_indices[i]), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 255), 1, lineType=cv2.LINE_AA)

                    # Add the point to the list if the probability is greater than the threshold
                    points.append(np.array([int(point[0]), int(point[1]),prob]))
                    # points_probs.append(prob)
                else :
                    # points_probs.append(0)
                    points.append(np.array([0, 0,0]))
            points = np.array(points)
            # Draw Skeleton
            for ii,pair in enumerate(self.POSE_PAIRS):
                partA = pair[0]
                partB = pair[1]
                # prob = points_probs[ii]
                if  np.all(points[partA]) and  np.all(points[partB]):
                    cv2.line(frame, tuple((points[partA][0:2]).astype(int)), tuple((points[partB][0:2]).astype(int)), (0, 255, 255), 2)
                    # cv2.circle(frame, points[partA][0:2], 8, (0, 0, int(255*prob)), thickness=-1, lineType=cv2.FILLED)
                    # cv2.circle(frame, points[partB][0:2], 8, (0, 0, int(255*prob)), thickness=-1, lineType=cv2.FILLED)

            if self.show_debug:
                cv2.imshow('Output-Skeleton', frame)
                print("Total time taken : {:.3f}".format(time.time() - t))

                cv2.waitKey(0)
            # cv2.imwrite(self.data_out + '\\'+self.output_file_name+'.png', frame)
            if self.min_number_of_points < sum(x is not None for x in points):
                ordered_points = np.array(points)[self.rearrange_finger_indices]
                ordered_points[:,0:2] = ordered_points[:,0:2]/self.resize_factor
                self.keypoints[0:self.nPoints-1,:] = ordered_points[:,0:3]
                # np.savez(self.data_out + '\\{}.npz'.format(output_file_name), num_hands = 1 , kp_coord_uv=self.keypoints[:,0:2], kp_visible=self.keypoints[:,2], )
                indices = self.keypoints[:, 2] > self.confidence_for_roi
                bb = {'minX': int((1 - self.roi_expansion) * min(self.keypoints[indices, 0])),
                      'maxX': int((1 + self.roi_expansion) * max(self.keypoints[indices, 0])),
                      'minY': int((1 - self.roi_expansion) * min(self.keypoints[indices, 1])),
                      'maxY': int((1 + self.roi_expansion) * max(self.keypoints[indices, 1])),
                      }
                # print(bb)
        self.debug_image = cv2.resize(frame,None,fx=1/self.resize_factor,fy=1/self.resize_factor)
        # import matplotlib.pyplot as plt
        # plt.close('all')
        # plt.imshow(self.debug_image)
        # plt.savefig('a.png')
        return bb

if __name__=='__main__':
    data_folder = r"P:\4Erez\david\test\raw_stream\stream001_00104RGB.png"
    show_debug = False
    hd = HandKeypointDetector("out/",show_debug)
    hd.detectKeyPoints(data_folder)
    print('%%%%%%%%%%% Done %%%%%%%%%%%%%%%')
