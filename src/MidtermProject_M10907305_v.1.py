from numpy.core.fromnumeric import size
import cv2
import numpy as np
import os

class Panorama_Stitching :
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        
    def sift_function(self,img_right,img_left,values,img):

        # use sift to find the descriptors of the two photos.
        self.kp2,self.des2 = self.sift.detectAndCompute(img_left,None)
        self.kp1,self.des1 = self.sift.detectAndCompute(img_right,None)
        self.matches = cv2.BFMatcher().knnMatch(self.des1,self.des2, k=2)
        self.good = [m for m, n in self.matches if m.distance < 0.25*n.distance]
        if(len(self.good) > 4):

            #Analyze descriptors into general coordinates.
            self.src_pts = np.float32([ self.kp1[m.queryIdx].pt for m in self.good ]).reshape(-1,1,2)
            self.dst_pts = np.float32([ self.kp2[m.trainIdx].pt for m in self.good ]).reshape(-1,1,2)
            print('================================ ({0}) image1 ================================\n'.format(str(values)),self.src_pts)
            print('\n ================================ ({0}) image2 ================================\n'.format(str(values)),self.dst_pts )

            #use Homography to find the H matrix.
            self.H, self.mask = cv2.findHomography(self.src_pts, self.dst_pts, cv2.RANSAC,5.0)
            print('================================ Homography ================================')
            print(self.H)
            self.Perspective_img1 = cv2.warpPerspective(img_right,self.H,(img_right.shape[1]+img_left.shape[1],img_right.shape[0]))
            if values ==4:
                self.Perspective_img2 = cv2.warpPerspective(img,self.H,(img_right.shape[1]+img_left.shape[1],img_right.shape[0]))
                cv2.imshow('1',self.Perspective_img2)
            # cv2.imshow('img1',self.Perspective_img1)
            
            #prcessing overlapping problem.
            img_ = cv2.copyMakeBorder(img_left.copy(),0,0,0,int(self.Perspective_img1.shape[1]-img_left.shape[1]),cv2.BORDER_CONSTANT)
            # print(img_.shape,self.Perspective_img1.shape)
            self.result = cv2.subtract(self.Perspective_img1,img_)
            cv2.imshow('img',self.result)
            self.result = cv2.add(self.result,img_)            
            cv2.imshow('dasd1',self.result)
            cv2.waitKey(0)

            #Draw corresponding MARKs on images.
            matchesMask = self.mask.ravel().tolist()
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = matchesMask, # draw only inliers
                flags = 2)
            img_match = cv2.drawMatches(img_right.copy(),self.kp1,img_left.copy(),self.kp2,self.good,None,**draw_params)
            cv2.imwrite("./result/drawMatches{0}.jpg".format(str(values)),img_match)
            cv2.imwrite("./result/stitching{0}.jpg".format(str(values)),self.result)
            return self.result
        else:
            return 0
if __name__=='__main__':
    #read image and resize image. 
    img1 = cv2.imread('img/DJI_1.JPG')
    img2 = cv2.imread('img/DJI_1.JPG') 
    # img3 = cv2.imread('002.JPG')
    # img4 = cv2.imread('001.JPG')

    #reduce computing time, but many messages will disappear.
    img1 = cv2.resize(img1,(375,500),interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2,(375,500),interpolation=cv2.INTER_AREA)
    # img3 = cv2.resize(img3,(375,500),interpolation=cv2.INTER_AREA)
    # img4 = cv2.resize(img4,(375,500),interpolation=cv2.INTER_AREA)
    # print(img1.shape)

    # Add image black border.
    img1 = cv2.copyMakeBorder(img1,int(img1.shape[0]/2),int(img1.shape[0]/2),0,0,cv2.BORDER_CONSTANT)
    img2 = cv2.copyMakeBorder(img2,int(img2.shape[0]/2),int(img2.shape[0]/2),0,0,cv2.BORDER_CONSTANT)
    # img3 = cv2.copyMakeBorder(img3,int(img3.shape[0]/2),int(img3.shape[0]/2),0,0,cv2.BORDER_CONSTANT)
    # img4 = cv2.copyMakeBorder(img4,int(img4.shape[0]/2),int(img4.shape[0]/2),0,0,cv2.BORDER_CONSTANT)

    # print(img1.shape,img2.shape,img3.shape,img4.shape)
    
    #Stitching processing
    Stitching = Panorama_Stitching()
    img_stitching1 = Stitching.sift_function(img2,img1,1,0)
    # img_stitching2 = Stitching.sift_function(img3,img_stitching1,2,0)
    # img_stitching3 = Stitching.sift_function(img4,img_stitching2,3,img1)
    # img_stitching2 = Stitching.sift_function(img3,img4,2,0)
    # img_stitching3 = Stitching.sift_function(img_stitching2,img_stitching1,3,img1)
    cv2.imwrite('./result/picture_stitching.jpg',img_stitching1)
    cv2.imshow('img',img_stitching1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
