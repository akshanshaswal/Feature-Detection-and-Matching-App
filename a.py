import cv2
import matplotlib.pyplot as plt

img0 = cv2.imread("picture1.png",cv2.IMREAD_GRAYSCALE)

img1 = cv2.imread("picture2.png",cv2.IMREAD_GRAYSCALE)


#extracting features

orb = cv2.ORB_create()

key0,des0 = orb.detectAndCompute(img0,None)

key1,des1 = orb.detectAndCompute(img1,None)


#matching descriptors using brute force

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)

matches = bf.knnMatch(des0,des1,k=2)


#sorting matches

matches = sorted(matches,key=lambda x:x[0].distance)



#performing ratio test

matches = [x[0] for x in matches if len(x)>1 and x[0].distance < 0.8*x[1].distance]

#drawing them


img_matches = cv2.drawMatches(img0,key0,img1,key1,matches[:25],img1,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_matches)
plt.show()