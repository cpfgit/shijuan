import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img = cv2.imread('4.pdf.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#灰度图像

edges = cv2.Canny(gray,1,1000)
cv2.imwrite('t.jpg',edges)
#hough transform
lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength=500,maxLineGap=10)
print(lines)
lines1 = lines[:,0,:]#提取为二维
#
# for x1,y1,x2,y2 in lines1[:]:
#     cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
# cv2.imwrite('tt.jpg',img)

# path='2.pdf.jpg'
# new_filename='4.pdf.jpg'
# img=cv2.imread(path)
# degree=-7
# im = Image.open(path)
# im2 = im.convert('RGBA')
# im_rotate = im2.rotate(degree,expand=1)
# fff = Image.new('RGBA', im_rotate.size, (255,)*4) #与旋转图像大小相同的白色图像
# out = Image.composite(im_rotate, fff, im_rotate)  #使用alpha层的rot作为掩码创建一个复合图像
# out=out.resize((5100,3300),Image.ANTIALIAS)
# out.convert(im.mode).save(new_filename)