import cv2
from PIL import Image
from math import *
import numpy as np
def find_degree(img):           #找到倾斜角度
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r,img2 = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))  # 膨胀，只留下定位点
    img2 = cv2.dilate(img2, kernel)
    cv2.imwrite('t.jpg',img2)
    height,width = img2.shape
    flag1,flag2,i_1,i_2,j_1,j_2,neg=[0,0,0,0,0,0,0]           #找到最左边2个边界点
    for i in range(width):
        for j in range(height):
            if img2[j,i]==0 and flag1==0:       #找到最左边的点
                i_1=i
                j_1=j
                flag1=1
            if img2[j,i]==0 and flag2==0 and j_1>height/2 and j<height/2: #左上
                i_2=i
                j_2=j
                flag2=1
            if img2[height-i-1,j]==0 and flag2==0 and j_1<height/2 and j>width/2:     #右下
                i_2 = i
                j_2 = j
                flag2 = 1
                neg=1
        if flag1==1 and flag2==1:
            break
    print(abs(i_2-i_1))
    print(abs(j_1-j_2))
    if i_2-i_1!=0 and neg==0:
        degree=atan(abs(i_2-i_1)/abs(j_1-j_2))*180/pi
    elif i_2-i_1!=0 and neg==1:
        degree = -atan(abs(i_2 - i_1)/ abs(j_1 - j_2) ) * 180 / pi
    else:
        degree=0
    print(degree)
    return degree
# def find_degree(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
#     edges = cv2.Canny(gray, 200, 1000)
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=340, maxLineGap=10)        #找出直线
#     i1=lines[0][0][0]
#     j1=lines[0][0][1]
#     i2=lines[0][0][2]
#     j2 = lines[0][0][3]
#     if j2-j1>0 and i2 - i1!=0:
#         degree = atan(abs(j2 - j1) / abs(i2 - i1)) * 180 / pi
#     elif j2-j1<0 and i2 - i1!=0:
#         degree = -atan(abs(j2 - j1) / abs(i2 - i1)) * 180 / pi
#     else:
#         degree=0
#     print(degree)
#     return degree

def find_pix(img):          #找到二维码左上坐标
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    r,img2 = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)
    height,width = img2.shape
    norm_i=0
    norm_j=0
    flag=1
    for i in range(width):
        for j in range(height):
            if img2[j,i]==0:
                norm_i=i
                norm_j=j
                flag=0
                break
        if flag==0:
            break
    return norm_j,norm_i

def get_norm_img(path,new_filename):     #获得矫正后的图像并保存
    img=cv2.imread(path)
    degree=find_degree(img)
    im = Image.open(path)
    im2 = im.convert('RGBA')
    im_rotate = im2.rotate(degree,expand=1)
    fff = Image.new('RGBA', im_rotate.size, (255,)*4) #与旋转图像大小相同的白色图像
    out = Image.composite(im_rotate, fff, im_rotate)  #使用alpha层的rot作为掩码创建一个复合图像
    out=out.resize((5100,3300),Image.ANTIALIAS)
    out.convert(im.mode).save(new_filename)

def get_img(true_img,new_filname,final_filename):      #校正后获取正确定位图像并保存
    img=cv2.imread(true_img)
    [j1,i1]=find_pix(img)
    img2=cv2.imread(new_filname)
    [j2,i2]=find_pix(img2)
    M = np.float32([[1,0,i1-i2],[0,1,j1-j2]])
    dst = cv2.warpAffine(img2,M,(img2.shape[1],img2.shape[0]))
    cv2.imwrite(final_filename,dst)

if __name__ == '__main__':
    new_filename='5.jpg'
    true_img='2.pdf.jpg'
    un_img='1.pdf.jpg'
    final_filename='final_img.jpg'
    get_norm_img(un_img,new_filename)
    get_img(true_img,new_filename,final_filename)