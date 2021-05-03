#=====================================
#DEFINING EXTERNAL FUNCTIONS
#=====================================
def myregionfill(img, seed,pix):                            #FUNCTION FOR REGION FILLING
    r = img.shape[0]
    c = img.shape[1]
    perm_mask = np.zeros((r + 2, c + 2), dtype=np.uint8)
    cv2.floodFill(img, perm_mask, seed , pix)
def mytessdata(r):                                          #FUNCTION TO CONVERT TESSERACT UNSTRUCTURED DATA INTO STRUCTURED DICTIONARIES
    with open("rrrrrrrrrr.csv", "w") as f:
        f.write(r)
    data = np.loadtxt("rrrrrrrrrr.csv", dtype=str, delimiter="\t")
    data = np.transpose(data)
    d = dict()
    for i in range(0, data.shape[0] - 1):
        d[data[i, 0]] = np.array(data[i, 1:], dtype=np.int32)
    d[data[data.shape[0] - 1, 0]] = np.array(data[data.shape[0] - 1, 1:], dtype=str)
    return(d)
def subimage(image, center, theta, width, height):           #FUNCTION TO ROTATE IMAGES WITH A CERTAIN DEGREE
    theta *= 3.14159 / 180 # convert to rad
    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)
    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])
    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)

# =====================================
# IMPORTING PYTHON LIBRARIES
# =====================================
import imutils
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import numpy as np                                                                              #IMPORTING NUMPY
import cv2                                                                                      #IMPORTING CV2
import os
import pytesseract                                                                              #IMPORTING PYTESSERACT
pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # DIRECTORY LOCATION OF TESSERACT PROGRAM

#=========================================================
#READING SOURCE IMAGES AND CREATING COPIES OF SOURCE IMAGE
#=========================================================
num = int(input("Enter Photo Number:\n"))                                            #ASKING USERS FOR INPUT NUMBER OF THE IMAGE
img_src = cv2.imread(str(num)+".jpg")                                                #SOURCE IMAGE IN COLOUR
img_src=cv2.resize(img_src,(int(img_src.shape[1]*0.25),int(img_src.shape[0]*0.25)))  #RESIZING SOURCE IMAGE IN COLOUR
ori_img = cv2.imread(str(num)+'.jpg',0)                                              #READING THE FILE IMAGE IN GRAYSCALE
ori_img=cv2.resize(ori_img,(int(ori_img.shape[1]*0.25),int(ori_img.shape[0]*0.25)))  #RESIZING IMAGE

img_copy1 = np.copy(ori_img)          #CREATE A COPIES OF THE GRAY IMAGE FOR PROCESSING PURPOSES
img_copy2 = np.copy(ori_img)
img_copy3 = np.copy(ori_img)
img_copy4_col = np.copy(img_src)          #CREATING A COPY OF THE COLORED IMAGE
#======================================================
#DEFINING KERNELS FOR IMPLMENTATION
#======================================================
rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT,(30,10))
cross_kern = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
square_kern = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

#==========================================================================
#IMAGE PROCESSING STAGE 1: PRODUCING BLACKHAT IMAGE WHICH IS THRESHOLDED
#ISOLATES THE REGION AROUND THE CHARACTERS ON THE NUMBER PLATE
#==========================================================================
blackhat = cv2.morphologyEx(ori_img,cv2.MORPH_BLACKHAT,rect_kern,iterations=1) #PERFORM BLACKHAT OPERATION ON THE ori_img
_,blackhat = cv2.threshold(blackhat,100,255,cv2.THRESH_BINARY)                 #THRESHOLDING THE BLACKHAT IMAGE WITH BINARY OPERATION
# cv2.imshow("STAGE 1 IMAGE PROCESSING",blackhat)
#====================================================================================================
#IMAGE PROCESSING STAGE 2: PERFORMING CLOSING OPERATION ON THE ori_img TO ISOLATE THE NUMBER PLATE
# ISOLATES THE CHARACTERS ON THE NUMBER PLATE
#====================================================================================================
light = cv2.morphologyEx(ori_img,cv2.MORPH_CLOSE,square_kern)                   #PERFORMING LIGHT OPERATION TO ISOLATE NUMBER PLATE
_,light = cv2.threshold(light,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)       #PERFORMING OTSU BINARY THRESHOLDING
light = 255-light                                                              #INVERTING THE IMAGE
light = cv2.morphologyEx(light,cv2.MORPH_ERODE,cross_kern,iterations=1)        #ERODING THE IMAGE TO CLARIFY THE CHARACTERS ON NUMBER PLATE
# cv2.imshow("STAGE 2 IMAGE PROCESSING",light)
#===========================================================================================
#IMAGE PROCESSING STAGE 3: PERFORMING BITWISE OPERATION BETWEEN THE light & blackhat IMAGES
# THE COMBINATION OF THESE 2 IMAGES PRODUCES A CLEARER IMAGE IN BLACK AND WHITE
#===========================================================================================
L_BH_and = cv2.bitwise_xor(light, blackhat)     #BITWISE AND OPERATION - FOCUSES ON THE AREA AROUND THE NUMBER PLATE
L_BH_and = 255-L_BH_and                         #INVERTING THE IMAGE
L_BH_or = cv2.bitwise_or(light, blackhat)       #BITWISE OR OPERATION - FOCUSES ON THE CHARACTERS THEMSEVLES
L_BH_or = 255 - L_BH_or                         #INVERING THE IMAGE
# cv2.imshow("STAGE 3 IMAGE PROCESSING",L_BH_or)
#===============================================================================================================================
#IMAGE PROCESSING STAGE 4: COMPUTES THE SCHARR GRADIENT REPRESENTATION OF THE BLACKHAT IMAGE IN X DIRECTION
# THEN SCALES THE RANGE BACK BETWEEN 0 - 255
# THIS STEP BRINGS OUT THE EDGE AREAS AROUND OF THE CHARACTERS IN THE NUMBER PLATE
#===============================================================================================================================
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,dx=1, dy=0, ksize=-1)    #CALCULATING THE SCHARR GRADIENT
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = 255 * ((gradX - minVal) / (maxVal - minVal))                    #SCALING THE VALUES BACK TO 0-255
gradX = gradX.astype("uint8")                                           #CONVERTING THE DATA INTO UNSIGNED 8 BIT

#===============================================================================================================
#IMAGE PROCESSING STAGE 5: blur the gradient representation, applying a closing operation and threshold the image using OTSU;s method.
#===============================================================================================================
gradX = cv2.GaussianBlur(gradX, (5, 5), 0)                    #BLURRRING THE IMAGE USING GAUSSIAN KERNEL
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rect_kern)   #PERFORMING CLOSING
gradX_thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] #THREHOLDING THE IMAGE USING OTSU BINARY METHOD
gradX_thresh = cv2.morphologyEx(gradX_thresh, cv2.MORPH_DILATE,cross_kern,iterations = 5) #DILATE THE IMAGE TO REDUCE BLACK AREAS COVERING THE PLATE
gradX_thresh = cv2.bitwise_and(gradX_thresh,L_BH_or) #BITWISE / OVERLAYING THE gradX_thresh IMAGE ON THE L_BH_or IMAGE TO BLACKEN AREAS NOT RELATED TO NUMBER PLATE

#===============================================================================================================
#IMAGE PROCESSING STAGE 6: DARKENING A PORTION OF THE IMAGE THAT IS IRRELEVANT
#===============================================================================================================
q1x = np.int64(0.41*gradX_thresh.shape[0])  #OBTAINING LENGTH OF 41% OF THE ROWS IN THE IMAGE
q1y = np.int64(0.5*gradX_thresh.shape[1])   #OBTAINING LENGTH OF 50% OF THE COLUMNS IN THE IMAGE

#IT IS ASSUMED THAT IN A CONTROLLED ENVIRONMENT THAT THE CAR NUMBERPLATE IS ALWAYS LOCATED AT THE BOTTOM RIGHT CORNER OF THE IMAGE
#HENCE THE TOP ROWS AND LEFT SIDE OF THE IMAGE IS BLACKENED
for i in range(gradX_thresh.shape[0]):
    for j in range (gradX_thresh.shape[1]):
        if i<=q1x:
            gradX_thresh[i,j] = 0
        elif i>=q1x and j<=q1y:
            gradX_thresh[i, j] = 0

#===================================================================================================================================================
#IMAGE PROCESSING STAGE 7: CONTOUR DETECTION AND FINE TUNING

#CONTOUR DETECTION IS DONE ITERATIVELY IN STAGES ON THE PROCESSED IMAGE.
#OPERATION INVOLVES FILTERING OUT NOISES AND CLUMPING AREAS TO CREATE CONSOLIDATED RECTANGLES TO FOCUS ON AREA AROUND THE NUMBER PLATE
#===================================================================================================================================================

#===================================================================================================================================================
#CONTOUR DETECTION AND FILTERING STAGE 1
#FIRST STAGE OF CONTOUR DETECTION AND FILTERING

cnts1,h1 = cv2.findContours(gradX_thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)     #DETECTING CONTOURS ON gradX_thresh
mask1 = np.zeros_like(gradX_thresh)         # CREATING mask1 FOR DRAWING CONTOURS OF gradX_thresh

#FILTERING CONTOURS IN gradX_thresh THAT MEETS CERTAIN CRITERIA
#DRAWING FILTERED CONTOURS ON img_copy1 AND mask1
for i in range(len(cnts1)):
    rect1 = cv2.minAreaRect(cnts1[i])
    box1 = cv2.boxPoints(rect1)
    box1 = np.int0(box1)
    (x1, y1, w1, h1) = cv2.boundingRect(box1)
    area1 = cv2.contourArea(box1)
    ar1 = w1 / float(h1)
    if area1> 150 and area1 < 5000:                     #AREA SIZE CRITERION FOR FILTERING
        img_copy1 = cv2.drawContours(img_copy1, [box1], 0, (255, 255, 255), 2)
        mask1 = cv2.drawContours(mask1, [box1], -1, color = 255)

myregionfill(mask1, (0, 0), 255)
mask1 = 255 - mask1

mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE,cross_kern,iterations = 5)     #DILATING mask1 TO SOFTEN THE EDGES FOR ANOTHER STAGE OF CONTOUR DETECTION
gradX_mask1 = cv2.bitwise_and(gradX_thresh, gradX_thresh, mask=mask1)  #PERFORMING BITWISE OPERATION ON gradX_thresh WITH mask1 TO OVERLAY ON img_copy2
img_copy2 = cv2.bitwise_and(img_copy2, img_copy2, mask=gradX_mask1)  #OVERLAY ON img_copy2 FOR VISUALIZATION OF RESULTS.

#===================================================================================================================================================
#CONTOUR DETECTION AND FILTERING STAGE 2
#2ND STAGE OF CONTOUR DETECTION AND FILTERING. REMAINING CONTOURS ON MASK 1 IS COMBINED INTO A CONSOLIDATED CONTOURS TO BE FURTHER FILTERED

cnts2,h2 = cv2.findContours(mask1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  #DETECTING CONTOURS ON mask1
mx = []     #EMPTY LIST TO STORE FILTERED CONTOURS' X-CENTROID
my = []     #EMPTY LIST TO STORE FILTERED CONTOURS' Y-CENTROID
cnts2a = [] #TO STORED CONTOURS POINTS OF FILTERED CONTOURS.

for i in range(len(cnts2)):
    rect2 = cv2.minAreaRect(cnts2[i])
    box2 = cv2.boxPoints(rect2)
    box2 = np.int0(box2)
    area2 = cv2.contourArea(box2)
    if area2 > 0:                   #FILTERING AREAS WHICH HAVE AREA MORE THAN 0. THERE ARE SOME AREAS WITH NEGATIVE VALUES NOTICED
        cnts2a.append(cnts2[i])     #SAVING FILTERED CONTOURS TO cnts2a
        M2 = cv2.moments(box2)      #CALCULATING CENTROID OF EACH SHAPE AND SAVING THEM INTO mx AND my
        cx2 = int(M2['m10'] / M2['m00'])
        cy2 = int(M2['m01'] / M2['m00'])
        mx.append(cx2)
        my.append(cy2)

mask2 = np.zeros_like(mask1)    #CREATING A 2ND MASK FOR DRAWING

for i in range(len(cnts2a)):
    dist_arr = []
    rect2a = cv2.minAreaRect(cnts2a[i])
    box2a = cv2.boxPoints(rect2a)
    box2a = np.int0(box2a)
    area2a = cv2.contourArea(box2a)
    (x2a, y2a, w2a, h2a) = cv2.boundingRect(box2a)
    ar2a = w2a / float(h2a)
    for ii in range(len(mx)):
        if ii != i:
            # CALCULATING THE DISTANCE OF THE CENTROID OF THE CONTOUR TO ALL NEIGHBOURING CONTOURS
            # THE DISTANCE BETWEEN THE CONTOUR TO IT'S NEXT CLOSEST CONTOUR IS USED AS A CRITERIA FOR FILTERING
            #THIS IS TO CAPTURE CONTOURS THAT SORROUND EACH CHARACTER IN NUMBER PLATE
            distance = np.sqrt((mx[i]-mx[ii])**2 + (my[i]-my[ii])**2)
            dist_arr.append(distance)
    if area2a> 1000 and min(dist_arr) < 200 and ar2a > 0 and ar2a < 3: #MODIFIABLE PARAMETER
        mask2 = cv2.drawContours(mask2, [box2a], -1, color=255)

myregionfill(mask2, (0, 0), 255)

#STEP TO DARKEN CONTOURS THAT FALL AT THE 4 EDGES OF THE IMAGE WHICH FAR AWAY FROM THE NUMBER PLATE.
for i in range(mask2.shape[1]):
    for j in range(mask2.shape[0]):
        if i == 0 or i == mask2.shape[1] -1 or j == 0 or j == mask2.shape[0]-1:
            myregionfill(mask2, (i, j), 255)


#PERFORMING BITWISE AND OPERATION BETWEEN mask2 AND img_copy3
#img_copy3 HAS MOST OF THE NOISE IMAGE FILTERED AWAY COVERING ONLY THE MINIMAL AREAS OF THE IMAGE SORROUNDING THE NUMBER PLATE
mask2 = 255 - mask2
img_copy3 = cv2.bitwise_and(img_copy3, img_copy3, mask=mask2)

#===================================================================================================================
#SAVING mask2 AND img_copy3 AS JPG FILES WHICH IS THEN REREAD INTO THE PROGRAM.
cv2.imwrite(str(num)+"_mask2.jpg",mask2)            #WRITING THE IMAGE AS A JPEG SAVED IN PROJECT DIRECTORY
cv2.imwrite(str(num)+"_img_copy3.jpg",img_copy3)

mask2 = cv2.imread(str(num)+"_mask2.jpg",0)         #READING THE IMAGE AS JPEG FORMAT AS SAVED IN PROJECT DIRECTORY
img_copy3 = cv2.imread(str(num)+"_img_copy3.jpg",0)

os.remove(str(num)+"_mask2.jpg")                    #REMOVING THE FILE FROM DIRECTORY AS IT IS NO LONGER REQUIRED.
os.remove(str(num)+"_img_copy3.jpg")

#===================================================================================================================
#CONTOUR DETECTION AND FILTERING STAGE 3
#3RD STAGE OF CONTOUR DETECTION AND FILTERING.
#CONSOLIDATING SMALLER RECTANGLES IN mask2 TO BE FURTHER FILTERED OUT

mask3 = np.zeros_like(mask2)                                                   #CREATING AN EMPTY ARRAY WITH SAME SIZE AS mask2
cnts3a = []                                                                    #AN EMPTY ARRAY TO STORE CORNERS INFORMATION OF CONTOURS IN mask2
box_area = []                                                                  #AN EMPTY ARRAY TO STORE THE AREA OF THE CONTOURS IN mask2
cnts3,h3 = cv2.findContours(mask2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  #DETECTING CONTOURS FROM THE IMPORTED MASK USING cv2.findContours()
for i in range(len(cnts3)):                                                    #LOOPING THROUGH EVERY CONTOUR AND ONLY FILTERING CONTOURS WITH
    rect3 = cv2.minAreaRect(cnts3[i])                                          #LARGE ENOUGH AREA CRITERIA
    box3 = cv2.boxPoints(rect3)
    box3 = np.int0(box3)
    area3 = cv2.contourArea(box3)
    if area3>2000:                                                             #IF AREA OF RECTANGLE MORE THAN 2000 SAVE THE POINTS AND ITS AREA
        cnts3a.append(cnts3[i])
        box_area.append(area3)

a_max_index = box_area.index(max(box_area))                                   #IDENTIFYING THE INDEX OF CONTOUR WITH THE MAX AREA IN THE IMAGE
rect3a = cv2.minAreaRect(cnts3a[a_max_index])
box3a = cv2.boxPoints(rect3a)
box3a = np.int0(box3a)
M3a = cv2.moments(box3a)                                                         #CALCULATING THE CENTROID LOCATION OF THE RECTANGLE CENTROID
cx3a = int(M3a['m10'] / M3a['m00'])                                              #CENTROID X
cy3a = int(M3a['m01'] / M3a['m00'])                                              #CENTROID Y
(x3a, y3a, w3a, h3a) = cv2.boundingRect(box3a)
mask3 = cv2.drawContours(mask3, [box3a], -1, color=255)                       #DRAWING THE LARGEST RECTANGLE ONLY ON mask3
myregionfill(mask3, (0, 0), 255)                                              #CONVERTING THE IMAGE TO BLACK BACKGROUND AND WHITE RECTANGLE
mask3 = 255-mask3
img_copy4_col = cv2.bitwise_and(img_copy4_col,img_copy4_col,mask = mask3)     #OVERLAYING MASK3 ON THE SOURCE COLOURED IMAGE

#mask3 IS AN OVERLAY WITH ONLY A SINGLE RECTANGLE CONTOUR FILTERED
#img_copy4_col IS THE mask3 OVERLAY OVER THE ORIGINAL COLOURED IMAGE.

#CROPPING IMAGE: TO REDUCE THE AREA OF IMAGE FOR EASIER PROCESSING
x = [i[0] for i in box3a]                                                     #GETTING THE X-COORDINATES IN THE RECTANGLE
y = [i[1] for i in box3a]                                                     #GETTING THE Y COORDINATES IN THE RECTANGLE
crop_img = img_copy4_col[min(y):max(y),min(x):max(x)]                         #CROPPING THE COLOURED IMAGE SOURCE
crop_img = crop_img[:,:,0]                                                    #GRAYSCALE CROPPED IMAGE
_,crop_img = cv2.threshold(crop_img,100,255,cv2.THRESH_BINARY)                #THRESHOLDING THE GRAY IMAGE

#====================================================================================================
#CONTOUR DETECTION AND FILTERING STAGE 4
#4TH STAGE OF CONTOUR DETECTION AND FILTERING.
#IMAGE PROCESSING THE CROPPED IMAGE TO REDUCE THE SIZE OF THE FINAL RECTANGLE.
#THE FINAL RESULT SHOULD BE SMALLER RECTANGLE FOCUSED SPECIFICALLY AROUND THE NUMBER PLATE CHARACTERS.

cnts4,h4 = cv2.findContours(crop_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)    #CONTOURS DETECTION ON THE CROPPED THRESHOLD IMAGE
mask4 = np.zeros_like(crop_img)                                                     #CREATING A MASK WITH THE SAME SIZE OF THE CROPPED IMAGE

for i in range(len(cnts4)):                  #DETECTING RECTANGLE CONTOURS AND FILTERING THEM BASED ON SIZE TO DRAW ON mask4
    rect4 = cv2.minAreaRect(cnts4[i])
    box4 = cv2.boxPoints(rect4)
    box4 = np.int0(box4)
    area4 = cv2.contourArea(box4)
    (x4, y4, w4, h4) = cv2.boundingRect(box4)
    ar4 = w4 / float(h4)
    perimeter4 = cv2.arcLength(cnts4[i], True)
    if  area4>100 and area4 < 8000 and h4 >20 and h4 <80 and ar4 >0 and ar4<2.2 and w4<150:
        mask4 = cv2.drawContours(mask4, [box4], 0,255, 2)

myregionfill(mask4, (0, 0), 255)
mask4 = 255-mask4

#MORPHOLOGICAL OPERATION TO COMBINE RECTANGLES AND REMOVE NOISE
mask4 = cv2.morphologyEx(mask4,cv2.MORPH_DILATE,cross_kern,iterations =15)
mask4 = cv2.morphologyEx(mask4,cv2.MORPH_ERODE,cross_kern,iterations =7)

crop_img = cv2.bitwise_and(crop_img,crop_img,mask = mask4)

final_image = np.copy(crop_img)            #CREATING A COPY OF THE crop_img USED FOR PERSPECTIVE TRANSFORM
mask5 = np.zeros_like(final_image)         #CREATING mask5 OF SIZE SAME AS FINAL IMAGE

cnts4a,h4a = cv2.findContours(mask4, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   #DETECTING CONTOURS FROM MASK 4
for i in range(len(cnts4a)):
    rect4a = cv2.minAreaRect(cnts4a[i])
    box4a = cv2.boxPoints(rect4a)
    box4a = np.int0(box4a)
    area4a = cv2.contourArea(box4a)
    (x4a, y4a, w4a, h4a) = cv2.boundingRect(box4a)
    ar4a = w4a / float(h4a)
    perimeter4a = cv2.arcLength(cnts4a[i], True)
    if area4a>5000 and ar4a >1:                                 #FILTERING OUT DESIRED RECTANGLES
        mask5 = cv2.drawContours(mask5, [box4a], 0,255, 2)
        corners = box4a
        # FOR TESTING PURPOSES
        # print("width:", w, " height: ", h, "Aspect ratio: ",ar," perimeter: ", perimeter, " area: ", area)

myregionfill(mask5, (0, 0), 255)
mask5 = 255-mask5
final_image = cv2.bitwise_and(final_image,final_image,mask = mask5)   #MASKING THE final_image WITH mask5

#====================================================================================================
#IMAGE PROCESSING STAGE 8: PERFORMING PERSPECTIVE TRANSFORM ON FINAL IMAGE
#====================================================================================================
try:
    corners=np.float32(corners)                #CONVERTING CORNERS TO DATA TYPE FLOAT32
except:                                        #EXCEPTION IF FILTERING REMOVES ALL THE RECTANGLE AND A BLANK CANVAS IS RETURNED WITH NO CORNERS
    corners = np.float32([[[1,1]],[[5,1]],[[5,2]],[[1,2]]])

# print(len(corners))                            #FOR CHECKING PURPOSES
newcorners = np.array([[[80,100]],[[580,100]],[[620,250]],[[120,250]],],dtype=np.float32) #DEFINING NEW COORDINATES FOR PERSPECTIVE TRANSFORM
M=cv2.getPerspectiveTransform(corners,newcorners)   #CARRYING OUT PERSPECTIVE TRANSFORM
result=cv2.warpPerspective(final_image,M,(800,800)) #SIZE OF THE TRANSFORMED IMAGE

cv2.imshow("Final Image",final_image)   #FOR VIEWING PURPOSES
cv2.imshow("result",result)             #FOR VIEWING PURPOSES

#==============================================================================================================================
#2 OCR RESULTS ARE CONSIDERED FROM PERSPECTIVE TRANSFORMED IMAGE (result) & AND WITHOUT PERSPECTIVE TRANSFORM (final_image)
#THE 2 RESULTANT OCRS ARE CLEANED AND COMPARED
#PRECEDENCE IS GIVEN TO THE OCR FROM THE PERSPECTIVE TRANSFORMED IMAGE (result).
#IF THE OUTPUT FROM result IS NOT SATISFACTORY THEN RESULTANT OCR FROM final_image IS SELECTED AS THE final_plate
#==============================================================================================================================

#=====================================================================================================================================
#IMAGE PROCESSING STAGE 9: FEEDING THE PERSPETIVE TRANSFORMED IMAGE AND final_image INTO THE PYTESSERACT LIBRARY FOR OCR DETECTION
#=====================================================================================================================================

tess_result = pytesseract.image_to_data(result)              #OCR ON THE PERSPECTIVE TRANSFORMED IMAGE (result)
tess_result2 = pytesseract.image_to_data(final_image)        #OCR ON THE FINAL FILTERED IMAGE (final_image)
# print(tess_result2)      #PRINTING THE RESULTS
# print(tess_result)

#FEEDING THE OCR RESULTS INTO mytessdata FUNCTION TO RETURN THE RESULTS IN DICTIONARY FORMAT
tess_data = mytessdata(tess_result)
tess_data2 = mytessdata(tess_result2)

#FILTERING OUT ONLY THE TEXT FROM THE DICTIONARY OF DATA AS LIST
words_transform = tess_data["text"]     #DATA FROM PERSPECTIVE TRANSFORM IMAGE (result)
words_cropped = tess_data2["text"]      #DATA FROM FINAL FILTERED IMAGE (final_image)

#====================================================================================================
#IMAGE PROCESSING STAGE 9: PROCESSING OCR RESULTS
#====================================================================================================

#TYPICAL SIGNS THAT CAN APPEAR FROM OCR AS NOISE OR TEXT THAT CAN BE MISINTERPRETED AS SIGNS
#USED FOR COMPARISON WITH THE OCR DATA FOR DATA PROCESSING
signs = ["~","`","!","@","#","$","%","^","&","*","(",")","_","-","+","=","|","}","{","]","[","'",";","\"","\"",":","/",".",",","?",">","<"]

#====================================================================================================
#PROCESSING words_transform
words_transform = [i for i in words_transform if i!='']        #REMOVING EMPTY FIELDS IN THE LIST

#IF THE FIRST ELEMENT IN THE LIST OF RETURNED OCRs IS A SIGN AND IS OF LENGTH 1, THEN IT IS REMOVED
for i in words_transform:
    if i in signs and len(i) == 1 and words_transform.index(i) != len(words_transform)-1:
        words_transform.remove(i)

#AFTER FILTERING OUT SIGNS IF THE FINAL OCR OUTPUT FOR words_transform DOES NOT MEET THE LENGTH REQUIREMENTS, THE LIST IS EMPTIED
if len(words_transform) == 1 and len(words_transform[0])<=4:
    words_transform = []

#IF THERE ARE SIGNS AT THE END OF THE OCR DATA AND THE SIGNS MEET THE CONDITIONS,
#THEY ARE CONVERTED INTO SIMILAR SHAPED ALPHABETS
for i in range(len(words_transform)):
    if i == len(words_transform)-1 and len(words_transform[i]) == 1:
        if words_transform[i] == "$":
            words_transform[i] = "S"
        elif words_transform[i] == "!":
            words_transform[i] = "I"
        elif words_transform[i] == "&":
            words_transform[i] = "B"
        elif words_transform[i] == "(" or words_transform[i] == "[":
            words_transform[i] = "C"

#====================================================================================================
#PROCESSING words_cropped
words_cropped = [i for i in words_cropped if i!='']

for i in words_cropped:
    if i in signs and len(i) == 1 and words_cropped.index(i) != len(words_cropped)-1:
        words_cropped.remove(i)
    if any(c.islower() for c in i):
        words_cropped.remove(i)

if len(words_cropped) == 1:
    words_cropped = []

for i in range(len(words_cropped)):
    if i == len(words_cropped)-1 and len(words_cropped[i]) == 1:
        if words_cropped[i] == "$":
            words_cropped[i] = "S"
        elif words_cropped[i] == "!":
            words_cropped[i] = "I"
        elif words_cropped[i] == "&":
            words_cropped[i] = "B"
        elif words_cropped[i] == "(" or words_cropped[i] == "[":
            words_cropped[i] = "C"

#====================================================================================================
# CONVERTING THE PROCESSED NUMBER PLATE INTO A LIST WITH ELEMENTS AS INDIVIDUAL ELEMENTS
words_transform = list("".join(words_transform))
words_cropped = list("".join(words_cropped))

#SIGNS IN THE COMBINED STRING IS CHECKED AND FILTERED
#THE FIRST ELEMENT OF THE STRING / NUMBER PLATE NEEDS TO BE A ALPHABET.
#CONDITIONS ARE SET SUCH THAT ANY SIGNS OR NUMBERS THAT RESEMBLE AN ALPHABET ARE CONVERTED INTO THE SIMILAR LOOKING ALPHABET
for i in range(len(words_transform)):
    if i == 0:
        if words_transform[i] == "$":
            words_transform[i] = "S"
        elif words_transform[i] == "1":
            words_transform[i] = "I"
        elif words_transform[i] == "5":
            words_transform[i] = "S"
        elif words_transform[i] == "4":
            words_transform[i] = "A"
        elif words_transform[i] == "3" or words_transform[i] == "8":
            words_transform[i] = "B"
        elif words_transform[i] == "6" or words_transform[i] == "&":
            words_transform[i] = "G"
        elif words_transform[i] == "(" or words_transform[i] == "[":
            words_transform[i] = "C"

for i in range(len(words_cropped)):
    if i == 0:
        if words_cropped[i] == "$":
            words_cropped[i] = "S"
        elif words_cropped[i] == "1":
            words_cropped[i] = "I"
        elif words_cropped[i] == "5":
            words_cropped[i] = "S"
        elif words_cropped[i] == "4":
            words_cropped[i] = "A"
        elif words_cropped[i] == "3" or words_cropped[i] == "8":
            words_cropped[i] = "B"
        elif words_cropped[i] == "6" or words_cropped[i] == "&":
            words_cropped[i] = "G"
        elif words_cropped[i] == "(" or words_cropped[i] == "[":
            words_cropped[i] = "C"

#JOINING IN THE FINAL REMAINING CHARACTERS AS A SINGLE STRING
words_transform = "".join(words_transform)
words_cropped = "".join(words_cropped)

print(words_transform)  #PRINTING RESULTS AFTER FILTERING FOR COMPARISON
print(words_cropped)

#====================================================================================================
#IMAGE PROCESSING STAGE 10: FINAL OUTPUT
#====================================================================================================

# THE FINAL NUMBER PLATE OUTPUT IS CHOSEN FROM words_transform
final_plate = words_transform
if len(final_plate) == 0:       #IF words_transform IS AN EMPTY ARRAY THE FINAL NUMBER PLATE IS TAKEN FROM words_cropped
    final_plate = words_cropped

print("Final Plate: ",final_plate) #PRINTING THE FINAL OUTPUT
cv2.waitKey(0)









