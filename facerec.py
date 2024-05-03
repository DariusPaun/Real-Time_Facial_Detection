import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model

model =load_model('model.file.h5')

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

label_dict={0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Neutral",5:"Sad",6:"Surpise"}

frame = cv2.imread("Assets\Denisa_s_mojo.png")
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray,1.3,3)
for x,y,w,h in faces:
    sub_face_img=gray[y:y+h,x:x+w]
    resized=cv2.resize(sub_face_img,(48,48))
    normalize = resized/255.0
    reshaped=np.reshape(normalize,(1,48,48,1))
    result =model.predict(reshaped)
    label=np.argmax(result,axis=1)[0]
    print(label)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
    cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
    cv2.putText(frame,label_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

height, width = frame.shape[:2]
scale_factor = 0.3  # You can adjust this scale factor as needed
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)
resized_frame = cv2.resize(frame, (new_width, new_height))

cv2.imshow("Frame",resized_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()



