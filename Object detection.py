import numpy as np
import argparse
import cv2
import os
from subprocess import call
import RPi.GPIO as GPIO
import time

ap = argparse.ArgumentParser()
ap.add_argument(&quot;-c&quot;, &quot;--confidence&quot;, type=float, default=0.2,
help=&quot;minimum probability to filter weak detections&quot;)
args = vars(ap.parse_args())

CLASSES = [&quot;background&quot;, &quot;aeroplane&quot;, &quot;bicycle&quot;, &quot;bird&quot;, &quot;boat&quot;,
&quot;bottle&quot;, &quot;bus&quot;, &quot;car&quot;, &quot;cat&quot;, &quot;chair&quot;, &quot;cow&quot;, &quot;diningtable&quot;,
&quot;dog&quot;, &quot;horse&quot;, &quot;motorbike&quot;, &quot;person&quot;, &quot;pottedplant&quot;, &quot;sheep&quot;,
&quot;sofa&quot;, &quot;train&quot;, &quot;tvmonitor&quot;, &quot;mobile&quot;]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print(&quot;[INFO] loading model...&quot;)
net = cv2.dnn.readNetFromCaffe(&quot;/home/pi/Downloads/MobileNetSSD_deploy.prototxt.txt&quot;,
&quot;/home/pi/Downloads/MobileNetSSD_deploy.caffemodel&quot;)

cam = cv2.VideoCapture(0)
cv2.namedWindow(&quot;test&quot;)

img_counter = 0
i = 0
distance = 0

def sense():
GPIO.setmode(GPIO.BCM)
TRIG = 23
ECHO = 24
print &quot;Distance..&quot;
GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)
GPIO.output(TRIG,False)
print &quot;Waiting for sensor to settle&quot;
GPIO.output(TRIG,True)
time.sleep(0.00001)
GPIO.output(TRIG,False)

while GPIO.input(ECHO)==0:
pulse_start = time.time()

while GPIO.input(ECHO)==1:
pulse_end = time.time()
pulse_duration=pulse_end - pulse_start


distance = pulse_duration * 17150
distance = round(distance, 2)
print &quot;Distance:&quot;,distance,&quot;cm&quot;
GPIO.cleanup()
time.sleep(1)
return distance

while(cam.isOpened()):
ret, frame = cam.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow(&quot;test&quot;, gray)
k = cv2.waitKey(1)
if(i%100)==0:
dist = sense()
if k%256 == 27:
print(&quot;Escape hit, closing...&quot;)
break

elif (i%100)==0:
img_name = &quot;image_{}.jpg&quot;.format(img_counter)
path = &quot;/home/pi/Downloads/Images&quot;
cv2.imwrite(os.path.join(path, img_name), gray)
print(&quot;{} written!&quot;.format(img_name))
img_counter += 1
image = cv2.imread(os.path.join(path, img_name))
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300),
127.5)
print(&quot;[INFO] computing object detections...&quot;)
net.setInput(blob)
detections = net.forward()
f = open(&quot;new.txt&quot;,&quot;w&quot;)
f.write(&quot;Caution\n&quot;)


for i in np.arange(0, detections.shape[2]):
confidence = detections[0, 0, i, 2]

if confidence &gt; args[&quot;confidence&quot;]:
idx = int(detections[0, 0, i, 1])
box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
(startX, startY, endX, endY) = box.astype(&quot;int&quot;)

label = &quot;{}&quot;.format(CLASSES[idx])
print(&quot;[INFO] {}&quot;.format(label))

cv2.rectangle(image, (startX, startY), (endX, endY),
COLORS[idx], 2)
y = startY - 15 if startY - 15 &gt; 15 else startY + 15
cv2.putText(image, label, (startX, y),
cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
f.write(&quot;There is a {}\n&quot;.format(label))
f.close()
with open(&#39;new.txt&#39;, &#39;r&#39;):
call(&#39;espeak -f new.txt&#39;,shell=True)
if(dist &lt; 100):
os.system(&quot;espeak -s70 &#39;You are close to the object&#39;&quot;)

i+=1
cam.release()
cv2.destroyAllWindows()