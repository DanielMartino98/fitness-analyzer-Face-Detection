import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os.path
from scipy.signal import savgol_filter
import time

def detect_faces(cascade, test_image, scaleFactor=1.3):
    #eine Kopie des Bildes erstellen, um Änderungen am Original zu verhindern.
    image_copy = test_image.copy()
    #Bild in Graustufenbild umwandeln, da der opencv-Gesichtsdetektor Graubilder erwartet
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    #haar classifier anwenden um GEsichter zu erkennen
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=4, minSize=(20, 20), maxSize=(60, 60))
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 5)
    return image_copy, y
def takeClosest(myList, myNumber, start, end):
    difference = abs(myNumber-myList[0])
    for i in range(start, end, 1):
        if abs(myList[i] - myNumber) < difference:
            difference = abs(myList[i] - myNumber)
            searched = i
    return searched

path = r'D:\...\NAME.mp4'
DIR = r'C:\...\NAME' #Ordner erstellen falls noch nicht vorhanden

for fname in os.listdir(DIR):  # so lösche ich alle Dateien die mit 'frame' beginnen
    if fname.startswith("frame"):
        os.remove(os.path.join(DIR, fname))

cascade = cv2.CascadeClassifier(
    r'C:\Users\Martino\PycharmProject\untitled\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    #r'C:\Users\Martino\PycharmProject\untitled\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
    #r'C:\Users\Martino\PycharmProject\untitled\venv\Lib\site-packages\cv2\data\lbpcascade_profileface.xml')
    #r'C:\Users\Martino\PycharmProject\untitled\venv\Lib\site-packages\cv2\data\haarcascade_profileface.xml')
    #r'C:\Users\Martino\PycharmProject\untitled\venv\Lib\site-packages\cv2\data\lbpcascade_frontalface.xml')

#Falls die Bearbeitungszeit gestoppt werden soll, muss folgende Zeile eingeblendet werden
#t0 = time.time()

height_list= []
nummer = 0

cap = cv2.VideoCapture(path)

while(True): #Jeden Frame laden
    ret, frame = cap.read()
    nummer += 1
    try:
        if frame.shape[0] > 4000:
            frame = cv2.resize(frame, (int(frame.shape[1] / 7), int(frame.shape[0] / 7)))
        elif frame.shape[0] > 2000:
            frame = cv2.resize(frame, (int(frame.shape[1] / 5), int(frame.shape[0] / 5)))
        elif frame.shape[0] > 1000:
            frame = cv2.resize(frame, (int(frame.shape[1] / 3), int(frame.shape[0] / 3)))
        #Falls das Bild um 90° gedreht werden soll, muss die folgende Zeile eingeblendet werden
        #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = frame.copy()
    except:
        break
    try:
        faces, height = detect_faces(cascade, frame)
    except:
        faces = frame
        try:
            height = height_list[-1]
        except:
            height = 0


    height_list.append(height)

#Falls die Frames während der Bearbeitung angezeigt werden sollen müssen die folgenden Zeilen eingeblendet werden
'''
    cv2.imshow("Analyse", faces)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''

#Falls die Bearbeitungszeit gestoppt werden soll, müssen folgende Zeilen eingeblendet werden
#t1 = time.time() #Bis zu dieser Zeile wird die benötigte Bearbeitungszeit gestoppt
#print("Bearbeitungszeit: ", t1-t0, " s")

x = range(0, len(height_list))
y = height_list
y_smoothed = savgol_filter(y, 51, 3) #Kurve glätten

average = sum(y_smoothed)/len(y_smoothed) #Mittelwert der ermitteln
maxima_log = find_peaks(y_smoothed, distance=50, height=(average, max(y_smoothed)))[0] #finde Maxima

#Falls die Kurve ausgegeben werden soll, müssen die folgenden Zeilen eingeblendet werden
'''
plt.plot(x,y)
plt.plot(x,y_smoothed, color='red')
plt.xlabel('Frame')
plt.ylabel('Höhe')
plt.show()
'''

minima_log = []
for x in range(0,len(maxima_log)-1, 1):
    minima_log.append(int((maxima_log[x] + maxima_log[x+1])/2))

zwischenpunkte_log = []

for x in range(0,len(maxima_log)-1, 1):
    maximum = y_smoothed[maxima_log[x + 1]]
    minimum = y_smoothed[minima_log[x]]
    searched_angle = int((maximum + minimum) / 2)
    try:
        zwischenpunkt = takeClosest(y_smoothed, searched_angle, minima_log[x] ,maxima_log[x+1]) #um Bild mit richtigem Winkel
    except:
        break
    zwischenpunkte_log.append(zwischenpunkt)
for x in range(0,len(maxima_log)-1, 1):
    maximum = y_smoothed[maxima_log[x]]
    minimum = y_smoothed[minima_log[x]]
    searched_angle = int((maximum + minimum) / 2)
    try:
        zwischenpunkt = takeClosest(y_smoothed, searched_angle, maxima_log[x], minima_log[x])
    except:
        break
    zwischenpunkte_log.append(zwischenpunkt)

writing_list = [] #Liste mit Framenummern dessen zugehoerige Frames abgespeichert werden sollen

for x in zwischenpunkte_log:
    writing_list.append(x)
for y in maxima_log:
    writing_list.append(y)
for z in minima_log:
    writing_list.append(z)
writing_list.sort() #Framenummern sortieren

for i in writing_list: #Bilder für chosen_pics aussuchen
    cap = cv2.VideoCapture(path)
    cap.set(1, i)  #i steht für Framenummer welche Geladen werden soll
    ret, img = cap.read()
    #wenn man die abzuspeichernden Bilder skalieren möchte, müssen folgende ausgeklammerte Zeilen eingeblendet werden
    '''
    if img.shape[0] > 4000:
        img = cv2.resize(img, (int(img.shape[1] / 7), int(img.shape[0] / 7)))
    elif img.shape[0] > 2000:
        img = cv2.resize(img, (int(img.shape[1] / 5), int(img.shape[0] / 5)))
    elif img.shape[0] > 1000:
        img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
    # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    '''
    cv2.imwrite(r'C:\Users\Martino\PycharmProject\untitled\chosen_pics\frame{}.jpg'.format(i), img) #abspeichern
