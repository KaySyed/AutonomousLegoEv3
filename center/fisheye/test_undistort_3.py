from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import rpyc
from time import sleep
import math
import pygame

motorenabled = True

if motorenabled:    
    con = rpyc.classic.connect('ev3dev.local')
    motors = con.modules['ev3dev2.motor']
    mediumMotor = motors.MediumMotor('outA')
    lMotor = motors.LargeMotor('outB')
    rMotor = motors.LargeMotor('outC')
    sensors = con.modules['ev3dev2.sensor.lego']
    us = sensors.UltrasonicSensor()
    pygame.mixer.init()
    #sound = con.modules['ev3dev2.sound']
    #s = sound()
    #s.speak('I am ON')
    #us = sensors.UltrasonicSensor()
    #ts = sensors.TouchSensor()

    mediumMotor.reset()

#Undistortion

camera = PiCamera()
camera.resolution = (1600,1200)
camera.framerate = 8
rawCapture = PiRGBArray(camera, size = camera.resolution)

DIM=(1600, 1200)

K=np.array([[774.4231548805052, 0.0, 822.6167410427034], [0.0, 769.3288387349592, 565.6990482106042], [0.0, 0.0, 1.0]])
D=np.array([[-0.17875854240547795], [0.02726679508811555], [-0.010188123245693159], [0.0024264322841337192]])

dim2 = (880,660)
dim3 = (880,880) #False

dim1 = DIM
    
if not dim2:
    dim2 = dim1
if not dim3:    
    dim3 = dim1
    
scaled_K = K * dim1[0] / DIM[0]
scaled_K[2][2] = 1

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=1)
    
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
#End Undistortion


# Instantiate OCV kalman filter
class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted
    
def run():
    lm1.run_forever(speed_sp= -150)
    lm2.run_forever(speed_sp= 150)
    sleep(1)

def stop():
    lMotor.stop()
    rMotor.stop()
    #sleep(2)


def func(nm):    
    previous_diff = 0
    totm = 0
    i = 0
    height, width = nm.shape[:2]
    p=10
    lx = []
    ly = []
    rx = []
    ry = []
    ma = []
    my = []
    #a= np.zeros((int((1000//p)+1), 2), dtype = "int32")
    #b= np.zeros((int((1000//p)+1), 2), dtype = "int32")
    added = 0
    roiy = 350
    
    #kfObj = KalmanFilter()
    #predictedCoords = np.zeros((2, 1), np.float32)

    for val in range(0, height, p):
        hist = np.sum(nm[height-val:height-(val-p),:], axis=0)
        left_max = np.argmax(hist[:height//2])
        right_max = np.argmax(hist[height//2:]) + height//2
        y = height-(val-p)
        #cv2.line(warpedorg, (0, roiy), (width,roiy), (0,255,255), 1)
        if y < roiy:
            continue
        #cv2.line(nm,(0,y),(width,y),(255,255,255),1)

        y = height-val
        avgm = 0

        #cv2.circle(warpedorg, (left_max,y), 3, (0,0,0), -1)
        #cv2.rectangle(warpedorg,(left_max-60, y-10),(left_max+60, y+10),(255,255,255),1)
        writeL = False
        writeR = False
        boxth = 70
        if len(lx) != 0:
            if left_max > (lx[-1] - boxth):
                if left_max < (lx[-1] + boxth):
                    cv2.circle(warpedorg, (left_max,y), 3, (0,0,0), -1)
                    #cv2.rectangle(warpedorg,(left_max-60, y-10),(left_max+60, y+10),(255,255,255),1)
                    lx.append(left_max)
                    ly.append(y)
                    writeL =True
        elif left_max > 7:
            lx.append(left_max)
            ly.append(y)
            writeL =True
            
            
        if len(rx) != 0 and right_max > 310:
            if right_max > (rx[-1] - boxth):
                if right_max < (rx[-1] + boxth):
                    cv2.circle(warpedorg, (right_max,y), 3, (0,0,0), -1)
                    #cv2.rectangle(warpedorg,(right_max-60, y-10),(right_max+60, y+10),(255,255,255),1)
                    rx.append(right_max)
                    ry.append(y)
                    writeR =True
                elif len(rx) == 1:
                    rx.append(right_max)
                    ry.append(y)
                    writeR =True
            elif len(rx) == 1:
                rx.append(right_max)
                ry.append(y)
                writeR =True
        elif right_max > 0 and right_max > 310:
            rx.append(right_max)
            ry.append(y)
            writeR =True
        
        #if len(ma) > 0:
         #   avgm = (totm//len(ma))
        if writeL and writeR and y > roiy:
            m = ((right_max - left_max) //2) + left_max
            ma.append((m, y))
            cv2.circle(warpedorg, (m, y), 3, (255,255,255), -1)
            totm = (totm + m)
            #predictedCoords = kfObj.Estimate(m, y)
            #avgm = totm//len(ma)
            #print(avgm)
        #elif y > roiy and int(predictedCoords[0]) > 0 :
            #predictedCoords = kfObj.Estimate(0, y)
            #predictedCoords = kfObj.Estimate(int(predictedCoords[0]), y)
            #cv2.circle(warpedorg, (predictedCoords[0], y), 3, (255,255,255), -1)
            #totm = int(totm + predictedCoords[0])
            #ma.append((predictedCoords[0], y))
            #if len(ma) > 0 and y > roiy and added > 3:
                #ma.append((ma[len(ma)-1][0],y))
                #avgm = ma[len(ma)-1][0] #int(((totm // len(ma))) + (ma[len(ma)-1][0]))# * 0.5) - ma[len(ma)-1][0])
                #ma.append((avgm, y))
                #totm = ma[len(ma)-1][0] +totm
                #cv2.circle(warpedorg, (int(avgm), y), 3, (255,0,255), -1)
        if len(ma) > 1:
            cv2.line(warpedorg, ma[len(ma)-1], ma[len(ma)-2],(0,0,255),2)
    #print(len(ma))
    #sum_lane_center = 0
    #len(ma) = len(ma) - 2
    
    if len(ma) > 1:        
        tan = math.degrees((math.atan2(ma[1][1]-ma[len(ma)-1][1], ma[1][0]-ma[len(ma)-1][0])))
        tan = (90-tan)
        print('tan = ' + str(tan))
        #for r in range (len(ma), 1, -1):
        #    cv2.line(warpedorg, ma[r-1], ma[r-2],(0,0,255),2)
        #    if r < len(ma):
        #        sum_lane_center = (ma[r-1][0] + sum_lane_center)
                #print(ma[r-1][0])
        avg_lane_center = totm // len(ma)
        diff = 265 - avg_lane_center
        print('difference = ' + str(diff))
        diff = int(diff *0.7)
        #print('difference scaled = ' + str(diff))
        if diff > 0 and diff > 70:
            diff = 70
        elif diff < 0 and diff < -70:
            diff = -70
        print('adjusted diff = ' + str(diff))
        diff = diff + tan
        print('adjusted after tangent calculated = ' + str(diff) )
        print('set steering to ' + str(diff)+ '. average is ' + str(avg_lane_center) + ' - needs motor moving')

        #print('reached here')
        if motorenabled:                
            if abs(diff - mediumMotor.position) > 8:
                print('position is ' + str(mediumMotor.position) + '. set steering to ' + str(diff)+ '. difference is ' + str(abs(diff - mediumMotor.position)) + '. average is ' + str(avg_lane_center) + ' - needs motor moving')
                #print('abs ' + str(abs(diff - mediumMotor.position)))
                #print('setting motor pos')
                #mediumMotor.stop()
                mediumMotor.on_to_position(speed = 8, position = int(diff) , brake = False)
                mediumMotor.stop()
                print('----')
    else:
        #if motorenabled:
            #stop()
        print('unable to detect proper lanes for driving')        
    '''
    plt.plot(lx, ly, 'o')
    #plt.imshow(nm, cmap="gray")
    lf = np.polyfit(lx, ly, 2)
    rf = np.polyfit(rx, ry, 2)
    plt.plot(lx, np.polyval(lf,lx), 'r-', linewidth = 4.0)
    plt.plot(rx, np.polyval(rf,rx), 'g-', linewidth = 4.0)
    print(lx)
    '''
    return nm

#run()
time.sleep(0.2)
t = 0
iteration = 0
stopped = 0
for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
    previous_diff = 0
    '''if ts.is_pressed:
        print('exiting')
        mediumMotor.stop()
       exit()]
    '''
    
    if motorenabled:
        if us.distance_centimeters < 30.0:
                stop()
                stopped = 1
                iteration = iteration + 1
                if iteration == 4:
                        pygame.mixer.music.load('why.mp3')
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() == True:
                            continue
                        #print('speak last')#time.sleep(2)
                        #s.speak('OBSTACLE NOT REMOVED. STOPPING REAR MOTORS NOW', play_type = Sound.Sound.PLAY_WAIT_FOR_COMPLETE)
                        #exit()
                if iteration < 3:
                        pygame.mixer.music.load('obstacle.mp3')
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() == True:
                            continue
                        #print('speak')#s.speak('OBSTACLE DETECTED. ' + str(iteration), play_type = Sound.PLAY_WAIT_FOR_COMPLETE)
        elif stopped == 1 and us.distance_centimeters > 30.0:
                #run()
                print('running')
                iteration = 0
                stopped = 0
    
    img = frame.array    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    point1 = (290, 320)
    point2 = (408, 320)
    point3 = (126, 413)
    point4 = (619, 413)
    cv2.circle(img, point1, 3, (0,0,0), -1)    
    cv2.circle(img, point2, 3, (0,0,0), -1)
    
    cv2.circle(img, point3, 3, (0,0,0), -1)
    cv2.circle(img, point4, 3, (0,0,0), -1)
    #cv2.imshow('img', img)
    orig_pts = np.float32([[290, 320], [408, 320], [126, 413],[619,413]])
    dest_pts = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])

    M = cv2.getPerspectiveTransform(orig_pts, dest_pts)
    warpedorg = cv2.warpPerspective(img, M, (600,600))
    #cv2.imshow('img', warpedorg) 
    th = 190
    #th = 80
    warpedorg[warpedorg < th] = 0    # Black
    warpedorg[warpedorg >= th] = 255 # White
    
    #cv2.imshow("bw", warpedorg)
        
    func(warpedorg)
    #cv2.imshow("un", warpedorg)    
    
    cv2.line(warpedorg,(300,0),(300,600),(255,0,0),2)
    
    cv2.imshow("undistorted", warpedorg)
    key = cv2.waitKey(1) & 0xFF
    
    rawCapture.truncate(0)
    if key == ord('q'):
        if motorenabled:
            mediumMotor.stop()
            stop()
        break
    elif key == ord('c'):
        cv2.imwrite("undist.jpg", warpedorg)
        break
    print((time.time() * 1000) - t)
    t = (time.time() * 1000)
