import rpyc
from time import sleep

con = rpyc.classic.connect('ev3dev.local')
ev3 = con.modules['ev3dev.ev3']
us = ev3.UltrasonicSensor()
assert us.connected, ev3.Sound.speak("Connect Ultrasonic Sensor").wait()
us.mode='US-DIST-CM'
units = us.units

ts = ev3.TouchSensor()
assert ts.connected, ev3.Sound.speak("Connect Touch Sensor").wait()

m = ev3.LargeMotor("outA")
m2 = ev3.LargeMotor("outD")

def run():
    m.run_forever(speed_sp=350)
    m2.run_forever(speed_sp=350)
    sleep(1)

def stop():
    m.stop(stop_action="hold")
    m2.stop(stop_action="hold")
    sleep(2)

def turn():
    m.run_timed(time_sp=1000, speed_sp=350, stop_action='brake')
    m.wait_while('running')
    
def reverse():
    m.run_timed(time_sp=1500, speed_sp=-250, stop_action='brake')
    m2.run_timed(time_sp=1500, speed_sp=-250, stop_action='brake')
    sleep(1)
    m.wait_while('running')
    m2.wait_while('running')
    #m2.speed_sp(-50)
ev3.Sound.beep()

run()
while not ts.value():
    distance = us.value()/10  # convert mm to cm
    #print(str(distance) + " " + units)
    if distance < 30.0:
        stop()
        reverse()
        turn()
        #ev3.Sound.speak("OBSTACLE!").wait()
        ev3.Sound.tone([(1000, 100, 500),(1000, 100, 500)])#.wait()
    else:
        run()
stop()
ev3.Sound.beep()