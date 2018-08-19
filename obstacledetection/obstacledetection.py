import rpyc

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
    m.run_forever(speed_sp=50)
    m2.run_forever(speed_sp=50)

def stop():
    m.stop(stop_action="hold")
    m2.stop(stop_action="hold")

run()
ev3.Sound.beep()
while not ts.value():
    distance = us.value()/10  # convert mm to cm
    print(str(distance) + " " + units)
    if distance < 10.0:
        stop()
        ev3.Sound.speak("OBSTACLE!").wait()
        ev3.Sound.tone([(1000, 100, 500),(1000, 100, 500)])#.wait()
    else:
        run()
stop()
ev3.Sound.beep()