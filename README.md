# AutonomousLegoEv3

This repository contains my attempts at buliding a self driving car using a RaspberryPi, Python and a LEGO EV3 Mindstorms educational unit
## Getting Started 
Following is a tutorial on getting started with LEGO EV3, EV3Dev, RaspberryPi and Python.
##### Good to know

  - Tutorial written by Syed Waqee Wali.
  - Shortcut to open a terminal on RaspberryPi: 
        ```
             Ctrl + Alt + T
        ```
### Setup RaspberryPi (Skip if raspbian, pip and pip3 is installed)

 - Download the latest version of Raspbian.
 - Google Etcher, Download, Install and follow the   [tutorial](https://www.raspberrypi.org/documentation/installation/installing-images/) to flash your SD card with Raspbian.
 - Install pip by using **JUST** the pip part  [here](https://www.raspberrypi.org/documentation/linux/software/python.md).
- Install pip3 by following [this](https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/)
	- Make sure you verify the pip3 version
- Close all terminals

### Setup EV3
##### On the EV3
  - Download the latest version of ev3dev, and flash it onto your SD card using Etcher
- Insert the SD card  in the EV3 and boot it
- Wait until  booting is complete
- Connect the RaspberryPi and EV3 using the USB cable
- Navigate to Wireless and networks > all network connections > Wired
- Click on Connect
- Click on Connect Automatically, check if the box is checked
- Wait until this screen shows Connected or an IP is shown on the top left corner of the display
- Connect the medium motor to any motor ports, prefereable A

##### On the RaspberryPi
- Open a new terminal and type in
        ```
             ssh robot@ev3dev.local	
        ```
- Enter the password: 
        ```
             maker	
        ```
- After you are greeted with the welcome EV3Dev screen, enter
	    ```
        sudo easy_install3 rpyc
        ```
- If asked for password: 
        ```
             maker	
        ```
- Enter 
        ```
	    pip3 show rpyc
	    ```
- Take a note of the version number
- Open a new terminal **WITHOUT** closing the older one and enter 
        ```
	sudo pip3 install rpyc==[the rpyc version you noted above, without the square brackets]
	    ```
- In the previous terminal window type in
        ```
	sudo nano rpyc_server.sh
	    ```
- Type the following exactly (copy and paste if possible)
    ```
    #!/bin/bash
    python3 `which rpyc_classic.py`			
    ```
- Press `Ctrl + X` then `Ctrl + Y` to save the file
- Enter
	`chmod +x rpyc_server.sh`
- Enter
			`./rpyc_server.sh`
- Wait until the confirmation as below is shown (* refers to some numbers)
    ```
    INFO:SLAVE/****:server started on ***:*** ``
    ```
- For subsequent connections to ev3
- ssh into the ev3 and enter
				`./rpyc_server.sh`
- This should show you the `'server started on ***'` as above


#### Points to take care about
- Never set the motor position to the same position as previous. Always make sure that the new position is +/- 10 the previous position
- Always set the brake parameter where available to False (will be discussed below)
- Never try to manually resist the motor from spinning
- Do not use any other method than on_to_position to move the motor (will be discussed below) - can be argued upon
- Never try the above; Unless you are willing to dismantle the ev3brick from the robot, unplugging the battery and doing the connection again


### Lets move some motors
- Verify that the medium motor is connected to one of the motor ports and not some sensor port
- Verify that the ev3 and raspberrypi are connected through usb
- Verify that the rpyc server is started on the ev3 ssh terminal should show
 `"INFO:SLAVE/****:server started on ***:***"`
##### On the RaspberryPi
- Open a terminal and enter
	`python3`
- This should get you into the python terminal which shows `>>>` followed by a blinking cursor
- Enter the following change the `A` in `outA` below to the port you connected your motor to
	```
    import rpyc
	con = rpyc.classic.connect('ev3dev.local')
	motors = con.modules['ev3dev2.motor']
	mediumMotor = motors.MediumMotor('outA')
	```
- This will get you the `mediumMotor` object to play with
- At this point, make sure that you center the front wheels if you are connected to steering motor of the car. You might need to unplug the battery and reconnect otherwise
- After the centering is done, enter			
	`mediumMotor.reset()`
- This will set the current position of the motor to `0`
- Verify by entering 
	`mediumMotor.position`
- If you are certain that the wheels are center, and still the position shows something else, run
	`mediumMotor.position = 0`
- `mediumMotor.position` refers to the current position of the motor in terms of degrees
- Refer to the points to take care about section above 
- On the car, a positive position will move the steering towards left and vice-versa for right
- Refer to the points to take care about section above again
- Enter the following to move the motor. If you are connected to the steering motor, make sure you center the wheels before this is executed
	`mediumMotor.on_to_position(speed = 10, position = 30, brake = False)`
- `speed` refers to the speed with which to move the motor. This any value between `1` and  `100` value. Best value is anything less than 26 based on experience. Other values can overshoot the motor and move to a different value of `position` than expected
- `position` refers to the position on which to move the motor form the current position. If the center is `0`, setting position to `30` will move the steering to `30` motor degrees towards the left. Setting it to `-30` will however, move the motor to the right
- Make sure you understand the following
    - `position` is an absolute position value on the motor. If set to `15`, and then `-15`, the current `mediumMotor.position` will not be `0`. it will be `-15`. i.e. it will have travelled `30` motor degrees towards right (`15` to `0` and then `0` to -`15`)
- Don't exceed the position on the steering motor by +/-65 (my ugly experience makes me say this)
