scratchpad.py is a quick write down of the Inverse Kinematics that was involved. It is no way essential to the actual running of the project.

In esp32_yolo_grid.py, you will need to pip install the following modules:

	opencv-python
	numpy
	serial (not pyserial)
	ultralytics

The ESP32_IP in the esp32 python script is the IP of the board on MY local network, not yours. Please change that by following the below steps (for Windows and Linux):

	-Connect the ESP32 to a network whose details you can easily access (like your mobile hotspot where you can see MAC addresses of each connected device)
	-Open terminal (with admin/root privileges) and run ifconfig(Linux) or ipconfig(Windows) to get your IP. You will most likely see your IP in the form of: XXX.XXX.XXX.XXX.
	-In terminal, run ping -b XXX.XXX.XXX.255 to ping your broadcast channel. You can specify number of packets to send, or just press Ctrl+C after a certain period of time, if you are confident of enough results.
	-Run arp -a. This command will most likely give you the IPs connected to your network and the corresponding MAC addresses. Reference the ESP32's MAC address that you got from the first step to get its IP.
	-Change the IP in the script to this IP.
	-If in anyway you are unable to get the IP, or the commands do not work, then either search it on Google, or switch to a more suited network (like your mobile hotspot with basic WPA protection). 
	OR 
	You can get IP from the ESP32 serial communication itself.

The board used for the robotic manipulator itself was an Arduino Nano. The code is in the folder titled Robbie (subdirectory within, not parent directory. I could not think of a better name at that moment). 
