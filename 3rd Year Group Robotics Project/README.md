This is the end result of our NovaBot which I had contributed to the development of.   
The task was specifically to create an automonous system capable of tracking 'rogue' droids, identify the error signal, to then immobilise safely and return to a designated 'maintenance' area.  
My specific contribution to this was with regards to the overhead live tracking of multiple droids, followed by the detection of their error codes via the use of OpenCV and Tensorflow. See 1:09 (Puck Detection) to 1:30 (Visual Error Code Detection) of the group project video.  
Files which specifically relate to each of these tasks can be found in their relevant folders.

For further videos relating to the development of our robotics solution, see the following Youtube channel: https://www.youtube.com/channel/UCl9ws_-HdhdBl4uOnCiEnHw/featured
The following videos will be of particular note:  
- [RVIZ via ROS - (Robot Operating System)](https://www.youtube.com/watch?v=94yw04b37Po) - which demonstrates the capability we were able to reach to automate the robots movement. Via the overhead camera, I was able to create a map with each boundary fed back to the central PC. RVIZ was then used to manage the automated pathing of the robot to final location (shown by the red arrow in the video)
- [Object detection and tracking system](https://www.youtube.com/watch?v=5fYSBhuBZVU) - which demonstrates the system's capability to automatically find the 'puck' and then consequently track its movement.
- [Training of CNN](https://www.youtube.com/watch?v=hqqe4yP7yQQ) - which shows the results of training the model. We see a fairly good accuracy, however this was very dependant on light conditions remaining fairly consistent. Due to the problem we were faced, the light conditions were fairly consistent so this issue was not tackled further as to make time for other more key aspects of the project. 
- [Visual Error Code detection](https://www.youtube.com/watch?v=81y13W2RRVI) - which shows the error code detection part of the system in action.

