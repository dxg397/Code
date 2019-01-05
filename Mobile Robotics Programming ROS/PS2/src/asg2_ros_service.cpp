#include <ros/ros.h>
#include <asg2_service/asg2_service.h> 
#include <iostream>
#include <string>
#include <math.h>
using namespace std;

bool callback(asg2_service::asg2_serviceRequest& request, asg2_service::asg2_serviceResponse& response)
{
    ROS_INFO("callback activated");
    float amp = request.amplitude;
    float freq = request.frequency;
    response.v_cmd  = amp*sin(freq*2.0*M_PI*0.01);
    ROS_INFO("received amplitude and frequency request with %d and %d ",amp,freq);
  return true;
}
int main(int argc, char **argv)
{
  ros::init(argc, argv, "asg2_service");
  ros::NodeHandle n;

  ros::ServiceServer service = n.advertiseService("asg2_service", callback);
  ROS_INFO("Ready accept amplitude and frequency.");
  ros::spin();

  return 0;
}

