#include <ros/ros.h>
#include <asg2_service/asg2_service.h> 
#include <iostream>
#include <string>
using namespace std;

int main(int argc, char **argv) {
    ros::init(argc, argv, "asg2_service");
    ros::NodeHandle n;
    ros::ServiceClient client = n.serviceClient<asg2_service::asg2_service>("asg2_service");
    asg2_service::asg2_service srv;
    bool found_on_list = false;
    float amp;
    float freq;
    while (ros::ok()) {
        cout<<endl;
        cout << "enter the amplitude (x to quit): ";
        cin>>amp;
        
        cout<<endl;
        cout << "enter the Frequency (x to quit): ";
        cin>>freq;

        srv.request.amplitude = amp; 
        srv.request.frequency = freq; 
        if (client.call(srv)) {
            
            cout << "Amplitude entered is:- "<<srv.request.amplitude;
            cout << "Frequency entered is:- "<<srv.request.frequency;
	    cout << "Caluclated Velocity command is :- "<<srv.response.v_cmd;
        } else {
            ROS_ERROR("Failed to call service asg2_service");
            return 1;
        }
    }
    return 0;
}

