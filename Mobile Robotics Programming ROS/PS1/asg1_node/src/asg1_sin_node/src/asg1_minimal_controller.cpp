#include<ros/ros.h> 
#include<std_msgs/Float64.h> 
//global variables for callback functions to populate for use in main program 

std_msgs::Float64 g_sin_vel;
std_msgs::Float64 g_vel_cmd;


void myCallbackSin(const std_msgs::Float64& message_holder) {
    // check for data on topic "Amplitude" 
    ROS_INFO("received Amplitude value is: %f", message_holder.data);
    g_sin_vel.data = message_holder.data; // post the received data in a global var for access by 
    //main prog. 
}

void myCallbackVelCmd(const std_msgs::Float64& message_holder) {
    // check for data on topic "vel_cmd" 
    ROS_INFO("received velocity command value is: %f", message_holder.data);
    g_vel_cmd.data = message_holder.data; // post the received data in a global var for access by 
    //main prog. 
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "asg1_sin_minimal_controller"); 
    ros::NodeHandle nh; 
    //create 2 subscribers: one for sinusoidal velocity and one for velocity commands 
    ros::Subscriber my_subscriber_object1 = nh.subscribe("sinusoidal", 1, myCallbackSin);
    ros::Subscriber my_subscriber_object2 = nh.subscribe("vel_cmd", 1, myCallbackVelCmd);
    //publish a force command computed by this controller; 
    ros::Publisher my_publisher_object = nh.advertise<std_msgs::Float64>("sin_cmd", 1);
    double Kv = 1.0; // velocity feedback gain 
    double dt_controller = 0.1; //specify 10Hz controller sample rate (pretty slow, but 
    //illustrative) 
    double sample_rate = 1.0 / dt_controller; 
    ros::Rate naptime(sample_rate); 
    g_sin_vel.data = 0.0; /
    g_force.data = 0.0;
    g_vel_cmd.data = 0.0; // init velocity command to zero 
    double vel_err = 0.0; // velocity error 

    while (ros::ok()) {
        vel_err = g_vel_cmd.data - g_sin_vel.data; /
        ros::spinOnce(); //allow data update from callback; 
        naptime.sleep(); // wait for remainder of specified period; 
    }
    return 0; // should never get here, unless roscore dies 
} 
