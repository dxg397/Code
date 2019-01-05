#include<ros/ros.h> 
#include<std_msgs/Float64.h> 
std_msgs::Float64 g_amp;
std_msgs::Float64 g_sin_vel;
std_msgs::Float64 g_disp; // displacement

void myCallback(const std_msgs::Float64& message_holder) {
    ROS_INFO("received force value is: %f", message_holder.data);
    g_disp.data = message_holder.data; // post the received data in a global var for access by 
    // main prog. 
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "asg1_sin_commander"); //name this node 
    // when this compiled code is run, ROS will recognize it as a node called "minimal_simulator" 
    ros::NodeHandle nh; // node handle 
    //create a Subscriber object and have it subscribe to the topic "force_cmd" 
    ros::Subscriber my_subscriber_object = nh.subscribe("sin_cmd", 1, myCallback);
    //simulate accelerations and publish the resulting velocity; 
    ros::Publisher my_publisher_object = nh.advertise<std_msgs::Float64>("velocity", 1);

    double dt = 0.01; //10ms integration time step 
    double sample_rate = 1.0 / dt; // compute the corresponding update frequency 
    ros::Rate naptime(sample_rate);
    g_sin_vel.data = 0.0; 
    g_disp.data = 0.0; 
    g_amp.data = 0.0; 
    while (ros::ok()) {
        g_sin_vel.data = (sample_rate * g_amp.data * g_disp.data); 
        my_publisher_object.publish(g_sin_vel); 
        ROS_INFO("sinusoidal velocity = %f", g_sin_vel.data);
        ros::spinOnce();  
        naptime.sleep();  

    }
    return 0; // should never get here, unless roscore dies 
} 
