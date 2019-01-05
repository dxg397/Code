//ps6 ROS client:
// first run: rosrun example_ROS_service example_ROS_service
// then start this node:  rosrun example_ROS_service example_ROS_client



// first install araic pkgs
// type rosrv show std_srvs/Trigger to check the Trigger.

// moving the conveyour without tsrating up the system
// rosservice call /ariac/conveyor/control "power:100"
// we have to integrate this in the code...
// so we get info about the service using rosservice info  /ariac/conveyor/control

// we will get a service with osrf_gear/ConveyorBeltControl which is to start the conveyor

#include <ros/ros.h>
//#include <example_ros_service/ExampleServiceMsg.h> // this message type is defined in the current package
#include <iostream>
#include <std_srvs/Trigger.h>
#include <osrf_gear/ConveyorBeltControl.h>
#include <osrf_gear/DroneControl.h>
#include <osrf_gear/LogicalCameraImage.h>
// we had to add this in the pacakage.xml file else will get an error.
#include <string>
using namespace std;


// if  we need to take a snapshot....
bool g_take_new_snapshot = false;


osrf_gear::LogicalCameraImage g_cam1_data;

void cam2B(const osrf_gear::LogicalCameraImage& message_holder)
{

    if(g_take_new_snapshot) {
       ROS_INFO_STREAM("Image from cam1: "<<message_holder<<endl);
       g_cam1_data = message_holder;
       g_take_new_snapshot = false;
    }
//    ROS_INFO("Received vale is: %f", message_holder.data);

}

int main(int argc, char **argv) 
{
    ros::init(argc, argv, "ps6");
    ros::NodeHandle n;
    ros::ServiceClient startup_client = n.serviceClient<std_srvs::Trigger>("/ariac/start_competition");
    std_srvs::Trigger startup_srv;
    startup_srv.response.success=false;
    ///////////////////////////////////////////////////////////////////////////
    // service and message for startig the conveyor 2nd client
    ros::ServiceClient conveyor_client = n.serviceClient<osrf_gear::ConveyorBeltControl>("/ariac/conveyor/control");
    // establish a container of correct datatype
    osrf_gear::ConveyorBeltControl conveyor_srv;
    conveyor_srv.response.success=false;
    /////////////////////////////////////////////////////////////////////////////////////////
  
    // logical camera service 
    ros::Subscriber cam2_subscriber = n.subscribe("/ariac/logical_camera_2",1,cam2B);

    /////////////////////////////////////////////////////////////////////////////////////////

    // drone service 

    ros::ServiceClient drone_client = n.serviceClient<osrf_gear::DroneControl>("/ariac/drone/control");
    // establish a container of correct datatype
    osrf_gear::DroneControl drone_srv;
    drone_srv.response.success=false;
    ///////////////////////////////////////////////////////////////////////////////////

    


    while(!startup_srv.response.success)
    {

        ROS_WARN("not successful starting up yet......");
        startup_client.call(startup_srv);
        ros::Duration(0.5).sleep();
    }
    ROS_INFO("got success response from statup client");

// the while loop to start the conveyor
    conveyor_srv.request.power = 100; // to start the conveyor with 100% power...
    while(!conveyor_srv.response.success)
    {

        ROS_WARN("not successful conveyor starting up yet......");
        conveyor_client.call(conveyor_srv);
        ros::Duration(0.5).sleep();
    }
    ROS_INFO("got success starting conveyor belt....");
    
    //////////////////////////////////////////////////////////////////////////////////////
    // logical camera
    ///////////////////////////////////////////////////////////////////////
    // rosmsg show osrf_gear/LogicalCameraImage
    // it will give the shiping boc label position and its own position wrt to the world
    g_take_new_snapshot =true;
    string box_name("shipping_box");
    cout<<g_cam1_data.models.size();
    int i=0;
    osrf_gear::Model model;
    geometry_msgs::Pose box_pose_wrt_cam;
    
    while(g_cam1_data.models.size()<1) // as long as it doesnt see any objects..s
    {
       // model = g_cam1_data.models;
        //string model_name(model.type);
        //cout<<model_name;
       g_take_new_snapshot =true;
        ros::spinOnce();
        ros::Duration(0.5).sleep();
    }

   
    
    ros::Duration(3.2).sleep();
    conveyor_srv.request.power = 0;
    ROS_INFO_STREAM("Z value at halt: - "<<g_cam1_data.models[0].pose.position.z<<endl);
    ROS_WARN("stopping the coveyor for 5 seconds......");
    conveyor_client.call(conveyor_srv);
    ROS_INFO("I see a box");
    ros::Duration(5.0).sleep(); // 5 seconds break
     ROS_INFO_STREAM("Z value at halt: - "<<g_cam1_data.models[0].pose.position.z<<endl);
    conveyor_srv.request.power =100;
    ROS_WARN("starting the coveyor after 5 seconds......");
    conveyor_client.call(conveyor_srv);



    drone_srv.request.shipment_type = "order_0_shipment_0"; // to start the drone with a shipment label...
    while(!drone_srv.response.success)
    {

        ROS_WARN("not successful drone starting up yet......");
        drone_client.call(drone_srv);
        ros::Duration(0.5).sleep();
        cout<<drone_srv.response.success;
    }
    ROS_INFO("got success starting drone service....");
    
    return 0;
}
