// illustrates use of a generic action client that communicates with
// an action server called "cartMoveActionServer"
// the actual action server can be customized for a specific robot, whereas
// this client is robot agnostic

//launch with roslaunch irb140_description irb140.launch, which places a block at x=0.5, y=0
// launch action server: rosrun irb140_planner irb140_cart_move_as
// then run this node

//for manual gripper control,  rosrun cwru_sticky_fingers finger_control_dummy_node /sticky_finger/link6 false

#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <arm_motion_action/arm_interfaceAction.h>
#include <cartesian_motion_commander/cart_motion_commander.h>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <xform_utils/xform_utils.h>
#include <std_srvs/SetBool.h>
#include <geometry_msgs/PoseStamped.h>
using namespace std;

//Global variable to hold block pose obtained in subscriber callback
geometry_msgs::PoseStamped block_pose_;
bool gotNewBlockPose_;

void blockPoseCb(const geometry_msgs::PoseStamped& newBlockPose) {
        block_pose_ = newBlockPose;
        gotNewBlockPose_ = true;
}

Eigen::MatrixXd quaternionToMatrix(geometry_msgs::Quaternion quat) {
        Eigen::MatrixXd m(3,3);
        double x = quat.x;
        double y = quat.y;
        double z = quat.z;
        double w = quat.w;
        double x2 = x*x;
        double y2 = y*y;
        double z2 = z*z;
        double w2 = w*w;

        m.row(0) << 1-(2*y2)-(2*z2), 2*x*y-2*z*w, 2*x*z+2*y*w;
        m.row(1) << 2*x*y+2*z*w, 1-2*x2-2*z2, 2*y*z-2*x*w;
        m.row(2) << 2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x2-2*y2;

        return m;
}

int main(int argc, char** argv) {
        ros::init(argc, argv, "example_arm_cart_move_ac"); // name this node
        ros::NodeHandle nh; //standard ros node handle
        CartMotionCommander cart_motion_commander;
        XformUtils xformUtils;
        ros::ServiceClient client = nh.serviceClient<std_srvs::SetBool>("/sticky_finger/link6");
        std_srvs::SetBool srv;
        srv.request.data = true;

        //Getting user input of where the gear should be moved to
        double x_cor, y_cor;
        cout << "Enter an X-coordinates:-";
        cin>>x_cor;
        cout << "Enter a Y-coordinates:-  ";
        cin>>y_cor;

        //Defining subscriber to get pose of the block.
        ros::Subscriber block_pose_sub = nh.subscribe("block_pose", 1, blockPoseCb);

        Eigen::VectorXd joint_angles;
        Eigen::Vector3d dp_displacement;
        int rtn_val;
        int njnts;
        int nsteps;
        double arrival_time;
        geometry_msgs::PoseStamped tool_pose, tool_pose_home;

        bool traj_is_valid = false;
        int rtn_code;

        nsteps = 10;
        arrival_time = 2.0;

        //REQUESTION DOWN POSE
        Eigen::Vector3d b_des, n_des, t_des, O_des;
        Eigen::Matrix3d R_gripper;
        b_des << 0, 0, -1;
        n_des << -1, 0, 0;
        t_des = b_des.cross(n_des);

        R_gripper.col(0) = n_des;
        R_gripper.col(1) = t_des;
        R_gripper.col(2) = b_des;

        O_des << 0.5, 0.3, 0.3;
        Eigen::Affine3d tool_affine;
        tool_affine.linear() = R_gripper;
        tool_affine.translation() = O_des;
        //   geometry_msgs::PoseStamped transformEigenAffine3dToPoseStamped(Eigen::Affine3d e,std::string reference_frame_id);

        tool_pose = xformUtils.transformEigenAffine3dToPoseStamped(tool_affine, "system_ref_frame");
        ROS_INFO("requesting plan to gripper-down pose:");
        xformUtils.printPose(tool_pose);
        rtn_val = cart_motion_commander.plan_jspace_traj_current_to_tool_pose(nsteps, arrival_time, tool_pose);
        if (rtn_val == arm_motion_action::arm_interfaceResult::SUCCESS) {
                ROS_INFO("successful plan; command execution of trajectory");
                rtn_val = cart_motion_commander.execute_planned_traj();
                ros::Duration(arrival_time + 0.2).sleep();
        } else {
                ROS_WARN("unsuccessful plan; rtn_code = %d", rtn_val);
        }

        //Transform of the gripper in the blocks frame (alligned x axes)
        Eigen::MatrixXd T_gb(4,4);
        T_gb.row(0) <<  1,  0,  0, 0;
        T_gb.row(1) <<  0, -1,  0, 0;
        T_gb.row(2) <<  0,  0, -1, 0.0343;
        T_gb.row(3) <<  0,  0,  0, 1;

        while (!gotNewBlockPose_) {
                ros::spinOnce();
        }
        gotNewBlockPose_ = false;

        //Transform of block in reference to robot (from find_block.cpp)
        Eigen::MatrixXd R_br = quaternionToMatrix(block_pose_.pose.orientation);
        Eigen::MatrixXd T_br(4,4);
        T_br.row(0) <<  R_br(0,0),  R_br(0,1),  R_br(0,2), block_pose_.pose.position.x;
        T_br.row(1) <<  R_br(1,0),  R_br(1,1),  R_br(1,2), block_pose_.pose.position.y;
        T_br.row(2) <<  R_br(2,0),  R_br(2,1),  R_br(2,2), 0;
        T_br.row(3) <<  0,          0,          0,         1;

        Eigen::MatrixXd T_gr(4,4);
        T_gr = T_br*T_gb;

        tool_affine.linear().row(0) <<  T_gr(0,0),  T_gr(0,1),  T_gr(0,2);
        tool_affine.linear().row(1) <<  T_gr(1,0),  T_gr(1,1),  T_gr(1,2);
        tool_affine.linear().row(2) <<  T_gr(2,0),  T_gr(2,1),  T_gr(2,2);
        tool_affine.translation() << T_gr(0,3), T_gr(1,3), T_gr(2,3); //x,y,z in transformation matrix

        ROS_INFO("enabling vacuum gripper");
        //enable the vacuum gripper:
        srv.request.data = true;
        while (!client.call(srv) && ros::ok()) {
                ROS_INFO("Sending command to gripper...");
                ros::spinOnce();
                ros::Duration(0.5).sleep();
        }

        //move to approach pose:
        ROS_INFO("moving to approach pose");
        //tool_pose.pose.position.x=block_pose_.pose.position.x;
        //tool_pose.pose.position.y=block_pose_.pose.position.y;
        //tool_pose.pose.position.z = 0.05;         //0.01;
        //tool_pose.pose.orientation = block_pose_.pose.orientation; //Orientation lined up with block
        tool_pose = xformUtils.transformEigenAffine3dToPoseStamped(tool_affine, "system_ref_frame");
        ROS_INFO("requesting plan to descend:");
        xformUtils.printPose(tool_pose);
        rtn_val = cart_motion_commander.plan_cartesian_traj_qprev_to_des_tool_pose(nsteps, arrival_time, tool_pose);
        if (rtn_val == arm_motion_action::arm_interfaceResult::SUCCESS) {
                ROS_INFO("successful plan; command execution of trajectory");
                rtn_val = cart_motion_commander.execute_planned_traj();
                ros::Duration(arrival_time + 0.2).sleep();
        } else {
                ROS_WARN("unsuccessful plan; rtn_code = %d", rtn_val);
        }

        ROS_INFO("requesting plan to depart with grasped object:");
        tool_pose.pose.position.z = 0.3;

        xformUtils.printPose(tool_pose);
        rtn_val = cart_motion_commander.plan_cartesian_traj_qprev_to_des_tool_pose(nsteps, arrival_time, tool_pose);
        if (rtn_val == arm_motion_action::arm_interfaceResult::SUCCESS) {
                ROS_INFO("successful plan; command execution of trajectory");
                rtn_val = cart_motion_commander.execute_planned_traj();
                ros::Duration(arrival_time + 0.2).sleep();
        } else {
                ROS_WARN("unsuccessful plan; rtn_code = %d", rtn_val);
        }

        ROS_INFO("requesting plan to rotate to correct orientation:");
        //tool_pose.pose.orientation =  xformUtils.convertPlanarPsi2Quaternion(-3.14159265359);
        tool_pose.pose.orientation.x =  0;
        tool_pose.pose.orientation.y =  -1;
        tool_pose.pose.orientation.z =  0;
        tool_pose.pose.orientation.w =  0;

        xformUtils.printPose(tool_pose);
        rtn_val = cart_motion_commander.plan_cartesian_traj_qprev_to_des_tool_pose(nsteps, arrival_time, tool_pose);
        if (rtn_val == arm_motion_action::arm_interfaceResult::SUCCESS) {
                ROS_INFO("successful plan; command execution of trajectory");
                rtn_val = cart_motion_commander.execute_planned_traj();
                ros::Duration(arrival_time + 0.2).sleep();
        } else {
                ROS_WARN("unsuccessful plan; rtn_code = %d", rtn_val);
        }

        ROS_INFO("requesting plan to move to desired location:");
        tool_pose.pose.position.x = x_cor;
        tool_pose.pose.position.y = y_cor;

        xformUtils.printPose(tool_pose);
        rtn_val = cart_motion_commander.plan_cartesian_traj_qprev_to_des_tool_pose(nsteps, arrival_time, tool_pose);
        if (rtn_val == arm_motion_action::arm_interfaceResult::SUCCESS) {
                ROS_INFO("successful plan; command execution of trajectory");
                rtn_val = cart_motion_commander.execute_planned_traj();
                ros::Duration(arrival_time + 0.2).sleep();
        } else {
                ROS_WARN("unsuccessful plan; rtn_code = %d", rtn_val);
        }

        ROS_INFO("requesting plan to descend on desired location:");
        tool_pose.pose.position.z = 0.036;         //0.035 is height of block

        xformUtils.printPose(tool_pose);
        rtn_val = cart_motion_commander.plan_cartesian_traj_qprev_to_des_tool_pose(nsteps, arrival_time, tool_pose);
        if (rtn_val == arm_motion_action::arm_interfaceResult::SUCCESS) {
                ROS_INFO("successful plan; command execution of trajectory");
                rtn_val = cart_motion_commander.execute_planned_traj();
                ros::Duration(arrival_time + 0.2).sleep();
        } else {
                ROS_WARN("unsuccessful plan; rtn_code = %d", rtn_val);
        }

        //disable the vacuum gripper:
        ROS_INFO("Disabling Vacuum Gripper:");
        srv.request.data = false;
        while (!client.call(srv) && ros::ok()) {
                ROS_INFO("Sending command to gripper...");
                ros::spinOnce();
                ros::Duration(0.5).sleep();
        }

        tool_pose.pose.position.z = 0.3;         //0.035 is height of block

        xformUtils.printPose(tool_pose);
        rtn_val = cart_motion_commander.plan_cartesian_traj_qprev_to_des_tool_pose(nsteps, arrival_time, tool_pose);
        if (rtn_val == arm_motion_action::arm_interfaceResult::SUCCESS) {
                ROS_INFO("successful plan; command execution of trajectory");
                rtn_val = cart_motion_commander.execute_planned_traj();
                ros::Duration(arrival_time + 0.2).sleep();
        } else {
                ROS_WARN("unsuccessful plan; rtn_code = %d", rtn_val);
        }
        //}

        return 0;
}
