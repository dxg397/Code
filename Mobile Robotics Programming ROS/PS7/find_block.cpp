//get images from topic "simple_camera/image_raw"; remap, as desired;
//search for red pixels;
// convert (sufficiently) red pixels to white, all other pixels black
// compute centroid of red pixels and display as a blue square
// publish result of processed image on topic "/image_converter/output_video"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <xform_utils/xform_utils.h>
#include <numeric>

static const std::string OPENCV_WINDOW = "Open-CV display window";
using namespace std;

int g_redratio; //threshold to decide if a pixel qualifies as dominantly "red"

const double BLOCK_HEIGHT=0.035; // hard-coded top surface of block relative to world frame

class ImageConverter {
ros::NodeHandle nh_;
image_transport::ImageTransport it_;
image_transport::Subscriber image_sub_;
image_transport::Publisher image_pub_;
ros::Publisher block_pose_publisher_;     // = n.advertise<std_msgs::Float64>("topic1", 1);
geometry_msgs::PoseStamped block_pose_;
XformUtils xformUtils;

public:

//Takes in x and y values. Use this equation that I learned in stats of signal processing:
//
double getSlope(const std::vector<double>& x, const std::vector<double>& y) {
        double n = x.size();

        double EX    = std::accumulate(x.begin(), x.end(), 0.0)/n;
        double EY    = std::accumulate(y.begin(), y.end(), 0.0)/n;
        double EXX   = std::inner_product(x.begin(), x.end(), x.begin(), 0.0)/n;
        double EXY   = std::inner_product(x.begin(), x.end(), y.begin(), 0.0)/n;
        double slope = (EXY-(EX*EY)) / (EXX-(EX*EX));
        return slope;
}

ImageConverter(ros::NodeHandle &nodehandle)
        : it_(nh_) {
        // Subscribe to input video feed and publish output video feed
        image_sub_ = it_.subscribe("simple_camera/image_raw", 1,
                                   &ImageConverter::imageCb, this);
        image_pub_ = it_.advertise("/image_converter/output_video", 1);
        block_pose_publisher_ = nh_.advertise<geometry_msgs::PoseStamped>("block_pose", 1, true);
        block_pose_.header.frame_id = "world"; //specify the  block pose in world coords
        block_pose_.pose.position.z = BLOCK_HEIGHT;
        block_pose_.pose.position.x = 0.5; //not true, but legal
        block_pose_.pose.position.y = 0.0; //not true, but legal

        // need camera info to fill in x,y,and orientation x,y,z,w
        //geometry_msgs::Quaternion quat_est
        //quat_est = xformUtils.convertPlanarPsi2Quaternion(yaw_est);
        block_pose_.pose.orientation = xformUtils.convertPlanarPsi2Quaternion(0); //not true, but legal

        cv::namedWindow(OPENCV_WINDOW);
}

~ImageConverter() {
        cv::destroyWindow(OPENCV_WINDOW);
}

//image comes in as a ROS message, but gets converted to an OpenCV type
void imageCb(const sensor_msgs::ImageConstPtr& msg);

}; //end of class definition

void ImageConverter::imageCb(const sensor_msgs::ImageConstPtr& msg){
        cv_bridge::CvImagePtr cv_ptr; //OpenCV data type
        try {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
        }

        std::vector<double> xVals;
        std::vector<double> yVals;
        // look for red pixels; turn all other pixels black, and turn red pixels white
        int npix = 0; //count the red pixels
        int isum = 0; //accumulate the column values of red pixels
        int jsum = 0; //accumulate the row values of red pixels
        int redval, blueval, greenval, testval;
        cv::Vec3b rgbpix; // OpenCV representation of an RGB pixel
        //comb through all pixels (j,i)= (row,col)
        for (int i = 0; i < cv_ptr->image.cols; i++) {
                for (int j = 0; j < cv_ptr->image.rows; j++) {
                        rgbpix = cv_ptr->image.at<cv::Vec3b>(j, i); //extract an RGB pixel
                        //examine intensity of R, G and B components (0 to 255)
                        redval = rgbpix[2] + 1; //add 1, to avoid divide by zero
                        blueval = rgbpix[0] + 1;
                        greenval = rgbpix[1] + 1;
                        //look for red values that are large compared to blue+green
                        testval = redval / (blueval + greenval);
                        //if red (enough), paint this white:
                        if (testval > g_redratio) {
                                cv_ptr->image.at<cv::Vec3b>(j, i)[0] = 255;
                                cv_ptr->image.at<cv::Vec3b>(j, i)[1] = 255;
                                cv_ptr->image.at<cv::Vec3b>(j, i)[2] = 255;
                                npix++; //note that found another red pixel
                                isum += i; //accumulate row and col index vals
                                jsum += j;

                                xVals.push_back(i-320);
                                yVals.push_back(240-j);
                        } else { //else paint it black
                                cv_ptr->image.at<cv::Vec3b>(j, i)[0] = 0;
                                cv_ptr->image.at<cv::Vec3b>(j, i)[1] = 0;
                                cv_ptr->image.at<cv::Vec3b>(j, i)[2] = 0;
                        }
                }
        }
        //cout << "npix: " << npix << endl;
        //paint in a blue square at the centroid:
        int half_box = 5; // choose size of box to paint
        int i_centroid, j_centroid;
        double x_centroid, y_centroid;
        if (npix > 0) {
                i_centroid = isum / npix; // average value of u component of red pixels
                j_centroid = jsum / npix; // avg v component
                x_centroid = ((double) isum)/((double) npix); //floating-pt version
                y_centroid = ((double) jsum)/((double) npix);
                //ROS_INFO("u_avg: %f; v_avg: %f",x_centroid,y_centroid);
                //cout << "i_avg: " << i_centroid << endl; //if,j centroid of red pixels
                //cout << "j_avg: " << j_centroid << endl;
                for (int i_box = i_centroid - half_box; i_box <= i_centroid + half_box; i_box++) {
                        for (int j_box = j_centroid - half_box; j_box <= j_centroid + half_box; j_box++) {
                                //make sure indices fit within the image
                                if ((i_box >= 0)&&(j_box >= 0)&&(i_box < cv_ptr->image.cols)&&(j_box < cv_ptr->image.rows)) {
                                        cv_ptr->image.at<cv::Vec3b>(j_box, i_box)[0] = 255; //(255,0,0) is pure blue
                                        cv_ptr->image.at<cv::Vec3b>(j_box, i_box)[1] = 0;
                                        cv_ptr->image.at<cv::Vec3b>(j_box, i_box)[2] = 0;
                                }
                        }
                }

        }
        // Update GUI Window; this will display processed images on the open-cv viewer.
        cv::imshow(OPENCV_WINDOW, cv_ptr->image);
        cv::waitKey(3); //need waitKey call to update OpenCV image window

        // Also, publish the processed image as a ROS message on a ROS topic
        // can view this stream in ROS with:
        //rosrun image_view image_view image:=/image_converter/output_video
        image_pub_.publish(cv_ptr->toImageMsg());


    

        //Normalizing pixels so that center is 0,0 and up and to the right is purely positive
        double x_normal_pix = x_centroid - 320;
        double y_normal_pix = 240 - y_centroid;

        //Achieved this variable through gazebo testing. Represents how many meters one pixel is in the robot's world for this specific simulation.
        //Testing was done by moving block a certain amount and then seeing how many pixels that made the centroid move
        double metersPerPixel = ((0.1/33.7178) + (0.05/16.9418) + (0.25/84.5498))/3;

        //Coords in meters of the block underneath the camera
        double x_camera = x_normal_pix*metersPerPixel;
        double y_camera = y_normal_pix*metersPerPixel;

        //Eigen Variables for finding position of object in reference to robot (not camera which we have already)
        Eigen::MatrixXd T_oc(4,4); //Transformation of object in reference to camera
        Eigen::MatrixXd T_cr(4,4); //Transformation of camera in reference to robot
        Eigen::MatrixXd T_or(4,4); //Transformation of object in reference to robot (WANT THIS)

        //Object in reference to camera using yaw rotation matrix with angle obtained from vision LATER
        double angleOc = atan(getSlope(xVals,yVals));
        //ROS_INFO("ANGLE (DEG): %f", angleOc*180/3.14515);

        T_oc.row(0) << cos(angleOc), -sin(angleOc), 0, x_camera;
        T_oc.row(1) << sin(angleOc),  cos(angleOc), 0, y_camera;
        T_oc.row(2) << 0,             0,            1, -1.75;
        T_oc.row(3) << 0,             0,            0, 1;

        //Camera in reference to the robot (obtained through Gazebo) using yaw, pitch, and roll rotation matrices (obtained from gazebo angles)
        double angleYaw   = -0.200; //Moved the box purely in x-direction, detected a rise in camera coordinates, detected slope of movement in the camera, find angle of slope, this was 0.2
        double anglePitch =  0;     //No Pitch Rotation
        double angleRoll  =  0;     //No roll Rotation
        Eigen::MatrixXd rCrYaw  (3,3);
        Eigen::MatrixXd rCrPitch(3,3);
        Eigen::MatrixXd rCrRoll (3,3);
        Eigen::MatrixXd R_cr    (3,3);

        rCrYaw.row(0) <<  cos(angleYaw), -sin(angleYaw), 0;
        rCrYaw.row(1) <<  sin(angleYaw),  cos(angleYaw), 0;
        rCrYaw.row(2) <<  0,              0,             1;

        rCrPitch.row(0) << cos(anglePitch),  0, sin(anglePitch);
        rCrPitch.row(1) << 0,                1, 0;
        rCrPitch.row(2) << -sin(anglePitch), 0, cos(anglePitch);

        rCrRoll.row(0) << 1, 0,               0,
                rCrRoll.row(1) << 0, cos(angleRoll), -sin(angleRoll),
                rCrRoll.row(2) << 0, sin(angleRoll),  cos(angleRoll);

        R_cr = rCrYaw*rCrPitch*rCrRoll; //Final rotation matrix for camera in reference to robot
        //Rightmost column (x,y of camera in robot coordinates) obtained by moving block in gazebo until the camera read 320,240 pixels (center of camera frame)
        T_cr.row(0) << R_cr(0,0), R_cr(0,1), R_cr(0,2), 0.545;
        T_cr.row(1) << R_cr(1,0), R_cr(1,1), R_cr(1,2), 0.319;
        T_cr.row(2) << R_cr(2,0), R_cr(2,1), R_cr(2,2), 1.75;
        T_cr.row(3) << 0,         0,         0,         1;

        //Finding object position in reference to robot and extracting x and y values
        T_or = T_cr*T_oc;
        double x_robot = T_or(0,3);
        double y_robot = T_or(1,3);

        //******************************************************

        block_pose_.pose.position.x = x_robot;
        block_pose_.pose.position.y = y_robot;
        double theta = -acos(T_or(0,0)); //Since its just a yaw rotation, element 0,0 is cos(theta)
        ROS_INFO("ROBO ANGLE (DEG): %f", theta*180/3.14515);

        // need camera info to fill in x,y,and orientation x,y,z,w
        //geometry_msgs::Quaternion quat_est
        //quat_est = xformUtils.convertPlanarPsi2Quaternion(yaw_est);
        block_pose_.pose.orientation = xformUtils.convertPlanarPsi2Quaternion(theta); //not true, but legal
        block_pose_publisher_.publish(block_pose_);
}

int main(int argc, char** argv) {
        ros::init(argc, argv, "red_pixel_finder");
        ros::NodeHandle n; //
        ImageConverter ic(n); // instantiate object of class ImageConverter
        //cout << "enter red ratio threshold: (e.g. 10) ";
        //cin >> g_redratio;
        g_redratio= 10; //choose a threshold to define what is "red" enough
        ros::Duration timer(0.1);
        double x, y, z;
        while (ros::ok()) {
                ros::spinOnce();
                timer.sleep();
        }
        return 0;
}
