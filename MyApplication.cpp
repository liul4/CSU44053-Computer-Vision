#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
    const char* file_location = "C:/Users/14871/Documents/Vision/assignment/assignment/balls/";
    const char* image_files[] = {
        "Ball1.jpg",
        "Ball2.jpg",
        "Ball3.jpg",
        "Ball4.jpg",
        "Ball5.jpg",
        "Ball6.jpg",
        "Ball7.jpg",
        "Ball8.jpg",
        "Ball9.jpg",
        "Ball10.jpg"
    };

    int count_images = sizeof(image_files) / sizeof(image_files[0]);
    for (int i = 0; i < count_images; i++)
    {
        // Load images
        string filename(file_location);
        filename.append(image_files[i]);
        Mat image = imread(filename, -1);
        if (image.empty())
        {
            cout << "Could not open " << filename << endl;
            return -1;
        }
        //convert to grey scale image
        Mat processed_image;
        cvtColor(image, processed_image, COLOR_BGR2GRAY);
        //smoothing the image
        medianBlur(processed_image, processed_image, 5);
        //use hough circle transform to do region detection
        vector<Vec3f> circles;
        HoughCircles(processed_image, circles, HOUGH_GRADIENT, 1, 220, 100, 30, 20, 70);
        //draw detected circles
        for (const auto& circle : circles) {
            cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
            int radius = cvRound(circle[2]);
            double area = CV_PI * radius * radius;
            if (center.y >= 80) {
                cv::circle(image, center, radius, cv::Scalar(10, 255, 10), 2);
            }
            
        }


        // display detected table tennis balls
        cv::imshow("detected table tennis balls", image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return 0;
}