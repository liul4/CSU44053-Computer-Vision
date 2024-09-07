#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

// part 1 ground truth
const string GROUND_TRUTH_BALL[][4] = {
    {"0", "564.5", "311.5", "83"},
    {"1", "432", "456.5", "136.5"},
    {"2", "414.5", "407", "95.5"},
    {"3", "363", "380", "113"},
    {"4", "146.5", "472", "85.5"},
    {"4", "440.5", "362", "84.5"},
    {"4", "711.5", "481.5", "84"},
    {"5", "383.5", "330.5", "96"},
    {"6", "529", "282", "70"},
    {"7", "523.5", "458.5", "61"},
    {"8", "531.5", "403.5", "59"},
    {"9", "494.5", "215.5", "41"}
};
// part 2 ground truth
const string GROUND_TRUTH_TABLE[][9] = {
    {"0", "426", "158",  "1121", "2798", "2889", "1115", "3768", "2265"},
    {"1", "163", "1642", "1234", "734", "2641", "2886", "3844", "539"},
    {"2", "466", "2640", "1068", "547", "2911", "2137", "3805", "1123"},
    {"3", "345", "3004", "1342", "768", "2645", "833", "3004", "2966"},
    {"4", "223", "2739", "1224", "490", "2562", "555", "3742", "2762"}
};
// part 3 ground truth
// Frame n(row, column) Bounce on Table
const string GROUND_TRUTH_VIDEO[][4]{
{"16", "367", "844", "Hit by Player"},
{"19", "431", "696", "Bounce on Table"},
{"35", "398", "327", "Bounce on Table"},
{"41", "334", "227", "Hit by Player"},
{"56", "390", "669", "Bounce on Table"},
{"64", "315", "774", "Hit by Player"},
{"78", "404", "418", "Bounce on Table"},
{"89", "346", "272", "Hit by Player"},
{"102", "408", "598", "Bounce on Table"},
{"111", "349", "700", "Hit by Player"},
{"129", "457", "211", "Bounce on Table"},
{"139", "365", "69", "Hit by Player"},
{"155", "401", "614", "Bounce on Table"},
{"164", "307", "725", "Hit by Player"},
{"184", "423", "145", "Bounce on Table"},
{"220", "378", "685", "Bounce on Table"},
{"229", "261", "799", "Hit by Player"},
{"253", "439", "193", "Bounce on Table"},
{"261", "287", "105", "Hit by Player"},
{"278", "419", "731", "Bounce on Table"},
{"286", "301", "853", "Hit by Player"},
{"400", "311", "133", "Hit by Player"},
{"404", "391", "249", "Bounce on Table"},
{"420", "399", "515", "Bounce on Table"},
{"431", "384", "674", "Hit by Player"},
{"446", "433", "242", "Bounce on Table"},
{"451", "377", "146", "Hit by Player"},
{"468", "435", "719", "Bounce on Table"},
{"475", "342", "848", "Hit by Player"},
{"494", "394", "215", "Bounce on Table"},
{"498", "321", "173", "Hit by Player"},
{"514", "408", "622", "Bounce on Table"},
{"523", "327", "745", "Hit by Player"},
{"538", "417", "271", "Bounce on Table"},
{"543", "346", "182", "Hit by Player"},
{"560", "410", "642", "Bounce on Table"},
{"568", "318", "752", "Hit by Player"},
{"584", "424", "262", "Bounce on Table"},
{"589", "349", "176", "Bounce on Table"},
{"604", "448", "747", "Bounce on Table"},
{"610", "365", "871", "Bounce on Table"},
{"629", "393", "251", "Bounce on Table"},
{"633", "323", "206", "Bounce on Table"},
{"647", "426", "657", "Bounce on Table"},
{"655", "372", "842", "Bounce on Table"},
{"674", "385", "158", "Bounce on Table"},
{"676", "369", "151", "Hit by Player"},
{"687", "380", "473", "Bounce on Table"},
{"697", "380", "451", "Bounce on Table"},
{"706", "377", "442", "Bounce on Table"},
{"715", "377", "435", "Bounce on Table"},
{"723", "375", "431", "Bounce on Table"},
{"730", "376", "428", "Bounce on Table"},
{"737", "375", "429", "Hit by Player"},
{"738", "372", "429", "Bounce on Table"},
{"740", "375", "422", "Hit by Player"},
{"742", "375", "417", "Bounce on Table"},
{"796", "374", "57", "Hit by Player"},
{"799", "433", "211", "Bounce on Table"},
{"814", "403", "610", "Bounce on Table"},
{"820", "344", "735", "Hit by Player"},
{"842", "414", "148", "Bounce on Table"},
{"875", "409", "708", "Bounce on Table"},
{"883", "270", "864", "Hit by Player"},
{"910", "438", "127", "Bounce on Table"},
{"951", "406", "612", "Bounce on Table"},
{"962", "267", "759", "Hit by Player"},
{"979", "421", "256", "Bounce on Table"},
{"986", "312", "141", "Hit by Player"},
{"1002", "401", "655", "Bounce on Table"},
{"1010", "302", "779", "Hit by Player"},
{"1025", "406", "233", "Bounce on Table"},
{"1030", "329", "156", "Hit by Player"},
{"1047", "389", "695", "Bounce on Table"},
{"1053", "306", "785", "Hit by Player"},
{"1073", "409", "147", "Bounce on Table"},
{"1077", "335", "70", "Hit by Player"},
{"1095", "397", "600", "Bounce on Table"},
{"1106", "301", "732", "Hit by Player"},
{"1122", "418", "265", "Bounce on Table"},
{"1128", "328", "180", "Hit by Player"},
{"1142", "397", "595", "Bounce on Table"},
{"1151", "315", "712", "Hit by Player"},
{"1168", "448", "208", "Bounce on Table"},
{"1174", "357", "104", "Hit by Player"},
{"1188", "392", "639", "Bounce on Table"},
{"1194", "309", "738", "Hit by Player"},
{"1210", "379", "208", "Bounce on Table"},
{"1213", "335", "148", "Hit by Player"},
{"1231", "447", "743", "Bounce on Table"},
{"1237", "363", "863", "Hit by Player"},
{"1446", "375", "898", "Hit by Player"},
{"1450", "448", "756", "Bounce on Table"},
{"1464", "421", "506", "Bounce on Table"},
{"1466", "393", "483", "Hit the Net"},
{"1474", "377", "481", "Hit the Net"},
{"1478", "407", "466", "Bounce on Table"},
{"1487", "401", "408", "Bounce on Table"},
{"1495", "390", "364", "Bounce on Table"},
{"1496", "393", "361", "Bounce on Table"},
{"1504", "388", "320", "Bounce on Table"},
{"1511", "383", "288", "Bounce on Table"},
{"1518", "379", "260", "Bounce on Table"},
{"1570", "327", "143", "Hit by Player"},
{"1575", "403", "323", "Bounce on Table"},
{"1589", "409", "691", "Bounce on Table"},
{"1595", "342", "781", "Hit by Player"},
{"1614", "418", "306", "Bounce on Table"},
{"1620", "335", "230", "Hit by Player"},
{"1636", "398", "725", "Bounce on Table"},
{"1642", "308", "812", "Hit by Player"},
{"1683", "388", "655", "Bounce on Table"},
{"1691", "292", "739", "Bounce on Table"},
{"1709", "445", "254", "Bounce on Table"},
{"1719", "334", "97)", "Bounce on Table"},
{"1733", "390", "604", "Bounce on Table"},
{"1742", "289", "715", "Hit by Player"},
{"1761", "447", "272", "Bounce on Table"},
{"1773", "362", "96)", "Hit by Player"},
{"1787", "405", "644", "Bounce on Table"},
{"1794", "308", "764", "Hit by Player"},
{"1816", "370", "177", "Bounce on Table"},
{"1825", "259", "105", "Hit by Player"},
{"1844", "406", "526", "Bounce on Table"},
{"1858", "305", "736", "Hit by Player"},
{"1876", "380", "269", "Bounce on Table"},
{"1882", "292", "210", "Hit by Player"},
{"1898", "423", "700", "Bounce on Table"},
{"1905", "342", "872", "Hit by Player"},
{"1923", "409", "322", "Bounce on Table"},
{"1932", "310", "194", "Hit by Player"},
{"1948", "413", "739", "Bounce on Table"},
{"1953", "323", "814", "Hit by Player"},
{"1972", "374", "250", "Bounce on Table"},
{"1977", "301", "187", "Hit by Player"},
{"1992", "400", "726", "Bounce on Table"},
{"1997", "331", "844", "Hit by Player"},
{"2021", "419", "154", "Bounce on Table"},
{"2050", "397", "523", "Bounce on Table"},
{"2060", "364", "728", "Hit by Player"},
{"2084", "460", "62", "Bounce on Table"}
};
// Calcultae aspect ratio to be used in part one
double calculateAspectRatio(const vector<Point>& contour) {
    Rect boundingBox = boundingRect(contour);
    // the ratio is calculate by width divide by height
    double aspectRatio = static_cast<double>(boundingBox.width) / boundingBox.height;
    return aspectRatio;
}
// Calcultae circularity to be used in part one
double calculateCircularity(const vector<Point>& contour) {
    double area = contourArea(contour);
    double perimeter = arcLength(contour, true);
    // calculate circularity using the formula
    double circularity = 4 * CV_PI * area / (perimeter * perimeter);
    return circularity;
}
// sort the contours based on x-coordinates of their centroids
bool compareContourX(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
    // Calculate the centroids of the contours
    cv::Moments moments1 = cv::moments(contour1);
    cv::Moments moments2 = cv::moments(contour2);
    // calculate x-coordinates of centroids
    double cx1 = moments1.m10 / moments1.m00;
    double cx2 = moments2.m10 / moments2.m00;
    // Compare contours based on x-coordinates of their centroids
    return cx1 < cx2;
}
// part one final version
int part_one() {
    const char* file_location = "balls/";
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
    int correct_count = 0;
    int incorrect_count = 0;
    int current_line = 0;
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
        // apply mean shift segmentation
        Mat clustered_image = image.clone();
        pyrMeanShiftFiltering(image, clustered_image, 40, 40, 2);
        // Convert the clustered image to the HSV color space
        Mat hsvImage;
        cvtColor(clustered_image, hsvImage, COLOR_BGR2HSV);
        // Define lower and upper HSV thresholds for white and orange balls
        Scalar lowerWhite(0, 0, 220);  // Lower HSV values for white color
        Scalar upperWhite(50, 50, 255); // Upper HSV values for white color
        Scalar lowerBound(0, 0, 180);  // Lower HSV values for ball color
        Scalar upperBound(180, 255, 255); // Upper HSV values for ball color
        // Create binary masks for white and orange balls
        Mat maskWhite, maskBalls;
        inRange(hsvImage, lowerWhite, upperWhite, maskWhite);
        inRange(hsvImage, lowerBound, upperBound, maskBalls);
        // Find contours in the binary masks
        vector<vector<Point>> contoursWhite, contoursBalls;
        findContours(maskWhite, contoursWhite, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        findContours(maskBalls, contoursBalls, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        // Define min and max area of table tennis balls
        float newWidth = image.cols / 10;
        float minWidth = image.cols / 50;
        float maxBallArea = 0.5 * 3.14 * newWidth * newWidth;
        float minBallArea = 0.5 * 3.14 * minWidth * minWidth;
        std::sort(contoursWhite.begin(), contoursWhite.end(), compareContourX);
        std::sort(contoursBalls.begin(), contoursBalls.end(), compareContourX);
        // Loop through white and orange ball contours and draw circles around them
        for (const auto& contour : contoursBalls) {
            // get center and radius of circles
            Point2f center;
            float radius;
            minEnclosingCircle(contour, center, radius);
            double aspectRatio = calculateAspectRatio(contour); // use calculateAspectRatio() function to calculate aspect ratio
            double circularity = calculateCircularity(contour); // use calculateCircularity() function to calculate circularity
            // circle detected table tennis balls
            if (0.5 * 3.14 * radius * radius < maxBallArea && 0.5 * 3.14 * radius * radius>minBallArea && aspectRatio > 0.5 && circularity > 0.5) {
                circle(image, center, static_cast<int>(radius), Scalar(0, 255, 0), 2); // Circle in white color
                // print ball info
                std::cout << "Ball location: " << center << std::endl;
                std::cout << "Ball diameter: " << radius * 2 << std::endl;
                // load ground truth
                double col = std::stod(GROUND_TRUTH_BALL[current_line][1]);
                double row = std::stod(GROUND_TRUTH_BALL[current_line][2]);
                double diameter = std::stod(GROUND_TRUTH_BALL[current_line][3]);
                // count correctness
                if (center.x<col + 6 && center.x>col - 6 && center.y > row - 2 && center.y < row + 2 && radius * 2 < diameter + 6 && radius * 2 > diameter - 6) {
                    correct_count++;
                }
                else {
                    incorrect_count++;
                }
                current_line++;
            }
        }
        // Loop through white ball contours and draw circles around them
        // Deal with special cases where part of the background has similar color as the table tennis ball
        std::vector<float> ball_x;
        std::vector<float> ball_y;
        std::vector<float> ball_diameter;
        for (const auto& contour : contoursWhite) {
            // get center and radius of circles
            Point2f center;
            float radius;
            minEnclosingCircle(contour, center, radius);
            double aspectRatio = calculateAspectRatio(contour); // use calculateAspectRatio() function to calculate aspect ratio
            double circularity = calculateCircularity(contour); // use calculateCircularity() function to calculate circularity
            // circle detected table tennis ball
            if (0.5 * 3.14 * radius * radius < maxBallArea && 0.5 * 3.14 * radius * radius>minBallArea && aspectRatio > 0.6 && aspectRatio < 0.65 && circularity > 0.6 && circularity < 0.7) {
                circle(image, center, static_cast<int>(radius), Scalar(0, 255, 0), 2);
                // print ball info
                std::cout << "Ball location: " << center << std::endl;
                std::cout << "Ball diameter: " << radius * 2 << std::endl;
                // load ground truth
                double col = std::stod(GROUND_TRUTH_BALL[current_line][1]);
                double row = std::stod(GROUND_TRUTH_BALL[current_line][2]);
                double diameter = std::stod(GROUND_TRUTH_BALL[current_line][3]);
                // count correctness
                if (center.x<col + 6 && center.x>col - 6 && center.y > row - 2 && center.y < row + 2 && radius * 2 < diameter + 6 && radius * 2 > diameter - 6) {
                    correct_count++;
                }
                else {
                    incorrect_count++;
                }
                current_line++;
            }
        }
        imshow("Detected Table Tennis Balls", image);
        //std::string name = "part_one_clustered_image_" + std::to_string(i) + ".jpg";
        //cv::imwrite(name, clustered_image);
        // print correctness
        std::cout << "Correct Count: " << correct_count << std::endl;
        std::cout << "Incorrect Count: " << incorrect_count << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    // print performance
    std::cout << "Correct Count: " << correct_count << std::endl;
    std::cout << "Incorrect Count: " << incorrect_count << std::endl;
    float accuracy = float(correct_count) / float(correct_count + incorrect_count);
    std::cout << "Accuracy: " << accuracy << std::endl;
    return 0;
}

// function to merge close lines
void mergeLines(std::vector<cv::Vec2f>& lines, double distanceThreshold) {
    for (size_t i = 0; i < lines.size(); ++i) {
        float rho1 = lines[i][0], theta1 = lines[i][1];
        cv::Point pt1_1, pt1_2;
        double a1 = cos(theta1), b1 = sin(theta1);
        pt1_1.x = cvRound(rho1 * a1 + 1000 * (-b1));
        pt1_1.y = cvRound(rho1 * b1 + 1000 * (a1));
        pt1_2.x = cvRound(rho1 * a1 - 1000 * (-b1));
        pt1_2.y = cvRound(rho1 * b1 - 1000 * (a1));

        for (size_t j = i + 1; j < lines.size(); ++j) {
            float rho2 = lines[j][0], theta2 = lines[j][1];
            cv::Point pt2_1, pt2_2;
            double a2 = cos(theta2), b2 = sin(theta2);
            pt2_1.x = cvRound(rho2 * a2 + 1000 * (-b2));
            pt2_1.y = cvRound(rho2 * b2 + 1000 * (a2));
            pt2_2.x = cvRound(rho2 * a2 - 1000 * (-b2));
            pt2_2.y = cvRound(rho2 * b2 - 1000 * (a2));

            // Calculate Euclidean distance between endpoints of the lines
            double distance = cv::norm(pt1_1 - pt2_1) + cv::norm(pt1_2 - pt2_2);

            // If the distance is below the threshold, merge the lines
            if (distance < distanceThreshold) {
                // Update the parameters of the first line to merge them
                lines[i][0] = (rho1 + rho2) / 2;
                lines[i][1] = (theta1 + theta2) / 2;

                // Remove the merged line
                lines.erase(lines.begin() + j);
                --j;  // Adjust the index
            }
        }
    }
}
// part two
int part_two() {
    const char* file_location = "tables/";
    const char* image_files[] = {
        "Table1.jpg",
        "Table2.jpg",
        "Table3.jpg",
        "Table4.jpg",
        "Table5.jpg"
    };
    int correct_count = 0;
    int incorrect_count = 0;
    int count_images = sizeof(image_files) / sizeof(image_files[0]);
    int current_line = 0;
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
        // Resize the image to the specified width and height
        float newWidth = image.cols / 6;
        float newHeight = image.rows / 6;
        resize(image, image, Size(newWidth, newHeight));
        Mat image_cpy = image.clone();
        // Convert the clustered image to the HSV color space
        Mat hsvImage;
        cvtColor(image, hsvImage, COLOR_BGR2HSV);
        // Define lower and upper HSV thresholds for table
        Scalar lowerBound(80, 10, 10);  // Lower HSV values for table
        Scalar upperBound(180, 50, 255);// upper HSV values for table
        Mat blueMask;
        inRange(hsvImage, lowerBound, upperBound, blueMask);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::dilate(blueMask, blueMask, kernel, cv::Point(-1, -1), 1);
        // Canny line detection
        Mat canny_edge_image;
        Canny(blueMask, canny_edge_image, 50, 150);
        std::vector<cv::Vec2f> lines;
        std::vector<cv::Vec2f> filter_lines;
        // hough line edge detection
        cv::HoughLines(canny_edge_image, lines, 1, CV_PI / 180, 100);
        // merge close lines
        mergeLines(lines, 80);
        // draw detected lines
        //for (size_t i = 0; i < lines.size(); ++i) {
        //    float rho = lines[i][0], theta = lines[i][1];
        //    cv::Point pt1, pt2;
        //    double a = cos(theta), b = sin(theta);
        //    double x0 = a * rho, y0 = b * rho;
        //    pt1.x = cvRound(x0 + 1000 * (-b));
        //    pt1.y = cvRound(y0 + 1000 * (a));
        //    pt2.x = cvRound(x0 - 1000 * (-b));
        //    pt2.y = cvRound(y0 - 1000 * (a));
        //    double slope = (pt2.y - pt1.y) / (double)(pt2.x - pt1.x);
        //    std::cout << "Angle: " << theta << std::endl;
        //    std::cout << "Slope: " << slope << std::endl;
        //    std::cout << "Rho: " << rho << std::endl;
        //    cv::line(image, pt1, pt2, cv::Scalar(255, 100, 100), 2, cv::LINE_AA); // Draws red lines on the image
        //    
        //}
        
        std::vector<cv::Point2f> intersections;  // Vector to store intersection points
        // calculate intersections of lines
        for (size_t i = 0; i < lines.size(); ++i) {
            for (size_t j = i + 1; j < lines.size(); ++j) {
                cv::Point2f intersection;
                float rho1 = lines[i][0], theta1 = lines[i][1];
                float rho2 = lines[j][0], theta2 = lines[j][1];
                double a1 = cos(theta1), b1 = sin(theta1);
                double a2 = cos(theta2), b2 = sin(theta2);
                double det = a1 * b2 - a2 * b1;
                if (det != 0) {  // Lines are not parallel
                    intersection.x = (b2 * rho1 - b1 * rho2) / det;
                    intersection.y = (-a2 * rho1 + a1 * rho2) / det;
                    // only store points within the screen
                    if (intersection.x >= 0 && intersection.x <= newWidth && intersection.y >= 0 && intersection.y <= newHeight) {
                        intersections.push_back(intersection);
                        //cv::circle(image, intersection, 2, cv::Scalar(0, 0, 255), -1);
                    }
                }
            }
        }

        // there should be nine detected intersection points in each picture
        // sort the intersaction according to value of x-axos
        std::sort(intersections.begin(), intersections.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
            });
        // the middle point
        //cv::circle(image, intersections[4], 5, cv::Scalar(0, 255, 0), -1); // Draw green circles at corner
        
        // filter table lines and middle lines
        std::vector<cv::Vec2f> table_lines;
        std::vector<cv::Vec2f> middle_lines;
        for (size_t i = 0; i < lines.size(); ++i) {
            float rho = lines[i][0], theta = lines[i][1];
            cv::Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));
            double slope = (pt2.y - pt1.y) / (double)(pt2.x - pt1.x);
            // if the middle intersection point is on the line, the line is filtered as middlle lines
            if (std::fabs(intersections[4].y - slope * intersections[4].x - rho / b) < 10) {
                cv::line(image, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                middle_lines.push_back(lines[i]);
            }
            // if the middle intersection point is not on the line, the line is filtered as edge lines
            else {
                cv::line(image, pt1, pt2, cv::Scalar(250, 20, 100), 2, cv::LINE_AA);
                table_lines.push_back(lines[i]);
            }
        }
        // remove middle point
        intersections.erase(intersections.begin() + 4);
        // find corners on the table and points on the net
        std::vector<cv::Point2f> corners;
        for (size_t i = 0; i < table_lines.size(); ++i) {
            for (size_t j = i + 1; j < table_lines.size(); ++j) {
                cv::Point2f intersection;
                float rho1 = table_lines[i][0], theta1 = table_lines[i][1];
                float rho2 = table_lines[j][0], theta2 = table_lines[j][1];
                double a1 = cos(theta1), b1 = sin(theta1);
                double a2 = cos(theta2), b2 = sin(theta2);
                double det = a1 * b2 - a2 * b1;
                if (det != 0) {  // Lines are not parallel
                    intersection.x = (b2 * rho1 - b1 * rho2) / det;
                    intersection.y = (-a2 * rho1 + a1 * rho2) / det;
                    // only store points within the screen
                    if (intersection.x >= 0 && intersection.x <= newWidth && intersection.y >= 0 && intersection.y <= newHeight) {
                        corners.push_back(intersection);
                    }
                }
            }
        }
        // sort the corners according to x coordinates
        std::sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
            });
        // Draw intersection points on the image; track correctness
        int num = 1;
        for (const auto& point : corners) {
            double col = std::stod(GROUND_TRUTH_TABLE[current_line][num]) / 6;
            double row = std::stod(GROUND_TRUTH_TABLE[current_line][num + 1]) / 6;
            cv::circle(image, point, 3, cv::Scalar(0, 255, 0), -1); // Draw green circles at corner
            std::cout << "corner: " << point << std::endl;
            std::cout << "ground truth: " << col << ", " << row << std::endl;
            //std::cout << "ground truth: " << GROUND_TRUTH_TABLE[current_line][num] << ", " << GROUND_TRUTH_TABLE[current_line][num+1] << std::endl;
            if (std::abs(point.x - col) <= float(150) && std::abs(point.y - row) <= float(150)) {
                //if (point.x*6 > float(col - 50) && point.x*6<float(col + 50) && point.y * 6 > float(row - 50) && point.y*6 > float(row + 50)) {
                correct_count++;
            }
            else {
                incorrect_count++;
            }
            std::cout << "absolute value: " << std::abs(point.x - col) << ", " << std::abs(point.y - row) << std::endl;
            num = num + 2;
        }
        imshow("detected table tennis table", image);
        //cv::imwrite("test.jpg", image);
        cv::waitKey(0);
        current_line++;
        // sort the corners according to x coordinates
        std::sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y;
            });
        // Define the width and height for the transformed image as float values
        double dist1 = cv::norm(corners[0].x - corners[1].x) + cv::norm(corners[0].y - corners[1].y);
        double dist2 = cv::norm(corners[1].x - corners[2].x) + cv::norm(corners[1].y - corners[2].y);
        float desiredWidth = 800.0f;  // Replace with your desired width
        float desiredHeight = 600.0f; // Replace with your desired height

        // Define the destination corners using the specified width and height as floats
        std::vector<cv::Point2f> dstPoints = { {0.0f, 0.0f}, {float(dist1), 0.0f}, {float(dist1), float(dist2)}, {0.0f, float(dist2)} };
        //std::vector<cv::Point2f> dstPoints = { {0.0f, 0.0f}, {desiredWidth, 0.0f}, {desiredWidth, desiredHeight}, {0.0f, desiredHeight} };
        cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(corners, dstPoints);

        // Apply perspective transformation
        cv::Mat transformedImage;
        cv::warpPerspective(image_cpy, transformedImage, perspectiveMatrix, Size(dist1, dist2), cv::INTER_LINEAR);

        // display detected 
        imshow("detected table tennis table", transformedImage);
        std::cout << "Correct Count: " << correct_count << std::endl;
        std::cout << "Incorrect Count: " << incorrect_count << std::endl;
        /*std::string name = "part_two_result_image_" + std::to_string(i) + ".jpg";
        cv::imwrite(name, transformedImage);*/
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    // print overall correctness
    std::cout << "Correct Count: " << correct_count << std::endl;
    std::cout << "Incorrect Count: " << incorrect_count << std::endl;
    float accuracy = float(correct_count) / float(correct_count + incorrect_count);
    std::cout << "Accuracy: " << accuracy << std::endl;
    return 0;
}

// Part 3 
int frame_track[1000] = {};
float col_x[1000] = {};
float row_y[1000] = {};
string type[1000] = {};
int track_count = 0;
// function to process the video
void process_video(VideoCapture& video, int starting_frame_number)
{
    double totalFrames = video.get(cv::CAP_PROP_FRAME_COUNT);  // calculate total number of frames
    int current_line = 0;
    int ground_truth_frame = std::stoi(GROUND_TRUTH_VIDEO[current_line][0]);
    bool left = true;
    bool up = false;
    int xpos1 = 1000, ypos1 = 1000;
    int xpos2 = 0, ypos2 = 0;
    if (video.isOpened()) {
        video.set(cv::CAP_PROP_POS_FRAMES, starting_frame_number);
        Mat current_frame, hls_image;
        video >> current_frame;
        int frame_number = starting_frame_number;

        // locate the table
        Mat hsvImage1;
        cvtColor(current_frame, hsvImage1, COLOR_BGR2HSV);
        // Define lower and upper HSV thresholds for table
        Scalar lowerTable(50, 10, 10);  
        Scalar upperTable(180, 50, 255);
        Mat blueMask;
        inRange(hsvImage1, lowerTable, upperTable, blueMask);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::dilate(blueMask, blueMask, kernel, cv::Point(-1, -1), 1);
        // Canny line detection
        Mat canny_edge_image;
        Canny(blueMask, canny_edge_image, 160, 150);
        std::vector<cv::Vec2f> lines;
        std::vector<cv::Vec2f> filter_lines;
        // hough line edge detection
        cv::HoughLines(canny_edge_image, lines, 1, CV_PI / 180, 100);
        mergeLines(lines, 200);
        // draw detected lines
        for (size_t i = 0; i < lines.size(); ++i) {
            float rho = lines[i][0], theta = lines[i][1];
            cv::Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));
            double slope = (pt2.y - pt1.y) / (double)(pt2.x - pt1.x);
            cv::line(current_frame, pt1, pt2, cv::Scalar(0, 255, 100), 2, cv::LINE_AA);
        }
        std::vector<cv::Point2f> intersections;  // Vector to store intersection points
        // find intersections
        for (size_t i = 0; i < lines.size(); ++i) {
            for (size_t j = i + 1; j < lines.size(); ++j) {
                cv::Point2f intersection;
                float rho1 = lines[i][0], theta1 = lines[i][1];
                float rho2 = lines[j][0], theta2 = lines[j][1];
                double a1 = cos(theta1), b1 = sin(theta1);
                double a2 = cos(theta2), b2 = sin(theta2);
                double det = a1 * b2 - a2 * b1;
                if (det != 0) {  // Lines are not parallel
                    intersection.x = (b2 * rho1 - b1 * rho2) / det;
                    intersection.y = (-a2 * rho1 + a1 * rho2) / det;
                    // only store points within the screen
                    if (intersection.x >= 0 && intersection.x <= current_frame.cols && intersection.y >= 0 && intersection.y <= current_frame.rows) {
                        intersections.push_back(intersection);
                    }
                }
            }
        }
        // sort the intersaction according to value of x-axos
        std::sort(intersections.begin(), intersections.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
            });
        // remove middle point
        intersections.erase(intersections.begin() + 2);
        for (const auto& point : intersections) {
            cv::circle(current_frame, point, 3, cv::Scalar(0, 0, 255), -1); // Draw green circles at corner
        }
        // calculate location of net
        int net = (intersections[1].x + intersections[2].x) / 2;
        int net_left = net - 40;
        int net_right = net + 70;
        cv::circle(current_frame, Point(net, intersections[1].y + 50), 3, cv::Scalar(255, 0, 255), -1);
        //while (!current_frame.empty() && frame_number < 300) {
        while (!current_frame.empty() && frame_number < totalFrames - 1) {
            //cvtColor(current_frame, hls_image, COLOR_BGR2HLS);
            Mat hsvImage;
            cvtColor(current_frame, hsvImage, COLOR_BGR2HSV);
            // Define lower and upper HSV thresholds for white and orange balls
            Scalar lowerBound(0, 140, 180); // Lower HSV values for ball color (red, green, blue)
            Scalar upperBound(80, 200, 255); // Upper HSV values for ball color
            // Create binary masks for white and orange balls
            Mat maskBall;
            inRange(hsvImage, lowerBound, upperBound, maskBall);
            // Find contours in the binary masks
            vector<vector<Point>> contoursBall;
            findContours(maskBall, contoursBall, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            float newWidth = current_frame.cols / 30;
            float minWidth = 2;
            float maxBallArea = 0.5 * 3.14 * newWidth * newWidth;
            float minBallArea = 0.5 * 3.14 * minWidth * minWidth;
            // Loop through ball contours and draw circles around them
            // Assume there's only one moving table tennis ball
            float largest_radius = 0;
            Point2f ball_center;
            // ball detection
            for (const auto& contour : contoursBall) {
                Point2f center;
                float radius;
                minEnclosingCircle(contour, center, radius);
                double aspectRatio = calculateAspectRatio(contour);
                double circularity = calculateCircularity(contour);
                // locate the ball
                if (0.5 * 3.14 * radius * radius < maxBallArea && 0.5 * 3.14 * radius * radius > minBallArea && aspectRatio > 0.5 && circularity > 0.5 && radius > 0.8 && radius > largest_radius) {
                    largest_radius = radius;
                    ball_center = center;
                    xpos2 = ball_center.x;
                    ypos2 = ball_center.y;
                }
            }
            //std::cout << "ball center: " << ball_center << std::endl;
            if (xpos2 == 0 && ypos2 == 0) {
                xpos2 = xpos1;
                ypos2 = ypos1;
            }
            // detect direction of ball
            if (left) {
                if (xpos2 > xpos1) {
                    left = false;
                    // change direction
                    string str = "right";
                    std::cout << "Change direction to: " << str << std::endl;
                    Point location(37, 28);
                    int line_step = 20;
                    cv::putText(current_frame, "Change direction to: right", location, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
                    // detect if the ball hit by player
                    if (xpos2 < intersections[1].x + 150) {
                        cv::putText(current_frame, "ball hit by player (left player)", Point(37, 58), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
                        cout << "Frame " << frame_number << " Hit by Player" << std::endl;
                        frame_track[track_count] = frame_number;
                        col_x[track_count] = xpos2;
                        row_y[track_count] = ypos2;
                        type[track_count] = "Hit by Player";
                        //cout << frame_track[track_count] << col_x[track_count] << row_y[track_count] << std::endl;
                        track_count++;
                    }
                    // detect if the ball hit the net
                    else if (xpos2 > net && xpos2 < net_right) {
                        cv::putText(current_frame, "ball hit the net", Point(37, 58), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
                        cout << "Frame " << frame_number << " Hit the Net" << std::endl;
                        frame_track[track_count] = frame_number;
                        col_x[track_count] = xpos2;
                        row_y[track_count] = ypos2;
                        type[track_count] = "Hit the Net";
                        track_count++;
                    }
                }
            }
            else {
                if (xpos2 < xpos1) {
                    left = true;
                    // change direction
                    string str = "left";
                    std::cout << "Change direction to: " << str << std::endl;
                    Point location(37, 28);
                    int line_step = 20;
                    cv::putText(current_frame, "Change direction to: left", location, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
                    // detect if the ball hit by player
                    if (xpos2 > intersections[2].x - 150) {
                        cv::putText(current_frame, "ball hit by player (right player)", Point(37, 58), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
                        cout << "Frame " << frame_number << " Hit by Player" << std::endl;
                        frame_track[track_count] = frame_number;
                        col_x[track_count] = xpos2;
                        row_y[track_count] = ypos2;
                        type[track_count] = "Hit by Player";
                        track_count++;
                    }
                    // detect if the ball hit the net
                    else if (xpos2 > net_left && xpos2 < net_right) {
                        cv::putText(current_frame, "ball hit the net", Point(37, 58), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
                        cout << "Frame " << frame_number << " Hit the Net" << std::endl;
                        frame_track[track_count] = frame_number;
                        col_x[track_count] = xpos2;
                        row_y[track_count] = ypos2;
                        type[track_count] = "Hit the Net";
                        track_count++;
                    }
                }
            }
            if (up) {
                if (ypos2 > ypos1) {
                    up = false;
                    // change direction
                    string str = "down";
                    std::cout << "Change direction to: " << str << std::endl;
                    Point location(37, 28);
                    cv::putText(current_frame, "Change direction to: down", location, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
                }
            }
            else {
                // detect if the ball bounce on table
                if (ypos2 < ypos1 && frame_number >5) {
                    up = true;
                    // change direction
                    string str = "up";
                    std::cout << "Change direction to: " << str << std::endl;
                    Point location(37, 28);
                    cv::putText(current_frame, "Change direction to: up", location, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
                    if (ypos2 < intersections[0].y && ypos2 > intersections[1].y) {
                        cv::putText(current_frame, "ball bounced on the table", Point(37, 58), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
                        cout << "Frame " << frame_number << " Bounce on Table" << std::endl;
                        frame_track[track_count] = frame_number;
                        col_x[track_count] = xpos2;
                        row_y[track_count] = ypos2;
                        type[track_count] = "Bounce on Table";
                        track_count++;
                    }
                }
            }
            // display current direction
            if (up) {
                if (left) {
                    cv::putText(current_frame, "Current direction: left, up", Point(37, 43), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
                }
                else {
                    cv::putText(current_frame, "Current direction: right, up", Point(37, 43), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
                }
            }
            else {
                if (left) {
                    cv::putText(current_frame, "Current direction: left, down", Point(37, 43), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
                }
                else {
                    cv::putText(current_frame, "Current direction: right, down", Point(37, 43), FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
                }
            }
            circle(current_frame, ball_center, largest_radius, Scalar(0, 255, 0), 1); // Circle the ball
            xpos1 = xpos2;
            ypos1 = ypos2;
            /*Point location(7, 13);
            int line_step = 13;
            cv::putText(current_frame, "press 1: ball", location, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
            location.y += line_step;*/
            imshow("Video", current_frame);
            video >> current_frame;
            //resize(current_frame, current_frame, Size(current_frame.cols / 1.5, current_frame.rows / 1.5));
            cvtColor(current_frame, hls_image, COLOR_BGR2HLS);
            frame_number++;
            cv::waitKey(1);
        }
        cv::destroyAllWindows();
    }
}
int part_three() {
    //Load video
    std::string file_location = "video/";
    std::string video_files[] = {
        "TableTennis.avi"
    };
    VideoCapture* video = new VideoCapture[1];
    string filename(file_location);
    filename.append(video_files[0]);
    video[0].open(filename);
    if (!video[0].isOpened())
    {
        cout << "Cannot open video file: " << filename << endl;
        //			return -1;
    }
    process_video(video[0], 1);
    // compare accuracy
    int detected_frame_count = 0;
    int ground_truth_count = 0;
    int correct_case = 0;
    int false_positive = 0;
    int false_negative = 0;
    cout << "start evaluating performance" << endl;
    while (ground_truth_count < 141) {
        int ground_truth_frame = stoi(GROUND_TRUTH_VIDEO[ground_truth_count][0]);
        string ground_truth_type = GROUND_TRUTH_VIDEO[ground_truth_count][3];
        int detected_frame = frame_track[detected_frame_count];
        //cout << ground_truth_frame << endl;
        //cout << detected_frame << endl;
        if (detected_frame - ground_truth_frame <= 3 && detected_frame - ground_truth_frame>=-1) {
            // true positive cases
            if (type[detected_frame_count].compare(ground_truth_type) == 0) {
                cout << "correct case: " << ground_truth_type << " Detected frame " << detected_frame << " Ground Truth frame " << ground_truth_frame << endl;
                correct_case++;
                detected_frame_count++;
                ground_truth_count++;
            }
            // false positive cases
            else {
                cout << "fp case: " <<type[detected_frame_count] << ground_truth_type << " Detected frame " << detected_frame << " Ground Truth frame " << ground_truth_frame << endl;
                false_positive++;
                detected_frame_count++;
            }
        }
        // false negative cases
        else if (detected_frame - ground_truth_frame > 3) {
            ground_truth_count++;
            false_negative++;
            cout << "fn case: " << type[detected_frame_count] << ground_truth_type << " Detected frame " << detected_frame << " Ground Truth frame " << ground_truth_frame << endl;
        }
        else if (detected_frame - ground_truth_frame < -1) {
            detected_frame_count++;
            false_positive++;
        }
    }
    // print performance
    float accuracy = correct_case / 141.0;
    float recall = float(correct_case) / float(correct_case + false_negative);
    float precision = float(correct_case) / float(correct_case + false_positive);
    cout << "Number of true positive: " << correct_case << endl;
    cout << "Number of false positive: " << false_positive << endl;
    cout << "Number of false negative: " << false_negative << endl;
    cout << "Accuracy: " << float(correct_case) / float(track_count - 1) << endl;
    cout << "Recall: " << recall << endl;
    cout << "Precision: " << precision << endl;
    cout << "Accuracy: " << accuracy << endl;
    return 0;
}
// main function to select which part to execute
int main(int argc, const char** argv) {
    cv::Mat startImage(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));
    Point location(7, 13);
    int line_step = 13;
    cv::putText(startImage, "press 1: ball", location, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
    location.y += line_step;
    cv::putText(startImage, "press 2: tables", location, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
    location.y += line_step;
    cv::putText(startImage, "press 3: video", location, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
    location.y += line_step;
    cv::putText(startImage, "press m: MyApplication", location, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
    location.y += line_step;
    cv::putText(startImage, "press x: exit", location, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
	int choice;
	do
	{
		imshow("Welcome", startImage);
		choice = cv::waitKey();
		cv::destroyAllWindows();
		switch (choice)
		{
		case '1':
			part_one();
			break;
        case '2':
            part_two();
            break;
        case '3':
            part_three();
            break;
        case 'm':
            imshow("Welcome", startImage);
            break;
		default:
			break;
		}
	} while ((choice != 'x') && (choice != 'X'));
    return 0;
}



