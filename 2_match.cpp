#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <stdlib.h>
using namespace cv;
using namespace std;
#define MAX_TESTS 4
struct Minutiae {
    int x, y;
    // 1 for ridge ending, 3 for bifurcation
    int type;
};

// Crossing Number Algorithm: Calculate Minutiae
int crossingNumber(const Mat& image, int x, int y) {
    int cn = 0;
    int p[9];
    p[0] = image.at<uchar>(y, x);
    p[1] = image.at<uchar>(y-1, x);
    p[2] = image.at<uchar>(y-1, x+1);
    p[3] = image.at<uchar>(y, x+1);
    p[4] = image.at<uchar>(y+1, x+1);
    p[5] = image.at<uchar>(y+1, x);
    p[6] = image.at<uchar>(y+1, x-1);
    p[7] = image.at<uchar>(y, x-1);
    p[8] = image.at<uchar>(y-1, x-1);

    for (int i = 1; i <= 8; i++) {
        cn += abs(p[i] - p[i % 8 + 1]);
    }
    return (cn / 2);
}

// Extract Minutae
vector<Minutiae> extractMinutiae(const Mat& binaryImage) {
    vector<Minutiae> minutiaePoints;
    for (int y = 1; y < binaryImage.rows - 1; y++) {
        for (int x = 1; x < binaryImage.cols - 1; x++) {
            if (binaryImage.at<uchar>(y, x) == 0) { // Check only black pixels
                int cn = crossingNumber(binaryImage, x, y);
                if (cn == 1 || cn == 3) {
                    Minutiae m = {x, y, cn};
                    minutiaePoints.push_back(m);
                }
            }
        }
    }

    return minutiaePoints;
}

int matchMinutiae(const vector<Minutiae>& minutiae1, const vector<Minutiae>& minutiae2) {
    int matches = 0;
    for (const auto& m1 : minutiae1) {
        for (const auto& m2 : minutiae2) {
            if (m1.x == m2.x && m1.y == m2.y && m1.type == m2.type) {
                matches++;
            }
        }
    }
    return matches;
}

void printResult(int arr[]) {
    std::cout << "Number of matching Minutiae Points with Control Image..." << std::endl;
    for(int index=0; index<MAX_TESTS; index++) {
        std::cout << std::endl << "Test Image " << (index+1) << ": " << arr[index];
    }
}

int main() {
    // Load the binary images
    Mat controlIMG = imread("./assets/binary/control.png", IMREAD_GRAYSCALE);
    Mat testIMG1 = imread("./assets/binary/test_1.png", IMREAD_GRAYSCALE);
    Mat testIMG2 = imread("./assets/binary/test_2.png", IMREAD_GRAYSCALE);
    Mat testIMG3 = imread("./assets/binary/test_3.png", IMREAD_GRAYSCALE);
    Mat testIMG4 = imread("./assets/binary/test_4.png", IMREAD_GRAYSCALE);

    if (controlIMG.empty() || testIMG1.empty() || testIMG2.empty() || testIMG3.empty() || testIMG4.empty()) {
        cout << "Could not open or find one of the images" << endl;
        return -1;
    }

    // Extract minutiae points from: Control Image
    vector<Minutiae> minutiaePointsControl = extractMinutiae(controlIMG);
    // Extract minutiae points from: Test Images 1 to 4
    vector<Minutiae> minutiaeTest1 = extractMinutiae(testIMG1);
    vector<Minutiae> minutiaeTest2 = extractMinutiae(testIMG2);
    vector<Minutiae> minutiaeTest3 = extractMinutiae(testIMG3);
    vector<Minutiae> minutiaeTest4 = extractMinutiae(testIMG4);

    // Match minutiae points
    // Total number of matches = Number of test images = 4
    int matches[4] = {matchMinutiae(minutiaePointsControl,minutiaeTest1),
                  matchMinutiae(minutiaePointsControl,minutiaeTest2),
                  matchMinutiae(minutiaePointsControl,minutiaeTest3),
                  matchMinutiae(minutiaePointsControl,minutiaeTest4)};
    
    printResult(matches);

    return EXIT_SUCCESS;
}
