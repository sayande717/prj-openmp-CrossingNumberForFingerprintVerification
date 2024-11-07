#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
namespace fs = std::filesystem;
#define MAX_TESTS 7

void printResult(int arr[]) {
    std::cout << "Number of matching Minutiae Points with Control Image..." << std::endl;
    for(int index=0; index<MAX_TESTS; index++) {
        std::cout << std::endl << "Test Image " << (index+1) << ": " << arr[index];
    }
}

int main() {
    string controlDir = "./assets/binary/control/";
    string binaryDir = "./assets/binary/";

    // Load control images
    map<string, Mat> controlImages;
    for (const auto& entry : fs::directory_iterator(controlDir)) {
        if (entry.path().extension() == ".tif") {
            string filename = entry.path().filename().string();
            controlImages[filename] = imread(entry.path().string(), IMREAD_GRAYSCALE);
        }
    }

    // Check if control images are loaded
    if (controlImages.empty()) {
        cout << "No control images found in " << controlDir << endl;
        return -1;
    }

    // Load test images and compare with control images
    map<string, vector<Mat>> testImages;
    for (const auto& entry : fs::directory_iterator(binaryDir)) {
        if (entry.path().extension() == ".tif") {
            string filename = entry.path().filename().string();
            string controlFilename = filename.substr(0, filename.find_last_of('_')) + "_1.tif";
            if (controlImages.find(controlFilename) != controlImages.end()) {
                testImages[controlFilename].push_back(imread(entry.path().string(), IMREAD_GRAYSCALE));
            }
        }
    }

    // Check if test images are loaded
    if (testImages.empty()) {
        cout << "No test images found in " << binaryDir << endl;
        return -1;
    }

    // Process and compare images
    for (const auto& controlImagePair : controlImages) {
        string controlFilename = controlImagePair.first;
        Mat controlIMG = controlImagePair.second;
        if (controlIMG.empty()) {
            cout << "Could not open or find the control image: " << controlFilename << endl;
            continue;
        }

        if (testImages.find(controlFilename) != testImages.end()) {
            vector<Mat> testIMGs = testImages[controlFilename];
            int matches[MAX_TESTS] = {0};

            for (size_t i = 0; i < testIMGs.size(); ++i) {
                Mat testIMG = testIMGs[i];
                if (testIMG.empty()) {
                    cout << "Could not open or find the test image: " << controlFilename << endl;
                    continue;
                }

                // Perform minutiae points comparison here
                // matches[i] = compareMinutiaePoints(controlIMG, testIMG);

                // For demonstration, we just print the filenames
                cout << "Comparing " << controlFilename << " with " << "test image " << i + 1 << endl;
            }

            printResult(matches);
        }
    }

    return EXIT_SUCCESS;
}