#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <omp.h> // Include OpenMP header
#include <iomanip>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// Function to calculate the Crossing Number for a pixel
int calculateCrossingNumber(const Mat& img, int x, int y) {
    int cn = 0;
    int p[9];
    p[0] = img.at<uchar>(x, y) > 0 ? 1 : 0;
    p[1] = img.at<uchar>(x - 1, y) > 0 ? 1 : 0;
    p[2] = img.at<uchar>(x - 1, y + 1) > 0 ? 1 : 0;
    p[3] = img.at<uchar>(x, y + 1) > 0 ? 1 : 0;
    p[4] = img.at<uchar>(x + 1, y + 1) > 0 ? 1 : 0;
    p[5] = img.at<uchar>(x + 1, y) > 0 ? 1 : 0;
    p[6] = img.at<uchar>(x + 1, y - 1) > 0 ? 1 : 0;
    p[7] = img.at<uchar>(x, y - 1) > 0 ? 1 : 0;
    p[8] = img.at<uchar>(x - 1, y - 1) > 0 ? 1 : 0;

    for (int i = 1; i <= 8; ++i) {
        cn += abs(p[i % 8] - p[(i - 1) % 8]);
    }
    return cn / 2;
}

// Function to find minutiae points using the Crossing Number algorithm
vector<Point> findMinutiaePoints(const Mat& img) {
    vector<Point> minutiaePoints;
    for (int i = 1; i < img.rows - 1; ++i) {
        for (int j = 1; j < img.cols - 1; ++j) {
            if (img.at<uchar>(i, j) > 0) {
                int cn = calculateCrossingNumber(img, i, j);
                if (cn == 1 || cn == 3) {
                    minutiaePoints.push_back(Point(j, i));
                }
            }
        }
    }
    return minutiaePoints;
}

// Function to compare minutiae points between two images
int compareMinutiaePoints(const Mat& controlIMG, const Mat& testIMG) {
    vector<Point> controlMinutiae = findMinutiaePoints(controlIMG);
    vector<Point> testMinutiae = findMinutiaePoints(testIMG);

    // Simple comparison: count the number of matching minutiae points
    int matchCount = 0;
    for (const auto& pt1 : controlMinutiae) {
        for (const auto& pt2 : testMinutiae) {
            if (norm(pt1 - pt2) < 5) { // Threshold distance for matching points
                matchCount++;
                break;
            }
        }
    }
    return matchCount;
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

    // IDs to process
    vector<string> ids = {"012", "013", "017", "022", "027", "076"};
    string controlID = "057";

// Load and compare images
for (size_t i = 0; i <= ids.size(); ++i) {
    vector<string> currentIDs(ids.begin(), ids.begin() + i);
    currentIDs.insert(currentIDs.begin(), controlID);

    map<string, vector<Mat>> testImages;
    for (const auto& entry : fs::directory_iterator(binaryDir)) {
        if (entry.path().extension() == ".tif") {
            string filename = entry.path().filename().string();
            for (const auto& id : currentIDs) {
                if (filename.find(id) == 0) {
                    string controlFilename = filename.substr(0, filename.find_last_of('_')) + "_1.tif";
                    if (controlImages.find(controlFilename) != controlImages.end()) {
                        testImages[controlFilename].push_back(imread(entry.path().string(), IMREAD_GRAYSCALE));
                    }
                }
            }
        }
    }

    // Process and compare images in increments of 8
    for (int groupSize = 8,iter=1; groupSize <= 520; groupSize += 8,iter++) {
        for (const auto& controlImagePair : controlImages) {
            string controlFilename = controlImagePair.first;
            Mat controlIMG = controlImagePair.second;
            if (controlIMG.empty()) {
                cout << "Could not open or find the control image: " << controlFilename << endl;
                continue;
            }

            if (testImages.find(controlFilename) != testImages.end()) {
                vector<Mat> testIMGs = testImages[controlFilename];
                if (testIMGs.size() < groupSize) {
                    continue; // Skip if there are not enough images for the current group size
                }

                vector<int> matches(groupSize, 0); // Initialize matches vector
                int bestMatch = 0;
                int bestMatchID = 0;

                // Serial run
                double startSerial = omp_get_wtime();
                for (size_t j = 0; j < groupSize; ++j) {
                    Mat testIMG = testIMGs[j];
                    if (testIMG.empty()) {
                        cout << "Could not open or find the test image: " << controlFilename << endl;
                        continue;
                    }

                    // Perform minutiae points comparison here
                    matches[j] = compareMinutiaePoints(controlIMG, testIMG);
                    if (matches[j] > bestMatch) {
                        bestMatch = matches[j];
                        bestMatchID = static_cast<int>(j);
                    }
                }
                double endSerial = omp_get_wtime();
                double elapsedSerial = endSerial - startSerial;

                // Parallel run
                double startParallel = omp_get_wtime();
                #pragma omp parallel for
                for (size_t j = 0; j < groupSize; ++j) {
                    Mat testIMG = testIMGs[j];
                    if (testIMG.empty()) {
                        cout << "Could not open or find the test image: " << controlFilename << endl;
                        continue;
                    }

                    // Perform minutiae points comparison here
                    matches[j] = compareMinutiaePoints(controlIMG, testIMG);
                }
                double endParallel = omp_get_wtime();
                double elapsedParallel = endParallel - startParallel;

                // Print results
                cout << fixed << setprecision(4);
                cout << "Iteration " << iter << ": " << endl;
                cout << "Number of images tested: " << groupSize << endl;
                cout << "Best match ID: " << bestMatchID << " with " << bestMatch << " matching minutiae points." << endl;
                cout << "Serial execution time: " << elapsedSerial << " seconds." << endl;
                cout << "Parallel execution time: " << elapsedParallel << " seconds." << endl;
            }
        }
    }
}
    return EXIT_SUCCESS;
}