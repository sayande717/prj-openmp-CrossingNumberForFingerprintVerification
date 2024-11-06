#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdlib.h>
#include <omp.h>

using namespace cv;

void preprocessImage(const char* inputPath, const char* outputPath) {
    // Load the image
    Mat image = imread(inputPath, IMREAD_GRAYSCALE);
    if (image.empty()) {
        printf("Could not open or find the image\n");
        return;
    }

    Mat binaryImage = Mat::zeros(image.size(), image.type());
    int rows = image.rows;
    int cols = image.cols;

    // Convert to binary image using thresholding in parallel
    #pragma omp parallel for collapse(2) schedule(static,50) firstprivate(binaryImage) lastprivate(binaryImage)
    // collapse(2) treats 2 nested loops as a single loop.
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            binaryImage.at<uchar>(i, j) = (image.at<uchar>(i, j) > 128) ? 255 : 0;
        }
    }

    // Save the binary image
    imwrite(outputPath, binaryImage);
}

int main() {
    const char* inputPath = "./assets/control.png";
    const char* outputPath = "./assets/binary/test_1.png";

    preprocessImage(inputPath, outputPath);

    std::cout << "Preprocessing completed. Binary image saved to" << outputPath << std::endl;
    return EXIT_SUCCESS;
}
