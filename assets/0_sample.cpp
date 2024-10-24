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
    // omp_set_num_threads(2);
    // Convert to binary image using thresholding
    #pragma omp parallel
    Mat binaryImage;
    threshold(image, binaryImage, 128, 255, THRESH_BINARY);
    
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
