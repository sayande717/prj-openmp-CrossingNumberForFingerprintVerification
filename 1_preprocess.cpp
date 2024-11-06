#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdlib.h>

using namespace cv;

void preprocessImage(const char* inputPath, const char* outputPath) {
    // Load the image
    Mat image = imread(inputPath, IMREAD_GRAYSCALE);
    if (image.empty()) {
        printf("Could not open or find the image\n");
        return;
    }

    // Convert to binary image using thresholding
    Mat binaryImage;
    threshold(image, binaryImage, 128, 255, THRESH_BINARY);

    // Save the binary image
    imwrite(outputPath, binaryImage);
}

int main() {
    const std::string inputDir = "./assets/";
    const std::string outputDir = "./assets/binary/";

    // Iterate through all files in the input directory
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".tif") {
            std::string inputPath = entry.path().string();
            std::string outputPath = outputDir + entry.path().filename().string();
            preprocessImage(inputPath, outputPath);
        }
    }

    std::cout << "Preprocessing completed for all TIFF files." << std::endl;
    return EXIT_SUCCESS;
}