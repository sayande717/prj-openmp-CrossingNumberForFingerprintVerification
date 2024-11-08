#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <omp.h>

using namespace cv;
namespace fs = std::filesystem;

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

    // Collect all file paths
    std::vector<std::string> inputPaths;
    std::vector<std::string> outputPaths;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".tif") {
            inputPaths.push_back(entry.path().string());
            outputPaths.push_back(outputDir + entry.path().filename().string());
        }
    }
    // SERIAL run
    double startSerial = omp_get_wtime();
    for (size_t i = 0; i < inputPaths.size(); ++i) {
        preprocessImage(inputPaths[i].c_str(), outputPaths[i].c_str());
    }
    double endSerial = omp_get_wtime();

    // PARALLEL run
    double startParallel = omp_get_wtime();
    #pragma omp parallel for
    for(size_t i = 0; i < inputPaths.size(); ++i) {
        preprocessImage(inputPaths[i].c_str(), outputPaths[i].c_str());
    }
    double endParallel = omp_get_wtime();

    std::cout << "Preprocessing completed for all TIFF files." << std::endl;
    
    // OUTPUT results
    std::cout << "Serial Execution Time: " << (endSerial - startSerial) << " seconds" << std::endl;
    std::cout << "Parallel Execution Time: " << (endParallel - startParallel) << " seconds" << std::endl;

    return EXIT_SUCCESS;
}