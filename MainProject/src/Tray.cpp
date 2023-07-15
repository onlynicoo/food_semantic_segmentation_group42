#include "../include/Tray.h"

#include <fstream>

#include "../include/FindFood.h"
#include "../include/SegmentFood.h"
#include "../include/Utils.h"

/**
 * The function "get_trayAfterPath" returns the trayAfterPath string.
 * 
 * @return a string value, which is the value of the variable "trayAfterPath".
 */
std::string Tray::get_trayAfterPath() {
    return trayAfterPath;
}


/**
 * The function `InitColorMap` initializes a map with integer keys and `cv::Vec3b` values representing
 * different colors.
 * 
 * @return The function `InitColorMap` returns a `std::map<int, cv::Vec3b>` object, which is a map with
 * integer keys and `cv::Vec3b` values.
 */
std::map<int, cv::Vec3b> InitColorMap() {

    std::map<int, cv::Vec3b> colors;

    colors[0] = cv::Vec3b{0, 0, 0};        // Black
    colors[1] = cv::Vec3b{0, 255, 124};    // Green
    colors[2] = cv::Vec3b{0, 0, 255};      // Red
    colors[3] = cv::Vec3b{255, 0, 0};      // Blue
    colors[4] = cv::Vec3b{0, 255, 255};    // Yellow
    colors[5] = cv::Vec3b{255, 255, 0};    // Cyan
    colors[6] = cv::Vec3b{255, 0, 255};    // Magenta
    colors[7] = cv::Vec3b{0, 165, 255};    // Orange
    colors[8] = cv::Vec3b{128, 0, 128};    // Purple
    colors[9] = cv::Vec3b{203, 192, 255};  // Pink
    colors[10] = cv::Vec3b{51, 102, 153};  // Brown
    colors[11] = cv::Vec3b{128, 128, 128}; // Gray
    colors[12] = cv::Vec3b{255, 255, 255}; // White
    colors[13] = cv::Vec3b{128, 128, 0};   // Olive

    return colors;
}

/**
 * The function "InsertBoundingBox" takes a binary mask image and a file path as input, and writes the
 * bounding box coordinates of each object in the mask to the specified file.
 * 
 * @param mask The "mask" parameter is a cv::Mat object representing a binary mask. It is used to
 * identify regions of interest in an image. Each region is assigned a unique ID from 1 to 13.
 * @param filePath The `filePath` parameter is a string that represents the path to the file where the
 * bounding box information will be written.
 */
void InsertBoundingBox(const cv::Mat mask, std::string filePath) {

    std::ofstream file(filePath, std::ios::trunc); // Open the file in append mode

    // For each possible label, extract the mask, compute the bounding box and write it in the file
    for (int i = 1; i < 14; i++) {
        cv::Mat binaryMask = (mask == i);
        if (cv::countNonZero(binaryMask) == 0)
            continue;

        cv::Rect bbox = cv::boundingRect(binaryMask);

        // Write the new line to the file
        file << "ID: " << i << "; [" << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << "]\n"; 
    }

    file.close();
}

/**
 * The function `getLabelsFound` takes a mask image as input and returns a vector of labels that are
 * found in the mask.
 * 
 * @param mask The "mask" parameter is a cv::Mat object, which represents a binary image. It is used to
 * identify regions of interest in an image. In this code, the function "getLabelsFound" takes this
 * mask as input and returns a vector of integers representing the labels found in the mask.
 * 
 * @return a vector of integers, which contains the labels found in the given mask.
 */
std::vector<int> getLabelsFound(const cv::Mat& mask) {
    std::vector<int> labelsFound;

    // For each possible food, if it's present in the mask than insert it in `labelsFoun`
    for (int i = 1; i < 14; i++)
        if (cv::countNonZero((mask == i)) > 0)
            labelsFound.push_back(i);

    return labelsFound;
}

/**
 * The function `segmentImage` takes an input image, segments different food items in the image, and
 * returns a gray scale mask representing the segmented regions.
 * 
 * @param src The source image on which the segmentation is performed.
 * @param labelsFound A vector of integers that will be filled with the labels found during the
 * segmentation process.
 * @param filePath The `filePath` parameter is a string that represents the file path where the
 * segmented image will be saved.
 * 
 * @return a cv::Mat object, which is the segmentation mask.
 */
cv::Mat Tray::segmentImage(const cv::Mat& src, std::vector<int>& labelsFound, const std::string& filePath) {

    cv::Mat segmentationMask(src.size(), CV_8UC1, cv::Scalar(0));

    // Segment food in plates
    cv::Mat plateFoodsMask;
    std::vector<cv::Vec3f> foodPlates = FindFood::findPlates(src);
    SegmentFood::getFoodMaskFromPlates(src, plateFoodsMask, foodPlates, labelsFound);
    segmentationMask += plateFoodsMask;

    // Segment salad
    cv::Mat saladMask;
    bool saladFound = (Utils::getIndexInVector(labelsFound, SegmentFood::SALAD_LABEL) != -1);

    std::vector<cv::Vec3f> saladBowls = FindFood::findSaladBowl(src, saladFound);
    if (saladBowls.size() != 0) {
        SegmentFood::getSaladMaskFromBowl(src, saladMask, saladBowls[0]);
        segmentationMask += saladMask;
    }

    // Segment bread
    cv::Mat breadArea;
    FindFood::findBread(src, breadArea);
    cv::Mat breadMask;
    SegmentFood::getBreadMask(src, breadArea, breadMask);
    segmentationMask += breadMask;

    // Store labels found
    labelsFound = getLabelsFound(segmentationMask);

    return segmentationMask;
}

/**
 * The function "extractTrayFromPath" takes an image path as input and returns the path to the output
 * directory based on the penultimate folder in the image path.
 * 
 * @param imagePath The imagePath parameter is a string that represents the path of an image file.
 * 
 * @return a string that represents the path to the output directory where the extracted tray should be
 * saved.
 */
std::string extractTrayFromPath(const std::string& imagePath) {
    
    // Find the penultimate occurrence of '/'
    size_t lastSlashPos = imagePath.rfind('/');
    size_t penultimateSlashPos = imagePath.rfind('/', lastSlashPos - 1);

    // Extract the substring between the penultimate and last slash
    std::string penultimatePath = imagePath.substr(penultimateSlashPos + 1, lastSlashPos - penultimateSlashPos - 1);

    std::string imageName = imagePath.substr(
        imagePath.find_last_of('/') + 1,
        imagePath.find_last_of('.') - 1 - imagePath.find_last_of('/'));

    return "../output/" + penultimatePath;
}

/**
 * The function "extractName" takes an image path as input and returns the name of the image file with
 * "_bounding_box" appended to it.
 * 
 * @param imagePath A string representing the path of an image file.
 * 
 * @return a modified version of the image name. It appends "_bounding_box" to the image name.
 */
std::string extractName(const std::string& imagePath) {
    
    std::string imageName = imagePath.substr(
        imagePath.find_last_of('/') + 1,
        imagePath.find_last_of('.') - 1 - imagePath.find_last_of('/'));

    return imageName.append("_bounding_box");
}


/**
 * The function `SaveSegmentedMask` saves a segmented mask image to a specified path with a modified
 * filename.
 * 
 * @param path The `path` parameter is a string that represents the file path where the segmented mask
 * image will be saved.
 * @param src The `src` parameter is a `cv::Mat` object representing the segmented mask image that
 * needs to be saved.
 */
void Tray::saveSegmentedMask(const std::string& path, const cv::Mat& src) {

    std::string imageName = extractName(path);

    if (imageName.compare("food_image_bounding_box") == 0)
        imageName = "food_image_mask";
    if (imageName.compare("leftover1_bounding_box") == 0)
        imageName = "leftover1";
    if (imageName.compare("leftover2_bounding_box") == 0)
        imageName = "leftover2";
    if (imageName.compare("leftover3_bounding_box") == 0)
        imageName = "leftover3";

    std::string filename = extractTrayFromPath(path) + "/" + "masks/" + imageName + ".png";

    bool success = cv::imwrite(filename, src);
}

/**
 * The Tray constructor takes two image paths as input, reads the images, performs segmentation on the
 * images, inserts bounding boxes, and saves the segmented masks.
 * 
 * @param trayBefore The `trayBefore` parameter is a string that represents the file path to an image
 * file that shows the tray before the meal.
 * @param trayAfter The `trayAfter` parameter is a string that represents the file path to an image
 * file that shows the tray after the meal.
 */
Tray::Tray(const std::string& trayBefore, const std::string& trayAfter) {
    
    cv::Mat before = cv::imread(trayBefore, cv::IMREAD_COLOR);
    cv::Mat after = cv::imread(trayAfter, cv::IMREAD_COLOR);

    trayBeforePath = trayBefore;
    trayAfterPath = trayAfter;

    trayBeforeImage = before;
    trayAfterImage = after;

    std::vector<int> labelsFound;
    trayBeforeBoundingBoxesPath = extractTrayFromPath(trayBefore) + "/" + "bounding_boxes/" + extractName(trayBefore) + ".txt";
    trayAfterBoundingBoxesPath = extractTrayFromPath(trayAfter) + "/" + "bounding_boxes/" + extractName(trayAfter) + ".txt";

    trayBeforeSegmentationMask = segmentImage(before, labelsFound, trayBeforeBoundingBoxesPath);
    trayAfterSegmentationMask = segmentImage(after, labelsFound, trayAfterBoundingBoxesPath);

    InsertBoundingBox(trayBeforeSegmentationMask, trayBeforeBoundingBoxesPath);
    InsertBoundingBox(trayAfterSegmentationMask, trayAfterBoundingBoxesPath);

    saveSegmentedMask(trayBeforePath, trayBeforeSegmentationMask);
    saveSegmentedMask(trayAfterPath, trayAfterSegmentationMask);
}

/**
 * The function takes a binary mask and returns a colored segmentation mask based on a predefined color
 * map.
 * 
 * @param mask The "mask" parameter is a grayscale image where each pixel represents a segment. The
 * value of each pixel indicates the segment to which it belongs.
 * 
 * @return a cv::Mat object, which represents the segmented image with colored regions.
 */
cv::Mat getColoredSegmentationMask(const cv::Mat& mask) {
    
    std::map<int, cv::Vec3b> colors = InitColorMap();

    // For each pixel of the mask, assign the correct color based on the label
    cv::Mat segmentedImage(mask.size(), CV_8UC3, cv::Scalar(0));
    for (int r = 0; r < mask.rows; r++)
        for (int c = 0; c < mask.cols; c++)
            if (mask.at<uchar>(r, c) != 0)
                segmentedImage.at<cv::Vec3b>(r, c) = colors[int(mask.at<uchar>(r, c))];
    return segmentedImage;
}

/**
 * The function `overimposeDetection` takes an input image, a file path, and an output image, and
 * overlays rectangles on the input image based on the coordinates specified in the file.
 * 
 * @param src The source image on which the rectangles will be drawn.
 * @param filePath The `filePath` parameter is a string that represents the path to a file containing
 * the detection information. This file is expected to have a specific format, where each line
 * represents a detection and contains the following information:
 * @param dst The `dst` parameter is a reference to a `cv::Mat` object. It is used to store the output
 * image after applying the overlay detection. The function modifies this parameter by assigning the
 * processed image to it.
 */
void overimposeDetection(const cv::Mat& src, std::string filePath, cv::Mat &dst) {

    cv::Mat out = src.clone();
    std::map<int, cv::Vec3b> colors = InitColorMap();

    std::ifstream file(filePath);

    // Check if the file is correctly open
    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line)) {

            cv::Point topLeft, bottomRight;

            // Extracting 'x'
            size_t xIndex = line.find("ID") + 3;
            size_t semicolonIndex = line.find(";", xIndex);
            int foodId = std::stoi(line.substr(xIndex, semicolonIndex - xIndex));

            // Extracting 'a', 'b', 'c', 'd'
            size_t openBracketIndex = line.find("[", semicolonIndex) + 1;
            size_t closeBracketIndex = line.find("]", openBracketIndex);
            std::string elementsStr = line.substr(openBracketIndex, closeBracketIndex - openBracketIndex);

            std::vector<int> elements;
            size_t commaIndex = 0;
            size_t nextCommaIndex = elementsStr.find(",", commaIndex);
            while (nextCommaIndex != std::string::npos) {
                elements.push_back(std::stoi(elementsStr.substr(commaIndex, nextCommaIndex - commaIndex)));
                commaIndex = nextCommaIndex + 2;  // Skip comma and space
                nextCommaIndex = elementsStr.find(",", commaIndex);
            }
            elements.push_back(std::stoi(elementsStr.substr(commaIndex)));  // Last element

            // Assigning values to points variables
            if (elements.size() == 4) {
                topLeft.x = elements[0];
                topLeft.y = elements[1];
                bottomRight.x = elements[0] + elements[2];
                bottomRight.y = elements[1] + elements[3];
            }
            // Draw the rectangle
            cv::rectangle(out, topLeft, bottomRight, colors[foodId], 10); // Draw the rectangle on the image
        }
    }

    dst = out;
}

/**
 * The function `showTray()` displays various images related to a tray, including segmentation masks,
 * detection boxes, and overlaid images.
 */
void Tray::showTray() {

    std::string window_name = "info tray";

    cv::Mat imageGrid, imageRow;
    cv::Size stdSize(0, 0);
    stdSize = trayBeforeImage.size();

    cv::Mat tmp1_1, tmp1_2, tmp1_3, tmp2_1, tmp2_2, tmp2_3;

    cv::Mat colorBeforeSegmented = getColoredSegmentationMask(trayBeforeSegmentationMask);
    cv::Mat colorAfterSegmented = getColoredSegmentationMask(trayAfterSegmentationMask);

    tmp1_1 = trayBeforeImage.clone();

    // Resize output to have all images of same size
    resize(trayAfterImage, tmp2_1, stdSize);
    // resize(OverimposeDetection(traysBefore, traysBeforeDetected), tmp1_2, stdSize);
    // resize(OverimposeDetection(traysAfter, traysAfterDetected), tmp2_2, stdSize);

    std::vector<cv::Vec3f> saladBefore = FindFood::findSaladBowl(trayBeforeImage, false);
    std::vector<cv::Vec3f> saladAfter;
    saladAfter = FindFood::findSaladBowl(trayAfterImage, false);
    if (saladBefore.size() != 0) {
        saladAfter = FindFood::findSaladBowl(trayAfterImage, true);
    }

    cv::Mat imgWithDetectionBoxesBefore, imgWithDetectionBoxesAfter;
    overimposeDetection(trayBeforeImage, trayBeforeBoundingBoxesPath, imgWithDetectionBoxesBefore);
    overimposeDetection(trayAfterImage, trayAfterBoundingBoxesPath, imgWithDetectionBoxesAfter);
    resize(imgWithDetectionBoxesBefore, tmp1_2, stdSize);
    resize(imgWithDetectionBoxesAfter, tmp2_2, stdSize);
    resize(colorBeforeSegmented, tmp1_3, stdSize);
    resize(colorAfterSegmented, tmp2_3, stdSize);

    // Add image to current image row
    tmp2_3.copyTo(imageRow);
    hconcat(tmp2_2, imageRow, imageRow);
    hconcat(tmp2_1, imageRow, imageRow);
    imageRow.copyTo(imageGrid);
    imageRow.release();

    tmp1_3.copyTo(imageRow);
    hconcat(tmp1_2, imageRow, imageRow);
    hconcat(tmp1_1, imageRow, imageRow);
    vconcat(imageRow, imageGrid, imageGrid);
    imageRow.release();

    // Resize the full image grid and display it
    resize(imageGrid, imageGrid, stdSize);
    imshow(window_name, imageGrid);

    cv::waitKey(0);
}