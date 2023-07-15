#include <fstream>
#include "../include/FindFood.h"
#include "../include/SegmentFood.h"
#include "../include/Utils.h"
#include "../include/Tray.h"

std::string Tray::get_trayAfterName()
{
    return traysAfterNames;
}

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

void InsertBoundingBox(cv::Mat src, std::string filePath) {

    std::ofstream file(filePath, std::ios::trunc); // Open the file in append mode

    for (int i = 1; i < 14; i++) {
        cv::Mat binaryMask = (src == i);

        if (cv::countNonZero(binaryMask) == 0)
            continue;
        
        cv::Rect bbox = cv::boundingRect(binaryMask);
        file << "ID: " << i << "; [" << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << "]\n"; // Write the new line to the file
    }

    file.close();
}

std::vector<int> getLabelsFound(const cv::Mat& mask) {
    std::vector<int> labelsFound;

    for (int i = 1; i < 14; i++)
        if (cv::countNonZero((mask == i)) > 0)
            labelsFound.push_back(i);
    
    return labelsFound;
}

cv::Mat Tray::SegmentImage(const cv::Mat& src, std::vector<int>& labelsFound, std::string filePath) {

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
    if (saladBowls.size() != 0)
    {
        SegmentFood::getSaladMaskFromBowl(src, saladMask, saladBowls[0]);
        segmentationMask += saladMask;
    }

    // Segment bread
    cv::Mat breadArea, breadMask;
    FindFood::findBread(src, breadArea);
    breadMask = SegmentFood::getBreadMask(src, breadArea);
    segmentationMask += breadMask;

    // Keep labels found
    labelsFound = getLabelsFound(segmentationMask);

    return segmentationMask;
}

std::string ExtractTrayFromPath(std::string imagePath)
{

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

std::string ExtractName(std::string imagePath)
{

    std::string imageName = imagePath.substr(
        imagePath.find_last_of('/') + 1,
        imagePath.find_last_of('.') - 1 - imagePath.find_last_of('/'));

    return imageName.append("_bounding_box");
}

// should make bool and check output
void Tray::SaveSegmentedMask(std::string path, cv::Mat src)
{

    std::string imageName = ExtractName(path);

    if (imageName.compare("food_image_bounding_box") == 0)
        imageName = "food_image_mask";
    if (imageName.compare("leftover1_bounding_box") == 0)
        imageName = "leftover1";
    if (imageName.compare("leftover2_bounding_box") == 0)
        imageName = "leftover2";
    if (imageName.compare("leftover3_bounding_box") == 0)
        imageName = "leftover3";

    std::string filename = ExtractTrayFromPath(path) + "/" + "masks/" + imageName + ".png";

    bool success = cv::imwrite(filename, src);
}

// constructor that does everything
Tray::Tray(std::string trayBefore, std::string trayAfter)
{

    cv::Mat before = cv::imread(trayBefore, cv::IMREAD_COLOR);
    cv::Mat after = cv::imread(trayAfter, cv::IMREAD_COLOR);

    traysBeforeNames = trayBefore;
    traysAfterNames = trayAfter;

    traysBefore = before;
    traysAfter = after;

    std::vector<int> labelsFound;
    traysBeforeDetected = ExtractTrayFromPath(trayBefore) + "/" + "bounding_boxes/" + ExtractName(trayBefore) + ".txt";
    traysAfterDetected = ExtractTrayFromPath(trayAfter) + "/" + "bounding_boxes/" + ExtractName(trayAfter) + ".txt";

    traysBeforeSegmented = SegmentImage(before, labelsFound, traysBeforeDetected);
    traysAfterSegmented = SegmentImage(after, labelsFound, traysAfterDetected);

    InsertBoundingBox(traysBeforeSegmented, traysBeforeDetected);
    InsertBoundingBox(traysAfterSegmented, traysAfterDetected);

    SaveSegmentedMask(traysBeforeNames, traysBeforeSegmented);
    SaveSegmentedMask(traysAfterNames, traysAfterSegmented);
}

cv::Mat getColoredSegmentationMask(cv::Mat src)
{
    std::map<int, cv::Vec3b> colors = InitColorMap();
    cv::Size segmentationMaskSize = src.size();
    cv::Mat segmentedImage(segmentationMaskSize, CV_8UC3, cv::Scalar(0));
    for (int r = 0; r < src.rows; r++)
        for (int c = 0; c < src.cols; c++)
            if (src.at<uchar>(r, c) != 0)
                segmentedImage.at<cv::Vec3b>(r, c) = colors[int(src.at<uchar>(r, c))];
    return segmentedImage;
}

void OverimposeDetection(cv::Mat src, std::string filePath, cv::Mat &dst)
{

    cv::Mat out = src.clone();
    std::map<int, cv::Vec3b> colors = InitColorMap();

    std::ifstream file(filePath);

    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
        {

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
            while (nextCommaIndex != std::string::npos)
            {
                elements.push_back(std::stoi(elementsStr.substr(commaIndex, nextCommaIndex - commaIndex)));
                commaIndex = nextCommaIndex + 2; // Skip comma and space
                nextCommaIndex = elementsStr.find(",", commaIndex);
            }
            elements.push_back(std::stoi(elementsStr.substr(commaIndex))); // Last element

            // Assigning values to points variables
            if (elements.size() == 4)
            {
                topLeft.x = elements[0];
                topLeft.y = elements[1];
                bottomRight.x = elements[0] + elements[2];
                bottomRight.y = elements[1] + elements[3];
            }

            cv::rectangle(out, topLeft, bottomRight, colors[foodId], 10); // Draw the rectangle on the image
        }
    }
    dst = out;
}

void Tray::ShowTray()
{

    std::string window_name = "info tray";

    cv::Mat imageGrid, imageRow;
    cv::Size stdSize(0, 0);
    stdSize = traysBefore.size();

    cv::Mat tmp1_1, tmp1_2, tmp1_3, tmp2_1, tmp2_2, tmp2_3;

    cv::Mat colorBeforeSegmented = getColoredSegmentationMask(traysBeforeSegmented);
    cv::Mat colorAfterSegmented = getColoredSegmentationMask(traysAfterSegmented);

    tmp1_1 = traysBefore.clone();

    // Resize output to have all images of same size
    resize(traysAfter, tmp2_1, stdSize);
    // resize(OverimposeDetection(traysBefore, traysBeforeDetected), tmp1_2, stdSize);
    // resize(OverimposeDetection(traysAfter, traysAfterDetected), tmp2_2, stdSize);

    std::vector<cv::Vec3f> saladBefore = FindFood::findSaladBowl(traysBefore, false);
    std::vector<cv::Vec3f> saladAfter;
    saladAfter = FindFood::findSaladBowl(traysAfter, false);
    if (saladBefore.size() == 0)
    {
    }
    else
    {
        saladAfter = FindFood::findSaladBowl(traysAfter, true);
    }

    cv::Mat imgWithDetectionBoxesBefore, imgWithDetectionBoxesAfter;
    OverimposeDetection(traysBefore, traysBeforeDetected, imgWithDetectionBoxesBefore);
    OverimposeDetection(traysAfter, traysAfterDetected, imgWithDetectionBoxesAfter);
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