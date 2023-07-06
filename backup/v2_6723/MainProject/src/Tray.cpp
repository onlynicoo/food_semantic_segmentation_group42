#include "../include/Tray.h"

cv::Mat Tray::DetectFoods(cv::Mat src) {
    cv::Mat out;
    // ... add code to detect food and return an image with bounding boxes
    return out;
}

cv::Mat Tray::SegmentFoods(cv::Mat src) {
    cv::Mat out;
    // ... add code to detect food and return an image with bounding boxes
    return out;
}

cv::Mat GetTrainedFeatures(std::string labelFeaturesPath) {
    cv::Mat labelFeatures;

    // Read template images
    cv::FileStorage fs(labelFeaturesPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cout << "Failed to open label features file." << std::endl;
    }
    fs["labelFeatures"] >> labelFeatures;
    fs.release();
    return labelFeatures;
}

std::map<int, cv::Vec3b> InitColorMap() {
    std::map<int, cv::Vec3b> colors;
    colors[0] = cv::Vec3b{0, 0, 0};  // Black
    colors[1] = cv::Vec3b{0, 255, 124};  // Green
    colors[2] = cv::Vec3b{0, 0, 255};  // Red
    colors[3] = cv::Vec3b{255, 0, 0};  // Blue
    colors[4] = cv::Vec3b{0, 255, 255};  // Yellow
    colors[5] = cv::Vec3b{255, 255, 0};  // Cyan
    colors[6] = cv::Vec3b{255, 0, 255};  // Magenta
    colors[7] = cv::Vec3b{0, 165, 255};  // Orange
    colors[8] = cv::Vec3b{128, 0, 128};  // Purple
    colors[9] = cv::Vec3b{203, 192, 255};  // Pink
    colors[10] = cv::Vec3b{42, 42, 165};  // Brown
    colors[11] = cv::Vec3b{128, 128, 128};  // Gray
    colors[12] = cv::Vec3b{255, 255, 255};  // White
    colors[13] = cv::Vec3b{128, 128, 0};  // Olive

    return colors;
}

cv::Mat performWatershedSegmentation(const cv::Mat& inputImage)
{
    // Convert the input image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);

    // Apply thresholding or other preprocessing if needed

    // Perform distance transform
    cv::Mat distTransform;
    distanceTransform(grayImage, distTransform, cv::DIST_L2, 3);
    // Normalize the distance transform
    normalize(distTransform, distTransform, 0, 1.0, cv::NORM_MINMAX);

    // Threshold the distance transform to obtain markers for watershed
    double thresholdValue = threshold(distTransform, distTransform, 0.7, 255, cv::THRESH_BINARY);

    // Convert the thresholded image to 8-bit
    distTransform.convertTo(distTransform, CV_8U);

    // Apply connected component analysis to label the markers
    cv::Mat markers;
    connectedComponents(distTransform, markers);

    std::cout << "ci sono \n\n";
    // Add 1 to all markers to mark background regions as 1
    markers = markers + 1;
    std::cout << "ci sono \n\n";
    
    // Mark the unknown regions with 0
    markers.setTo(0, inputImage == cv::Scalar(0, 0, 0));
    std::cout << "ci sono \n\n";
    // Apply watershed algorithm
    watershed(inputImage, markers);
    std::cout << "ci sono \n\n";

    // Generate random colors for visualization
    std::vector<cv::Vec3b> colors;
    for (int i = 0; i < markers.rows * markers.cols; ++i)
    {
        int r = 21;
        int g = 123;
        int b = 200;
        colors.push_back(cv::Vec3b(r, g, b));
    }

    // Create the segmented image using random colors
    cv::Mat segmented = cv::Mat::zeros(markers.size(), CV_8UC3);
    for (int y = 0; y < markers.rows; ++y)
    {
        for (int x = 0; x < markers.cols; ++x)
        {
            int label = markers.at<int>(y, x);
            segmented.at<cv::Vec3b>(y, x) = colors[label];
        }
    }

    return segmented;
}

void beliefPropagation(cv::Mat& inputImage, cv::Mat& segmentationMask, int maxIterations, float lambda)
{
    // Define the size of the graph
    int height = inputImage.rows;
    int width = inputImage.cols;

    // Create a graph with one node per pixel
    int numNodes = height * width;

    // Initialize the graph with random beliefs (for segmentation)
    cv::Mat_<float> beliefs(height, width);
    cv::randu(beliefs, 0, 1);

    // Initialize the messages
    cv::Mat_<float> messages(height, width);
    messages.setTo(0.1);

    // Compute the data term (unary potential) based on pixel intensities
    cv::Mat_<float> dataTerm(height, width);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            dataTerm(y, x) = abs(inputImage.at<uchar>(y, x) - 128) / 128.0f;
        }
    }

    // Perform belief propagation iterations
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        // Update messages from nodes to edges
        for (int i = 0; i < numNodes; ++i)
        {
            int y = i / width;
            int x = i % width;

            float belief = beliefs(y, x);
            float message = 0;

            // Compute the message by summing over neighboring beliefs and messages
            // Here, we consider the 4-neighborhood (top, bottom, left, right)
            if (y > 0)
                message += messages(y - 1, x);
            if (y < height - 1)
                message += messages(y + 1, x);
            if (x > 0)
                message += messages(y, x - 1);
            if (x < width - 1)
                message += messages(y, x + 1);

            // Update the message with a damping factor (lambda)
            message = lambda * message + (1 - lambda) * belief;

            messages(y, x) = message;
        }

        // Update beliefs from edges to nodes
        for (int i = 0; i < numNodes; ++i)
        {
            int y = i / width;
            int x = i % width;

            float message = messages(y, x);
            float belief = beliefs(y, x);

            // Update the belief by multiplying incoming messages and the data term
            // Here, we consider the 4-neighborhood (top, bottom, left, right)
            if (y > 0)
                belief *= messages(y - 1, x);
            if (y < height - 1)
                belief *= messages(y + 1, x);
            if (x > 0)
                belief *= messages(y, x - 1);
            if (x < width - 1)
                belief *= messages(y, x + 1);

            belief *= dataTerm(y, x);

            beliefs(y, x) = belief;
        }
    }

    // Generate the segmentation mask from the final beliefs
    segmentationMask = cv::Mat(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            // Set the segmentation mask based on the final beliefs
            if (beliefs(y, x) > 0.5)
                segmentationMask.at<uchar>(y, x) = 255;
            else
                segmentationMask.at<uchar>(y, x) = 0;
        }
    }
}

cv::Mat generateRadialKernel(int kernelSize, double sigma)
{
    // Generate a 1D Gaussian kernel
    cv::Mat gaussianKernel1D = cv::getGaussianKernel(kernelSize, sigma, CV_64F);

    // Compute the outer product of the Gaussian kernel with itself to obtain a 2D radial kernel
    cv::Mat radialKernel = gaussianKernel1D * gaussianKernel1D.t();

    // Normalize the radial kernel to sum up to 1.0
    radialKernel /= sum(radialKernel)[0];

    return radialKernel;
}

void Tray::ElaborateImage(const cv::Mat src, cv::Mat tmpDest[2], std::vector<int>& labelsFound) {
    // it contains
    // image detection | image segmentation
    std::string LABELS[14] = {
        "0. Background", "1. pasta with pesto", "2. pasta with tomato sauce", "3. pasta with meat sauce",
        "4. pasta with clams and mussels", "5. pilaw rice with peppers and peas", "6. grilled pork cutlet",
        "7. fish cutlet", "8. rabbit", "9. seafood salad", "10. beans", "11. basil potatoes", "12. salad", "13. bread"};

    std::string labelFeaturesPath = "../data/label_features.yml";
    std::vector<int> excludedLabels = {0, 12, 13};
    if(labelsFound.size() != 0) {
        for(int i = 0; i < 14; i ++) {
            if(std::find(std::begin(labelsFound), std::end(labelsFound), i) == std::end(labelsFound)) 
                if(std::find(std::begin(excludedLabels), std::end(excludedLabels), i) == std::end(excludedLabels))
                    excludedLabels.push_back(i);
        }
    }

    int firstPlatesLabel[] = {1, 2, 3, 4, 5};
    
    std::map<int, cv::Vec3b> colors = InitColorMap();


    cv::Size segmentationMaskSize = src.size();
    cv::Mat segmentationMask(segmentationMaskSize, CV_8UC3, cv::Scalar(0));


    cv::Mat labels = GetTrainedFeatures(labelFeaturesPath);
    std::vector<cv::Vec3f> plates = PlatesFinder::get_plates(src);
    tmpDest[0] = PlatesFinder::print_plates_image(src, plates);

    
    for(int i = 0; i < plates.size(); i++) {
        
        cv::Point center;
        int radius;
        center.x = plates[i][0];
        center.y = plates[i][1];
        radius = plates[i][2];

        cv::Mat tmpMask;
        
        // remove plates giving only food
        PlateRemover::getFoodMask(src, tmpMask, center, radius);
        
        // creates the features for the segmented patch
        cv::Mat patchFeatures = FeatureComparator::getImageFeatures(src, tmpMask);
        
        // compare the extracted features with the pretrained features
        int foodLabel = FeatureComparator::getFoodLabel(labels, excludedLabels, patchFeatures);
        std::cout << "Plate " << i << " label found: " << LABELS[foodLabel] << "\n";
        
        if(std::find(std::begin(firstPlatesLabel), std::end(firstPlatesLabel), foodLabel) != std::end(firstPlatesLabel)) {
            labelsFound.push_back(foodLabel);
            for(int r = 0; r < segmentationMask.rows; r++) {
                for(int c = 0; c < segmentationMask.cols; c++) {
                    if(tmpMask.at<uchar>(r,c) != 0)
                        segmentationMask.at<cv::Vec3b>(r,c) = colors[int(tmpMask.at<uchar>(r,c)*foodLabel)];
                }
            }
        }
        else {
            //add refinition of segmentation mask for multifood plates
            cv::Mat tmp, segmentation_mask;
            bitwise_and(src, src, tmp, tmpMask);
            imshow("test1", tmp);

            cv::Mat labImage;
            cv::cvtColor(tmp, labImage, cv::COLOR_BGR2Lab);
            // Perform mean shift color-based segmentation
            cv::Mat segmented;
            pyrMeanShiftFiltering(labImage, segmented, 40, 90);

            // Convert the segmented image back to the BGR color space
            cv::Mat outputImage;
            cvtColor(segmented, outputImage, cv::COLOR_Lab2BGR);

            //beliefPropagation(tmp, segmentation_mask, 10, 0.75);
            imshow("test2", outputImage);
            
            //std::cout<<"tmp size: " << tmp.size() << "\n";
            //cv::Mat tmp1 = performWatershedSegmentation(tmp);
            //std::cout<<"dammi size: " << tmp1.size() << "\n";
            
            imshow("test3", generateRadialKernel(250,100));
            cv::waitKey();

            //
            labelsFound.push_back(foodLabel);
        }
    }
    
    tmpDest[1] = segmentationMask;
    // ... add code ...
}

Tray::Tray(std::string trayBefore, std::string trayAfter) {
    
    cv::Mat before = cv::imread(trayBefore, cv::IMREAD_COLOR);
    cv::Mat after = cv::imread(trayAfter, cv::IMREAD_COLOR);

    traysBeforeNames = trayBefore;
    traysAfterNames = trayAfter;
    
    traysBefore = before;
    traysAfter = after;

    cv::Mat tmpDest[2];
    std::vector<int> labelsFound;
    ElaborateImage(before, tmpDest, labelsFound);
    traysBeforeDetected = tmpDest[0];
    traysBeforeSegmented = tmpDest[1];
    
    ElaborateImage(after, tmpDest, labelsFound);
    traysAfterDetected = tmpDest[0];
    traysAfterSegmented = tmpDest[1];
}

void Tray::PrintInfo() {
    std::string window_name_before = "info tray";

    cv::Mat imageGrid, imageRow;
    cv::Size stdSize(0,0);
    stdSize = traysBefore.size();

    cv::Mat tmp1_1, tmp1_2, tmp1_3, tmp2_1, tmp2_2, tmp2_3; 
    tmp1_1 = traysBefore.clone();

    // Resize output to have all images of same size
    resize(traysAfter, tmp2_1, stdSize);
    resize(traysBeforeDetected, tmp1_2, stdSize);
    resize(traysAfterDetected, tmp2_2, stdSize);
    resize(traysBeforeSegmented, tmp1_3, stdSize);
    resize(traysAfterSegmented, tmp2_3, stdSize);

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
    imshow(window_name_before, imageGrid);

    cv::waitKey();
}