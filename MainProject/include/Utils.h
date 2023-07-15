#pragma once
#include <vector>

class Utils {
    public:
        static std::vector<int> sortVectorByFreq(std::vector<int> values);
        static int getMostFrequentNeighbor(std::vector<std::vector<int>> matrix, int row, int col, int radius);
        static std::vector<std::vector<int>> getMostFrequentMatrix(std::vector<std::vector<int>> matrix, int radius);
        static std::vector<int> getVectorUnion(std::vector<int> a, std::vector<int> b);
        static std::vector<int> getVectorIntersection(std::vector<int> a, std::vector<int> b);
        static int getIndexInVector(std::vector<int> vector, int value);
};