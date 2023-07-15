#pragma once
#include <vector>

class Utils {
    private:
        static int getMostFrequentNeighbor(const std::vector<std::vector<int>>&, int, int, int);

    public:
        static void sortVectorByFreq(std::vector<int>&);
        static void getMostFrequentMatrix(std::vector<std::vector<int>>&, int);
        static std::vector<int> getVectorUnion(const std::vector<int>&, const std::vector<int>&);
        static std::vector<int> getVectorIntersection(const std::vector<int>&, const std::vector<int>&);
        static int getIndexInVector(const std::vector<int>&, int);
};