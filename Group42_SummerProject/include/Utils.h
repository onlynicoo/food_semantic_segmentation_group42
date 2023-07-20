// Author: Daniele Moschetta

#pragma once
#include <vector>

// Handles utility methods for vectors and matrices
class Utils {
    private:

        // Retrieves the most frequent neighbor for a certain cell in a matrix
        static int getMostFrequentNeighbor(const std::vector<std::vector<int>>&, int, int, int);

    public:

        // Sorts a vector by frequency
        static void sortVectorByFreq(std::vector<int>&);

        // Modifies a matrix assigning to each cell the value returned by getMostFrequentNeighbor
        static void getMostFrequentMatrix(std::vector<std::vector<int>>&, int);

        // Returns a vector containing the union of the given two
        static std::vector<int> getVectorUnion(const std::vector<int>&, const std::vector<int>&);

        // Returns a vector containing the intersection of the given two
        static std::vector<int> getVectorIntersection(const std::vector<int>&, const std::vector<int>&);

        // Returns the index of an element in a vector
        static int getIndexInVector(const std::vector<int>&, int);
};