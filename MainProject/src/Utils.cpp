// Author: Daniele Moschetta

#include "../include/Utils.h"

#include <algorithm>
#include <map>
#include <set>
#include <unordered_map>

/**
 * The function `sortVectorByFreq` takes a vector of integers and sorts it in descending order based on
 * the frequency of each value.
 * 
 * @param values values is a reference to a vector of integers.
 * @return The function does not return anything. It modifies the input vector `values` in-place.
 */
void Utils::sortVectorByFreq(std::vector<int>& values) {
    // Create frequency map
    std::map<int, int> freqMap;
    for (int value : values)
        freqMap[value]++;

    // Sort values descending based on frequencies
    std::vector<std::pair<int, int>> sortedFreq(freqMap.begin(), freqMap.end());
    std::sort(sortedFreq.begin(), sortedFreq.end(), [](std::pair<int, int> a, std::pair<int, int> b) {
        return a.second > b.second;
    });

    // Get n most frequent values
    std::vector<int> result;
    for (int i = 0; i < sortedFreq.size(); i++)
        result.push_back(sortedFreq[i].first);
    values = result;
}

/**
 * The function `getMostFrequentNeighbor` takes a matrix, a row and column index, and a radius, and
 * returns the most frequent value among the neighbors within the specified radius.
 * 
 * @param matrix A 2D vector representing the matrix of integers.
 * @param row The row parameter represents the row index of the current element in the matrix.
 * @param col The parameter "col" represents the column index of the element in the matrix for which we
 * want to find the most frequent neighbor.
 * @param radius The radius parameter represents the distance from the center (row, col) to the
 * neighboring cells. It determines the size of the neighborhood around the given cell.
 * @return The function `getMostFrequentNeighbor` returns the most frequent value among the neighbors
 * of a given element in a matrix.
 */
int Utils::getMostFrequentNeighbor(const std::vector<std::vector<int>>& matrix, int row, int col, int radius) {
    std::unordered_map<int, int> freqMap;
    int maxCount = -1, mostFreqValue = -1;

    // Iterate over the neighbors
    for (int i = row - radius; i <= row + radius; i++) {
        for (int j = col - radius; j <= col + radius; j++) {
            if (i < 0 || j < 0 || i >= matrix.size() || j >= matrix[0].size() || (i == row && j == col) || matrix[i][j] == -1)
                continue;

            int neighborValue = matrix[i][j];
            freqMap[neighborValue]++;
            int count = freqMap[neighborValue];

            if (count > maxCount) {
                maxCount = count;
                mostFreqValue = neighborValue;
            }
        }
    }
    return mostFreqValue;
}

/**
 * The function "getMostFrequentMatrix" takes a matrix and a radius as input, and replaces each element
 * in the matrix with the most frequent element among its neighbors within the given radius.
 * 
 * @param matrix A 2D vector representing the matrix. Each element of the matrix is an integer value.
 * @param radius The "radius" parameter in the given code represents the distance from a cell to its
 * neighboring cells. It determines the range within which the most frequent value is calculated for
 * each cell in the matrix.
 */
void Utils::getMostFrequentMatrix(std::vector<std::vector<int>>& matrix, int radius) {
    std::vector<std::vector<int>> newMatrix(matrix.size(), std::vector<int>(matrix[0].size()));

    for (int i = 0; i < matrix.size(); i++)
        for (int j = 0; j < matrix[0].size(); j++)
            newMatrix[i][j] = Utils::getMostFrequentNeighbor(matrix, i, j, radius);

    matrix = newMatrix;
}

/**
 * The function `getVectorUnion` takes two vectors of integers as input and returns a new vector that
 * contains the union of the two input vectors.
 * 
 * @param a The parameter "a" is a constant reference to a vector of integers.
 * @param b The parameter "b" is a constant reference to a vector of integers.
 * @return a vector of integers, which is the union of the two input vectors.
 */
std::vector<int> Utils::getVectorUnion(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> c;
    c.reserve(a.size() + b.size());
    std::copy(a.begin(), a.end(), std::back_inserter(c));
    std::copy(b.begin(), b.end(), std::back_inserter(c));
    return c;
}

/**
 * The function `getVectorIntersection` takes two vectors of integers as input and returns a new vector
 * containing the intersection of the two input vectors.
 * 
 * @param a a is a vector of integers.
 * @param b The parameter "b" is a vector of integers.
 * @return a vector of integers that represents the intersection of two input vectors.
 */
std::vector<int> Utils::getVectorIntersection(const std::vector<int>& a, const std::vector<int>& b) {
    std::set<int> setB(b.begin(), b.end());
    std::vector<int> c;
    for (int i = 0; i < a.size(); i++)
        if (setB.count(a[i]) > 0)
            c.push_back(a[i]);
    return c;
}

/**
 * The function `getIndexInVector` returns the index of a given value in a vector, or -1 if the value
 * is not found.
 * 
 * @param vector A constant reference to a vector of integers. This vector is the one in which we want
 * to find the index of a specific value.
 * @param value The "value" parameter is the integer value that we are searching for in the vector.
 * @return The function `getIndexInVector` returns the index of the first occurrence of the `value` in
 * the `vector`. If the `value` is found, the index is returned. If the `value` is not found, -1 is
 * returned.
 */
int Utils::getIndexInVector(const std::vector<int>& vector, int value) {
    auto it = std::find(vector.begin(), vector.end(), value);
    if (it != vector.end())
        return std::distance(vector.begin(), it);
    return -1;
}