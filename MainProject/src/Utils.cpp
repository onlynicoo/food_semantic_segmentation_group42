#include <map>
#include <unordered_map>
#include <algorithm>
#include <set>
#include "../include/Utils.h"

std::vector<int> Utils::sortVectorByFreq(std::vector<int> values) {
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
    return result;
}

int Utils::getMostFrequentNeighbor(std::vector<std::vector<int>> matrix, int row, int col, int radius) {
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

std::vector<std::vector<int>> Utils::getMostFrequentMatrix(std::vector<std::vector<int>> matrix, int radius) {
    std::vector<std::vector<int>> newMatrix(matrix.size(), std::vector<int>(matrix[0].size()));

    for (int i = 0; i < matrix.size(); i++)
        for (int j = 0; j < matrix[0].size(); j++)
                newMatrix[i][j] = Utils::getMostFrequentNeighbor(matrix, i, j, radius);

    return newMatrix;
}

std::vector<int> Utils::getVectorUnion(std::vector<int> a, std::vector<int> b){
    std::vector<int> c;
    c.reserve(a.size() + b.size());
    std::copy(a.begin(), a.end(), std::back_inserter(c));
    std::copy(b.begin(), b.end(), std::back_inserter(c));
    return c;
}

std::vector<int> Utils::getVectorIntersection(std::vector<int> a, std::vector<int> b){
    std::set<int> setB(b.begin(), b.end());
    std::vector<int> c;
    for (int i = 0; i < a.size(); i++)
        if (setB.count(a[i]) > 0)
            c.push_back(a[i]);
    return c;
}

int Utils::getIndexInVector(std::vector<int> vector, int value) {
    auto it = std::find(vector.begin(), vector.end(), value);
    if (it != vector.end())
        return std::distance(vector.begin(), it);
    return -1;
}