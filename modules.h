// modules.h
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#define NOMINMAX

/** @brief ���c�G���n�ֻP���ݩ� */
struct Kernel {
    cv::Mat matrix;
    bool normalize;
};

/** @brief ���c�G�t ghost zone ���v���϶� */
struct ImageBlock {
    cv::Mat data;
    int x;  ///< ���W�� x �y��
    int y;  ///< ���W�� y �y��
};

class InputLoader {
public:
    cv::Mat LoadImageFile(const std::string& filePath);
};

class Preprocessor {
public:
    cv::Mat NormalizeAndPrepare(const cv::Mat& image);
};

class FilterManager {
public:
    Kernel CreatePredefinedFilter(const std::string& filterName);
};

class BlockSplitter {
public:
    std::vector<ImageBlock> Split(
        const cv::Mat& image,
        const std::string& strategy,
        int blockSize,
        const cv::Size& kernelSize);
};

class ParallelConvolver {
public:
    std::vector<ImageBlock> ConvolveBlocks(
        const std::vector<ImageBlock>& blocks,
        const Kernel& kernel,
        const std::string& schedule);
};

class ResultIntegrator {
public:
    cv::Mat Integrate(
        const std::vector<ImageBlock>& blocks,
        int originalRows,
        int originalCols,
        const cv::Size& kernelSize);
};

class OutputModule {
public:
    void SaveImage(const cv::Mat& image, const std::string& filePath);
};
