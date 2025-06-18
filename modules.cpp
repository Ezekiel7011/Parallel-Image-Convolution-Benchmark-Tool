// modules.cpp
#include "modules.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <omp.h>

cv::Mat InputLoader::LoadImageFile(const std::string& filePath) {
    cv::Mat img = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("無法載入影像：" + filePath);
    }
    return img;
}

cv::Mat Preprocessor::NormalizeAndPrepare(const cv::Mat& image) {
    cv::Mat out;
    image.convertTo(out, CV_32F);
    cv::normalize(out, out, 0.0f, 1.0f, cv::NORM_MINMAX);
    return out;
}

Kernel FilterManager::CreatePredefinedFilter(const std::string& filterName) {
    cv::Mat kern;
    bool normalize = true;
    if (filterName == "blur")
        kern = cv::Mat::ones(3, 3, CV_32F) / 9.0f;
    else if (filterName == "sobel") {
        kern = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
        normalize = false;
    }
    else if (filterName == "sharpen")
        kern = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    else if (filterName == "laplacian") {
        kern = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
        normalize = false;
    }
    else
        throw std::invalid_argument("未知的預設濾鏡：" + filterName);
    return { kern, normalize };
}

std::vector<ImageBlock> BlockSplitter::Split(
    const cv::Mat& image,
    const std::string& strategy,
    int blockSize,
    const cv::Size& kernelSize)
{
    std::vector<ImageBlock> blocks;
    int rows = image.rows, cols = image.cols;
    int padY = kernelSize.height / 2, padX = kernelSize.width / 2;

    if (strategy == "rowWise") {
        for (int y = 0; y < rows; y += blockSize) {
            int y0 = std::max(0, y - padY);
            int y1 = std::min(rows, y + blockSize + padY);
            blocks.push_back({ image.rowRange(y0,y1).clone(), 0, y0 });
        }
    }
    else if (strategy == "blockWise") {
        for (int y = 0; y < rows; y += blockSize) {
            for (int x = 0; x < cols; x += blockSize) {
                int y0 = std::max(0, y - padY), y1 = std::min(rows, y + blockSize + padY);
                int x0 = std::max(0, x - padX), x1 = std::min(cols, x + blockSize + padX);
                blocks.push_back({
                    image(cv::Range(y0,y1), cv::Range(x0,x1)).clone(),
                    x0, y0
                    });
            }
        }
    }
    else {
        throw std::invalid_argument("未知的切割策略：" + strategy);
    }
    return blocks;
}

std::vector<ImageBlock> ParallelConvolver::ConvolveBlocks(
    const std::vector<ImageBlock>& blocks,
    const Kernel& kernel,
    const std::string& schedule)
{
    std::vector<ImageBlock> results(blocks.size());
    if (schedule == "static") {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)blocks.size(); ++i) {
            cv::Mat res;
            cv::filter2D(blocks[i].data, res, -1, kernel.matrix, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
            if (kernel.normalize)
                cv::normalize(res, res, 0.0f, 1.0f, cv::NORM_MINMAX);
            results[i] = { res, blocks[i].x, blocks[i].y };
        }
    }
    else if (schedule == "dynamic") {
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)blocks.size(); ++i) {
            cv::Mat res;
            cv::filter2D(blocks[i].data, res, -1, kernel.matrix, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
            if (kernel.normalize)
                cv::normalize(res, res, 0.0f, 1.0f, cv::NORM_MINMAX);
            results[i] = { res, blocks[i].x, blocks[i].y };
        }
    }
    else if (schedule == "guided") {
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < (int)blocks.size(); ++i) {
            cv::Mat res;
            cv::filter2D(blocks[i].data, res, -1, kernel.matrix, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
            if (kernel.normalize)
                cv::normalize(res, res, 0.0f, 1.0f, cv::NORM_MINMAX);
            results[i] = { res, blocks[i].x, blocks[i].y };
        }
    }
    else {
        throw std::invalid_argument("未知的排程策略：" + schedule);
    }
    return results;
}

cv::Mat ResultIntegrator::Integrate(
    const std::vector<ImageBlock>& blocks,
    int originalRows,
    int originalCols,
    const cv::Size& kernelSize)
{
    cv::Mat output(originalRows, originalCols, blocks[0].data.type());
    int padY = kernelSize.height / 2, padX = kernelSize.width / 2;
    for (auto& blk : blocks) {
        int y0 = blk.y + padY, x0 = blk.x + padX;
        int h = blk.data.rows - 2 * padY;
        int w = blk.data.cols - 2 * padX;
        blk.data(cv::Range(padY, padY + h), cv::Range(padX, padX + w))
            .copyTo(output(cv::Rect(x0, y0, w, h)));
    }
    return output;
}

void OutputModule::SaveImage(const cv::Mat& image, const std::string& filePath) {
    if (!cv::imwrite(filePath, image * 255))
        throw std::runtime_error("無法儲存影像：" + filePath);
}
