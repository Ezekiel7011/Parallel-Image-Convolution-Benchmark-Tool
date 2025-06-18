// main.cpp
#include "modules.h"
#include <windows.h>
#include <vector>
#include <iostream>
#include <direct.h>
#include <errno.h>
#include <fstream>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <atomic>
#include <omp.h>
#include <chrono>
#include <ctime>
#include <iomanip>

// 讓 PaddedImageBlock 恰好對齊到 64 bytes
struct alignas(64) PaddedImageBlock {
    ImageBlock block;
};
static_assert(sizeof(PaddedImageBlock) % 64 == 0,
    "PaddedImageBlock 必須是 64 bytes 的整數倍");

// 取得所有 Performance‐Core（有 SMT 標記）的邏輯處理器遮罩
DWORD_PTR GetPerformanceCoreMask() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<BYTE> buffer(len);
    auto info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
    GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len);

    DWORD_PTR mask = 0;
    BYTE* ptr = buffer.data();
    BYTE* end = ptr + len;
    while (ptr < end) {
        auto rel = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(ptr);
        if (rel->Processor.Flags & LTP_PC_SMT) {
            auto& pr = rel->Processor;
            for (WORD i = 0; i < pr.GroupCount; ++i) {
                mask |= pr.GroupMask[i].Mask;
            }
        }
        ptr += rel->Size;
    }
    return mask;
}

int main() {
    // —— 新增：建立 Job Object 並限制 CPU 使用率 90%
    HANDLE hJob = CreateJobObject(NULL, NULL);
    if (hJob) {
        JOBOBJECT_CPU_RATE_CONTROL_INFORMATION cpuInfo = {};
        cpuInfo.ControlFlags = JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP;
        cpuInfo.CpuRate = 90 * 100;  // 90% ×100 (Windows 單位為 1/100 percent)
        SetInformationJobObject(hJob, JobObjectCpuRateControlInformation, &cpuInfo, sizeof(cpuInfo));
        AssignProcessToJobObject(hJob, GetCurrentProcess());
    }

    // —— 新增：提升 Process 與 Thread 優先權
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

    // 讀取是否要儲存中間影像
    std::cout << "是否要儲存中間影像？(y/n)：";
    char saveChar;
    std::cin >> saveChar;
    bool saveImages = (saveChar == 'y' || saveChar == 'Y');
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // 綁定到 P-core
    DWORD_PTR procMask, sysMask;
    GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask);
    DWORD_PTR pMask = GetPerformanceCoreMask();
    SetProcessAffinityMask(GetCurrentProcess(), pMask);

    // 輸入影像路徑
    std::cout << "請輸入影像檔路徑：";
    std::string imagePath;
    std::getline(std::cin, imagePath);

    // 載入 + 正規化
    InputLoader loader;
    Preprocessor prep;
    cv::Mat img = loader.LoadImageFile(imagePath);
    cv::Mat normImg = prep.NormalizeAndPrepare(img);

    // 全 12 種卷積核
    FilterManager fm;
    std::vector<std::pair<std::string, Kernel>> allKernels = {
        {"blur_3x3",      fm.CreatePredefinedFilter("blur")},
        {"sobel_3x3",     fm.CreatePredefinedFilter("sobel")},
        {"sharpen_3x3",   fm.CreatePredefinedFilter("sharpen")},
        {"laplacian_3x3", fm.CreatePredefinedFilter("laplacian")},
        {"blur_5x5",      Kernel{cv::Mat::ones(5,5,CV_32F) / 25.0f,true}},
        {"sobel_5x5",     Kernel{[]() { cv::Mat kx,ky; cv::getDerivKernels(kx,ky,1,0,5,true,CV_32F); return ky * kx.t(); }(),false}},
        {"sharpen_5x5",   Kernel{[]() { cv::Mat id = cv::Mat::zeros(5,5,CV_32F); id.at<float>(2,2) = 2.0f; return id - cv::Mat::ones(5,5,CV_32F) / 25.0f; }(),true}},
        {"laplacian_5x5", Kernel{[]() { cv::Mat kx2,ky2; cv::getDerivKernels(kx2,ky2,2,0,5,true,CV_32F); cv::Mat lapx = ky2 * kx2.t(); cv::getDerivKernels(kx2,ky2,0,2,5,true,CV_32F); cv::Mat lapy = ky2 * kx2.t(); return lapx + lapy; }(),false}},
        {"blur_7x7",      Kernel{cv::Mat::ones(7,7,CV_32F) / 49.0f,true}},
        {"sobel_7x7",     Kernel{[]() { cv::Mat kx,ky; cv::getDerivKernels(kx,ky,1,0,7,true,CV_32F); return ky * kx.t(); }(),false}},
        {"sharpen_7x7",   Kernel{[]() { cv::Mat id = cv::Mat::zeros(7,7,CV_32F); id.at<float>(3,3) = 2.0f; return id - cv::Mat::ones(7,7,CV_32F) / 49.0f; }(),true}},
        {"laplacian_7x7", Kernel{[]() { cv::Mat kx2,ky2; cv::getDerivKernels(kx2,ky2,2,0,7,true,CV_32F); cv::Mat lapx = ky2 * kx2.t(); cv::getDerivKernels(kx2,ky2,0,2,7,true,CV_32F); cv::Mat lapy = ky2 * kx2.t(); return lapx + lapy; }(),false}}
    };

    // 選擇要測試的 kernel
    std::cout << "請選要測試的卷積核：\n";
    for (size_t i = 0; i < allKernels.size(); ++i)
        std::cout << " " << (i + 1) << ". " << allKernels[i].first << "\n";
    std::cout << " " << (allKernels.size() + 1) << ". 全部都測\n";
    int choice; std::cin >> choice;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::vector<std::pair<std::string, Kernel>> kernelsToTest;
    if (choice >= 1 && choice <= (int)allKernels.size())
        kernelsToTest.push_back(allKernels[choice - 1]);
    else if (choice == (int)allKernels.size() + 1)
        kernelsToTest = allKernels;
    else {
        std::cerr << "錯誤：無效編號\n";
        return 1;
    }

    // 參數列表：最小 4×4 起
    std::vector<int> blockSizes = { 4, 8, 16, 32, 64, 128, 256, 512 };
    std::vector<std::string> strategies = { "rowWise","blockWise" };
    std::vector<std::string> schedules = { "static","dynamic","guided" };
    std::vector<int> threadCounts = { 1,2,4,8,16 };
    const int repeatCount = 1;

    BlockSplitter     splitter;
    ParallelConvolver convolver;
    ResultIntegrator  integrator;
    OutputModule      output;

    // 計算基準時間
    std::map<int, double> baseAvg;
    for (int bs : blockSizes) {
        omp_set_num_threads(1);
        std::vector<double> tms;
        for (int r = 0; r < repeatCount; ++r) {
            auto blks = splitter.Split(normImg, "rowWise", bs,
                cv::Size(kernelsToTest[0].second.matrix.cols,
                    kernelsToTest[0].second.matrix.rows));
            auto t0 = std::chrono::high_resolution_clock::now();
            auto cvt = convolver.ConvolveBlocks(blks,
                kernelsToTest[0].second, "static");
            auto rs = integrator.Integrate(cvt,
                img.rows, img.cols,
                cv::Size(kernelsToTest[0].second.matrix.cols,
                    kernelsToTest[0].second.matrix.rows));
            auto t1 = std::chrono::high_resolution_clock::now();
            tms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        baseAvg[bs] = std::accumulate(tms.begin(), tms.end(), 0.0) / tms.size();
    }

    // 重複執行三次
    for (int trial = 1; trial <= 3; ++trial) {
        // 建 trial 目錄
        std::ostringstream dirss;
        dirss << "results" << trial;
        std::string trialDir = dirss.str();
        if (_mkdir(trialDir.c_str()) != 0 && errno != EEXIST) {
            std::cerr << "無法建立資料夾：" << trialDir << "\n";
            return 1;
        }

        // 開 CSV，加入詳細欄位：KernelSize, IsSMT, IsPCoreOnly, TotalBlocks, FalseSharingRate
        std::ostringstream csvss;
        csvss << trialDir << "/comparison_results.csv";
        std::ofstream csv(csvss.str());
        csv << "Kernel,KernelSize,BlockSize,Strategy,Schedule,ThreadCount,Padding,IsSMT,IsPCoreOnly,TotalBlocks,"
            "Avg_ms,Min_ms,Max_ms,StdDev_ms,Speedup,Efficiency,FalseSharingCount,FalseSharingRate\n";

        // 計算進度總數
        long total = (long)kernelsToTest.size()
            * blockSizes.size()
            * strategies.size()
            * schedules.size()
            * threadCounts.size()
            * 2  // padding
            * 2; // affinityMode
        long done = 0;
        const int numPcores = 8;

        // 主要測試迴圈
        for (int affinityMode = 0; affinityMode < 2; ++affinityMode) {
            SetProcessAffinityMask(GetCurrentProcess(),
                (affinityMode == 0 ? pMask : procMask));

            for (auto& kp : kernelsToTest) {
                const auto& kName = kp.first;
                const auto& kern = kp.second;

                for (int bs : blockSizes)
                    for (auto const& strat : strategies)
                        for (auto const& sched : schedules)
                            for (int tc : threadCounts) {
                                omp_set_num_threads(tc);
                                int hyper1 = (tc > numPcores) ? 1 : 0;

                                for (int padFlag = 0; padFlag < 2; ++padFlag) {
                                    // 計算區塊總數
                                    size_t nBlocks = splitter.Split(normImg, strat, bs,
                                        cv::Size(kern.matrix.cols,
                                            kern.matrix.rows)).size();

                                    // 設置 false-sharing 統計
                                    size_t blockSizeBytes = sizeof(ImageBlock);
                                    size_t nGroups = (nBlocks * blockSizeBytes + 63) / 64;
                                    std::vector<std::atomic<int>> groupOwner(nGroups);
                                    for (size_t g = 0; g < nGroups; ++g) groupOwner[g].store(-1);
                                    std::atomic<long> falseCount(0);

                                    // 時間測量
                                    std::vector<double> tms;
                                    cv::Mat firstImg;
                                    for (int r = 0; r < repeatCount; ++r) {
                                        auto blks = splitter.Split(normImg, strat, bs,
                                            cv::Size(kern.matrix.cols,
                                                kern.matrix.rows));
                                        auto t0 = std::chrono::high_resolution_clock::now();

                                        std::vector<ImageBlock> convResult;
                                        if (padFlag == 0) {
                                            // 無 padding：統計 false-sharing
                                            std::vector<ImageBlock> tmp(blks.size());
#pragma omp parallel for schedule(static)
                                            for (int i = 0; i < (int)blks.size(); ++i) {
                                                int tid = omp_get_thread_num();
                                                size_t offset = i * blockSizeBytes;
                                                size_t g = offset / 64;
                                                int prev = groupOwner[g].exchange(tid);
                                                if (prev != -1 && prev != tid) {
                                                    falseCount.fetch_add(1);
                                                }
                                                cv::Mat res;
                                                cv::filter2D(blks[i].data, res, -1,
                                                    kern.matrix, cv::Point(-1, -1),
                                                    0, cv::BORDER_DEFAULT);
                                                if (kern.normalize)
                                                    cv::normalize(res, res, 0.0f, 1.0f, cv::NORM_MINMAX);
                                                tmp[i] = { res, blks[i].x, blks[i].y };
                                            }
                                            convResult.swap(tmp);
                                        }
                                        else {
                                            // 有 padding：不統計 false-sharing
                                            std::vector<PaddedImageBlock> tmp(blks.size());
#pragma omp parallel for schedule(static)
                                            for (int i = 0; i < (int)blks.size(); ++i) {
                                                cv::Mat res;
                                                cv::filter2D(blks[i].data, res, -1,
                                                    kern.matrix, cv::Point(-1, -1),
                                                    0, cv::BORDER_DEFAULT);
                                                if (kern.normalize)
                                                    cv::normalize(res, res, 0.0f, 1.0f, cv::NORM_MINMAX);
                                                tmp[i].block = { res, blks[i].x, blks[i].y };
                                            }
                                            convResult.reserve(tmp.size());
                                            for (auto& p : tmp) convResult.push_back(p.block);
                                        }

                                        auto rs = integrator.Integrate(convResult,
                                            img.rows, img.cols,
                                            cv::Size(kern.matrix.cols,
                                                kern.matrix.rows));
                                        auto t1 = std::chrono::high_resolution_clock::now();
                                        tms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                                        if (r == 0) firstImg = rs.clone();
                                    }

                                    // 統計結果
                                    double avg = std::accumulate(tms.begin(), tms.end(), 0.0) / tms.size();
                                    double mn = *std::min_element(tms.begin(), tms.end());
                                    double mx = *std::max_element(tms.begin(), tms.end());
                                    double var = 0; for (auto v : tms) var += (v - avg) * (v - avg);
                                    double sd = std::sqrt(var / tms.size());
                                    double sp = baseAvg[bs] / avg;
                                    double ef = sp / tc;
                                    long   fsCount = falseCount.load();
                                    double fsRate = static_cast<double>(fsCount) / nBlocks;

                                    // 存圖（如果啟用）
                                    if (saveImages) {
                                        std::ostringstream fn;
                                        fn << trialDir << "/" << kName << "_" << bs << "_" << strat << "_" << sched
                                            << "_t" << tc << (padFlag ? "_pad" : "")
                                            << (affinityMode ? "_all" : "_p") << ".png";
                                        output.SaveImage(firstImg, fn.str());
                                    }

                                    // 寫入 CSV
                                    csv << kName << ","
                                        << kern.matrix.rows << ","
                                        << bs << ","
                                        << strat << ","
                                        << sched << ","
                                        << tc << ","
                                        << padFlag << ","
                                        << hyper1 << ","
                                        << (affinityMode == 0 ? 1 : 0) << ","
                                        << nBlocks << ","
                                        << avg << ","
                                        << mn << ","
                                        << mx << ","
                                        << sd << ","
                                        << sp << ","
                                        << ef << ","
                                        << fsCount << ","
                                        << fsRate
                                        << "\n";

                                    // 更新進度
                                    ++done;
                                    int pct = int(done * 100 / total);
                                    std::cout << "\rRun " << trial << " Progress: " << pct << "%   " << std::flush;
                                }
                            }
            }
        }

        csv.close();
        std::cout << "\nTrial " << trial << " 完成，CSV 存於 " << csvss.str() << "\n";
    }

    return 0;
}
