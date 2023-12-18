
#include <CL/cl.hpp>
#include "calibration.h"
#include "../cameraio/HikCamera.h"
#include "../LF.h"
#include <fstream>
#include <direct.h>


void CalcGroup(int x1, int x2, int& g1, int& g2, size_t max_group_size)
{
    int m = x1 * x2, tmp = 1;
    g1 = 1;
    g2 = 1;
    int start = std::min((int)ceil(sqrt(m)), (int)max_group_size);  
    for (int i = start; i > 0; --i)
        if (m % i == 0)   
        {
            tmp = i;
            break;
        }

    for (int i = tmp; i > 0; --i)  
        if (x1 % i == 0)
        {
            g1 = i;
            break;
        }
    g2 = tmp / g1;  

}

int main()
{
    
    cv::Mat src_img = cv::imread("Image_20230530161250068.bmp");      
    anakin::GrabImage stream(-1, src_img);

    char buffer11[100];
    _getcwd(buffer11, 100);
    printf("The current directory is: %s\n\n\n", buffer11);//打印当前运行目录

    const cv::Scalar colors[] = { cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 0), cv::Scalar(0,255,255), cv::Scalar(255,0,255) };
    int left(15), right(15), top(15), bottom(15), medianBlurSize(3), thresh(10), blockSize(31), C(1), openOpBlockSize(13);
    cv::namedWindow("Parameter");
    //cv::createTrackbar("medianBlurSize", "Parameter", &medianBlurSize, 60);
    cv::createTrackbar("blockSize", "Parameter", &blockSize, 60);
    cv::createTrackbar("thresh", "Parameter", &thresh, 255);
    cv::createTrackbar("C", "Parameter", &C, 60);
    //cv::createTrackbar("enrodBlockSize", "Parameter", &openOpBlockSize, 60);

    cv::createTrackbar("left", "Parameter", &left, 100);
    cv::createTrackbar("right", "Parameter", &right, 100);
    cv::createTrackbar("top", "Parameter", &top, 100);
    cv::createTrackbar("bottom", "Parameter", &bottom, 100);

    std::shared_ptr<char> kernel_code(new char[1024 * 1024]{ 0 });
    FILE* pFILE = fopen("../../../../calibration/kernel.ocl", "r");
    size_t kernel_len = fread(kernel_code.get(), 1, 1024 * 1024, pFILE);
    cl_int err = CL_SUCCESS;
    cl::Buffer param_buf, img_buf;
    std::vector<cl::Platform> platforms;
    cl::Context context;
    std::vector<cl::Device> devices;
    cl::Program program_;
    cl::Program::Sources source;
    cl::CommandQueue queue;
    cl::Kernel kernel;
    try
    {
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "Platform size 0\n";
            return -1;
        }
        for (auto& it : platforms)
        {
            std::cout << it.getInfo<CL_PLATFORM_NAME>() << "\n";
            it.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            for (auto& dev_it : devices)
            {
                std::cout << "\t" << dev_it.getInfo<CL_DEVICE_NAME>() << "\n";
                std::cout << "\t" << dev_it.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
                std::cout << "\t";
                for (auto& param_it : dev_it.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>())
                    std::cout << param_it << " ";
                std::cout << "\n";
                std::cout << "\t" << dev_it.getInfo<CL_DEVICE_TYPE>() << "\n";
            }
        }
        cl_context_properties properties[] =
        { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
        context = cl::Context(CL_DEVICE_TYPE_DEFAULT, properties);

        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        queue = cl::CommandQueue(context, devices[0], 0, &err);

        source = cl::Program::Sources(1,
            std::make_pair(kernel_code.get(), kernel_len));
        program_ = cl::Program(context, source);
        program_.build(devices);

        std::cout << "\n***************OpenCL Build INFO***************\n\n";
        std::cout << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << "\n";
        std::cout << "\n===============================================\n\n";
        kernel = cl::Kernel(program_, "drawMLA", &err);

    }
    catch (const std::exception& excp)
    {
        std::cerr
            << "ERROR: "
            << excp.what()
            << std::endl;
    }
    try
    {
        cv::Mat src_mat_gray, src_use_mat;
        cv::Mat dst_binary_img, viewImage;
        std::vector<cv::Point2d> centers;
        double avg_dists, angle, x0, y0, max_row, max_col;
        while (true)
        {
            stream >> src_img;
            std::cout << src_img.size() << "\n";

            src_img.copyTo(viewImage);
            int ratio = viewImage.rows / 900;
            cv::Size sz(viewImage.cols / ratio, viewImage.rows / ratio);
            cv::resize(viewImage, viewImage, sz);
            cv::imshow("src", viewImage);
            char key = cv::waitKey();

            if (key == '1')
            {
                cv::cvtColor(src_img, src_mat_gray, cv::COLOR_BGR2GRAY);
                cv::imwrite("raw.bmp", src_mat_gray);
                cv::adaptiveThreshold(src_mat_gray, dst_binary_img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, blockSize * 2 + 3, static_cast<double>(C) / 100.0);
                cv::imwrite("raw.bmp", src_img);
                dst_binary_img.copyTo(viewImage);
                cv::cvtColor(viewImage, viewImage, cv::COLOR_GRAY2BGR);
                cv::imwrite("1.bmp", viewImage);
                int ratio = dst_binary_img.rows / 900;
                cv::Size sz(dst_binary_img.cols / ratio, dst_binary_img.rows / ratio);
                cv::resize(viewImage, viewImage, sz);
                cv::imshow("Calibration", viewImage);
            }
            else if (key == '2')
            {
                std::chrono::steady_clock::time_point tp_start = std::chrono::steady_clock::now();
                anakin::DetectCenters(dst_binary_img, centers, 10, 38);
                std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();
                centers = anakin::CropBorder(centers, dst_binary_img.cols, dst_binary_img.rows, left, right, top, bottom);
                std::cout << "centers len:" << centers.size() << "\t";
                std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();
                avg_dists = anakin::EstimateAverageDist(centers, 36, 5);
                std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();

                std::cout << "avg_distsavg_distsavg_distsavg_dists:" << avg_dists << "\n";

                std::cout << "TP1:" << std::chrono::duration_cast<std::chrono::milliseconds>(tp1 - tp_start).count() << "\t";
                std::cout << "TP2:" << std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1).count() << "\t";
                std::cout << "TP3:" << std::chrono::duration_cast<std::chrono::milliseconds>(tp3 - tp2).count() << "\n";
                src_mat_gray.copyTo(viewImage);
                cv::cvtColor(viewImage, viewImage, cv::COLOR_GRAY2BGR);
                for (auto& it : centers)
                    cv::circle(viewImage, it, avg_dists / 2, cv::Scalar(0, 0, 255), 3);

                cv::imwrite("2.bmp", viewImage);
                int ratio = viewImage.rows / 900;
                cv::Size sz(viewImage.cols / ratio, viewImage.rows / ratio);
                cv::resize(viewImage, viewImage, sz);
                cv::imshow("Calibration", viewImage);
            }
            else if (key == '3')
            {

                anakin::DetectMLA(centers, avg_dists, angle, x0, y0);//这里返回的angel是正确的的  xy是串联的最小的那个索引的像素坐标 并不是左上角的那个
                std::cout << "Average distance: " << avg_dists << " pixel\n"
                    << "Angel: " << angle * 180 / M_PI << " degree\n";
                cv::cvtColor(src_mat_gray, viewImage, cv::COLOR_GRAY2BGR);


                MLAParam param;

                param.Rotation = angle;
                param.Diameter = avg_dists;
                cv::Mat GridMatrix = param.Diameter * (cv::Mat_<double>(2, 2) <<
                    std::cos(param.Rotation), std::cos(param.Rotation + 120. * CL_M_PI / 180.),
                    std::sin(param.Rotation), std::sin(param.Rotation + 120. * CL_M_PI / 180.));
                cv::Mat InvGridMatrix = GridMatrix.inv(cv::DECOMP_SVD);
                memcpy(param.Matrix, GridMatrix.data, sizeof(double) * 4);
                memcpy(param.InvMatrix, InvGridMatrix.data, sizeof(double) * 4);

                std::cout << "M MATRIX " << GridMatrix << "\n";


                
                double pts[4][2] = { {0 - x0,0 - y0},{viewImage.cols - 1 - x0, 0 - y0},{viewImage.cols - 1 - x0,viewImage.rows - 1 - y0},{0 - x0,viewImage.rows - 1 - y0} };
                int min_row(100000), min_col(100000);
                max_row = 0;
                max_col = 0;
                for (int i = 0; i < 4; ++i)
                {
                    double idx = param.InvMatrix[0] * pts[i][0] + param.InvMatrix[1] * pts[i][1];
                    double idy = param.InvMatrix[2] * pts[i][0] + param.InvMatrix[3] * pts[i][1];
                    min_row = std::min<int>(min_row, std::floor(idy));
                    min_col = std::min<int>(min_col, std::floor(idx));
                    max_row = std::max<int>(max_row, std::ceil(idy));
                    max_col = std::max<int>(max_col, std::ceil(idx));
                }
                std::cout << "Range: [" << min_row << ", " << max_row << "] [" << min_col << ", " << max_col << "]\n";
                x0 = x0 + (param.Matrix[0] * min_col + param.Matrix[1] * min_row);
                y0 = y0 + (param.Matrix[2] * min_col + param.Matrix[3] * min_row);
                max_row -= min_row;
                max_col -= min_col;
                param.Width = viewImage.cols;
                param.Height = viewImage.rows;
                param.Origin[0] = x0;
                param.Origin[1] = y0;
                try
                {
                    img_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, static_cast<size_t>((size_t)param.Width * param.Height * 3), viewImage.data, NULL);
                    param_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(MLAParam), &param, NULL);
                    err = kernel.setArg(0, img_buf);
                    std::cout << "setArg " << err << "\n";
                    err = kernel.setArg(1, param_buf);
                    std::cout << "setArg " << err << "\n";
                    int x1(viewImage.cols), x2(viewImage.rows), g1(1), g2(1);
                    CalcGroup(x1, x2, g1, g2, devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());

                    cl::Event event;
                    err = queue.enqueueNDRangeKernel(
                        kernel,
                        cl::NullRange,
                        cl::NDRange(x1, x2),
                        cl::NDRange(g1, g2),
                        NULL,
                        &event);
                    std::cout << "enqueueNDRangeKernel " << err << "\n";
                    event.wait();

                    err = queue.enqueueReadBuffer(img_buf, CL_TRUE, 0, img_buf.getInfo<CL_MEM_SIZE>(), viewImage.data);
                    std::cout << "enqueueReadBuffer " << err << "\n";

                    cv::imwrite("3.bmp", viewImage);
                    int ratio = dst_binary_img.rows / 900;
                    cv::Size sz(dst_binary_img.cols / ratio, dst_binary_img.rows / ratio);
                    cv::resize(viewImage, viewImage, sz);
                    cv::imshow("Calibration", viewImage);
                }
                catch (const std::exception& e)
                {
                    std::cerr << e.what() << "\n";
                }

            }
            else if (key == '5')
            {
                cv::FileStorage fs("my_test_0530modify.yaml", cv::FileStorage::WRITE);
                fs << "width" << src_mat_gray.cols;
                fs << "height" << src_mat_gray.rows;
                fs << "offsetx" << x0;
                fs << "offsety" << y0;
                fs << "diameter" << avg_dists;
                fs << "rotation" << angle;
                fs << "max_col" << max_col;
                fs << "max_row" << max_row;
                fs.release();

            }
            else if (key == 27)
                break;

        }
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << "\n";
    }

}