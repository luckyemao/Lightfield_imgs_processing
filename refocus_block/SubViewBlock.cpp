
#include <CL/cl.hpp>
#include <opencv2/opencv.hpp>
#include "../LF.h"
#include <fstream>
#include <direct.h>//输出当前目录用

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

    //g1 = 64;
    //g2 = 4;

}


int main(int argc, char** argv)
{

 
    const int OFFSET_MAX(50);
    int zoom(150), all_focus(0), offset_x(OFFSET_MAX / 2), offset_y(OFFSET_MAX / 2);
    int view_ox(0), view_oy(0), ratio(21);
    int blocksize(7), blocksizelast(7);
    cv::namedWindow("Parameter");
    cv::createTrackbar("ZOOM", "Parameter", &zoom, 5000);//yuanlai 1200
    cv::createTrackbar("all_focus", "Parameter", &all_focus, 1);
    cv::createTrackbar("re_focus_block_size", "Parameter", &blocksize, 35);
    char buffer11[100];
    _getcwd(buffer11, 100);
    printf("The current directory is: %s\n\n\n", buffer11);//打印当前运行目录
    
 
    std::ifstream in("kernelsRefucal.ocl");
    std::istreambuf_iterator<char> beg(in), end;
    std::string code1(beg, end);

    cl_int err = CL_SUCCESS;
    
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices; 
    cl::Context context; 
    cl::Program::Sources source;
    cl::CommandQueue queue;
    cl::Kernel kernel, view_depth_kernel, disp_kernel;
    cl::Event event;
    try
    {
        err = cl::Platform::get(&platforms);
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
                std::cout << "\t" << dev_it.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
                std::cout << "\t" << dev_it.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << "\n";
                std::cout << "\t" << dev_it.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
                
                std::cout << "\t";
                for (auto& param_it : dev_it.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>())
                    std::cout << param_it << " ";
                std::cout << "\n";
                std::cout << "\t" << dev_it.getInfo<CL_DEVICE_TYPE>() << "\n";
            }
        }
        size_t plat_id(0), dev_id(0);
        cl_context_properties properties[] =
        { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[plat_id])(), 0 };
        context = cl::Context(CL_DEVICE_TYPE_DEFAULT, properties);

        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        queue = cl::CommandQueue(context, devices[dev_id], 0, &err);


        source.push_back(std::make_pair(code1.c_str(), code1.size()));
        cl::Program program_(context, source);
        err = program_.build(devices);
        std::cout << "\n***************OpenCL Build INFO***************\n\n";
        std::cout << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[dev_id]) << "\n";
        std::cout << "\n===============================================\n\n";
        
        kernel = cl::Kernel(program_, "render", &err);
        std::cout << "render " << err << "\n";
        view_depth_kernel = cl::Kernel(program_, "view_depth_kernel", &err);
        std::cout << "view_depth_kernel " << err << "\n";
        disp_kernel = cl::Kernel(program_, "Disp", &err);
        std::cout << "Disp " << err << "\n";

        std::cout << "value of  CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:   " << view_depth_kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(devices[0]) << "\n";
        std::cout << "value of  CL_KERNEL_WORK_GROUP_SIZE:   " << view_depth_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices[0]) << "\n";
        
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
        cv::FileStorage fs("../../../../calibration/my_test_0527modify.yaml", cv::FileStorage::READ);//0215modify
        MLAParam param;
        fs["width"] >> param.Width;
        fs["height"] >> param.Height;
        fs["offsetx"] >> param.Origin[0];
        fs["offsety"] >> param.Origin[1];
        fs["diameter"] >> param.Diameter;
        fs["rotation"] >> param.Rotation;
        fs["max_col"] >> param.Cols;
        fs["max_row"] >> param.Rows;
        cv::Mat GridMatrix = param.Diameter * (cv::Mat_<double>(2, 2) <<
            std::cos(param.Rotation), std::cos(param.Rotation + 120. * CL_M_PI / 180.),
            std::sin(param.Rotation), std::sin(param.Rotation + 120. * CL_M_PI / 180.));
        cv::Mat InvGridMatrix = GridMatrix.inv(cv::DECOMP_SVD);
        memcpy(param.Matrix, GridMatrix.data, sizeof(double) * 4);
        memcpy(param.InvMatrix, InvGridMatrix.data, sizeof(double) * 4);
        
        std::cout << param.Matrix[0] << "\n";
        std::cout << param.Matrix[1] << "\n";
        std::cout << param.Matrix[2] << "\n";
        std::cout << param.Matrix[3] << "\n";
        std::cout << "attention: above is param.Matrix " << "\n";
        std::cout << GridMatrix << "\n";

        StereoParam sparam;
        sparam.max_disp = 20;//20
        sparam.sobel_thresh = 20;//20
        sparam.block_size = 7;//7
        sparam.block_avg_thresh = 15;//15

        cv::Mat src_mat, src_gray, viewImage;
        src_mat = cv::imread("Image_20230527181053111-8v-0.8.bmp"); 

        cv::cvtColor(src_mat, src_gray, cv::COLOR_BGR2GRAY);       
        unsigned char* GridDepth = new unsigned char[param.Cols * param.Rows];
        double rendering_ratio = 1./1;
        int x1(param.Cols), x2(param.Rows), g1(1), g2(1);
        CalcGroup(x1, x2, g1, g2, 256);
        std::cout << "Use ocl nd range: " << x1 << " " << x2 << " " << g1 << " " << g2 << "\n";
        cl::Buffer src_buffer(context, CL_MEM_READ_WRITE, static_cast<size_t>((size_t)param.Width* param.Height * 3)),
            src_gray_buffer(context, CL_MEM_READ_WRITE, static_cast<size_t>((size_t)param.Width* param.Height )),
            param_buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(MLAParam), &param, NULL),
            sparam_buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(sparam), &sparam, NULL), 
            depth_map_buffer(context, CL_MEM_READ_WRITE, static_cast<size_t>((size_t)param.Width* param.Height)),
            view_depth_buffer(context, CL_MEM_READ_WRITE, (size_t)param.Cols* param.Rows);

        std::chrono::time_point<std::chrono::steady_clock> tp_start, tp_end;

        err = queue.enqueueWriteBuffer(src_buffer, CL_TRUE, 0, static_cast<size_t>((size_t)param.Width * param.Height * 3), src_mat.data);
        err = queue.enqueueWriteBuffer(src_gray_buffer, CL_TRUE, 0, static_cast<size_t>((size_t)param.Width * param.Height ), src_gray.data);
        err = disp_kernel.setArg(0, param_buffer);
        err = disp_kernel.setArg(1, sparam_buffer);
        err = disp_kernel.setArg(2, src_gray_buffer); 
        err = disp_kernel.setArg(3, depth_map_buffer);

        tp_start = std::chrono::steady_clock::now();      
        err = queue.enqueueNDRangeKernel(
            disp_kernel,
            cl::NullRange,
            cl::NDRange(param.Width, param.Height),
            cl::NullRange,
            NULL,
            &event);
        err = event.wait();
        tp_end = std::chrono::steady_clock::now();
        
        std::cout << "Disparity Cost " << std::chrono::duration_cast<std::chrono::milliseconds>(tp_end - tp_start).count() << "(ms)\n";       
        cv::Mat depth_map(param.Height, param.Width, CV_8U);
                                                        
        err = queue.enqueueReadBuffer(depth_map_buffer, CL_TRUE, 0, depth_map_buffer.getInfo<CL_MEM_SIZE>(), depth_map.data);
        cv::imwrite("depthMapPic.bmp", depth_map );
        cv::pyrDown(depth_map, depth_map); 
        cv::pyrDown(depth_map, depth_map);
        cv::imshow("depthMap", depth_map *10); 
        err = view_depth_kernel.setArg(0, param_buffer);
        err = view_depth_kernel.setArg(1, depth_map_buffer);
        err = view_depth_kernel.setArg(2, view_depth_buffer);
        err = queue.enqueueNDRangeKernel(
            view_depth_kernel,
            cl::NullRange,
            cl::NDRange(param.Cols, param.Rows),
            cl::NullRange,
            NULL,
            &event);

        err = event.wait();                        
        err = queue.enqueueReadBuffer(view_depth_buffer, CL_TRUE, 0, view_depth_buffer.getInfo<CL_MEM_SIZE>(), GridDepth);
                for (int i = 0; i < param.Rows; i+=4)
                {
                    for (int j = 0; j < param.Cols; j+=4)
                    {
                        if ((int)GridDepth[i * param.Cols + j] != 12)
                        {
                            std::cout << "[" << i << ", " << j << "]" << (int)GridDepth[i * param.Cols + j] << "\n";
                        }
                    }
                }
        cv::VideoWriter writer;
        bool rec = false;
        bool con_flag = false;
        double fps = 0.0;
        cv::Mat view_dst_new(param.Height, param.Width, CV_8UC3);
        std::stringstream NumberToString;
        std::string sv_fn = "1216v1.bmp";
        std::string sv_fn_con;
        
        while (true)
        {
            
            cv::Mat dst_mat(x2*blocksize, x1* blocksize, CV_8UC3, cv::Scalar(0, 0, 0));

            // define the buffer
            cl::Buffer dst_buffer(context, CL_MEM_READ_WRITE, static_cast<size_t>((size_t)x1 * blocksize * x2 * blocksize * 3));
            
            //////////////////////////////////////////  “render1” kernel ////////////////////// 
            tp_start = std::chrono::steady_clock::now();
            err = kernel.setArg(0, param_buffer);
            err = kernel.setArg(1, src_buffer);
            err = kernel.setArg(2, dst_buffer);
            err = kernel.setArg(3, 3);
            err = kernel.setArg(4, all_focus==0? (double)zoom / 1000.: 0.9);
            err = kernel.setArg(5, 2.0 * offset_x / OFFSET_MAX - 1);
            err = kernel.setArg(6, 2.0 * offset_y / OFFSET_MAX - 1);
            err = kernel.setArg(7, view_depth_buffer);
            err = kernel.setArg(8, blocksize);
			
            err = queue.enqueueNDRangeKernel(
                kernel,
                cl::NullRange,
                cl::NDRange(x1, x2),
                cl::NDRange(g1, g2),
                NULL,
                &event);

            err = event.wait();                        
            err = queue.enqueueReadBuffer(dst_buffer, CL_TRUE, 0, dst_buffer.getInfo<CL_MEM_SIZE>(), dst_mat.data);
            tp_end = std::chrono::steady_clock::now();
            fps = 0.95 * fps + 0.05 * 1000000 / std::chrono::duration_cast<std::chrono::microseconds>(tp_end - tp_start).count();
           
            std::cout << fps << " " << std::chrono::duration_cast<std::chrono::microseconds>(tp_end - tp_start).count() << "(us)\n";
           
            dst_mat.copyTo(view_dst_new);
            cv::imshow("111", view_dst_new);
            char key = cv::waitKey(1);

            if (con_flag)
            {
                if (blocksize > 35)
                {
                    std::cout << "Record finish\n";
                    std::cout << "Record finish\n";
                    std::cout << "Record finish\n";
                    std::cout << "Record finish\n";
                    con_flag = false;
                    //offset_y = OFFSET_MAX / 2;

                }
                else
                {
                    NumberToString << blocksize;
                    sv_fn_con ="k" + NumberToString.str() + ".bmp";
                    cv::imwrite(sv_fn_con, dst_mat);
                    blocksize = (blocksize + 1);
                    NumberToString.str("");
                }
            }

            
            if (key == 27)
                break;
            else if (key == '5')
            {
                cv::imwrite(sv_fn, dst_mat);
            }
            else if (key == '7')
            {
                writer.open("VideoTest1.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(dst_mat.cols, dst_mat.rows));//(param.Width, param.Height)
                rec = true;
                zoom = 1000;
                all_focus = 0;
            }
            else if (key == '3')
            {
                con_flag = true;
                blocksize = 1;                
            }

            dst_mat.release();

        }
        delete GridDepth;
        
    }
    catch (const std::exception& excp)
    {
        std::cerr
            << "ERROR: "
            << excp.what()
            << std::endl;
    }
    return 0;
}



