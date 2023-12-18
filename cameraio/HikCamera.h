#ifndef HIKCAMERA_INCLUDE_H
#define HIKCAMERA_INCLUDE_H
#include <opencv2/core/types.hpp>
namespace anakin
{
    class GrabImage
    {
    public:
        GrabImage();
        GrabImage(int MODE, cv::Mat img);
        ~GrabImage();
        cv::Mat& operator>>(cv::Mat& img);
    private:
        cv::Mat m_Image;
        int m_GrabMode;
        void* m_HikHandle;
        const int IMAGEBUFFERSIZE = 300000000;
        std::shared_ptr<unsigned char> m_pImageData;
    };
}

#endif // !HIKCAMERA_INCLUDE_H
