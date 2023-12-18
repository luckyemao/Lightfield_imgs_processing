
#include "opencv2/opencv.hpp"
#include "HikCamera.h"
#include <MvCameraControl.h>
namespace anakin
{
    bool PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo)
    {
        if (NULL == pstMVDevInfo)
        {
            printf("The Pointer of pstMVDevInfo is NULL!\n");
            return false;
        }
        if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE)
        {
            int nIp1 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
            int nIp2 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
            int nIp3 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
            int nIp4 = (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);

            // ch:打印当前相机ip和用户自定义名字 | en:print current ip and user defined name
            printf("CurrentIp: %d.%d.%d.%d\n", nIp1, nIp2, nIp3, nIp4);
            printf("UserDefinedName: %s\n\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
        }
        else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE)
        {
            printf("UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
            printf("Serial Number: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chSerialNumber);
            printf("Device Number: %d\n\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.nDeviceNumber);
        }
        else
        {
            printf("Not support.\n");
        }

        return true;
    }

    GrabImage::GrabImage():GrabImage(0, cv::Mat())
    {
    }
    GrabImage::GrabImage(int MODE, cv::Mat img) : m_GrabMode(MODE), m_Image(img)
    {


        m_pImageData = std::shared_ptr<unsigned char>(new unsigned char[IMAGEBUFFERSIZE]);

        if (m_GrabMode != -1)
        {
            // ch:枚举设备 | en:Enum device
            MV_CC_DEVICE_INFO_LIST stDeviceList;
            memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
            int nRet = 0;
            nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
            if (MV_OK != nRet)
            {
                printf("Enum Devices fail! nRet [0x%x]\n", nRet);
                throw std::exception("Enum Devices fail!");
            }

            if (stDeviceList.nDeviceNum > 0)
            {
                for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++)
                {
                    printf("[device %d]:\n", i);
                    MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
                    if (NULL == pDeviceInfo)
                    {
                        break;
                    }
                    PrintDeviceInfo(pDeviceInfo);
                }
            }
            else
            {
                printf("Find No Devices!\n");
                throw std::exception("Find No Devices!");
            }

            printf("Please Input camera index(0-%d):", stDeviceList.nDeviceNum - 1);
            unsigned int nIndex = 0;
            scanf_s("%d", &nIndex);

            if (nIndex >= stDeviceList.nDeviceNum)
            {
                printf("Input error!\n");
                throw std::exception("Input error!");
            }


            // ch:选择设备并创建句柄 | en:Select device and create handle
            nRet = MV_CC_CreateHandleWithoutLog(&m_HikHandle, stDeviceList.pDeviceInfo[nIndex]);
            if (MV_OK != nRet)
            {
                printf("Create Handle fail! nRet [0x%x]\n", nRet);
                throw std::exception("Create Handle fail!");
            }

            // ch:打开设备 | en:Open device
            nRet = MV_CC_OpenDevice(m_HikHandle);
            if (MV_OK != nRet)
            {
                printf("Open Device fail! nRet [0x%x]\n", nRet);
                throw std::exception("Open Device fail!");
            }

            // ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if (stDeviceList.pDeviceInfo[nIndex]->nTLayerType == MV_GIGE_DEVICE)
            {
                int nPacketSize = MV_CC_GetOptimalPacketSize(m_HikHandle);
                if (nPacketSize > 0)
                {
                    nRet = MV_CC_SetIntValue(m_HikHandle, "GevSCPSPacketSize", nPacketSize);
                    if (nRet != MV_OK)
                    {
                        printf("Warning: Set Packet Size fail nRet [0x%x]!", nRet);
                    }
                }
                else
                {
                    printf("Warning: Get Packet Size fail nRet [0x%x]!", nPacketSize);
                }
            }

            // ch:设置触发模式为off | en:Set trigger mode as off
            nRet = MV_CC_SetEnumValue(m_HikHandle, "TriggerMode", 0);
            if (MV_OK != nRet)
            {
                printf("Set Trigger Mode fail! nRet [0x%x]\n", nRet);
                throw std::exception("Set Trigger Mode fail!");
            }

            // ch:开始取流 | en:Start grab image
            nRet = MV_CC_StartGrabbing(m_HikHandle);
            if (MV_OK != nRet)
            {
                printf("Start Grabbing fail! nRet [0x%x]\n", nRet);
                throw std::exception("Start Grabbing fail!");
            }
            
        }
    }

    GrabImage::~GrabImage()
    {
        if (m_GrabMode != -1)
        {
            // ch:停止取流 | en:Stop grab image
            int nRet = MV_CC_StopGrabbing(m_HikHandle);
            if (MV_OK != nRet)
            {
                printf("Stop Grabbing fail! nRet [0x%x]\n", nRet);
                throw std::exception("Stop Grabbing fail!");
            }

            // ch:关闭设备 | Close device
            nRet = MV_CC_CloseDevice(m_HikHandle);
            if (MV_OK != nRet)
            {
                printf("ClosDevice fail! nRet [0x%x]\n", nRet);
                throw std::exception("ClosDevice fail!");
            }

            // ch:销毁句柄 | Destroy handle
            nRet = MV_CC_DestroyHandle(m_HikHandle);
            if (MV_OK != nRet)
            {
                printf("Destroy Handle fail! nRet [0x%x]\n", nRet);
                throw std::exception("Destroy Handle fail!");
            }
        }
    }

    cv::Mat& GrabImage::operator>>(cv::Mat& img)
    {
        MV_FRAME_OUT frame = { 0 };
        if (m_GrabMode == -1)
        {
            m_Image.copyTo(img);
        }
        else
        {
            //MV_CC_GetImageForBGR(m_HikHandle, m_pImageData.get(), 300000000, &frame_info, 1000);
            int ret = MV_CC_GetImageBuffer(m_HikHandle, &frame, 100);
            if (ret == 0)
            {
                m_Image = cv::Mat(frame.stFrameInfo.nHeight, frame.stFrameInfo.nWidth, CV_8UC1, frame.pBufAddr);
                cv::cvtColor(m_Image, img, cv::COLOR_BayerBG2BGR);
            }
            MV_CC_FreeImageBuffer(m_HikHandle, &frame);
        }
        return img;
    }
}