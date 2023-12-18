#ifndef CALIBRATION_INCLUDE_H
#define CALIBRATION_INCLUDE_H

#include <opencv2/opencv.hpp>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace anakin
{
    template<typename T>
    struct MicroLensUnit
    {
        T p;
        int row, col;
    };

    template <typename PType>
    void DetectCenters(cv::Mat& binMat, std::vector<PType>& centers, double dist_thresh1, double dist_thresh2)
    {
        centers.clear();
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierachy;

        cv::findContours(binMat, contours, hierachy, cv::CHAIN_APPROX_SIMPLE, cv::RETR_TREE);
        size_t sz = contours.size();
        //cv::RotatedRect r;
        for (size_t i = 0; i < sz; ++i)
        {
            if (hierachy[i][3] >= 0)
                continue;
            cv::RotatedRect r = cv::minAreaRect(contours[i]);
            if (r.size.width < dist_thresh1 || r.size.width > dist_thresh2 || r.size.height < dist_thresh1 || r.size.height > dist_thresh2)
                continue;
            centers.push_back(r.center);
        }
    }

    template <typename PType>
    std::vector<PType> CropBorder(std::vector<PType> centers, size_t w, size_t h, size_t left, size_t right, size_t top, size_t bottom)
    {
        for (auto i = centers.begin(); i != centers.end();)
            i = (i->x < left || i->x >= w - right || i->y < top || i->y >= h - bottom) ? centers.erase(i) : i + 1;
        return centers;
    }

    template <typename PType>
    int IsHexNeighbor(const PType& a, const PType& b, double estiname_dist, double dist_thresh)
    {
        static double cos60 = cos(60 * M_PI / 180), sin60 = sin(60 * M_PI / 180);
        static PType dir[6] = { PType(-estiname_dist, 0), 
            PType(-estiname_dist * cos60, -estiname_dist * sin60),
            PType(estiname_dist * cos60, -estiname_dist * sin60),
            PType(estiname_dist, 0),
            PType(estiname_dist * cos60, estiname_dist * sin60),
            PType(-estiname_dist * cos60, estiname_dist * sin60)};//��������֪ĳ��Բ��A������Բ��A������6��Բ��Բ�ĵ�����ֵ���ٰ�����ֵ��ʵ��ֵ��������
                                                                  //Ϊʲô��ôд������ֱ�������  
                                                                  // ���ֱ������� �������37 �ܱ�֤һ�������Բ���𣿿�������һ�������Ѿ�������  ����Բ��  �����Ҳ����Բ����
                                                                  //  ���ʵ��λ��������λ�����3������ �϶���������׼Բ���� 
                                                                
        
        for (int i = 0; i < 6; ++i)
        {
            if (cv::norm(a + dir[i] - b) < dist_thresh)
                return i;
        }
        return -1;
    }

    template <typename PType>
    double EstimateAverageDist(std::vector<PType> centers, double est_diameter, double thresh)
    {
        std::vector<double> dist_vec;
        for (auto& it : centers)
        {
            double dst_sum = 0;
            int dst_cnt = 0;
            for (auto& it1 : centers)
            {
                if (dst_cnt >= 6)
                    break;
                if (&it != &it1 && IsHexNeighbor(it, it1, est_diameter, thresh) != -1)
                {
                    dst_sum += cv::norm(it - it1);
                    dst_cnt++;
                }
            }
            if (dst_cnt != 0)
            {
                dist_vec.push_back(dst_sum / dst_cnt);
            }
        }
        //std::sort(dist_vec.begin(), dist_vec.end());
        return cv::sum(dist_vec)[0]/dist_vec.size();
    }

    template <typename PType>
    void DetectMLA(std::vector<PType> centers, double estimate_dist, double& angle, double& x, double& y)
    {

        angle = 0;
        x = 0;
        y = 0;
        if (centers.empty())
        {
            std::cerr << __FILE__ << "(" << __LINE__ << "): centers.empty()==true\n";
            return ;
        }

        struct Aperture
        {
            Aperture(PType p_, int row_, int col_) :p(p_), row(row_), col(col_) {}
            PType p;
            int row, col;
        };

        if (abs(estimate_dist) <= DBL_EPSILON)
        {
            std::cerr << __FILE__ << "(" << __LINE__ << "): abs(estimate_dist) <= DBL_EPSILON\n";
            return;
        }

        cv::Mat A, B;
        int items_num = 0;

        std::queue<Aperture> q;
        q.push(Aperture(centers[2500], 0, 0));//centers.back()  �����е�bug  ��Ҫһ������԰���ܱ���⵽
        centers.pop_back();//centers���������еĿ׾����ģ�һ��һ���ļ��ص�����queue�У�ÿ����һ����centers�����о͵���һ��
        std::cout << "centers.back:WWWWWWW" << centers[2500] << "\n";//centers.back()
        std::cout << "centers.size:WWWWWWW" << centers.size() << "\n";
        
        int minXIdx = 10000, minYIdx = 10000;
        while (!q.empty())
        {
            Aperture aper = q.front();
            q.pop();
            items_num++;
            minXIdx = std::min(minXIdx, aper.col);
            minYIdx = std::min(minYIdx, aper.row);
            //std::cout << aper.row << " " << aper.col << "\n";
            A.push_back(static_cast<double>(aper.col));//�����Ӧ�����еĹ�ʽ4-6  p64
            A.push_back(static_cast<double>(aper.row));
            A.push_back(1.0);
            B.push_back(aper.p.x);
            B.push_back(aper.p.y);
            static int diridx[6][2] = { {-1, 0}, {-1, -1}, {0, -1}, {1, 0}, {1, 1}, {0, 1} };
            for (auto it = centers.begin(); it != centers.end();)
            {
                int dir = IsHexNeighbor(aper.p, *it, estimate_dist, estimate_dist * 0.15);
                if (dir >= 0)
                {
                    q.push(Aperture(*it, aper.row + diridx[dir][1], aper.col + diridx[dir][0]));
                    it = centers.erase(it);//�ҵ�һ������ �ͱ�Ŵ���q������Ȼ��Ѹ����Ĵ�center����ɾ��  Ȼ���q�������һ��һ��������ȥcenter���Ҹ�����
                }
                else
                {
                    it++;
                }
            }
        }

        std::cout << "items_num:" << items_num << "\n";
        std::cout << "A.size:" << A.size << "\n";
        std::cout << "B.size:" << B.size << "\n";
        std::cout << "XIdx:" << minXIdx << "\n";
        std::cout << "YIdx:" << minYIdx << "\n";

        try
        {
            A = A.reshape(1, items_num);
            B = B.reshape(1, items_num);
            A.col(0) -= minXIdx;
            A.col(1) -= minYIdx;
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            std::cerr << items_num << "\n" << A.size << "\n" << B.size << "\n";
            return ;
        }

        
        
        
        //����A B��� 
        cv::Mat test_Pcenters = cv::imread("D:\\YEM\\work\\new\\practice\\LFHexArray\\my_field\\biaoding1008v3.bmp");
        cv::Mat test_Pcenters_view;
        int xx(0);
        std::string str_cols, str_rows;
        for (xx = 0; xx < 1000; xx++)
        {
            if (xx % 8 == 0)
            {
                cv::Point2d center_pint(B.at<double>(xx, 0), B.at<double>(xx, 1));
                cv::circle(test_Pcenters, center_pint, 37.377412 / 2, cv::Scalar(0, 255, 0), 3);
                //std::cout << "center_pint" << center_pint << "\n";
                str_cols = std::to_string((int)A.at<double>(xx, 0));
                str_rows = std::to_string((int)A.at<double>(xx, 1));
                str_cols += ",";
                str_cols += str_rows;
                //sprintf(str_cols, "%d", it[0]);
                //sprintf(str_rows, "%d", it[1]);
                cv::putText(test_Pcenters, str_cols, center_pint, cv::FONT_HERSHEY_SIMPLEX, 1, (200, 255, 155), 3, cv::LINE_AA);
            }
        }
        test_Pcenters.copyTo(test_Pcenters_view);
        cv::imwrite("�����׾���ţ�ԭʼ��С��.bmp", test_Pcenters_view);
        int ratio = test_Pcenters_view.rows / 900;
        cv::Size sz(test_Pcenters_view.cols / ratio, test_Pcenters_view.rows / ratio);
        cv::resize(test_Pcenters_view, test_Pcenters_view, sz);
        cv::imshow("ԭʼ���ģ�������center", test_Pcenters_view);
        std::cout << "���������  ���Կ�" << "\n";
        std::cout << "items_num:" << items_num << "\n";
        std::cout << "A.size:" << A.size << "\n";
        std::cout << "B.size:" << B.size << "\n";

        
        //std::cout << "AAAAA:" << A << "\n";
        //std::cout << "BBBBB:" << B << "\n";


        cv::Mat Matrix;
        cv::solve(A, B, Matrix, cv::DECOMP_SVD);

        std::cout  <<"�������ԭ����" << Matrix << "\n";

        while (Matrix.at<double>(2, 0) > 0 && Matrix.at<double>(2, 1) > 0)
        {
            if (Matrix.at<double>(2, 0) > 0)
            {
                Matrix.row(2) -= Matrix.row(0);
            }
        }

        std::cout << Matrix << "\n";
        angle = std::atan((Matrix.at<double>(0, 1) + Matrix.at<double>(1, 1)) / (Matrix.at<double>(0, 0) + Matrix.at<double>(1, 0))) - 60 * M_PI / 180;
       //������㷽��������sina��sin(a+pi 2/3) ��������������cosa��cos(a+pi 2/3) Ȼ���ȥ����֮pai  �������׼ .��Ϊ�����õ��ǺͲ����ʽ�õ�tan�Ƕ� �ٷ���Ƕ� ��ֱһ��
        //angle = (std::atan(Matrix.at<double>(0, 1) / Matrix.at<double>(0, 0)) + std::abs(std::atan(Matrix.at<double>(1, 0) / Matrix.at<double>(1, 1))) - M_PI / 6) / 2;
       

        //һ��Ҫע�������matrix�ͺ�����ʵ��matrix�Ǹ�ת�ù�ϵ  �����matrix����ʵM��ת��
        std::cout << "Matrix.at<double>(0, 0):" << Matrix.at<double>(0, 0) << "\n";
        std::cout << "Matrix.at<double>(0, 1):" << Matrix.at<double>(0, 1) << "\n";
        std::cout << "Matrix.at<double>(1, 0):" << Matrix.at<double>(1, 0) << "\n";
        std::cout << "Matrix.at<double>(1, 1):" << Matrix.at<double>(1, 1) << "\n";
        
        std::cout << "angle" << angle*180/ M_PI << "\n";

        x = Matrix.at<double>(2, 0);//�������һ���������   һ��Angel  һ��x һ��y  ����xy�Ǵ�������ԲȦ�����Ͻǵ�һ�� ��������Ȼ����ͼ���������Ͻ��Ǹ�
        y = Matrix.at<double>(2, 1);

    }
}

#endif // !CALIBRATION_INCLUDE_H
