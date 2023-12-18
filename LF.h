#ifndef LF_INCLUDE_H
#define LF_INCLUDE_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
struct MLAParam
{
    double Matrix[4], InvMatrix[4], Origin[2], Rotation, Diameter;
    int Width, Height, Cols, Rows;
};

struct StereoParam
{
    int max_disp, block_size;
    int block_avg_thresh, sobel_thresh;
};

#endif // !LF_INCLUDE_H
