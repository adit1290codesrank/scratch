#include "../../../include/core/conv_ops.h"
#include <cuda_runtime.h>

__global__ void im2col_kernel(const int n,const float* data_im,const int height,const int width,const int kernel_size,const int pad,const int stride,const int height_col,const int width_col,float* data_col)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<n)
    {
        int w_out=index%width_col;
        int h_index=index/width_col;
        int h_out=h_index%height_col;
        int channel_in=h_index/height_col;
        int channel_out=channel_in*kernel_size*kernel_size;
        int h_in=h_out*stride-pad;
        int w_in=w_out*stride-pad;

        float* data_col_ptr=data_col+(channel_out*height_col+h_out)*width_col+w_out;
        const float* data_im_ptr=data_im+(channel_in*height+h_in)*width+w_in;

        for(int i=0;i<kernel_size;i++)
        {
            for(int j=0;j<kernel_size;j++)
            {
                int h=h_in+i;
                int w=w_in+j;
                *data_col_ptr=(h>=0&&w>=0&&h<height&&w<width)?data_im_ptr[i*width+j]:0.0f;
                data_col_ptr+=height_col*width_col;
            }
        }
    }
}

void im2col_gpu(const float* data_im,int channels,int height,int width,int kernel_size,int pad,int stride,float* data_col)
{
    int height_col=(height+2*pad-kernel_size)/stride+1;
    int width_col=(width+2*pad-kernel_size)/stride+1;
    int num_kernels=channels*height_col*width_col;
    
    int threads=256;
    int blocks=(num_kernels+threads-1)/threads;
    im2col_kernel<<<blocks,threads>>>(num_kernels,data_im,height,width,kernel_size,pad,stride,height_col,width_col,data_col);
}

__global__ void col2im_kernel(const int n,const float* data_col,const int height,const int width,const int kernel_size,const int pad,const int stride,const int height_col,const int width_col,float* data_im)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<n)
    {
        int w_out=index%width_col;
        int h_index=index/width_col;
        int h_out=h_index%height_col;
        int channel_in=h_index/height_col;
        int channel_out=channel_in*kernel_size*kernel_size;
        int h_in=h_out*stride-pad;
        int w_in=w_out*stride-pad;

        const float* data_col_ptr=data_col+(channel_out*height_col+h_out)*width_col+w_out;
        float* data_im_ptr=data_im+(channel_in*height+h_in)*width+w_in;

        for(int i=0;i<kernel_size;i++)
        {
            for(int j=0;j<kernel_size;j++)
            {
                int h=h_in+i;
                int w=w_in+j;
                if(h>=0&&w>=0&&h<height&&w<width) atomicAdd(&data_im_ptr[i*width+j],*data_col_ptr);
                data_col_ptr+=height_col*width_col;
            }
        }
    }
}

void col2im_gpu(const float* data_col,int channels,int height,int width,int kernel_size,int pad,int stride,float* data_im)
{
    int height_col=(height+2*pad-kernel_size)/stride+1;
    int width_col=(width+2*pad-kernel_size)/stride+1;
    int num_kernels=channels*height_col*width_col;
    
    int threads=256;
    int blocks=(num_kernels+threads-1)/threads;
    col2im_kernel<<<blocks,threads>>>(num_kernels,data_col,height,width,kernel_size,pad,stride,height_col,width_col,data_im);
}