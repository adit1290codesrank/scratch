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

void im2col_gpu(const float* data_im,int batch,int channels,int height,int width,int kernel_size,int pad,int stride,float* data_col)
{
    int height_col=(height+2*pad-kernel_size)/stride+1;
    int width_col=(width+2*pad-kernel_size)/stride+1;
    int num_kernels=channels*height_col*width_col;
    int fan_in=channels*kernel_size*kernel_size;
    int spatial=height_col*width_col;
    
    int threads=256;
    int blocks=(num_kernels+threads-1)/threads;
    for(int n=0;n<batch;n++)
    {
        const float* im_ptr=data_im+n*(channels*height*width);
        float* col_ptr=data_col+n*(fan_in*spatial);
        im2col_kernel<<<blocks,threads>>>(num_kernels,im_ptr,height,width,kernel_size,pad,stride,height_col,width_col,col_ptr);
    }
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

void col2im_gpu(const float* data_col,int batch,int channels,int height,int width,int kernel_size,int pad,int stride,float* data_im)
{
    int height_col=(height+2*pad-kernel_size)/stride+1;
    int width_col=(width+2*pad-kernel_size)/stride+1;
    int num_kernels=channels*height_col*width_col;
    int fan_in=channels*kernel_size*kernel_size;
    int spatial=height_col*width_col;
    
    int threads=256;
    int blocks=(num_kernels+threads-1)/threads;
    for(int n=0;n<batch;n++)
    {
        const float* col_ptr=data_col+n*(fan_in*spatial);
        float* im_ptr=data_im+n*(channels*height*width);
        col2im_kernel<<<blocks,threads>>>(num_kernels,col_ptr,height,width,kernel_size,pad,stride,height_col,width_col,im_ptr);
    }
}

__global__ void maxpool_forward_kernel(const int n_threads,const float* X,float* Y,float* mask,int channels,int height,int width,int hout,int wout,int k,int s)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<n_threads)
    {
        int pw=index%wout;
        int ph=(index/wout)%hout;
        int c=(index/(wout*hout))%channels;
        int n=index/(channels*wout*hout);

        int hstart=ph*s;
        int wstart=pw*s;
        int hend=hstart+k<height?hstart+k:height;
        int wend=wstart+k<width?wstart+k:width;

        int offset=(n*channels+c)*height*width;
        const float* X_slice=X+offset;

        float maxval=-1e20f;
        int maxidx=-1;

        for(int h=hstart;h<hend;h++)
        {
            for(int w=wstart;w<wend;w++)
            {
                int idx=h*width+w;
                if(X_slice[idx]>maxval)
                {
                    maxval=X_slice[idx];
                    maxidx=idx; // Remember the winning pixel!
                }
            }
        }
        Y[index]=maxval;
        mask[index]=(float)maxidx;
    }
}

void maxpool_forward_gpu(const float* X,float* Y,float* mask,int batch,int channels,int height,int width,int hout,int wout,int k,int s)
{
    int total=batch*channels*hout*wout;
    int threads=256;
    int blocks=(total+threads-1)/threads;
    maxpool_forward_kernel<<<blocks,threads>>>(total,X,Y,mask,channels,height,width,hout,wout,k,s);
}

__global__ void maxpool_backward_kernel(const int n_threads,const float* dY,const float* mask,float* dX,int channels,int height,int width,int hout,int wout)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<n_threads)
    {
        int c=(index/(wout*hout))%channels;
        int n=index/(channels*wout*hout);

        int maxidx=(int)mask[index]; // Pull the winning index back out
        int offset=(n*channels+c)*height*width;
        
        // Pass 100% of the gradient directly to the winner
        atomicAdd(&dX[offset+maxidx],dY[index]); 
    }
}

void maxpool_backward_gpu(const float* dY,const float* mask,float* dX,int batch,int channels,int height,int width,int hout,int wout)
{
    int total=batch*channels*hout*wout;
    int threads=256;
    int blocks=(total+threads-1)/threads;
    maxpool_backward_kernel<<<blocks,threads>>>(total,dY,mask,dX,channels,height,width,hout,wout);
}