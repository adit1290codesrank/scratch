#include "../../include/core/dataset.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>

int32_t swap_endian(int32_t val){return ((val>>24)&0xff)|((val<<8)&0xff0000)|((val>>8)&0xff00)|((val<<24)&0xff000000);}

void Dataset::load_mnist(const std::string& path,const std::string& labelp,Tensor& X,Tensor& Y)
{
    std::ifstream img(path,std::ios::binary),label(labelp,std::ios::binary);

    if(!img.is_open()||!label.is_open())
    {
        std::cerr<<"Failed to open MNIST files!"<<std::endl;
        return;
    }

    int32_t magic,num,r,c,magicl,numl;

    img.read((char*)&magic,4);
    img.read((char*)&num,4);
    img.read((char*)&r,4);
    img.read((char*)&c,4);

    label.read((char*)&magicl,4);
    label.read((char*)&numl,4);

    num=swap_endian(num);r=swap_endian(r);c=swap_endian(c);

    std::cout<<"Loading "<<num<<" MNIST images..."<<std::endl;

    std::vector<float> hX(num*r*c),hY(num*10,0.0f);
    std::vector<uint8_t> raw_img(num*r*c),raw_label(num);

    img.read((char*)raw_img.data(),raw_img.size());label.read((char*)raw_label.data(),raw_label.size());

    for(int i=0;i<num;i++)
    {
        for(int j=0;j<r*c;j++) hX[i*r*c+j]=raw_img[i*r*c+j]/255.0f;
        hY[i*10+raw_label[i]]=1.0f;
    }
    X=Tensor::zeros({num,1,r,c});Y=Tensor::zeros({num,10});
    X.copy_from_host(hX.data());Y.copy_from_host(hY.data());
}

void Dataset::load_emnist(const std::string& path,const std::string& labelp,Tensor& X,Tensor& Y,int nc)
{
    std::ifstream img(path,std::ios::binary),label(labelp,std::ios::binary);
    if(!img.is_open()||!label.is_open())
    {
        std::cerr<<"Failed to open MNIST files!"<<std::endl;
        return;
    }

    int32_t magic,num,r,c,magicl,numl;
    img.read((char*)&magic,4);
    img.read((char*)&num,4);
    img.read((char*)&r,4);
    img.read((char*)&c,4);

    label.read((char*)&magicl,4);
    label.read((char*)&numl,4);

    num=swap_endian(num);r=swap_endian(r);c=swap_endian(c);

    std::cout<<"Loading "<<num<<" EMNIST images..."<<std::endl;

    std::vector<float> hX(num*r*c),hY(num*10,0.0f);
    std::vector<uint8_t> raw_img(num*r*c),raw_label(num);

    img.read((char*)raw_img.data(),raw_img.size());label.read((char*)raw_label.data(),raw_label.size());

    for(int i=0;i<num;i++)
    {
        for(int j=0;j<r*c;j++) hX[i*r*c+j]=raw_img[i*r*c+j]/255.0f;
        hY[i*nc+raw_label[i]]=1.0f;
    }
    X=Tensor::zeros({num,1,r,c});Y=Tensor::zeros({num,nc});
    X.copy_from_host(hX.data());Y.copy_from_host(hY.data());
}