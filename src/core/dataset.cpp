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

    std::vector<float> hX(num*r*c),hY(num*nc,0.0f);
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

void Dataset::load_cifar10(const std::string& path,Tensor& X,Tensor& Y,bool is_test)
{
    std::vector<std::string> files;
    if(is_test) files.push_back(path + "/test_batch.bin");
    else
    {
        files.push_back(path + "/data_batch_1.bin");
        files.push_back(path + "/data_batch_2.bin");
        files.push_back(path + "/data_batch_3.bin");
        files.push_back(path + "/data_batch_4.bin");
        files.push_back(path + "/data_batch_5.bin");
    }

    int num=files.size()*10000;
    std::vector<float> hX(num*3*32*32),hY(num*10,0.0f);

    float mean[3]={0.4914f, 0.4822f, 0.4465f},std[3]={0.2023f, 0.1994f, 0.2010f};

    int index=0;
    for(const auto& file:files)
    {
        std::ifstream f(file,std::ios::binary);
        if(!f.is_open())
        {
            std::cerr << "Failed to open CIFAR-10 file: " << file << std::endl;
            return;
        }

        for(int i=0;i<10000;i++)
        {
            uint8_t label;
            f.read((char*)&label,1);
            hY[index*10+label]=1.0f;

            uint8_t img[3072];
            f.read((char*)img,3072);

            for(int j=0;j<3;j++)
            {
                for(int k=0;k<1024;k++)
                {
                    float val=img[j*1024+k]/255.0f;
                    val=(val-mean[j])/std[j];
                    hX[index*3072+j*1024+k]=val;
                }
            }
            index++;
        }
        f.close();
    }

    std::cout << "Loaded " << num << " CIFAR-10 images." << std::endl;
    X=Tensor::zeros({num,3,32,32}),Y=Tensor::zeros({num,10});
    X.copy_from_host(hX.data());Y.copy_from_host(hY.data());
}