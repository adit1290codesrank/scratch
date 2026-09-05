#include <iostream>                                                                                                                                                                                                            
#include <fstream>                                                                                                                                                                                                             
#include <vector>                                                                                                                                                                                                              
#include <cmath>                                                                                                                                                                                                               
#include <string>                                                                                                                                                                                                              
#include "include/core/network.h"                                                                                                                                                                                              
#include "include/layer/conv2d.h"                                                                                                                                                                                              
#include "include/layer/gap.h"                                                                                                                                                                                                 
#include "include/layer/linear.h"                                                                                                                                                                                              
#include "include/layer/activation.h"                                                                                                                                                                                          
#include "include/layer/batchnorm.h"                                                                                                                                                                                           
#include "include/layer/res.h"                                                                                                                                                                                                 
#include "include/layer/dropout.h"                                                                                                                                                                                             
#include "include/core/dataset.h"                                                                                                                                                                                              
#include "include/core/memory.h"   
#include "include/layer/augment.h"
#include <filesystem>
#include <ctime>
#include <sstream>
#include <iomanip>

void save_pgm(const std::vector<float>& data,int c,int h,int w,const std::string& filename)
{
    int hout=256;
    int wout=256;

    std::vector<float> mean(h*w,0.0f);
    float max=-1e9f,min=1e9f;

    for(int j=0;j<h;j++)
    {
        for(int k=0;k<w;k++)
        {
            float sum=0;
            for(int i=0;i<c;i++)
            {
                sum+=data[i*h*w+j*w+k];
            }
            float val=sum/c;
            mean[j*w+k]=val;
            if(val>max) max=val;
            if(val<min) min=val;
        }
    }
    float r=max-min;
    if(r<1e-5f) r=1.0f;

    std::vector<unsigned char> img(hout*wout,0);

    for(int y=0;y<hout;y++)
    {
        for(int x=0;x<wout;x++)
        {
            float gx=((float)x/wout)*(w-1);
            float gy=((float)y/hout)*(h-1);

            int x1=(int)gx;
            int y1=(int)gy;
            int x2=std::min(x1+1,w-1);
            int y2=std::min(y1+1,h-1);

            float dx=gx-x1;
            float dy=gy-y1;

            float p11=mean[y1*w+x1];
            float p12=mean[y1*w+x2];
            float p21=mean[y2*w+x1];
            float p22=mean[y2*w+x2];

            float top=p11*(1.0f-dx)+p12*dx;
            float bot=p21*(1.0f-dx)+p22*dx;
            float val=top*(1.0f-dy)+bot*dy;

            unsigned char pixel=(unsigned char)(((val-min)/r)*255.0f);
            img[y*wout+x]=pixel;
        }
    }

    std::ofstream f(filename,std::ios::binary);
    f<<"P5\n"<<wout<<" "<<hout<<"\n255\n";
    f.write((char*)img.data(),img.size());
    f.close();
}

int main(int argc, char** argv)
{
    Network net;

    net.add_layer(new Augment(false, 0));

    net.add_layer(new Conv2D(3,32,3,1,1));
    net.add_layer(new BatchNorm(32));
    net.add_layer(new ReLU());

    Res* res1=new Res();
    res1->add_main(new Conv2D(32,32,3,1,1));
    res1->add_main(new BatchNorm(32));
    res1->add_main(new ReLU());
    res1->add_main(new Conv2D(32,32,3,1,1));
    res1->add_main(new BatchNorm(32));
    net.add_layer(res1);
    net.add_layer(new ReLU());

    Res* pres2=new Res();
    pres2->add_main(new Conv2D(32,64,3,2,1));
    pres2->add_main(new BatchNorm(64));
    pres2->add_main(new ReLU());
    pres2->add_main(new Conv2D(64,64,3,1,1));
    pres2->add_main(new BatchNorm(64));
    pres2->add_skip(new Conv2D(32,64,1,2,0));
    pres2->add_skip(new BatchNorm(64));
    net.add_layer(pres2);
    net.add_layer(new ReLU());

    Res* res2=new Res();
    res2->add_main(new Conv2D(64,64,3,1,1));
    res2->add_main(new BatchNorm(64));
    res2->add_main(new ReLU());
    res2->add_main(new Conv2D(64,64,3,1,1));
    res2->add_main(new BatchNorm(64));
    net.add_layer(res2);
    net.add_layer(new ReLU());

    Res* pres3=new Res();
    pres3->add_main(new Conv2D(64,128,3,2,1));
    pres3->add_main(new BatchNorm(128));
    pres3->add_main(new ReLU());
    pres3->add_main(new Conv2D(128,128,3,1,1));
    pres3->add_main(new BatchNorm(128));
    pres3->add_skip(new Conv2D(64,128,1,2,0));
    pres3->add_skip(new BatchNorm(128));
    net.add_layer(pres3);
    net.add_layer(new ReLU());

    Res* res3=new Res();
    res3->add_main(new Conv2D(128,128,3,1,1));
    res3->add_main(new BatchNorm(128));
    res3->add_main(new ReLU());
    res3->add_main(new Conv2D(128,128,3,1,1));
    res3->add_main(new BatchNorm(128));
    net.add_layer(res3);
    net.add_layer(new ReLU());

    Res* pres4=new Res();
    pres4->add_main(new Conv2D(128,256,3,2,1));
    pres4->add_main(new BatchNorm(256));
    pres4->add_main(new ReLU());
    pres4->add_main(new Conv2D(256,256,3,1,1));
    pres4->add_main(new BatchNorm(256));
    pres4->add_skip(new Conv2D(128,256,1,2,0));
    pres4->add_skip(new BatchNorm(256));
    net.add_layer(pres4);
    net.add_layer(new ReLU());

    Res* res4=new Res();
    res4->add_main(new Conv2D(256,256,3,1,1));
    res4->add_main(new BatchNorm(256));
    res4->add_main(new ReLU());
    res4->add_main(new Conv2D(256,256,3,1,1));
    res4->add_main(new BatchNorm(256));
    net.add_layer(res4);
    net.add_layer(new ReLU());

    net.add_layer(new GAP());
    net.add_layer(new Dropout(0.3f));
    net.add_layer(new Linear(256,10));
    net.add_layer(new Softmax());

    net.load_weights("weights/cifar10_50.bin");

    Tensor img({1,3,32,32});
    if(argc>1)
    {
        std::cout<<"Loading custom image from: "<<argv[1]<<std::endl;
        std::vector<float> h_img(3072);
        std::ifstream f(argv[1],std::ios::binary);
        f.read((char*)h_img.data(),3072*sizeof(float));
        f.close();
        img.copy_from_host(h_img.data());
    }
    else
    {
        Tensor X_test,Y_test;
        Dataset::load_cifar10("data/cifar-10-batches-bin",X_test,Y_test,true);
        img=X_test.slice(0,1);
    }

    std::time_t t=std::time(nullptr);
    std::tm* now=std::localtime(&t);
    std::stringstream ss;
    ss<<"run_" 
       <<(now->tm_year+1900)
       <<std::setw(2)<<std::setfill('0')<<(now->tm_mon+1)
       <<std::setw(2)<<std::setfill('0')<<now->tm_mday<<"_"
       <<std::setw(2)<<std::setfill('0')<<now->tm_hour
       <<std::setw(2)<<std::setfill('0')<<now->tm_min
       <<std::setw(2)<<std::setfill('0')<<now->tm_sec;
       
    std::string unique_folder="output/images/"+ss.str();
    std::filesystem::create_directories(unique_folder);

    std::cout<<"Pushing image theough network (saving to: "<<unique_folder<<")"<<std::endl;
    net.eval();

    Tensor curr=img;
    auto layers=net.get_layers();

    for(int i=0;i<layers.size();i++)
    {
        curr=layers[i]->forward(curr);
        if(curr.shape.size()==4 && curr.shape[1]>3)
        {
            int c=curr.shape[1],h=curr.shape[2],w=curr.shape[3];
            std::vector<float> hd(curr.total_elements());
            curr.copy_to_host(hd.data());

            std::string fname=unique_folder+"/vis_layer_"+std::to_string(i)+"_c"+std::to_string(c)+".pgm";
            save_pgm(hd,c,h,w,fname);
            std::cout<<"Saved Feature Map: "<<fname<<std::endl;
        }
    }
    
    float max_prob=-1e9f;
    int pred_class=-1;
    std::vector<float> out_h(10);
    curr.copy_to_host(out_h.data());
    for(int i=0;i<10;i++)
    {
        if(out_h[i]>max_prob)
        {
            max_prob=out_h[i];
            pred_class=i;
        }
    }
    
    std::vector<std::string> classes={"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};
    std::cout<<"PREDICTION: "<<classes[pred_class]<<std::endl;

    clear_memory_pool();
    std::cout<<"Done!"<<std::endl;
    return 0;
}
                                     
