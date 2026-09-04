#pragma once
#include "layer.h"
#include "../core/gap_ops.h"

class GAP:public Layer
{
    private:
        int n,c,h,w;

    public:
        GAP()=default;
        ~GAP() override=default;

        Tensor forward(const Tensor& X) override;
        Tensor backward(const Tensor& dY) override;
};

