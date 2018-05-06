/**
 * Copyright (c) 2018 by Marek Wydmuch, Robert Istvan Busa-Fekete
 * All rights reserved.
 */

#pragma once

#include <iostream>
#include <ostream>
#include <vector>
#include <memory>

#include "real.h"
#include "vector.h"
#include "args.h"
#include "dictionary.h"

namespace fasttext {

class Model;

class LossLayer {
public:
    LossLayer(std::shared_ptr<Args>);
    virtual ~LossLayer();

    bool isMultilabel();
    int64_t getShift();
    void setShift(int64_t);

    virtual int32_t getSize() = 0;
    virtual void setup(std::shared_ptr<Args>, std::shared_ptr<Dictionary>) = 0;

    virtual real loss(int32_t target, real lr, Model *model_);
    virtual real loss(const std::vector <int32_t> &labels, real lr, Model *model_);
    virtual real loss(const std::vector <int32_t> &input, const std::vector <int32_t> &labels, real lr, Model *model_);
    virtual void findKBest(int32_t top_k, std::vector <std::pair<real, int32_t>> &heap, Vector &hidden, const Model *model_) = 0;
    virtual real getLabelP(int32_t label, Vector &hidden, const Model *model_);

    virtual void save(std::ostream&) = 0;
    virtual void load(std::istream&) = 0;

    virtual void printInfo();

protected:
    std::shared_ptr<Args> args_;

    int64_t shift;
    bool multilabel;
};

std::shared_ptr<LossLayer> lossLayerFactory(std::shared_ptr<Args> args);
std::shared_ptr<LossLayer> lossLayerFactory(std::shared_ptr<Args> args, loss_name loss);

}
