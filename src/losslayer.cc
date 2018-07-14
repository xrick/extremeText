/**
 * Copyright (c) 2018 by Marek Wydmuch, Robert Istvan Busa-Fekete
 * All rights reserved.
 */

#include "losslayer.h"
#include "model.h"

#include "plt.h"
#include "ensemble.h"


namespace fasttext {


std::shared_ptr<LossLayer> lossLayerFactory(std::shared_ptr<Args> args, loss_name loss){
    if (loss == loss_name::plt)
        return std::static_pointer_cast<LossLayer>(std::make_shared<PLT>(args));

    return nullptr;

    // First we need to wrap HSM as LossLayer
    //std::cerr << "Unknown loss type!\n";
    //exit(1);
}

std::shared_ptr<LossLayer> lossLayerFactory(std::shared_ptr<Args> args){
    if(args->ensemble > 1)
        return std::static_pointer_cast<LossLayer>(std::make_shared<Ensemble>(args));
    else
        return lossLayerFactory(args, args->loss);
}

LossLayer::LossLayer(std::shared_ptr<Args> args){
    args_ = args;
    multilabel = false;
    shift = 0;
}

LossLayer::~LossLayer(){

}

// Move label selections to this functions;
real LossLayer::loss(int32_t target, real lr, Model *model_){
    if(multilabel){
        std::vector <int32_t> target_ = {target};
        return loss(target_, lr, model_);
    }

    std::cerr << "Multiclass LossLayer doesn't have multilabel loss function!\n";
    return 0;
}

real LossLayer::loss(const std::vector <int32_t> &labels, real lr, Model *model_){
    return 0;
}

real LossLayer::loss(const std::vector <int32_t> &input, const std::vector <int32_t> &labels, real lr, Model *model_){
    return loss(labels, lr, model_);
}

real LossLayer::getLabelP(int32_t label, Vector &hidden, const Model *model_){
    return 0;
}

bool LossLayer::isMultilabel(){
    return multilabel;
}

void LossLayer::setShift(int64_t shift_){
    shift = shift_;
}

int64_t LossLayer::getShift(){
    return shift;
}

void LossLayer::printInfo(){

}

}
