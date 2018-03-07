/**
 * Copyright (c) 2018 by Marek Wydmuch, Robert Istvan Busa-Fekete
 * All rights reserved.
 */

#pragma once

#include <iostream>
#include <ostream>
#include <vector>
#include <utility>
#include <memory>
#include <boost/unordered_map.hpp>
#include <queue>
#include <cstdlib>
#include <random>
#include <string>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <random>

#include "vector.h"
#include "real.h"
#include "dictionary.h"
#include "losslayer.h"

namespace fasttext {

class Model;

struct NodePLT{
    uint32_t n; //id of the base predictor
    uint32_t label;

    NodePLT* parent; // pointer to the parent node
    std::vector<NodePLT*> children; // pointers to the children nodes
    bool internal; // internal or leaf
    float t;
    float p; // prediction value
    bool operator < (const NodePLT& r) const { return p < r.p; }
};

class FreqTuple{
public:
    int64_t f;
    NodePLT* node;
public:
    FreqTuple(int64_t f_, NodePLT* node_){
        f=f_; node=node_;
    }
    int64_t getFrequency() const { return f;}
};

struct DereferenceCompareNode : public std::binary_function<FreqTuple*, FreqTuple*, bool>{
    bool operator()(const FreqTuple* lhs, const FreqTuple* rhs) const {
        return lhs->getFrequency() > rhs->getFrequency();
    }
};


class PLT: public LossLayer{
private:

    uint32_t k; // number of labels
    uint32_t t; // number of tree nodes
    uint32_t ti; // number of internal nodes
    uint32_t arity;
    bool separate_lr;
    bool prob_norm;
    bool neg_sample;
    bool sh_loos;

    int n_in_vis_count;
    int n_vis_count;
    int y_count;
    int x_count;

    NodePLT *tree_root;
    std::vector<NodePLT*> tree; // pointers to tree nodes
    std::unordered_map<uint32_t, NodePLT*> tree_leaves; // leaves map

    real base_lr;
    real power_t;
    uint32_t *labels_nodes_map;
    uint32_t *nodes_labels_map;

    Model *model_;

    real learnNode(NodePLT *n, real label, real lr, Model *model_);
    void buildCompletePLTree(int32_t);
    void buildHuffmanPLTree(const std::vector<int64_t>&);
    void loadTreeStructureFromPaths(std::string filename);
    void loadTreeStructure(std::string filename);
    void permLabels(std::vector<int64_t>&);

    void buildTree(Model *model_);
    void balancedKMeans(Matrix);

public:
    PLT(std::shared_ptr<Args> args);
    ~PLT();

    void setup(std::shared_ptr<Args>, std::shared_ptr<Dictionary>);
    real loss(const std::vector<int32_t>& labels, real lr, Model *model_);
    void findKBest(int32_t top_k, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, const Model *model_);
    real getLabelP(int32_t label, Vector &hidden, const Model *model_);

    int32_t getSize();

    void save(std::ostream&);
    void load(std::istream&);

    void printInfo();
};

}
