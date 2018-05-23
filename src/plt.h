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
  int32_t label;
  NodePLT* parent; // pointer to the parent node
  std::vector<NodePLT*> children; // pointers to the children nodes

  // training
  uint32_t n_updates;
  uint32_t n_positive_updates;
  //real minWeight;
  //real minLabel;

  // prediction
  float p; // probability

  bool operator < (const NodePLT& r) const { return p < r.p; }
  bool operator > (const NodePLT& r) const { return p > r.p; }
};

struct NodeFrequency{
  NodePLT* node;
  int64_t frequency;

  bool operator<(const NodeFrequency& r) const { return frequency < r.frequency; }
  bool operator>(const NodeFrequency& r) const { return frequency > r.frequency; }
};

class PLT: public LossLayer{
private:
    uint32_t k; // number of labels
    uint32_t t; // number of tree nodes
    bool separate_lr;
    bool prob_norm;
    bool neg_sample;

    uint64_t n_in_vis_count;
    uint64_t n_vis_count;
    uint64_t y_count;
    uint64_t x_count;

    NodePLT *tree_root;
    std::vector<NodePLT*> tree; // pointers to tree nodes
    std::unordered_map<int32_t, NodePLT*> tree_leaves; // leaves map

    real base_lr;
    real power_t;
    uint32_t *labels_nodes_map;
    uint32_t *nodes_labels_map;

    Model *model_;

    real learnNode(NodePLT *n, real label, real lr, real l2, Model *model_);
    real predictNode(NodePLT *n, Vector& hidden, const Model *model_);

    void buildCompletePLTree(int32_t);
    void buildHuffmanPLTree(const std::vector<int64_t>&);
    void loadTreeStructure(std::string filename);

    NodePLT* createNode(NodePLT *parent = nullptr, int32_t label = -1);

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
