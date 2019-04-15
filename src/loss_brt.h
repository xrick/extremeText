/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <iostream>
#include <ostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <string>

#include "vector.h"
#include "real.h"
#include "dictionary.h"
#include "losslayer.h"
#include "smatrix.h"
#include "kmeans.h"
#include "loss_plt.h"

namespace fasttext {

class Model;

class BRT: public LossLayer{
 private:
  uint32_t t; // number of tree nodes

  uint64_t n_in_vis_count;
  uint64_t n_vis_count;
  uint64_t y_count;
  uint64_t x_count;

  TreeNode *tree_root;
  std::vector<TreeNode*> tree; // pointers to tree nodes
  std::unordered_map<int32_t, TreeNode*> tree_labels; // labels map (nodes with labels)

  real learnNode(TreeNode *n, real label, real lr, real l2, Model *model_);
  real predictNode(TreeNode *n, Vector& hidden, const Model *model_, real threshold = 0.5);

  void buildCompleteBRTree(int32_t);
  void buildHuffmanBRTree(const std::vector<int64_t>&);
  void buildKMeansBRTree(std::shared_ptr<Args>, std::shared_ptr<Dictionary>);
  void loadTreeStructure(std::string filename, std::shared_ptr<Dictionary>);

  TreeNode* createNode(TreeNode *parent = nullptr, int32_t label = -1);

 public:
  BRT(std::shared_ptr<Args> args);
  ~BRT();

  void setup(std::shared_ptr<Dictionary>, uint32_t seed);
  real loss(const std::vector<int32_t>& labels, real lr, Model *model_);
  NodeProb getNext(std::queue<NodeProb>& n_queue, Vector& hidden, const Model *model_, real threshold);
  void findKBest(int32_t top_k, real threshold, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, const Model *model_);
  real getLabelP(int32_t label, Vector &hidden, const Model *model_);

  int32_t getSize();

  void save(std::ostream&);
  void load(std::istream&);

  void printInfo();
};

}
