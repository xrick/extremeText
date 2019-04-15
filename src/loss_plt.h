/**
 * Copyright (c) 2018 by Marek Wydmuch, Róbert Busa-Fekete, Krzysztof Dembczyński
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

namespace fasttext {

class Model;

struct TreeNode{
  uint32_t index; //id of the base predictor
  int32_t label;
  TreeNode* parent; // pointer to the parent node
  std::vector<TreeNode*> children; // pointers to the children nodes

  // training
  uint32_t n_updates;
  uint32_t n_positive_updates;
};

struct NodeFreq{
  TreeNode* node;
  int64_t freq; // frequency

  bool operator<(const NodeFreq& r) const { return freq < r.freq; }
  bool operator>(const NodeFreq& r) const { return freq > r.freq; }
};

struct NodeProb{
  TreeNode* node;
  real prob; // probability

  bool operator<(const NodeProb& r) const { return prob < r.prob; }
  bool operator>(const NodeProb& r) const { return prob > r.prob; }
};

// For K-Means based trees

struct NodePartition{
  TreeNode* node;
  std::vector<Assignation>* partition;
};

class PLT: public LossLayer{
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
  real predictNode(TreeNode *n, Vector& hidden, const Model *model_);

  void buildCompletePLTree(int32_t);
  void buildHuffmanPLTree(const std::vector<int64_t>&);
  void buildKMeansPLTree(std::shared_ptr<Args>, std::shared_ptr<Dictionary>);
  void loadTreeStructure(std::string filename, std::shared_ptr<Dictionary>);

  TreeNode* createNode(TreeNode *parent = nullptr, int32_t label = -1);

 public:
  PLT(std::shared_ptr<Args> args);
  ~PLT();

  void setup(std::shared_ptr<Dictionary>, uint32_t seed);
  real loss(const std::vector<int32_t>& labels, real lr, Model *model_);
  NodeProb getNextBest(std::priority_queue<NodeProb, std::vector<NodeProb>, std::less<NodeProb>>& n_queue,
                        Vector& hidden, const Model *model_);
  void findKBest(int32_t top_k, real threshold, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, const Model *model_);
  real getLabelP(int32_t label, Vector &hidden, const Model *model_);

  int32_t getSize();

  void save(std::ostream&);
  void load(std::istream&);

  void printInfo();
};

}
