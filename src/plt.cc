/**
 * Copyright (c) 2018 by Marek Wydmuch, Robert Istvan Busa-Fekete
 * All rights reserved.
 */

#include <iostream>
#include <fstream>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <vector>
#include <queue>
#include <list>
#include <chrono>
#include <random>

#include "plt.h"
#include "model.h"

namespace fasttext {

bool compare_node_ptr_func(const NodePLT* l, const NodePLT* r) { return (*l < *r); }

struct compare_node_ptr_functor{
    bool operator()(const NodePLT* l, const NodePLT* r) const { return (*l < *r); }
};

bool compare_node_freq_ptr_func(const NodeFrequency* l, const NodeFrequency* r) { return (*l > *r); }

struct compare_node_freq_ptr_functor{
    bool operator()(const NodeFrequency* l, const NodeFrequency* r) const { return (*l > *r); }
};


PLT::PLT(std::shared_ptr<Args> args) : LossLayer(args){
    power_t = 0.5;
    base_lr = 1;
    separate_lr = false;
    prob_norm = true;
    neg_sample = 0;
    multilabel = true;

    n_in_vis_count = 0;
    n_vis_count = 0;
    y_count = 0;
    x_count = 0;
}

PLT::~PLT() {
    for(size_t i = 0; i < tree.size(); ++i){
        delete tree[i];
    }
}

void PLT::buildHuffmanPLTree(const std::vector<int64_t>& freq){
    std::cout << "  Building PLT with Huffman tree ...\n";

    k = freq.size();
    t = 2 * k - 1; // size of the tree

    std::priority_queue<NodeFrequency*, std::vector<NodeFrequency*>, compare_node_freq_ptr_functor> freqheap;
    for(int i = 0; i < k; ++i) {
        NodePLT *n = createNode(nullptr, i);
        NodeFrequency* f = new NodeFrequency();
        *f = {n, freq[i]};
        freqheap.push(f);

        //std::cout << "Leaf: " << n->label << ", Node: " << n->n << ", Freq: " << freq[i] << "\n";
    }

    while (true) {
        std::vector<NodeFrequency *> toMerge;
        for (int a = 0; a < args_->arity; ++a) {
            NodeFrequency *tmp = freqheap.top();
            freqheap.pop();
            toMerge.push_back(tmp);
            if (freqheap.empty()) break;
        }

        NodePLT *parent = createNode();

        int64_t aggregatedFrequency = 0;
        for (NodeFrequency *e : toMerge) {
            e->node->parent = parent;
            parent->children.push_back(e->node);
            aggregatedFrequency += e->frequency;
            delete e;
        }

        if (freqheap.empty()) {
            tree_root = parent;
            tree_root->parent = nullptr;
            break;
        }

        NodeFrequency *tup = new NodeFrequency();
        *tup = {parent, aggregatedFrequency};
        freqheap.push(tup);
    }

    t = tree.size();
    std::cout << "    Nodes: " << tree.size() << ", leaves: " << tree_leaves.size() << ", arity: " << args_->arity << "\n";
}

void PLT::buildCompletePLTree(int32_t k_) {
  std::cout << "  Building PLT with complete tree ...\n";

  // Build complete tree

  std::default_random_engine rng(time(0) * shift);
  k = k_;
  t = static_cast<int>(ceil(static_cast<double>(args_->arity * k - 1) / (args_->arity - 1)));
  uint32_t ti = t - k;

  std::vector<int32_t> labels_order;
  if (args_->randomTree){
    for (auto i = 0; i < k; ++i)
      labels_order.push_back(i);
    std::shuffle(labels_order.begin(), labels_order.end(), rng);
  }

  for(size_t i = 0; i < t; ++i){
    NodePLT *n = createNode();

    if(i >= ti){
      if(args_->randomTree) n->label = labels_order[i - ti];
      else n->label = i - ti;
      tree_leaves.insert(std::make_pair(n->label, n));
    }

    if(i > 0){
      n->parent = tree[static_cast<int>(floor(static_cast<float>(n->n - 1) / args_->arity))];
      n->parent->children.push_back(n);
    }
  }

  tree_root = tree[0];
  tree_root->parent = nullptr;

  std::cout << "   Nodes: " << tree.size() << ", leaves: " << tree_leaves.size() << ", arity: " << args_->arity << "\n";
}

void PLT::loadTreeStructure(std::string filename){
    std::cout << "Loading PLT structure from file ...\n";
    std::ifstream treefile(filename);

    treefile >> k >> t;

    for (auto i = 0; i < t; ++i)
        NodePLT *n = createNode();
    tree_root = tree[0];

    for (auto i = 0; i < t - 1; ++i) {
        int parent, child, label;
        treefile >> parent >> child >> label;

        if(parent == -1){
            tree_root = tree[child];
            --i;
            continue;
        }

        NodePLT *parentN = tree[parent];
        NodePLT *childN = tree[child];
        parentN->children.push_back(childN);
        childN->parent = parentN;

        if(label >= 0){
            childN->label = label;
            tree_leaves.insert(std::make_pair(childN->label, childN));
        }
    }
    treefile.close();

    std::cout << "  Nodes: " << tree.size() << ", leaves: " << tree_leaves.size() << "\n";
    assert(tree.size() == t);
    assert(tree_leaves.size() == k);
}

real PLT::learnNode(NodePLT *n, real label, real lr, real l2, Model *model_){
    if(n->label < 0) ++n_in_vis_count;
    ++n_vis_count;
    ++n->n_updates;

    //real score = model_->sigmoid(model_->wo_->dotRow(model_->hidden_, n->n));

    real score = model_->wo_->dotRow(model_->hidden_, shift + n->n);
    if(score > MAX_SIGMOID) score = MAX_SIGMOID;
    else if(score < -MAX_SIGMOID) score = -MAX_SIGMOID;
    score = model_->sigmoid(score);

    double lambda = 0.0001;
    double gamma = 0.001;
    double lr_tmp = gamma /(1.0 + gamma * lambda * n->n_updates);
    //double lr_tmp = lr;
    //real alpha = lr * (label - score);
    real diff = (label - score);
    //model_->updateGrad(shift + n->n, alpha);
    //model_->grad_.addRow(*model_->wo_, shift + n->n, (lr_tmp * diff) / args_->nbase);//
    model_->grad_.addRowL2(*model_->wo_, shift + n->n, lr_tmp, diff / args_->nbase, 0.0001);//
    //model_->wo_->addRow(model_->hidden_, shift + n->n, alpha);
    //model_->wo_->addRowL1(model_->hidden_, shift + n->n, alpha, l1);
    model_->wo_->addRowL2(model_->hidden_, shift + n->n, lr_tmp, diff, l2);

    if (label) {
        ++n->n_positive_updates;
        return -log(score);
    } else {
        return -log(1.0 - score);
    }
}

real PLT::predictNode(NodePLT *n, Vector& hidden, const Model *model_){
    if(n->n_updates == 0 || n->n_positive_updates == 0) return 0;
    else if(n->n_positive_updates == n->n_updates) return 1;
    else return model_->sigmoid(model_->wo_->dotRow(hidden, shift + n->n));
}

NodePLT* PLT::createNode(NodePLT *parent, int32_t label){
    NodePLT *n = new NodePLT();
    n->n = tree.size();
    n->label = label;
    n->parent = parent;
    n->n_updates = 0;
    n->n_positive_updates = 0;

    tree.push_back(n);
    if(label >= 0) tree_leaves[n->label] = n;
    if(parent != nullptr) parent->children.push_back(n);
    return n;
}


// public
//----------------------------------------------------------------------------------------------------------------------

real PLT::loss(const std::vector<int32_t>& labels, real lr, Model *model_) {

    double l2 = model_->args_->l2;

    real loss = 0.0;



    std::unordered_set<NodePLT*> n_positive; // positive nodes
    std::unordered_set<NodePLT*> n_negative; // negative nodes

    if (labels.size() > 0) {
        for (uint32_t i = 0; i < labels.size(); ++i) {
            NodePLT *n = tree_leaves[labels[i]];
            n_positive.insert(n);
            while (n->parent) {
                n = n->parent;
                n_positive.insert(n);
            }
        }

        std::queue<NodePLT*> n_queue; // nodes queue
        n_queue.push(tree_root); // push root

        while(!n_queue.empty()) {
            NodePLT* n = n_queue.front(); // current node index
            n_queue.pop();

            if (n->label < 0) {
                for(auto child : n->children) {
                    if (n_positive.count(child)) n_queue.push(child);
                    else n_negative.insert(child);
                }
            }
        }
    }
    else
        n_negative.insert(tree_root);

    real label = 1.0;
    for (auto &n : n_positive){
        loss += learnNode(n, label, lr, l2, model_);
    }

    label = 0.0;
    for (auto &n : n_negative){
        loss += learnNode(n, label, lr, l2, model_);
    }

    //std::cout << "    Loss: " << loss << ", Loss sum: " << model_->loss_ << "\n";
    y_count += labels.size();
    ++x_count;
    return loss;
}


void PLT::findKBest(int32_t top_k, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, const Model *model_) {

    std::vector<NodePLT*> best_labels, found_leaves;
    std::priority_queue<NodePLT*, std::vector<NodePLT*>, compare_node_ptr_functor> n_queue;

    tree_root->p = predictNode(tree_root, hidden, model_);
    n_queue.push(tree_root);

    while (!n_queue.empty()) {
        NodePLT *n = n_queue.top(); // current node
        n_queue.pop();

        float cp = n->p;

        if(!prob_norm) {
            if (n->label < 0) {
                float sumOfP = 0.0f;
                for (auto child : n->children) {
                    child->p = cp * predictNode(child, hidden, model_);
                    n_queue.push(child);
                }
            } else {
                heap.push_back(std::make_pair(n->p, n->label));
                if (heap.size() >= top_k)
                    break;
            }
        } else {
            if (n->label < 0) {
                float sumOfP = 0.0f;
                for (auto child : n->children) {
                    child->p = cp * predictNode(child, hidden, model_);
                    sumOfP += child->p;
                }
                if ((sumOfP < cp) && (sumOfP > 10e-6)) {
                    for (auto child : n->children) {
                        child->p = (child->p * cp) / sumOfP;
                    }
                }
                for (auto child : n->children) {
                    if (child->p > n_queue.top()->p * 0.01)
                        n_queue.push(child);
                }
            } else {
                heap.push_back(std::make_pair(n->p, n->label));
                if (heap.size() >= top_k)
                    break;
            }
        }
    }
}

real PLT::getLabelP(int32_t label, Vector &hidden, const Model *model_){
    float p = 1.0;
    float parentProb = -1.0f;

    std::vector<NodePLT*> path;
    NodePLT *n = tree_leaves[label];

    if(!prob_norm){
        while(n != tree_root)
            p *= predictNode(n, hidden, model_);

        return p;
    }

    path.push_back(n);
    while (n->parent) {
        n = n->parent;
        path.push_back(n);
    }

    assert(tree_root == n);
    assert(tree_root == path.back());

    tree_root->p = predictNode(tree_root, hidden, model_);
    for(auto n = path.rbegin(); n != path.rend(); ++n){
        float cp = (*n)->p;

        if ((*n)->label < 0) {
            float sumOfP = 0.0f;
            for (auto child : (*n)->children) {
                child->p = cp * predictNode(child, hidden, model_);
                sumOfP += child->p;
            }
            if ((sumOfP < cp) && (sumOfP > 10e-6)) {
                for (auto child : (*n)->children) {
                    child->p = (child->p * cp) / sumOfP;
                }
            }
        }
    }

    assert(tree_root == n);

    return path.front()->p;
}

void PLT::setup(std::shared_ptr<Args> args, std::shared_ptr<Dictionary> dict){
    args_ = args;
    if(args_->treeStructure != ""){
        args_->treeType = tree_type_name::custom;
        loadTreeStructure(args_->treeStructure);
        return;
    }

    if (args_->treeType == tree_type_name::complete)
        buildCompletePLTree(dict->nlabels());

    else if (args_->treeType == tree_type_name::huffman){
        buildHuffmanPLTree(dict->getCounts(entry_type::label));
    }
}

int32_t PLT::getSize(){
    assert(t == tree.size());
    return tree.size();
}

void PLT::printInfo(){
    std::cout << "  Avg n vis: " << static_cast<float>(n_vis_count) / x_count << "\n";
    std::cout << "  Avg n in vis: " << static_cast<float>(n_in_vis_count) / x_count << "\n";
    std::cout << "  Avg y: " << static_cast<float>(y_count) / x_count << "\n";
}

void PLT::save(std::ostream& out){
    if(args_->verbose > 2)
        std::cerr << "Saving PLT model ...\n";
    out.write((char*) &shift, sizeof(shift));
    out.write((char*) &k, sizeof(int32_t));

    t = tree.size();
    out.write((char*) &t, sizeof(t));
    for(size_t i = 0; i < t; ++i) {
        NodePLT *n = tree[i];
        out.write((char*) &n->n, sizeof(n->n));
        out.write((char*) &n->label, sizeof(n->label));
        out.write((char*) &n->n_updates, sizeof(n->n_updates));
        out.write((char*) &n->n_positive_updates, sizeof(n->n_positive_updates));
    }

    uint32_t root_n = tree_root->n;
    out.write((char*) &root_n, sizeof(root_n));

    for(size_t i = 0; i < t; ++i) {
        NodePLT *n = tree[i];

        int parent_n;
        if(n->parent) parent_n = n->parent->n;
        else parent_n = -1;

        out.write((char*) &parent_n, sizeof(parent_n));
    }
}

void PLT::load(std::istream& in){
    if(args_->verbose > 2)
        std::cerr << "Loading PLT model ...\n";

    in.read((char*) &shift, sizeof(shift));
    in.read((char*) &k, sizeof(int32_t));

    in.read((char*) &t, sizeof(t));
    tree.resize(t);
    for(size_t i = 0; i < t; ++i) {
        NodePLT *n = new NodePLT();
        in.read((char*) &n->n, sizeof(n->n));
        in.read((char*) &n->label, sizeof(n->label));
        in.read((char*) &n->n_updates, sizeof(n->n_updates));
        in.read((char*) &n->n_positive_updates, sizeof(n->n_positive_updates));

        tree[i] = n;
        if (n->label >= 0) tree_leaves[n->label] = n;
    }

    uint32_t root_n;
    in.read((char*) &root_n, sizeof(root_n));
    tree_root = tree[root_n];

    for(size_t i = 0; i < t; ++i) {
        NodePLT *n = tree[i];

        int parent_n;
        in.read((char*) &parent_n, sizeof(parent_n));
        if(parent_n >= 0) {
            tree[parent_n]->children.push_back(n);
            n->parent = tree[parent_n];
        }
    }
}

}
