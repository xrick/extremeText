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

// Comperators for priority queues

PLT::PLT(std::shared_ptr<Args> args) : LossLayer(args){
    multilabel = true;

    // Stats
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

    std::priority_queue<NodeFreq, std::vector<NodeFreq>, std::less<NodeFreq>> freqheap;
    for(int i = 0; i < k; ++i) {
        NodePLT *n = createNode(nullptr, i);
        freqheap.push({n, freq[i]});

        //std::cout << "Leaf: " << n->label << ", Node: " << n->n << ", Freq: " << freq[i] << "\n";
    }

    while (true) {
        std::vector<NodeFreq> toMerge;
        for (int a = 0; a < args_->arity; ++a) {
            NodeFreq tmp = freqheap.top();
            freqheap.pop();
            toMerge.push_back(tmp);
            if (freqheap.empty()) break;
        }

        NodePLT *parent = createNode();

        int64_t aggregatedFrequency = 0;
        for (NodeFreq e : toMerge) {
            e.node->parent = parent;
            parent->children.push_back(e.node);
            aggregatedFrequency += e.freq;
        }

        if (freqheap.empty()) {
            tree_root = parent;
            tree_root->parent = nullptr;
            break;
        }

        freqheap.push({parent, aggregatedFrequency});
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
      n->parent = tree[static_cast<int>(floor(static_cast<float>(n->index - 1) / args_->arity))];
      n->parent->children.push_back(n);
    }
  }

  tree_root = tree[0];
  tree_root->parent = nullptr;

  std::cout << "    Nodes: " << tree.size() << ", leaves: " << tree_leaves.size() << ", arity: " << args_->arity << "\n";
}

void PLT::loadTreeStructure(std::string filename){
    std::cout << "  Loading PLT structure from file ...\n";
    std::ifstream treefile(filename);

    treefile >> k >> t;

    for (auto i = 0; i < t; ++i) NodePLT *n = createNode();
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

    std::cout << "    Nodes: " << tree.size() << ", leaves: " << tree_leaves.size() << "\n";
    assert(tree.size() == t);
    assert(tree_leaves.size() == k);
}

real PLT::learnNode(NodePLT *n, real label, real lr, real l2, Model *model_){

    //real score = model_->sigmoid(model_->wo_->dotRow(model_->hidden_, n->index));
    real score = model_->wo_->dotRow(model_->hidden_, shift + n->index);
    if(score > MAX_SIGMOID) score = MAX_SIGMOID;
    else if(score < -MAX_SIGMOID) score = -MAX_SIGMOID;
    score = model_->sigmoid(score);
    real diff = (label - score);

    // Original update
    /*
    real alpha = lr * (label - score);
    model_->grad_.addRow(*model_->wo_, shift + n->index, (lr * diff) / args_->nbase)
    model_->wo_->addRow(model_->hidden_, shift + n->index, alpha);
     */

    if(args_->fobos){
        model_->grad_.addRowL2Fobos(*model_->wo_, shift + n->index, lr, diff / args_->nbase, l2);
        model_->wo_->addRowL2Fobos(model_->hidden_, shift + n->index, lr, diff, l2);
    } else {
        model_->grad_.addRowL2(*model_->wo_, shift + n->index, lr, diff / args_->nbase, l2);
        model_->wo_->addRowL2(model_->hidden_, shift + n->index, lr, diff, l2);
    }

    if(n->label < 0) ++n_in_vis_count;
    ++n_vis_count;
    ++n->n_updates;

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
    else return model_->sigmoid(model_->wo_->dotRow(hidden, shift + n->index));
}

NodePLT* PLT::createNode(NodePLT *parent, int32_t label){
    NodePLT *n = new NodePLT();
    n->index = tree.size();
    n->label = label;
    n->parent = parent;
    n->n_updates = 0;
    n->n_positive_updates = 0;
    //n->minWeight = 0;

    tree.push_back(n);
    if(label >= 0) tree_leaves[n->label] = n;
    if(parent != nullptr) parent->children.push_back(n);
    return n;
}


// public
//----------------------------------------------------------------------------------------------------------------------

real PLT::loss(const std::vector<int32_t>& labels, real lr, Model *model_) {

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

    real loss = 0.0;
    double l2 = args_->l2;

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
    std::priority_queue<NodeProb, std::vector<NodeProb>, std::less<NodeProb>> n_queue;

    n_queue.push({tree_root, predictNode(tree_root, hidden, model_)});

    while (!n_queue.empty()) {
        NodeProb np = n_queue.top(); // current node
        n_queue.pop();

        if(!args_->probNorm) {
            if (np.node->label < 0) {
                for (auto& child : np.node->children)
                    n_queue.push({child, np.prob * predictNode(child, hidden, model_)});
            } else {
                heap.push_back({np.prob, np.node->label});
                if (heap.size() >= top_k)
                    break;
            }
        } else {
            if (np.node->label < 0) {
                float sumOfP = 0.0f;
                std::vector<NodeProb> normChildren;
                for (auto& child : np.node->children) {
                    real p = predictNode(child, hidden, model_);
                    normChildren.push_back({child, p});
                    sumOfP += p;
                }
                if (sumOfP < 1.0){ //&& (sumOfP > 10e-6)) {
                    for (auto& child : normChildren) {
                        child.prob = child.prob / sumOfP;
                    }
                }
                for (auto& child : normChildren){
                    child.prob *= np.prob;
                    n_queue.push(child);
                }
            } else {
                heap.push_back({np.prob, np.node->label});
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

    if(!args_->probNorm){
        while(n->parent) {
            p *= predictNode(n, hidden, model_);
            n = n->parent;
        }
        return p;
    }

    path.push_back(n);
    while (n->parent) {
        n = n->parent;
        path.push_back(n);
    }

    assert(tree_root == n);
    assert(tree_root == path.back());

    p = predictNode(tree_root, hidden, model_);
    for(auto n = path.rbegin(); n != path.rend(); ++n){
        if ((*n)->label < 0) {

            //TODO: rewrite
            /*
            for (auto child : (*n)->children) {
                normChildren.push_back({child, })
                child->p = cp * predictNode(child, hidden, model_);
                sumOfP += child->p;
            }
            if ((sumOfP < cp) //&& (sumOfP > 10e-6)) {
                for (auto child : (*n)->children) {
                    child->p = (child->p * cp) / sumOfP;
                }
            }
            float sumOfP = 0.0f;
             */
        }
    }

    return p;
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
        out.write((char*) &n->index, sizeof(n->index));
        out.write((char*) &n->label, sizeof(n->label));
        out.write((char*) &n->n_updates, sizeof(n->n_updates));
        out.write((char*) &n->n_positive_updates, sizeof(n->n_positive_updates));
    }

    uint32_t root_n = tree_root->index;
    out.write((char*) &root_n, sizeof(root_n));

    for(size_t i = 0; i < t; ++i) {
        NodePLT *n = tree[i];

        int parent_n;
        if(n->parent) parent_n = n->parent->index;
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
        in.read((char*) &n->index, sizeof(n->index));
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
