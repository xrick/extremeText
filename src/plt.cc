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


PLT::PLT(std::shared_ptr<Args> args) : LossLayer(args){
    power_t = 0.5;
    base_lr = 1;
    separate_lr = false;
    prob_norm = true;
    neg_sample = 0;
    sh_loos = false;
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
    std::cout << "Building PLT with Huffman tree ...\n";

    k = freq.size();
    t = 2 * k - 1; // size of the tree

    ti = k - 1;
    std::priority_queue<FreqTuple*, std::vector<FreqTuple*>, DereferenceCompareNode> freqheap;
    for(int i=0; i<k; i++) {
        NodePLT *n = new NodePLT();
        n->n = i;
        n->t = 0;
        n->internal = false;
        n->label=i;
        tree_leaves.insert(std::make_pair(n->label, n));
        tree.push_back(n);

        FreqTuple* f = new FreqTuple(freq[i], n);
        freqheap.push(f);

        //std::cout << "Leaf: " << n->label << ", Node: " << n->n << ", Freq: " << freq[i] << "\n";
    }
    int i = 0;
    while (1) {
        std::vector<FreqTuple*> toMerge;
        for(int a = 0; a < args_->arity; ++a){
            FreqTuple* tmp = freqheap.top();
            freqheap.pop();
            toMerge.push_back(tmp);

            if (freqheap.empty()) break;
        }

        NodePLT* parent = new NodePLT();
        parent->n = k + i;
        ++i;
        parent->t = 0;
        parent->internal = true;

        int64_t aggregatedFrequency = 0;
        for( FreqTuple* e : toMerge){
            e->node->parent = parent;
            parent->children.push_back(e->node);
            aggregatedFrequency += e->getFrequency();
        }

        tree.push_back(parent);

        if (freqheap.empty()) {
            tree_root = parent;
            tree_root->parent = nullptr;
            break;
        }

        FreqTuple* tup = new FreqTuple(aggregatedFrequency,parent);
        freqheap.push(tup);
    }

    std::cout << "  Nodes: " << tree.size() << ", leaves: " << tree_leaves.size() << ", arity: " << args_->arity << "\n";
}

void PLT::loadTreeStructureFromPaths(std::string filename){
    std::cout << "Loading PLT structure from file ...\n";
    std::ifstream treefile(filename);

    treefile >> k >> t;

    for (auto i = 0; i < t; ++i) {
        NodePLT *n = new NodePLT();
        n->n = i;
        n->t = 0;
        n->internal = true;
        n->parent = nullptr;
        tree.push_back(n);
    }

    for (auto i = 0; i < k + 1; ++i) {
        int label, path_size, node;
        treefile >> label >> path_size;

        NodePLT *n = tree[label];
        n->label = label;
        n->internal = false;
        tree_leaves.insert(std::make_pair(n->label, n));

        for(auto j = 0; j < path_size; ++j) {
            treefile >> node;
            if(!n->parent) {
                n->parent = tree[k + node];
                n->parent->children.push_back(n);
                if(j == path_size - 1) tree_root = n->parent;
                n = n->parent;
            }
        }
    }
    treefile.close();

    std::cout << "  Nodes: " << tree.size() << ", leaves: " << tree_leaves.size() << "\n";
}

void PLT::loadTreeStructure(std::string filename){
    std::cout << "Loading PLT structure from file ...\n";
    std::ifstream treefile(filename);

    treefile >> k >> t;

    for (auto i = 0; i < t; ++i) {
        NodePLT *n = new NodePLT();
        n->n = i;
        n->t = 0;
        n->internal = true;
        n->parent = nullptr;
        tree.push_back(n);
    }
    tree_root = tree[0];

    for (auto i = 0; i < t - 1; ++i) {
        int parent, child, label;
        treefile >> parent >> child >> label;

//        if(parent == -1){
//            tree_root = tree[child];
//            continue;
//        }

        NodePLT *parentN = tree[parent];
        NodePLT *childN = tree[child];
        parentN->children.push_back(childN);
        childN->parent = parentN;

        if(label >= 0){
            childN->internal = false;
            childN->label = label;
            tree_leaves.insert(std::make_pair(childN->label, childN));
        }
    }
    treefile.close();

    std::cout << "  Nodes: " << tree.size() << ", leaves: " << tree_leaves.size() << "\n";
    assert(tree.size() == t);
    assert(tree_leaves.size() == k);
}

void PLT::buildCompletePLTree(int32_t k_) {
    std::cout << "Building PLT with complete tree ...\n";

    std::default_random_engine rng(time(0) * shift);

    k = k_;

    // Build complete tree
    if (args_->arity > 2) {
        double a = pow(args_->arity, floor(log(k) / log(args_->arity)));
        double b = k - a;
        double c = ceil(b / (args_->arity - 1.0));
        double d = (args_->arity * a - 1.0) / (args_->arity - 1.0);
        double e = k - (a - c);
        t = static_cast<uint32_t>(e + d);
    } else {
        args_->arity = 2;
        t = 2 * k - 1;
    }

    ti = t - k;

    std::vector<int32_t> labels_order;
    if (args_->randomTree){
        for (auto i = 0; i < k; ++i){
            labels_order.push_back(i);
        }

        std::random_shuffle(labels_order.begin(), labels_order.end());
    }

    for(size_t i = 0; i < t; ++i){
        NodePLT *n = new NodePLT();
        n->n = i;
        n->t = 0;
        if(i < ti) n->internal = true;
        else{
            n->internal = false;
            if(args_->randomTree) n->label = labels_order[i - ti];
            else n->label = i - ti;
            tree_leaves.insert(std::make_pair(n->label, n));
        }
        if(i > 0){
            n->parent = tree[static_cast<int>(floor(static_cast<float>(n->n - 1) / args_->arity))];
            n->parent->children.push_back(n);
        }
        tree.push_back(n);
    }

    tree_root = tree[0];
    tree_root->parent = nullptr;

    std::cout << "  Nodes: " << tree.size() << ", leaves: " << tree_leaves.size() << ", arity: " << args_->arity << "\n";
}

real PLT::learnNode(NodePLT *n, real label, real lr, Model *model_){
    if(n->internal) ++n_in_vis_count;
    ++n_vis_count;

    //real score = model_->sigmoid(model_->wo_->dotRow(model_->hidden_, n->n));

    real score = model_->wo_->dotRow(model_->hidden_, shift + n->n);
    if(score > 8) score = 8;
    else if(score < -8) score = -8;
    score = model_->sigmoid(score);

    real alpha = lr * (label - score);
    //model_->updateGrad(shift + n->n, alpha);
    model_->grad_.addRow(*model_->wo_, shift + n->n, alpha);
    model_->wo_->addRow(model_->hidden_, n->n, alpha);
    //model_->wo_->addRowL1(model_->hidden_, shift + n->n, alpha, l1);

    if (label) {
        return -log(score);
    } else {
        return -log(1.0 - score);
    }
}

void PLT::permLabels(std::vector<int64_t>& perm){
    for( int i=0; i<perm.size(); i++ )
        tree_leaves[i]->label = perm[i];
}


// public
//----------------------------------------------------------------------------------------------------------------------

real PLT::loss(const std::vector<int32_t>& labels, real lr, Model *model_) {
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

            if (n->internal) {
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
        loss += learnNode(n, label, lr, model_);
    }

    label = 0.0;
    for (auto &n : n_negative){
        loss += learnNode(n, label, lr, model_);
    }

    //std::cout << "    Loss: " << loss << ", Loss sum: " << model_->loss_ << "\n";
    y_count += labels.size();
    ++x_count;
    return loss;
}


void PLT::findKBest(int32_t top_k, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, const Model *model_) {

    std::vector<NodePLT*> best_labels, found_leaves;
    std::priority_queue<NodePLT*, std::vector<NodePLT*>, compare_node_ptr_functor> n_queue;

    tree_root->p = model_->sigmoid(model_->wo_->dotRow(hidden, shift + tree_root->n));
    n_queue.push(tree_root);

    while (!n_queue.empty()) {
        NodePLT *n = n_queue.top(); // current node
        n_queue.pop();

        float cp = n->p;

        if (n->internal) {
            float sumOfP = 0.0f;
            for (auto child : n->children) {
                child->p = cp * model_->sigmoid(model_->wo_->dotRow(hidden, shift + child->n));
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

real PLT::getLabelP(int32_t label, Vector &hidden, const Model *model_){
    float p = 1.0;
    float parentProb = -1.0f;

    std::vector<NodePLT*> path;
    NodePLT *n = tree_leaves[label];

//    if(!prob_norm){
//        while(n != tree_root){
//            p *= model_->sigmoid(model_->wo_->dotRow(hidden, shift + n->n));
//        }
//
//        return p;
//    }

    path.push_back(n);
    while (n->parent) {
        n = n->parent;
        path.push_back(n);
    }

    assert(tree_root == n);
    assert(tree_root == path.back());

    tree_root->p = model_->sigmoid(model_->wo_->dotRow(hidden, shift + tree_root->n));
    for(auto n = path.rbegin(); n != path.rend(); ++n){
        float cp = (*n)->p;

        if ((*n)->internal) {
            float sumOfP = 0.0f;
            for (auto child : (*n)->children) {
                child->p = cp * model_->sigmoid(model_->wo_->dotRow(hidden, shift + child->n));
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
        //loadTreeStructureFromPaths(args_->treeStructure);
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
        out.write((char*) &n->internal, sizeof(n->internal));
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
    for(size_t i = 0; i < t; ++i) {
        NodePLT *n = new NodePLT();
        in.read((char*) &n->n, sizeof(n->n));
        in.read((char*) &n->label, sizeof(n->label));
        in.read((char*) &n->internal, sizeof(n->internal));

        tree.push_back(n);
        if (!n->internal) tree_leaves[n->label] = n;
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
