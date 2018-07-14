/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "args.h"

#include <stdlib.h>

#include <iostream>
#include <stdexcept>

namespace fasttext {

Args::Args() {
  seed = time(0);
  train = false;
  lr = 0.05;
  dim = 100;
  ws = 5;
  epoch = 5;
  minCount = 5;
  minCountLabel = 0;
  neg = 0;
  wordNgrams = 1;
  loss = loss_name::ns;
  model = model_name::sg;
  bucket = 2000000;
  minn = 3;
  maxn = 6;
  thread = 12;
  lrUpdateRate = 100;
  t = 1e-4;
  label = "__label__";
  verbose = 5;

  // Save args
  saveOutput = false;
  saveVectors = false;
  saveDocuments = false;

  // Vectors init
  pretrainedVectors = "";
  freezeVectors = false;
  initZeros = false;

  // Features args
  wordsWeights = false;
  tfidf = false;
  labelsOrder = false;

  // Quantization args
  qout = false;
  retrain = false;
  qnorm = false;
  cutoff = 0;
  dsub = 2;

  // PLT args
  arity = 2;
  bias = 0;
  treeType = tree_type_name::complete;
  treeStructure = "";
  randomTree = false;
  probNorm = false;
  maxLeaves = 100;

  // KMeans
  kMeansEps = 0.0001;
  kMeansBalanced = true;

  // Update args
  l2 = 0;
  fobos = false;
  labelsWeights = false;

  // Bagging args
  bagging = 1.0;
  ensemble = 1;

}

std::string Args::lossToString(loss_name ln) const {
  switch (ln) {
    case loss_name::hs:
      return "hs";
    case loss_name::ns:
      return "ns";
    case loss_name::softmax:
      return "softmax";
    case loss_name::plt:
      return "plt";
  }
  return "Unknown loss!"; // should never happen
}

std::string Args::boolToString(bool b) const {
  if (b) {
    return "true";
  } else {
    return "false";
  }
}

std::string Args::modelToString(model_name mn) const {
  switch (mn) {
    case model_name::cbow:
      return "cbow";
    case model_name::sg:
      return "sg";
    case model_name::sup:
      return "sup";
  }
  return "Unknown model name!"; // should never happen
}

void Args::parseArgs(const std::vector<std::string>& args) {
  std::string command(args[1]);
  if (command == "supervised") {
    model = model_name::sup;
    loss = loss_name::softmax;
    minCount = 1;
    minn = 0;
    maxn = 0;
    lr = 0.1;
  } else if (command == "cbow") {
    model = model_name::cbow;
  }
  for (int ai = 2; ai < args.size(); ai += 2) {
    if (args[ai][0] != '-') {
      std::cerr << "Provided argument without a dash! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    try {
      if (args[ai] == "-h") {
        std::cerr << "Here is the help! Usage:" << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      } else if (args[ai] == "-input") {
        input = std::string(args.at(ai + 1));
      } else if (args[ai] == "-output") {
        output = std::string(args.at(ai + 1));
      } else if (args[ai] == "-lr") {
        lr = std::stof(args.at(ai + 1));
      } else if (args[ai] == "-lrUpdateRate") {
        lrUpdateRate = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-dim") {
        dim = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-ws") {
        ws = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-epoch") {
        epoch = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minCount") {
        minCount = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minCountLabel") {
        minCountLabel = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-neg") {
        neg = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-wordNgrams") {
        wordNgrams = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-loss") {
        if (args.at(ai + 1) == "hs") {
          loss = loss_name::hs;
        } else if (args.at(ai + 1) == "ns") {
          loss = loss_name::ns;
          if(neg == 0) neg = 5;
        } else if (args.at(ai + 1) == "softmax") {
          loss = loss_name::softmax;
        } else if (args.at(ai + 1) == "plt") {
          loss = loss_name::plt;
        } else {
          std::cerr << "Unknown loss: " << args.at(ai + 1) << std::endl;
          printHelp();
          exit(EXIT_FAILURE);
        }
      } else if (args[ai] == "-bucket") {
        bucket = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-minn") {
        minn = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-maxn") {
        maxn = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-thread") {
        thread = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-t") {
        t = std::stof(args.at(ai + 1));
      } else if (args[ai] == "-label") {
        label = std::string(args.at(ai + 1));
      } else if (args[ai] == "-verbose") {
        verbose = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-pretrainedVectors") {
        pretrainedVectors = std::string(args.at(ai + 1));
      } else if (args[ai] == "-saveOutput") {
        saveOutput = true;
        ai--;
      } else if (args[ai] == "-saveVectors") {
        saveVectors = true;
        ai--;
      } else if (args[ai] == "-saveDocuments") {
        saveDocuments = true;
        ai--;
      } else if (args[ai] == "-freezeVectors") {
        freezeVectors = true;
        ai--;
      } else if (args[ai] == "-initZeros") {
        initZeros = true;
        ai--;
      } else if (args[ai] == "-wordsWeights") {
        wordsWeights = true;
        ai--;
      } else if (args[ai] == "-tfidf") {
        tfidf = true;
        ai--;
      } else if (args[ai] == "-unitNorm") {
        unitNorm = true;
        ai--;
      } else if (args[ai] == "-bias") {
        bias = 1;
        ai--;
      } else if (args[ai] == "-labelsOrder") {
        labelsOrder = true;
        ai--;
      } else if (args[ai] == "-probNorm") {
        probNorm = true;
        ai--;

      // Quantization args
      } else if (args[ai] == "-qnorm") {
        qnorm = true;
        ai--;
      } else if (args[ai] == "-retrain") {
        retrain = true;
        ai--;
      } else if (args[ai] == "-qout") {
        qout = true;
        ai--;
      } else if (args[ai] == "-cutoff") {
        cutoff = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-dsub") {
        dsub = std::stoi(args.at(ai + 1));

      // PLT args
      } else if (args[ai] == "-arity") {
        arity = std::stoi(args.at(ai + 1));
      } else if (args[ai] == "-l2") {
        l2 = std::stof(args.at(ai + 1));
      } else if (args[ai] == "-fobos") {
        fobos = true;
        ai--;
      } else if (args[ai] == "-labelsWeights") {
        labelsWeights = true;
        ai--;
      } else if (args[ai] == "-treeStructure") {
        treeStructure = std::string(args.at(ai + 1));
      } else if (args[ai] == "-randomTree") {
        randomTree = true;
        ai--;
      } else if (args[ai] == "-treeType") {
        if (args[ai + 1] == "complete") {
          treeType = tree_type_name::complete;
        } else if (args[ai + 1] == "huffman") {
          treeType = tree_type_name::huffman;
        } else if (args[ai + 1] == "kmeans") {
          treeType = tree_type_name::kmeans;
        } else {
          std::cout << "Unknown tree: " << args[ai] << std::endl;
          printHelp();
          exit(EXIT_FAILURE);
        }

      // Bagging args
      } else if (args[ai] == "-bagging") {
        bagging = std::stof(args.at(ai + 1));
      } else if (args[ai] == "-ensemble") {
        ensemble = std::stoi(args.at(ai + 1));
      } else {
        std::cerr << "Unknown argument: " << args[ai] << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      }
    } catch (std::out_of_range) {
      std::cerr << args[ai] << " is missing an argument" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
  }
  if (input.empty() || output.empty()) {
    std::cerr << "Empty input or output path." << std::endl;
    printHelp();
    exit(EXIT_FAILURE);
  }
  if (wordNgrams <= 1 && maxn == 0) {
    bucket = 0;
  }
}

void Args::printHelp() {
  printBasicHelp();
  printDictionaryHelp();
  printTrainingHelp();
  printQuantizationHelp();
}


void Args::printBasicHelp() {
  std::cerr
    << "\nThe following arguments are mandatory:\n"
    << "  -input              training file path\n"
    << "  -output             output file path\n"
    << "\nThe following arguments are optional:\n"
    << "  -verbose            verbosity level [" << verbose << "]\n";
}

void Args::printDictionaryHelp() {
  std::cerr
    << "\nThe following arguments for the dictionary are optional:\n"
    << "  -minCount           minimal number of word occurences [" << minCount << "]\n"
    << "  -minCountLabel      minimal number of label occurences [" << minCountLabel << "]\n"
    << "  -wordNgrams         max length of word ngram [" << wordNgrams << "]\n"
    << "  -bucket             number of buckets [" << bucket << "]\n"
    << "  -minn               min length of char ngram [" << minn << "]\n"
    << "  -maxn               max length of char ngram [" << maxn << "]\n"
    << "  -t                  sampling threshold [" << t << "]\n"
    << "  -label              labels prefix [" << label << "]\n";
}

void Args::printTrainingHelp() {
  std::cerr
    << "\nThe following arguments for training are optional:\n"
    << "  -lr                 learning rate [" << lr << "]\n"
    << "  -lrUpdateRate       change the rate of updates for the learning rate [" << lrUpdateRate << "]\n"
    << "  -l2                 l2 regularization [" << l2 << "]\n"
    << "  -dim                size of word vectors [" << dim << "]\n"
    << "  -ws                 size of the context window [" << ws << "]\n"
    << "  -epoch              number of epochs [" << epoch << "]\n"
    << "  -neg                number of negatives sampled [" << neg << "]\n"
    << "  -loss               loss function {ns, hs, softmax} [" << lossToString(loss) << "]\n"
    << "  -thread             number of threads [" << thread << "]\n"
    << "  -pretrainedVectors  pretrained word vectors for supervised learning ["<< pretrainedVectors <<"]\n"
    << "  -wordsWeights       TODO"
    << "  -saveOutput         whether output params should be saved [" << boolToString(saveOutput) << "]\n";
}

void Args::printQuantizationHelp() {
  std::cerr
    << "\nThe following arguments for quantization are optional:\n"
    << "  -cutoff             number of words and ngrams to retain [" << cutoff << "]\n"
    << "  -retrain            whether embeddings are finetuned if a cutoff is applied [" << boolToString(retrain) << "]\n"
    << "  -qnorm              whether the norm is quantized separately [" << boolToString(qnorm) << "]\n"
    << "  -qout               whether the classifier is quantized [" << boolToString(qout) << "]\n"
    << "  -dsub               size of each sub-vector [" << dsub << "]\n";
}

void Args::printInfo(){
  std::cerr << "  Model: " << modelToString(model) << ", loss: " << lossToString(loss) << "\n  Features: ";
  if(model == model_name::sup){
      if(tfidf) std::cerr << "tf-idf\n";
      else if(wordsWeights) std::cerr << "word weights\n";
      else std::cerr << "bow\n";
  }
  if(ensemble > 1) std::cerr << "  Ensemble: " << ensemble << ", bagging ratio: " << bagging << "\n";
  std::cerr << "  Lr: " << lr << ", L2: " << l2 << ", dims: " << dim << ", epochs: " << epoch
            << ", buckets: " << bucket << ", neg: " << neg << "\n";
  //std::cerr << "  Fobos: " << fobos << ", prob. norm.: " << probNorm << "\n";
}

void Args::save(std::ostream& out) {
  out.write((char*) &(lr), sizeof(double));
  out.write((char*) &(dim), sizeof(int));
  out.write((char*) &(ws), sizeof(int));
  out.write((char*) &(epoch), sizeof(int));
  out.write((char*) &(minCount), sizeof(int));
  out.write((char*) &(neg), sizeof(int));
  out.write((char*) &(wordNgrams), sizeof(int));
  out.write((char*) &(loss), sizeof(loss_name));
  out.write((char*) &(model), sizeof(model_name));
  out.write((char*) &(bucket), sizeof(int));
  out.write((char*) &(minn), sizeof(int));
  out.write((char*) &(maxn), sizeof(int));
  out.write((char*) &(lrUpdateRate), sizeof(int));
  out.write((char*) &(t), sizeof(double));
  out.write((char*) &(wordsWeights), sizeof(bool));
  out.write((char*) &(tfidf), sizeof(bool));
  out.write((char*) &(saveDocuments), sizeof(bool));

  // PLT args
  out.write((char*) &(arity), sizeof(int));
  out.write((char*) &(l2), sizeof(real));
  out.write((char*) &(unitNorm), sizeof(bool));
  out.write((char*) &(probNorm), sizeof(bool));

  // Ensemble args
  out.write((char*) &(ensemble), sizeof(int));
  out.write((char*) &(bagging), sizeof(real));
}

void Args::load(std::istream& in) {
  in.read((char*) &(lr), sizeof(double));
  in.read((char*) &(dim), sizeof(int));
  in.read((char*) &(ws), sizeof(int));
  in.read((char*) &(epoch), sizeof(int));
  in.read((char*) &(minCount), sizeof(int));
  in.read((char*) &(neg), sizeof(int));
  in.read((char*) &(wordNgrams), sizeof(int));
  in.read((char*) &(loss), sizeof(loss_name));
  in.read((char*) &(model), sizeof(model_name));
  in.read((char*) &(bucket), sizeof(int));
  in.read((char*) &(minn), sizeof(int));
  in.read((char*) &(maxn), sizeof(int));
  in.read((char*) &(lrUpdateRate), sizeof(int));
  in.read((char*) &(t), sizeof(double));
  in.read((char*) &(wordsWeights), sizeof(bool));
  in.read((char*) &(tfidf), sizeof(bool));
  in.read((char*) &(saveDocuments), sizeof(bool));

  // PLT args
  in.read((char*) &(arity), sizeof(int));
  in.read((char*) &(l2), sizeof(real));
  in.read((char*) &(unitNorm), sizeof(bool));
  in.read((char*) &(probNorm), sizeof(bool));

  // Ensemble args
  in.read((char*) &(ensemble), sizeof(int));
  in.read((char*) &(bagging), sizeof(real));
}

void Args::dump(std::ostream& out) const {
  out << "dim" << " " << dim << std::endl;
  out << "ws" << " " << ws << std::endl;
  out << "epoch" << " " << epoch << std::endl;
  out << "minCount" << " " << minCount << std::endl;
  out << "neg" << " " << neg << std::endl;
  out << "wordNgrams" << " " << wordNgrams << std::endl;
  out << "loss" << " " << lossToString(loss) << std::endl;
  out << "model" << " " << modelToString(model) << std::endl;
  out << "bucket" << " " << bucket << std::endl;
  out << "minn" << " " << minn << std::endl;
  out << "maxn" << " " << maxn << std::endl;
  out << "lrUpdateRate" << " " << lrUpdateRate << std::endl;
  out << "t" << " " << t << std::endl;
}

}
