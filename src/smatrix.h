/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <unordered_map>
#include <assert.h>

#include "real.h"

namespace fasttext {

class Vector;
class SVector;

class SMatrix {
 protected:
  typedef std::unordered_map<int32_t, std::unordered_map<int32_t, real>> SMatrixData;
  SMatrixData data_;

 public:
  SMatrix() {};

  inline SMatrixData &data() {
    return data_;
  };

  inline real at(int64_t i, int64_t j) {
    if (data_.count(i) && data_[i].count(j))
      return data_[i][j];
    return 0;
  };

  inline void set(int64_t i, int64_t j, real a) {
    data_[i][j] = a;
  };

  inline void add(int64_t i, int64_t j, real a) {
    data_[i][j] += a;
  };

  inline void mul(int64_t i, int64_t j, real a) {
    data_[i][j] *= a;
  };


  int64_t size() const;
  void zero();
  void zero(int64_t i);
  void zero(int64_t i, int64_t j);

  real dotRow(const SVector &, int64_t);
  void addRow(const SVector &, int64_t, real);

  void save(std::ostream &);
  void load(std::istream &);
};

}
