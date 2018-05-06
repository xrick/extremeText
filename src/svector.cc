/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include "svector.h"

#include <cmath>

namespace fasttext {

void SVector::zero() {
  data_.clear();
}

real SVector::norm() const {
  real sum = 0;
  for (auto &it : data_) {
    sum += it.second * it.second;
  }
  return std::sqrt(sum);
}

void SVector::mul(real a) {
  for (auto &it : data_) {
    it.second *= a;
  }
}

}