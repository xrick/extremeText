/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include "smatrix.h"
#include "svector.h"

namespace fasttext {

int64_t SMatrix::size() const {
  int64_t size;
  for (auto const &it : data_)
    size += it.second.size();
  return size;
}

void SMatrix::zero() {
  data_.clear();
}

void SMatrix::zero(int64_t i) {
  data_.erase(i);
}

void SMatrix::zero(int64_t i, int64_t j) {
  data_[i].erase(j);
  if (data_[i].size() == 0)
    data_.erase(i);
}

real SMatrix::dotRow(const SVector &vec, int64_t i){
  assert(i >= 0);
  if (!data_.count(i))
    return 0;

  real d = 0.0;
  for (auto &it : vec.data()) {
    d += at(i, it.first) * it.second;
  }
  if (std::isnan(d)) {
    throw std::runtime_error("Encountered NaN.");
  }
  return d;
}

void SMatrix::addRow(const SVector &vec, int64_t i, real a) {
  assert(i >= 0);
  for (auto it : vec.data())
    add(i, it.first, it.second);
}

}