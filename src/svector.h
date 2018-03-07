/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <unordered_map>

#include "real.h"

namespace fasttext {

class SVector {
 protected:
  //typedef std::vector<std::pair<int32_t, real>> SVectorData;
  typedef std::unordered_map<int32_t, real> SVectorData;
  SVectorData data_;

 public:
  SVector() {};

  inline SVectorData &data() {
    return data_;
  };
  inline const SVectorData &data() const{
    return data_;
  };

  inline real& operator[](int64_t i) {
    return data_[i];
  };
//  inline const real &operator[](int64_t i) const{
//    return data_[i];
//  };

  inline real at(int64_t i) {
    if (data_.count(i))
      return data_[i];
    return 0;
  };

  inline void set(int64_t i, real a) {
    data_[i] = a;
  };

  inline void add(int64_t i, real a) {
    data_[i] += a;
  };

  inline void mul(int64_t i, real a) {
    data_[i] *= a;
  };

  inline int64_t size() const {
    return data_.size();
  }

  void zero();
  void mul(real);
  real norm() const;
};

//std::ostream &operator<<(std::ostream &, const SVector &);

}
