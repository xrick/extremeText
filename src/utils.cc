/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "utils.h"

#include <ios>
#include <thread>
#include <sstream>
#include <iomanip>

namespace fasttext {

namespace utils {

  int64_t size(std::ifstream& ifs) {
    ifs.seekg(std::streamoff(0), std::ios::end);
    return ifs.tellg();
  }

  void seek(std::ifstream& ifs, int64_t pos) {
    ifs.clear();
    ifs.seekg(std::streampos(pos));
  }

  size_t cpuCount(){
    return std::thread::hardware_concurrency();
  }

  std::string itos(int32_t number, int32_t leadingZeros){
      std::stringstream ss;
      if (leadingZeros != 0) ss << std::setw(leadingZeros) << std::setfill('0');
      ss << number;
      return ss.str();
  }

  void printProgress(float progress, std::ostream& log_stream) {
    progress = progress * 100;
    log_stream << std::fixed;
    log_stream << "Progress: ";
    log_stream << std::setprecision(1) << std::setw(5) << progress << "%\r";
    log_stream << std::flush;
  }
}

}
