#ifndef ISAAC_TYPES_H
#define ISAAC_TYPES_H

#include <algorithm>
#include <vector>
#include <cstddef>
#include "isaac/defines.h"

namespace isaac
{

typedef long long int_t;
typedef std::vector<int_t> size4;

inline int_t prod(size4 x){ return std::accumulate(x.begin(), x.end(), 1, std::multiplies<int>()); }
inline int_t max(size4 x){ return std::accumulate(x.begin(), x.end(), -INFINITY, [](int_t a, int_t b){ return std::max(a, b); }); }

static const int_t start = 0;
static const int_t end = -1;
struct slice
{
  slice(int_t _start) : start(_start), end(_start + 1), stride(1){}
  slice(int_t _start, int_t _end, int_t _stride = 1) : start(_start), end(_end), stride(_stride) { }

  int_t size(int_t bound) const
  {
    int_t effective_end = (end < 0)?bound - (end + 1):end;
    return (effective_end - start)/stride;
  }

  int_t start;
  int_t end;
  int_t stride;
};
static const slice all = slice(start, end, 1);

}
#endif
