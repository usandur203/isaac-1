#include <cassert>

#include "isaac/algorithms/lasr.h"
#include "isaac/array.h"
#include "isaac/symbolic/execute.h"

namespace isaac
{

  void lasr(char side, char pivot, char direct, array_base const & cos, array_base const & sin, array_base & A)
  {
    using isaac::_i0;
    assert(A.dim()==2 && "A must be 2D");
    int_t M = A.shape()[0];
    int_t N = A.shape()[1];
    if(side=='R' && pivot=='V' && direct=='B')
      execute(sfor(_i0 = N-2, _i0 >= 0, _i0-=1, rot(col(A, _i0), col(A, _i0 + 1), cos[_i0], sin[_i0])));
    else if(side=='R' && pivot=='V' && direct=='F')
      execute(sfor(_i0 = 0, _i0 <= N-2, _i0+=1, rot(col(A, _i0), col(A, _i0 + 1), cos[_i0], sin[_i0])));
    else if(side=='L' && pivot=='V' && direct=='B')
      execute(sfor(_i0 = M-2, _i0 >= 0, _i0-=1, rot(row(A, _i0), row(A, _i0 + 1), cos[_i0], sin[_i0])));
    else if(side=='L' && pivot=='V' && direct=='F')
      execute(sfor(_i0 = 0, _i0 <= M-2, _i0+=1, rot(row(A, _i0), row(A, _i0 + 1), cos[_i0], sin[_i0])));


  }

}
