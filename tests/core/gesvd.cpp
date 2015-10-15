#include "isaac/symbolic/execute.h"
#include "isaac/symbolic/io.h"
#include "isaac/algorithms/householder.h"
#include "common.hpp"
#include "external/f2c.h"
#include "external/clapack.h"

namespace sc = isaac;

template<class T>
T* ptr(std::vector<T> & x) { return x.data(); }

int main()
{
  typedef float T;
  using sc::_i0;

  long int M = 10;
  long int N = 10;
  long int SUBM = 7;
  long int SUBN = 11;

  isaac::driver::Context const & ctx = isaac::driver::backend::contexts::get_default();

  INIT_MATRIX(M, SUBM, 5, 3, N, SUBN, 7, 2, cA, A, ctx);
  std::vector<float> tauq(M);
  std::vector<float> taup(M);
  std::vector<float> d(M);
  std::vector<float> e(M);

  sc::gebrd(A, ptr(tauq), ptr(taup), ptr(d), ptr(e), 4);
//  std::vector<float> work(max(M, N));
//  long int lda = cA.ld();
//  long int lwork = work.size();
//  sgebrd_(&M, &N, ptr(cA.data()), &lda, ptr(d), ptr(e), ptr(tauq), ptr(taup), ptr(work), &lwork, NULL);







//  char side = 'R';
//  char pivot = 'V';
//  char direct = 'B';
//  INIT_VECTOR(N-1, SUBN-1, 5, 3, ccos, cos, ctx);
//  INIT_VECTOR(M-1, SUBM-1, 5, 3, csin, sin, ctx);
//  float* pcos = ccos.data().data();
//  float* psin = csin.data().data();
//  long int lda = cA.ld();

//  std::cout << A << std::endl;
//  std::cout << std::endl;

//  slasr_(&side, &pivot, &direct, &M, &M, pcos, psin, pA, &lda);
//  sc::math_expression tree = sfor(_i0 = 8, _i0 >= 0, _i0-=1, rot(col(A, _i0), col(A, _i0 + 1), cos[_i0], sin[_i0]));
//  std::cout << to_string(tree) << std::endl;
//  sc::execute(tree);

//  std::cout << cos << std::endl;
//  std::cout << sin << std::endl;
//  std::cout << std::endl;

//  std::cout << sc::array(M, N, cA.data(), ctx) << std::endl;
//  std::cout << std::endl;
//  std::cout << A << std::endl;
}
