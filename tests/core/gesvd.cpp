#include <iterator>
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

  long int M = 7;
  long int N = 7;
  long int SUBM = 2;
  long int SUBN = 2;

  isaac::driver::Context const & ctx = isaac::driver::backend::contexts::get_default();

  INIT_MATRIX(M, SUBM, 5, 3, N, SUBN, 7, 2, cA, A, ctx);
  std::vector<float> tauq(M);
  std::vector<float> taup(M);
  std::vector<float> d(M);
  std::vector<float> e(M-1);

  sc::array G = A;
  sc::gebrd(A, ptr(tauq), ptr(taup), ptr(d), ptr(e), 4);
  std::cout << "d: "; std::copy(d.begin(), d.end(), std::ostream_iterator<float>(std::cout, " ")); std::cout << std::endl;
  std::cout << "e: "; std::copy(e.begin(), e.end(), std::ostream_iterator<float>(std::cout, " ")); std::cout << std::endl;
  sc::array Q = A;
  std::cout << "BEFORE " << Q << std::endl;
  sc::orgbr('Q', Q, M, ptr(tauq));
  std::cout << "AFTER " << Q << std::endl;
  sc::array PT = A;
  sc::orgbr('P', PT, M, ptr(taup));

  sc::array BD = sc::zeros(M, N, sc::FLOAT_TYPE);
  sc::diag(BD) = d;
  sc::diag(BD, 1) = e;
//  sc::array tmp = sc::dot(BD, PT);
//  std::cout << sc::dot(Q, tmp) << std::endl;
//  std::cout << G - sc::dot(U, tmp) << std::endl;

    long int lda = cA.ld();

  std::vector<float> work(1);
  long int lwork = -1;
  long int info;
  sgebrd_(&M, &N, ptr(cA.data()), &lda, ptr(d), ptr(e), ptr(tauq), ptr(taup), ptr(work), &lwork, &info);
  lwork = work[0];
  work.resize(lwork);
  sgebrd_(&M, &N, ptr(cA.data()), &lda, ptr(d), ptr(e), ptr(tauq), ptr(taup), ptr(work), &lwork, &info);
  std::cout << "d: "; std::copy(d.begin(), d.end(), std::ostream_iterator<float>(std::cout, " ")); std::cout << std::endl;
  std::cout << "e: "; std::copy(e.begin(), e.end(), std::ostream_iterator<float>(std::cout, " ")); std::cout << std::endl;


//  using sc::_i0;
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
