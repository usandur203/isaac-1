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


  std::vector<float> tauq(M);
  std::vector<float> taup(M);
  std::vector<float> d(M);
  std::vector<float> e(M-1);
  simple_matrix<T> cA(M, N);
  for(unsigned int i = 0 ; i < M ; ++i)
    for(unsigned int j = 0 ; j < N ; ++j)
      cA(i, j) = (float)rand()/RAND_MAX;
  sc::array A(M, N, cA.data());

  sc::array G = A;
  sc::gebrd(A, ptr(tauq), ptr(taup), ptr(d), ptr(e), 4);
  std::cout << "d: "; std::copy(d.begin(), d.end(), std::ostream_iterator<float>(std::cout, " ")); std::cout << std::endl;
  std::cout << "e: "; std::copy(e.begin(), e.end(), std::ostream_iterator<float>(std::cout, " ")); std::cout << std::endl;
  std::cout << "tauq: "; std::copy(tauq.begin(), tauq.end(), std::ostream_iterator<float>(std::cout, " ")); std::cout << std::endl;
  std::cout << "taup: "; std::copy(taup.begin(), taup.end(), std::ostream_iterator<float>(std::cout, " ")); std::cout << std::endl;

  sc::array Q = A;
  sc::orgbr('Q', Q, M, ptr(tauq));
  sc::array PT = A;
  sc::orgbr('P', PT, M, ptr(taup));
  sc::array BD = sc::zeros(M, N, sc::FLOAT_TYPE);
  sc::diag(BD) = d;
  sc::diag(BD, 1) = e;
  sc::array tmp = sc::dot(BD, PT);
  sc::array res = sc::dot(Q, tmp);
  res -= G;
  std::cout << res << std::endl;

    long int lda = cA.ld();

  std::vector<float> work(1);
  long int lwork = -1;
  long int info;
  sgebrd_(&M, &N, ptr(cA.data()), &lda, ptr(d), ptr(e), ptr(tauq), ptr(taup), ptr(work), &lwork, &info);
  lwork = work[0];
  work.resize(lwork);
  sgebrd_(&M, &N, ptr(cA.data()), &lda, ptr(d), ptr(e), ptr(tauq), ptr(taup), ptr(work), &lwork, &info);
//  std::cout << std::endl;
//  std::cout << "d: "; std::copy(d.begin(), d.end(), std::ostream_iterator<float>(std::cout, " ")); std::cout << std::endl;
//  std::cout << "e: "; std::copy(e.begin(), e.end(), std::ostream_iterator<float>(std::cout, " ")); std::cout << std::endl;
//  std::cout << "tauq: "; std::copy(tauq.begin(), tauq.end(), std::ostream_iterator<float>(std::cout, " ")); std::cout << std::endl;
//  std::cout << "taup: "; std::copy(taup.begin(), taup.end(), std::ostream_iterator<float>(std::cout, " ")); std::cout << std::endl;
//  sorgbr_("P", &M, &N, &M, ptr(cA.data()), &lda, ptr(taup), ptr(work), &lwork, &info);
//  for(unsigned int i = 0 ; i < M ; ++i)
//  {
//    for(unsigned int j = 0 ; j < N ; ++j)
//      std::cout << cA(i, j) << " ";
//    std::cout << std::endl;
//  }

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
