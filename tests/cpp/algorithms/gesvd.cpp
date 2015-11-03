#include <iterator>
#include "isaac/symbolic/execute.h"
#include "isaac/symbolic/io.h"
#include "isaac/algorithms/householder.h"
#include "isaac/algorithms/lasr.h"
#include "common.hpp"
#include "lapacke.h"

namespace sc = isaac;

template<class T>
T* p(std::vector<T> & x) { return x.data(); }

extern "C"{
int slasr_(char *side, char *pivot, char *direct, lapack_int *m, lapack_int *n, float *c__, float *s, float *a, lapack_int *lda);
}

template<class T>
void test(int M, int N, int nb, T epsilon)
{
  simple_matrix<T> cA(M, N);
  for(int i = 0 ; i < M ; ++i)
    for(int j = 0 ; j < N ; ++j)
      cA(i, j) = (T)rand()/RAND_MAX;
  sc::array A(M, N, cA.data());
  std::vector<T> tmp(M*N);

  /* GEBRD */
  std::cout << "GEBRD...";
  //ISAAC
  std::vector<T> tauq(M), taup(M), d(M), e(M-1);
  sc::gebrd(A, p(tauq), p(taup), p(d), p(e), nb);
  //LAPACK
  std::vector<T> htauq(M), htaup(M), hd(M), he(M-1), hwork(1);
  int lwork = -1;
  int info;
  sgebrd_(&M, &N, p(cA.data()), &M, p(hd), p(he), p(htauq), p(htaup), p(hwork), &lwork, &info);
  lwork = hwork[0];
  hwork.resize(lwork);
  sgebrd_(&M, &N, p(cA.data()), &M, p(hd), p(he), p(htauq), p(htaup), p(hwork), &lwork, &info);
  {
  sc::copy(A, tmp);
  std::vector<std::string> errors;
  if(diff(d, hd, epsilon))
    errors.push_back("d");
  if(diff(e, he, epsilon))
    errors.push_back("e");
  if(diff(tauq, htauq, epsilon))
    errors.push_back("tauq");
  if(diff(taup, htaup, epsilon))
    errors.push_back("taup");
  if(diff(tmp, cA.data(), epsilon))
    errors.push_back("A");
  if(errors.size())
    std::cout << " [Failure!: " << join(errors.begin(), errors.end(), ", ") << "]" << std::endl;
  else
    std::cout << std::endl;
  }

  /* ORGBR */
  std::cout << "ORGBR-Q...";
  sc::array Q = A;
  sc::orgbr('Q', Q, M, p(tauq));
  simple_matrix<T> cQ(M, N, cA.data());
  sorgbr_((char*)"Q", &M, &N, &M, p(cQ.data()), &M, p(htauq), p(hwork), &lwork, &info);
  sc::copy(Q, tmp);
  if(diff(tmp, cQ.data(), epsilon))
    std::cout << "[Failure!]" << std::endl;
  else
    std::cout << std::endl;

  std::cout << "ORGBR-P...";
  sc::orgbr('P', A, M, p(taup));
  sorgbr_((char*)"P", &M, &N, &M, p(cA.data()), &M, p(htaup), p(hwork), &lwork, &info);
  sc::copy(A, tmp);
  if(diff(tmp, cA.data(), epsilon))
    std::cout << "[Failure!]" << std::endl;
  else
    std::cout << std::endl;

  std::cout << "SLASR-RVB...";
  std::vector<T> hcos = random<T>(N-1);
  std::vector<T> hsin = random<T>(N-1);
  sc::array cos = hcos;
  sc::array sin = hsin;
  char side = 'R';
  char pivot = 'V';
  char direct = 'B';
  slasr_(&side, &pivot, &direct, &M, &N, p(hcos), p(hsin), p(cA.data()), &M);
  sc::lasr(side, pivot, direct, cos, sin, A);
  sc::copy(A, tmp);
  if(diff(tmp, cA.data(), epsilon))
    std::cout << "[Failure!]" << std::endl;
  else
    std::cout << std::endl;
}

int main()
{
  test<float>(5, 5, 4, 1e-4);
}
