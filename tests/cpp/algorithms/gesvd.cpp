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
int sbdsqr_debug_(char *uplo, int *n, int *ncvt, int *
  nru, int *ncc, float *d__, float *e, float *vt, int *ldvt, float *
  u, int *ldu, float *c__, int *ldc, float *work, int *info);
int slasr_(char *side, char *pivot, char *direct, lapack_int *m, lapack_int *n, float *c__, float *s, float *a, lapack_int *lda);
void slas2_( float* f, float* g, float* h, float* ssmin, float* ssmax);
void slasv2_(float* f, float* g, float* h, float *ssmin, float *ssmax, float *snr, float * csr, float *snl, float *csl);
int slartg_(float* f, float* g, float* cs, float *sn, float *r);
}

template<class T>
void test(int M, int N, int nb, T epsilon)
{
  int _0 = 0;
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

#define TEST_LASR(SIDE, PIVOT, DIRECT) \
  {\
  std::cout << "LASR-" << SIDE << PIVOT << DIRECT << "...";\
  std::vector<T> hcos = random<T>(N-1);\
  std::vector<T> hsin = random<T>(N-1);\
  sc::array cos = hcos;\
  sc::array sin = hsin;\
  char side = SIDE;\
  char pivot = PIVOT;\
  char direct = DIRECT;\
  slasr_(&side, &pivot, &direct, &M, &N, p(hcos), p(hsin), p(cA.data()), &M);\
  sc::lasr(side, pivot, direct, cos, sin, A);\
  sc::copy(A, tmp);\
  if(diff(tmp, cA.data(), epsilon))\
    std::cout << "[Failure!]" << std::endl;\
  else\
    std::cout << std::endl;\
  }

  TEST_LASR('R', 'V', 'B')
  TEST_LASR('R', 'V', 'F')
  TEST_LASR('L', 'V', 'B')
  TEST_LASR('L', 'V', 'F')


  //BDSQR related
  T f = (T)rand()/RAND_MAX, g = (T)rand()/RAND_MAX, h = (T)rand()/RAND_MAX;
  T ssmin, ssmax, snr, csr, snl, csl, r;
  T issmin, issmax, isnr, icsr, isnl, icsl, ir;
  {
    std::cout << "LAS2...";
    slas2_(&f, &g, &h, &ssmin, &ssmax);
    sc::las2(f, g, h, &issmin, &issmax);
    std::vector<std::string> errors;
    if(diff(ssmin,issmin,epsilon)) errors.push_back("ssmin");
    if(diff(ssmax,issmax,epsilon)) errors.push_back("ssmax");
    if(errors.size())
      std::cout << " [Failure!: " << join(errors.begin(), errors.end(), ", ") << "]";
    std::cout << std::endl;
  }

  {
    std::cout << "LASV2...";
    slasv2_(&f, &g, &h, &ssmin, &ssmax, &snr, &csr, &snl, &csl);
    sc::lasv2(f, g, h, &issmin, &issmax, &isnr, &icsr, &isnl, &icsl);
    std::vector<std::string> errors;
    if(diff(ssmin,issmin,epsilon)) errors.push_back("ssmin");
    if(diff(ssmax,issmax,epsilon)) errors.push_back("ssmax");
    if(diff(snr,isnr,epsilon)) errors.push_back("snr");
    if(diff(csr,icsr,epsilon)) errors.push_back("csr");
    if(diff(snl,isnl,epsilon)) errors.push_back("snl");
    if(diff(csl,icsl,epsilon)) errors.push_back("csl");
    if(errors.size())
      std::cout << " [Failure!: " << join(errors.begin(), errors.end(), ", ") << "]";
    std::cout << std::endl;
  }

  {
    std::cout << "LARTG...";
    slartg_(&f, &g, &csl, &snl, &r);
    sc::lartg(f, g, &icsl, &isnl, &ir);
    std::vector<std::string> errors;
    if(diff(csl,icsl,epsilon)) errors.push_back("cs");
    if(diff(snl,isnl,epsilon)) errors.push_back("sn");
    if(diff(r,ir,epsilon)) errors.push_back("r");
    if(errors.size())
      std::cout << " [Failure!: " << join(errors.begin(), errors.end(), ", ") << "]";
    std::cout << std::endl;
  }

  d = hd;
  e = he;
  std::vector<T> hU = random<T>(M*M);
  std::vector<T> hVT = random<T>(N*N);
  sc::array U(M, M, hU);
  sc::array VT(N, N, hVT);
  int MN =  d.size();
  hwork.resize(4*N);
  int _1 = 1;
  {
    std::cout << "BDSQR-U...";
    std::cout << std::endl;
    sbdsqr_debug_("U", &MN, &M, &N, &_0, p(hd), p(he), p(hVT), &M, p(hU), &N, NULL, &_1, p(hwork), &info);
    std::cout << std::endl;
    sc::bdsqr('U', MN, p(d), p(e), &VT, &U);
    std::cout << std::endl;
    for(unsigned int i = 0 ; i < M ; ++i){
      for(unsigned int j = 0 ; j < N ; ++j)
        std::cout << hU[j*M + i] << "," << std::flush;
      std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << U << std::endl;
    std::cout << std::endl;
    std::vector<std::string> errors;
    if(diff(d, hd, epsilon)) errors.push_back("d");
    if(errors.size())
      std::cout << " [Failure!: " << join(errors.begin(), errors.end(), ", ") << "]";
    std::cout << std::endl;
  }
}

int main()
{
  test<float>(5, 5, 4, 1e-4);
}
