#ifndef ISAAC_ALGORTITHMS_HOUSEHOLDER_H
#define ISAAC_ALGORTITHMS_HOUSEHOLDER_H

#include "isaac/array.h"

double slamch_(char cmach);
double dlamch_(char cmach);

namespace isaac
{
  template<typename real> double lamch(char cmach);
  template<> double lamch<float>(char cmach) { return slamch_(cmach); }
  template<> double lamch<double>(char cmach) { return dlamch_(cmach); }

  void larf(char side, view v, float tau, view C);
  void larfg(view x, float* tau, float* alpha);
  void labrd(view A, float* tauq, float* taup, float* d, float* e, view X, view Y);

  void gebd2(view A, float* tauq, float* taup, float* d, float* e);
  void gebrd(array& A, float* tauq, float* taup, float* d, float* e, int_t nb);

  void org2r(view A, int_t K, float* tau);
  void orgqr(view A, int_t K, float* tau);
  void orgl2(view A, int_t K, float* tau);
  void orglq(view A, int_t K, float* tau);
  void orgbr(char flag, array& A, int_t K, float* tau);
}

#endif
