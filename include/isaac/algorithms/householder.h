#ifndef ISAAC_ALGORTITHMS_HOUSEHOLDER_H
#define ISAAC_ALGORTITHMS_HOUSEHOLDER_H

#include "isaac/array.h"

namespace isaac
{
  void larfg(isaac::view x, float* tau, float* alpha);
  void labrd(isaac::view A, float* tauq, float* taup, float* d, float* e, isaac::view X, isaac::view Y);

  void gebd2(isaac::view A, float* tauq, float* taup, float* d, float* e);
  void gebrd(array& A, float* tauq, float* taup, float* d, float* e, int_t nb);
  //orgxr
  void org2r(char flag, array& A, int_t K, float* tau);
  void orgqr(char flag, array& A, int_t K, float* tau);
  void orgbr(char flag, array& A, int_t K, float* tau);
}

#endif
