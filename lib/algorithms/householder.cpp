#include <cmath>

#include "isaac/array.h"
#include "isaac/algorithms/householder.h"
#include "isaac/symbolic/execute.h"

namespace isaac
{
  void larfg(view x, float* tau, float* a)
  {
    float xnorm = value_scalar(norm(x));
    if(xnorm==0)
      *tau = 0;
    else
    {
      float alpha = *a;
      float sign = alpha>0?1:-1;
      float beta = -sign*std::sqrt(alpha*alpha + xnorm*xnorm);
      *tau = (beta - alpha)/beta;
      x /= (beta - alpha);
      *a = beta;
    }
  }

  void labrd(view A, float* tauq, float* taup, float* d, float* e, view X, view Y, int_t NB)
  {
    int_t N = A.shape()[1];
    copy(diag(A), d, false);
    copy(diag(A, 1), e, true);
    for(int_t i = 0 ; i < NB ; ++i)
    {
        //Helper slices
        slice __i = {0, i}, __ip1 = {0, i+1};
        slice i__ = {i, end}, ip1__ = {i+1, end}, ip2__ = {i+2, end};
        //Update A[i:, i]
        A(i__, i) -= dot(A(i__, __i), Y(i, __i));
        A(i__, i) -= dot(X(i__, __i), A(__i, i));
        //Householder A[i:, i]
        larfg(A(ip1__, i), &tauq[i], &d[i]);
        A(i, i) = (float)1;
        if(i < N - 1)
        {
            //Compute Y[i+1:,i]
            Y(ip1__, i)   = dot(A(i__, ip1__).T, A(i__, i));
            Y(__i, i)     = dot(A(i__, __i)  .T, A(i__, i));
            Y(ip1__, i)  -= dot(Y(ip1__, __i)    , Y(__i, i));
            Y(__i, i)     = dot(X(i__, __i)  .T, A(i__, i));
            Y(ip1__, i)  -= dot(A(__i, ip1__).T, Y(__i, i));
            Y(ip1__, i)  *= tauq[i];
            //Update A[i,i+1:]
            A(i, ip1__)  -= dot(Y(ip1__, __ip1), A(i, __ip1));
            A(i, ip1__)  -= dot(A(__i, ip1__).T, X(i, __i));
            //Householder of A[i, i+1:]
            larfg(A(i, ip2__), &taup[i], &e[i]);
            A(i, i+1)     = (float)1;
            //Compute X[i+1:, i]
            X(ip1__, i)   = dot(A(ip1__, ip1__)  , A(i, ip1__));
            X(__ip1, i)   = dot(Y(ip1__, __ip1).T, A(i, ip1__));
            X(ip1__, i)  -= dot(A(ip1__, __ip1)  , X(__ip1, i));
            X(__i,   i)   = dot(A(__i, ip1__)    , A(i, ip1__));
            X(ip1__, i)  -= dot(X(ip1__,__i)     , X(__i, i));
            X(ip1__, i)  *= taup[i];
        }
    }
  }

  void gebd2(view A, float* tauq, float* taup, float* d, float* e)
  {
      int_t N = A.shape()[1];
      copy(diag(A), d, false);
      copy(diag(A, 1), e, true);
      for(int_t i = 0 ; i < N ; ++i)
      {
        //Householder vector
        larfg(A({i+1,end}, i), &tauq[i], &d[i]);
        A(i, i) = (float)1;
        //Apply H(i) to A[i:, i+1:] from the left
        larf('L', A({i,end}, i), tauq[i], A({i,end}, {i+1,end}));
        if(i < N - 1)
        {
            //Householder vector
            larfg(A(i, {i+2,end}), &taup[i], &e[i]);
            A(i, i+1) = (float)1;
            //Apply G(i) to A(i+1:, i_1:) from the right
            larf('R', A(i, {i+1,end}), taup[i], A({i+1,end}, {i+1,end}));
        }
        else
            taup[i] = 0;
      }
  }

  void gebrd(array& A, float* tauq, float* taup, float* d, float* e, int_t nb)
  {
      std::cout << A << std::endl;
      int_t M = A.shape()[0];
      int_t N = A.shape()[1];
      array X = zeros(M, nb, A.dtype(), A.context());
      array Y = zeros(N, nb, A.dtype(), A.context());
      int_t i = 0;
//      //Blocked computations
//      while(N - i >= nb)
//      {
//          slice i__(i, end), ipnb__(i+nb, end), __i_ipnb(i, i+nb);
//          //Update nb rows/cols of A[i:, i:]
//          labrd(A(i__,i__), tauq + i, taup + i, d + i, e + i, X(i__, all), Y(i__, all), nb);
//          //Updates remainded A[i+nb:, i+nb:]
//          A(ipnb__, ipnb__) -= dot(A(ipnb__, __i_ipnb), Y(ipnb__, all).T);
//          A(ipnb__, ipnb__) -= dot(X(ipnb__, all)     , A(__i_ipnb, ipnb__));
//          i+= nb;
//      }
//      //Cleanup
      slice i__(i, end);
      gebd2(A(i__, i__), tauq + i, taup + i, d + i, e + i);
  }


  void larf(char side, view v, float tau, view C)
  {
    if(side=='L')
    {
      array x = dot(C.T, v);
      C -= tau*outer(v, x);
    }
    else
    {
      array x = dot(C, v);
      C -= tau*outer(x, v);
    }
  }

  void org2r(view A, int_t K, float* tau)
  {
    int_t M = A.shape()[0];
    int_t N = A.shape()[1];
    for(int_t i = K-1 ; i >= 0 ; ++i)
    {
      if(i < N - 1){
        A(i, i) = 1;
        larf('L', A({i, end}, i), tau[i], A({i, end},{i+1, end}));
      }
      if(i < M - 1)
        A({i+1, end}, i) *= -tau[i];
      A(i, i) = 1 - tau[i];
      A({0, i}, i) = 0;
    }
  }

  void orgl2(view A, int_t K, float* tau)
  {
    int_t M = A.shape()[0];
    int_t N = A.shape()[1];
    for(int_t i = K-1 ; i >= 0 ; ++i)
    {
      if(i < M - 1){
        A(i, i) = 1;
        larf('R', A(i, {i+1, end}), tau[i], A({i+1, end}, {i, end}));
      }
      if(i < N - 1)
        A(i, {i+1, end}) *= -tau[i];
      A(i, i) = 1 - tau[i];
      A(i, {0, i}) = 0;
    }
  }

  void orgqr(view A, int_t K, float* tau)
  {
    org2r(A, K, tau);
  }

  void orglq(view A, int_t K, float* tau)
  {
    orgl2(A, K, tau);
  }

  void orgbr(char flag, array& A, int_t K, float* tau)
  {
      if(flag=='Q'){
        orgqr(A({0, end}, {0, end}), K, tau);
      }
      else{
        execute(sfor(_i0 = A.shape()[0] - 1, _i0 >= 0, _i0-=1, assign(row(A, _i0 + 1),row(A, _i0))));
        A({0, end}, 0) = (float)0;
        A(0, {0, end}) = (float)0;
        A(0, 0) = (float)1;
        orglq(A({1, end}, {1, end}), K-1, tau);
      }
  }
}
