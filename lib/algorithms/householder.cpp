#include <cmath>

#include "isaac/array.h"
#include "isaac/algorithms/householder.h"
#include "isaac/algorithms/lasr.h"
#include "isaac/symbolic/execute.h"

extern "C" /* Subroutine */ int slartg_(float *, float *, float *, float *, float * );
extern "C" void slasv2_(float *, float *, float *, float *, float *, float *, float *, float *, float *);

namespace isaac
{

  template<typename T, typename U>
  inline T sign(T a, U b)
  {
    float x =  a >= 0 ? a : -a;
    return b >= 0 ? x : -x;
  }

  void larfg(view x, float* tau, float* a)
  {
    float xnorm = value_scalar(norm(x));
    if(xnorm==0)
      *tau = 0;
    else
    {
      float alpha = *a;
      float sign = alpha>=0?1:-1;
      float beta = -sign*std::sqrt(alpha*alpha + xnorm*xnorm);
      float safmin = lamch<float>('S')/lamch<float>('E');
      if(std::abs(beta) < safmin)
      {
        int knt = 0;
        while(std::abs(beta) < safmin){
          x /= safmin;
          beta /= safmin;
          alpha /= safmin;
          knt++;
        }
        xnorm = value_scalar(norm(x));
        sign = alpha>=0?1:-1;
        beta = -sign*std::sqrt(alpha*alpha + xnorm*xnorm);
        for(int j = 0 ; j < knt ; ++j)
          alpha *= safmin;
      }
      *tau = (beta - alpha)/beta;
      x /= (alpha - beta);
      *a = beta;
    }
  }

  void labrd(view A, float* tauq, float* taup, float* d, float* e, view X, view Y, int_t NB)
  {
    int_t N = A.shape()[1];
    for(int_t i = 0 ; i < NB ; ++i)
    {
        //Helper slices
        slice __i = {0, i}, __ip1 = {0, i+1};
        slice i__ = {i, end}, ip1__ = {i+1, end}, ip2__ = {i+2, end};
        //Update A[i:, i]
        A(i__, i) -= dot(A(i__, __i), Y(i, __i));
        A(i__, i) -= dot(X(i__, __i), A(__i, i));
        //Householder A[i:, i]
        d[i] = (float)A(i,i)[0];
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
            e[i] = (float)A(i,i+1)[0];
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
      for(int_t i = 0 ; i < N ; ++i)
      {
        //Householder vector
        d[i] = (float)A(i,i)[0];
        larfg(A({i+1,end}, i), &tauq[i], &d[i]);
        A(i, i) = (float)1;
        //Apply H(i) to A[i:, i+1:] from the left
        larf('L', A({i,end}, i), tauq[i], A({i,end}, {i+1,end}));
        if(i < N - 1)
        {
            //Householder vector
            e[i] = (float)A(i,i+1)[0];
            larfg(A(i, {i+2,end}), &taup[i], &e[i]);
            A(i, i+1) = (float)1;
            //Apply G(i) to A(i+1:, i_1:) from the right
            larf('R', A(i, {i+1,end}), taup[i], A({i+1,end}, {i+1,end}));
        }
        else
            taup[i] = 0;
      }
  }

  void gebrd(array_base& A, float* tauq, float* taup, float* d, float* e, int_t nb)
  {
      int_t M = A.shape()[0];
      int_t N = A.shape()[1];
      array X = zeros(M, nb, A.dtype(), A.context());
      array Y = zeros(N, nb, A.dtype(), A.context());
      int_t i = 0;
      //Blocked computations
      while(N - i >= nb)
      {
          //Update nb rows/cols of A[i:, i:]
          labrd(A({i,end},{i,end}), &tauq[i], &taup[i], &d[i], &e[i], X({i,end}, all), Y({i,end}, all), nb);
          //Updates remainded A[i+nb:, i+nb:]
          i+= nb;
          A({i,end}, {i,end}) -= dot(A({i,end}, {i-nb, i}), Y({i,end}, all).T);
          A({i,end}, {i,end}) -= dot(X({i,end}, all)      , A({i-nb, i}, {i,end}));
      }
      //Cleanup
      gebd2(A({i,end}, {i,end}), &tauq[i], &taup[i], &d[i], &e[i]);
      diag(A) = std::vector<float>(d, d + N);
      diag(A, 1) = std::vector<float>(e, e + N - 1);
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
    for(int_t i = K-1 ; i >= 0 ; --i)
    {
      if(i < N - 1){
        A(i, i) = (float)1;
        larf('L', A({i, end}, i), tau[i], A({i, end},{i+1, end}));
      }
      if(i < M - 1)
        A({i+1, end}, i) *= -tau[i];
      A(i, i) = 1 - tau[i];
      A({0, i}, i) = (float)0;
    }
  }

  void orgl2(view A, int_t K, float* tau)
  {
    int_t M = A.shape()[0];
    int_t N = A.shape()[1];
    for(int_t i = K-1 ; i >= 0 ; --i)
    {
      if(i < M - 1){
        A(i, i) = (float)1;
        larf('R', A(i, {i, end}), tau[i], A({i+1, end}, {i, end}));
      }
      if(i < N - 1)
        A(i, {i+1, end}) *= -tau[i];
      A(i, i) = 1 - tau[i];
      A(i, {0, i}) = (float)0;
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

  void orgbr(char flag, array_base& A, int_t K, float* tau)
  {
      if(flag=='Q'){
        orgqr(A({0, end}, {0, end}), K, tau);
      }
      else{
        execute(sfor(_i0 = A.shape()[0] - 2, _i0 >= 0, _i0-=1, assign(row(A, _i0 + 1),row(A, _i0))));
        A({0, end}, 0) = (float)0;
        A(0, {0, end}) = (float)0;
        A(0, 0) = (float)1;
        orglq(A({1, end}, {1, end}), K-1, tau);
      }
  }

  /*-----------------------------
   * Bidiagonalization
   *-----------------------------*/

  void las2(float f, float g, float h, float* ssmin, float* ssmax)
  {
      using std::min;
      using std::max;
      using std::abs;
      using std::sqrt;

      float fa = abs(f);
      float ga = abs(g);
      float ha = abs(h);
      float fhmn = min(fa, ha);
      float fhmx = max(fa, ha);
      if(fhmn==0)
      {
          *ssmin = 0;
          if(fhmx==0)
              *ssmax = 0;
          else{
              float d1 = min(fhmx, ha) / max(fhmx, ga);
              *ssmax = max(fhmx, ga)*sqrt(d1*d1 + 1);
          }
      }
      else
      {
          if(ga < fhmx)
          {
              float as = fhmn/fhmx + 1.;
              float at = (fhmx - fhmn)/fhmx;
              float d1 = ga/fhmx;
              float au = d1*d1;
              float c = 2./(sqrt(as*as + au) + sqrt(at*at + au));
              *ssmin = fhmn*c;
              *ssmax = fhmx/c;
          }
          else
          {
              float au = fhmx/ga;
              if(au==0){
                  *ssmin = fhmn*fhmx/ga;
                  *ssmax = ga;
              }
              else{
                  float as = fhmn/fhmx + 1.;
                  float at = (fhmx - fhmn)/fhmx;
                  float d1 = as*au;
                  float d2 = at*au;
                  float c = 1./(sqrt(d1*d1 + 1.) + sqrt(d2*d2 + 1.));
                  *ssmin = fhmn*c*au;
                  *ssmin += *ssmin;
                  *ssmax = ga/(c + c);
              }
          }
      }
  }

  void lasv2(float f, float g, float h,
            float *ssmin, float *ssmax, float *snr, float *
            csr, float *snl, float *csl)
  {
      using std::min;
      using std::max;
      using std::abs;
      using std::sqrt;

      static float clt, crt, slt, srt;
      static bool gasmal;

      float ft = f, ht = h, gt = g;
      float fa = abs(ft), ha = abs(ht), ga = abs(gt);

      /* PMAX points to the maximum absolute element of matrix
         PMAX = 1 if F largest in absolute values
         PMAX = 2 if G largest in absolute values
         PMAX = 3 if H largest in absolute values */
      int pmax = 1;
      bool swap = ha > fa;
      if (swap) {
          pmax = 3;
          std::swap(ft, ht);
          std::swap(fa, ha);
      }
      if (ga == 0.)
      {
          /* Diagonal matrix */
          *ssmin = ha;
          *ssmax = fa;
          clt = 1.;
          crt = 1.;
          slt = 0.;
          srt = 0.;
      }
      else
      {
          gasmal = true;
          if (ga > fa)
          {
              pmax = 2;
              if (fa / ga < lamch<float>('E'))
              {
                  /* Case of very large GA */
                  gasmal = false;
                  *ssmax = ga;
                  if (ha > 1.)
                      *ssmin = fa / (ga / ha);
                  else
                      *ssmin = fa / ga * ha;
                  clt = 1.;
                  slt = ht / gt;
                  srt = 1.;
                  crt = ft / gt;
              }
          }
          if (gasmal) {
              /* Normal case */
              float d = fa - ha;
              float l = (d==fa)?1:d/fa;
              /* Note that 0 .le. L .le. 1 */
              float m = gt / ft;
              /* Note that abs(M) .le. 1/macheps */
              float t = 2. - l;
              /* Note that T .ge. 1 */
              float mm = m * m;
              float tt = t * t;
              float s = sqrt(tt + mm);
              /* Note that 1 .le. S .le. 1 + 1/macheps */
              float r = (l==0)?abs(m):sqrt(l*l + mm);
              /* Note that 0 .le. R .le. 1 + 1/macheps */
              float a = (s + r) * .5;
              /* Note that 1 .le. A .le. 1 + abs(M) */
              *ssmin = ha / a;
              *ssmax = fa * a;
              if (mm == 0.){
                  /* Note that M is very tiny */
                  if (l == 0.)
                      t = sign(2, ft) * sign(1, gt);
                  else
                      t = gt / sign(d, ft) + m/t;
              }
              else {
                  t = (m / (s + t) + m / (r + l)) * (a + 1.);
              }
              l = sqrt(t * t + 4.);
              crt = 2. / l;
              srt = t / l;
              clt = (crt + srt * m) / a;
              slt = ht / ft * srt / a;
          }
      }
      if (swap) {
          *csl = srt;
          *snl = crt;
          *csr = slt;
          *snr = clt;
      } else {
          *csl = clt;
          *snl = slt;
          *csr = crt;
          *snr = srt;
      }

      /* Correct signs of SSMAX and SSMIN */
      float _1 = 1.;
      float tsign;
      if (pmax == 1)
          tsign = sign(_1, *csr) * sign(_1, *csl) * sign(_1, f);
      if (pmax == 2)
          tsign = sign(_1, *snr) * sign(_1, *csl) * sign(_1, g);
      if (pmax == 3)
          tsign = sign(_1, *snr) * sign(_1, *snl) * sign(_1, h);
      *ssmax = sign(*ssmax, tsign);
      float d1 = tsign * sign(_1, f) * sign(_1, h);
      *ssmin = sign(*ssmin, d1);
  }

  int lartg(float f, float g, float *cs, float *sn, float *r)
  {
    using std::pow;
    using std::max;
    using std::abs;
    using std::log;
    using std::sqrt;
    float safmin = lamch<float>('S');
    float eps = lamch<float>('E');
    float B = lamch<float>('B');
    float safmn2 = pow(B, static_cast<int>(log((double)safmin/eps) / log((double)B) / (float)2));
    float safmx2 = 1 / safmn2;
    if(g==0)
    {
        *cs = 1;
        *sn = 0;
        *r = f;
    }
    else if(f==0)
    {
        *cs = 0;
        *sn = 1;
        *r = g;
    }
    else
    {
        float f1 = f;
        float g1 = g;
        float scale = max(f1, g1);
        if(scale >= safmx2)
        {
            int count = 0;
            while(scale >= safmx2)
            {
                count++;
                f1*=safmn2;
                g1*=safmn2;
                scale = max(abs(f1), abs(g1));
            }
            *r = sqrt((f1*f1 + g1*g1));
            *cs = f1 / *r;
            *sn = g1 / *r;
            for(int i = 0 ; i < count ; ++i)
                *r *= safmx2;

        }
        else if(scale <= safmn2)
        {
            int count = 0;
            while(scale <= safmn2)
            {
                count++;
                f1 *= safmx2;
                g1 *= safmx2;
                scale = max(abs(f1), abs(g1));
            }
            *r = sqrt(f1*f1 + g1*g1);
            *cs = f1 / *r;
            *sn = g1 / *r;
            for(int i = 0 ; i < count ; ++i)
                *r *= safmn2;
        }
        else
        {
            *r = sqrt(f1*f1 + g1*g1);
            *cs = f1 / *r;
            *sn = g1 / *r;
        }
        if(abs(f) > abs(g) && *cs < 0){
            *cs = -*cs;
            *sn = -*sn;
            *r = -*r;
        }
    }
    return 0;
  }

  void bdsqr(char uplo, int N, float* d, float* e, array_base * VT, array_base * U)
  {
    using std::pow;
    using std::min;
    using std::max;
    using std::sqrt;
    using std::abs;

    bool rotate = VT || U;
    bool lower = uplo=='L';
    if(!rotate){
        //TODO
    }
    float eps = lamch<float>('E');
    float unfl = lamch<float>('S');
    eps = 5.96046e-08;
    unfl = 1.17549e-38;
    if(lower){
        //TODO
    }
    float d3 = 100, d4 = pow((double)eps, -.125);
    float d1 = 10, d2 = min(d3, d4);
    float tolmul = max(d1, d2);
    float tol = tolmul*eps;
    //Compute approximate maximum singular value
    float smax = 0;
    for(int i = 0; i < N ; ++i)
        smax = max(smax, abs(d[i]));
    for(int i = 0 ; i < N - 1; ++i)
        smax = max(smax, abs(e[i]));

    int maxit = N*6*N;

    float thresh;
    if(tol >= 0){
        float sminoa = abs((double)d[0]);
        float mu = sminoa;
        for(int i = 1 ; i < N && sminoa > 0 ; ++i){
            mu = abs((double)d[i])*(mu / (mu + abs(e[i-1])));
            sminoa = min(sminoa, mu);
        }
        sminoa /= sqrt((float)N);
        thresh = max(tol*sminoa, maxit*unfl);
    }
    else{
        thresh = max(abs(tol)*smax, maxit*unfl);
    }

    int oldm = -1;
    int oldll = -1;
    //M points to the last element of unconverged part of matrix
    int M = N - 1;
    std::vector<float> hcosr(M), hcosl(M), hsinr(M), hsinl(M);
    array gcosr(M, FLOAT_TYPE), gcosl(M, FLOAT_TYPE), gsinr(M, FLOAT_TYPE), gsinl(M, FLOAT_TYPE);

    float smin;
    int iter = 0;
    int idir = 0;

    while(iter < maxit && M >= 1)
    {
        //Find diagonal block of matrix to work on
        if(tol < 0 && abs(d[M]) <= thresh)
            d[M] = 0;
        smax = abs(d[M]);
        smin = smax;
        //Approximate min/max singular value
        int ll;
        for(ll = M - 1; ll >= 0 ; --ll)
        {
            float abss = abs(d[ll]);
            float abse = abs(e[ll]);
            if(tol < 0 && abss <= thresh)
                d[ll] = 0;
            if(abse <= thresh)
                break;
            smin = min(smin, abss);
            smax = max(max(smax,abss), abse);
        }
        if(ll == M - 1){
            /* Convergence of bottom singular value, return to top of loop */
            --M;
            continue;
        }
        ll++;

        /* E(LL) through E(M-1) are nonzero, E(LL-1) is zero */
        float sinr, cosr, sinl, cosl;
        if(ll==M-1)
        {
            float sigmn, sigmx;
            slasv2_(&d[M-1], &e[M-1], &d[M], &sigmn, &sigmx, &sinr, &cosr, &sinl, &cosl);
            d[M-1] = sigmx;
            e[M-1] = 0;
            d[M] = sigmn;
            if(VT)
                rot((*VT)(M-1, {1,end}), (*VT)(M, {1,end}), cosr, sinr);
            if(U)
                rot((*U)({1,end}, M-1), (*U)({1,end}, M), cosl, sinl);
            M -= 2;
            continue;
        }

        /* If working on new submatrix, choose shift direction
         * (from larger end diagonal element towards smaller)*/
        if(ll >oldm || M < oldll){
            if(abs(d[ll]) >= abs(d[M]))
                /* Chase bulge from top to bottom */
                idir = 1;
            else
                /* Chase bulge from bottom to top */
                idir = 2;
        }
        /* Apply convergence test */
        float sminl;
        if(idir==1)
        {
            /* Run convergence test in forward direction
             * First apply standard test to bottom of matrix*/
            if(  (abs(e[M-1]) <= abs(tol)*abs(d[M]))
              || (tol < 0 && abs(e[M-1]) <= thresh))
            {
                e[M-1] = 0;
                continue;
            }

            /* If relative accuracy desired, apply convergence criterion forward */
            if(tol >= 0)
            {
                float mu = abs(d[ll]);
                sminl = mu;
                int lll;
                for(lll = ll; lll < M ; ++lll)
                {
                    if(abs(e[lll]) <= tol*mu){
                        e[lll] = 0;
                        break;
                    }
                    mu = (double)abs(d[lll+1])*(mu/(mu + (double)abs(e[lll])));
                    sminl = min(sminl, mu);
                }
                if(lll < M)
                    continue;
            }
        }
        else
        {
            /* Run convergence test in backward direction
               First apply standard test to top of matrix */
            if(  (abs(e[ll]) <= abs(tol)*abs(d[ll]))
              || (tol < 0 && abs(e[ll]) <= thresh))
            {
                e[ll] = 0;
                continue;
            }

            /* If relative accuracy desired, apply convergence criterion backward */
            if(tol >= 0)
            {
                float mu = abs(d[M]);
                sminl = mu;
                int lll;
                for(lll = M-2 ; lll >= ll ; --lll)
                {
                    if(abs(e[lll]) <= tol*mu){
                        e[lll] = 0;
                        break;
                    }
                    mu = abs(d[lll])*(mu/(mu + abs(e[lll])));
                    sminl = min(sminl, mu);
                }
                if(lll >= ll)
                    continue;
            }
        }

        oldll = ll;
        oldm = M;

        float shift, r;
        /* Compute shift.  First, test if shifting would ruin relative,
         * accuracy, and if so set the shift to zero. */
        if(tol >= 0 && N*tol*(sminl/smax) <= max(eps, (float)0.01*tol))
            shift = 0;
        else
        {
            if(idir==1)
                las2(d[M-1], e[M-1], d[M], &shift, &r);
            else
                las2(d[ll], e[ll], d[ll + 1], &shift, &r);
            float sll = abs((idir==1)?d[ll]:d[M]);
            if(sll > 0 && pow(shift/sll, 2) < eps)
                shift = 0;
        }

        /* If SHIFT = 0, do simplified QR iteration */
        iter += M - ll;
        float sn, oldsn;
        if(shift==0)
        {

            if(idir==1)
            {
                /* chase bulge from bottom to top */
                float cs = 1;
                float oldcs = 1;
                for(int i = ll ; i < M  ; ++i){
                    lartg(d[i]*cs, e[i], &cs, &sn, &r);
                    if(i > ll)
                        e[i-1] = oldsn*r;
                    lartg(oldcs*r, d[i+1]*sn, &oldcs, &oldsn, &d[i]);
                    /* store for later */
                    hcosr[i - ll] = cs;
                    hsinr[i - ll] = sn;
                    hcosl[i - ll] = oldcs;
                    hsinl[i - ll] = oldsn;
                }
                float h = d[M]*cs;
                d[M] = h*oldcs;
                e[M-1] = h*oldsn;
                /* update singular vectors */
                if(VT){
                    copy(hcosr, gcosr);
                    copy(hsinr, gsinr);
                    lasr('L', 'V', 'F', gcosr, gsinr, (*VT)({ll,M+1}, {1, end}));
                }
                if(U){
                    copy(hcosl, gcosl);
                    copy(hsinl, gsinl);
                    lasr('R', 'V', 'F', gcosl, gsinl, (*U)({1, end}, {ll, M+1}));
                }
                /* test convergence */
                if(abs(e[M-1]) <= thresh)
                    e[M-1] = 0;
            }
            else
            {
                /* chase bulge from top to bottom */
                float cs = 1;
                float oldcs = 1;
                for(int i = M ; i >= ll + 1  ; --i){
                    lartg(d[i]*cs, e[i - 1], &cs, &sn, &r);
                    if(i < M)
                        e[i-1] = oldsn*r;
                    lartg(oldcs*r, d[i-1]*sn, &oldcs, &oldsn, &d[i]);
                    /* store for later */
                    hcosr[i - ll] = cs;
                    hsinr[i - ll] = sn;
                    hcosl[i - ll] = oldcs;
                    hsinl[i - ll] = oldsn;
                }
                float h = d[ll]*cs;
                d[ll] = h*oldcs;
                e[ll] = h*oldsn;
                /* update singular vectors */
                if(VT){
                    copy(hcosr, gcosr);
                    copy(hsinr, gsinr);
                    lasr('L', 'V', 'B', gcosr, gsinr, (*VT)({ll,M+1}, {1, end}));
                }
                if(U){
                    copy(hcosl, gcosl);
                    copy(hsinl, gsinl);
                    lasr('R', 'V', 'B', gcosl, gsinl, (*U)({1, end}, {ll, M+1}));
                }
                /* test convergence */
                if(abs(e[ll]) <= thresh)
                    e[ll] = 0;
            }
        }
        else
        {
            /* Use nonzero shift */
            if(idir==1)
            {
                float f = (abs((double)d[ll]) - shift)*(sign(1.,d[ll]) + shift/d[ll]);
                float g = e[ll];
                for(int i = ll; i < M ; ++i)
                {
                    lartg(f, g, &cosr, &sinr, &r);
                    if(i > ll)
                        e[i - 1] = r;
                    f = cosr*d[i] + sinr*e[i];
                    e[i] = cosr*e[i] - sinr*d[i];
                    g = sinr*d[i+1];
                    d[i + 1] = cosr*d[i + 1];
                    lartg(f, g, &cosl, &sinl, &r);
                    d[i] = r;
                    f = cosl * e[i] + sinl * d[i + 1];
                    d[i + 1] = cosl * d[i + 1] - sinl * e[i];
                    if (i < M - 1) {
                        g = sinl * e[i + 1];
                        e[i + 1] = cosl * e[i + 1];
                    }
                    hcosr[i - ll] = cosr;
                    hsinr[i - ll] = sinr;
                    hcosl[i - ll] = cosl;
                    hsinl[i - ll] = sinl;
                }
                e[M-1] = f;

                /* update singular vectors */
                if(VT){
                    copy(hcosr, gcosr);
                    copy(hsinr, gsinr);
                    lasr('L', 'V', 'F', gcosr, gsinr, (*VT)({ll, M+1}, {0, end}));
                }
                if(U){
                    copy(hcosl, gcosl);
                    copy(hsinl, gsinl);
                    lasr('R', 'V', 'F', gcosl, gsinl, (*U)({0, end}, {ll, M+1}));
                }

//                int ii, jj;
//                for(ii = 0 ; ii < N -1  ; ++ii)
//                  printf("%f ", hcosr[ii]);
//                printf("\n");

//                for(ii = 0 ; ii < N -1  ; ++ii)
//                  printf("%f ", hsinr[ii]);
//                printf("\n");
//                printf("%d %d\n", ll, M+1-ll);
//                std::cout << *VT << std::endl;

                /* Test convergence */
                if(abs(e[M-1]) <= thresh)
                    e[M-1] = 0;
            }
            else
            {
                /* Chase bulge from bottom to top */
                float f = (abs(d[M]) - shift)*(sign(1, d[M]) + shift/d[M]);
                float g = e[M - 1];
                for (int i = M; i >= ll + 1; --i)
                {
                    lartg(f, g, &cosr, &sinr, &r);
                    if (i < M)
                        e[i] = r;
                    f = cosr * d[i] + sinr * e[i - 1];
                    e[i - 1] = cosr * e[i - 1] - sinr * d[i];
                    g = sinr * d[i - 1];
                    d[i - 1] = cosr * d[i - 1];
                    lartg(f, g, &cosl, &sinl, &r);
                    d[i] = r;
                    f = cosl * e[i - 1] + sinl * d[i - 1];
                    d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
                    if (i > ll + 1) {
                        g = sinl * e[i - 2];
                        e[i - 2] = cosl * e[i - 2];
                    }
                    hcosr[i - ll] = cosr;
                    hsinr[i - ll] = -sinr;
                    hcosl[i - ll] = cosl;
                    hsinl[i - ll] = -sinl;
                }
                e[ll] = f;
                /* update singular vectors */
                if(VT){
                    copy(hcosr, gcosr);
                    copy(hsinr, gsinr);
                    lasr('L', 'V', 'B', gcosr, gsinr, (*VT)({ll,M+1}, {1, end}));
                }
                if(U){
                    copy(hcosl, gcosl);
                    copy(hsinl, gsinl);
                    lasr('R', 'V', 'B', gcosl, gsinl, (*U)({1, end}, {ll, M+1}));
                }
                /* Test convergence */
                if(abs(e[ll]) <= thresh)
                    e[ll] = 0;
            }
        }
    }
    /* All singular values converged, so make them positive */
    for(int i = 0 ; i < N ; ++i)
        if(d[i] < 0){
            d[i] = -d[i];
            if (VT)
              (*VT)(i,{0,end}) *= -1;
        }

    /* Sort singular values into decreasing order */
    for(int i = 0 ; i < N - 1 ; ++i){
        int isub = 0;
        float smin = d[0];
        for(int j = 1 ; j < N - i ; ++j){
            if(d[j] <= smin){
                isub = j;
                smin = d[j];
            }
        }
        /* Swap singular values */
        std::swap(d[isub], d[N - i - 1]);

        /* Swap singular vectors */
        if(VT)
            swap((*VT)(isub, {1, end}), (*VT)(N - i - 1, {1, end}));
        if(U)
            swap((*U)({1, end}, isub), (*U)({1, end}, N - i - 1));
    }


  }

}
