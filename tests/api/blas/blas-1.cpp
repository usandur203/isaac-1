#include <cmath>
#include <iostream>
#include <type_traits>
#include "api.hpp"
#include "isaac/array.h"
#include "isaac/driver/common.h"
#include "clBLAS.h"
#include "cublas.h"

namespace sc = isaac;
typedef isaac::int_t int_t;


template<typename T>
void test_impl(std::string const & ST, simple_vector_base<T> & cx, simple_vector_base<T>& cy, sc::array_base& x, sc::array_base& y, int& nfail, int& npass)
{
  std::string DT = std::is_same<T, float>::value?"S":"D";
  T a = -4.3;
  int_t N = cy.size();
  sc::driver::CommandQueue & queue = sc::driver::backend::queues::get(x.context(),0);
  //Real scalar
  T cs = 0;
  sc::scalar ds(cs, y.context());
  //Int scalar
  int ci = 0;
  sc::scalar di(ci, y.context());
  //Scratch
  sc::array scratch(N, y.dtype());
  // AXPY
  auto axpy = [&]() { for(int i = 0 ; i < cy.size() ; ++i) cy[i] = a*cx[i] + cy[i]; };
  auto copy = [&]() { for(int i = 0 ; i < cy.size() ; ++i) cy[i] = cx[i]; };
  auto scal = [&]() { for(int i = 0 ; i < cy.size() ; ++i) cy[i] = a*cy[i]; };
  auto dot = [&]() { cs = 0; for(int i = 0 ; i < cy.size() ; ++i) cs += cx[i]*cy[i]; };
  auto asum = [&]() { cs = 0; for(int i = 0 ; i < cy.size() ; ++i) cs += std::fabs(cx[i]); };
  auto iamax = [&]() { ci = 0; cs = -INFINITY; for(int i = 0 ; i < cx.size() ; ++i){ cs = std::max(cs, std::abs(cx[i])); if(cs==std::abs(cx[i])) ci = i; } };
  auto iamin = [&]() { ci = 0; cs = INFINITY; for(int i = 0 ; i < cx.size() ; ++i){ cs = std::min(cs, std::abs(cx[i])); if(cs==std::abs(cx[i])) ci = i; } };
  if(queue.device().backend()==sc::driver::OPENCL)
  {
      cl_command_queue clqueue = queue.handle().cl();
      test<T>(DT+"AXPY"+ST, axpy, [&]{ BLAS<T>::F(clblasSaxpy, clblasDaxpy)(N, a, cl(x), off(x), inc(x), cl(y), off(y), inc(y), 1, &clqueue, 0, NULL, NULL); }, y, cy, nfail, npass);
      test<T>(DT+"COPY"+ST, copy, [&]{ BLAS<T>::F(clblasScopy, clblasDcopy)(N, cl(x), off(x), inc(x),  cl(y), off(y), inc(y),  1, &clqueue, 0, NULL, NULL); }, y, cy, nfail, npass);
      test<T>(DT+"SCAL"+ST, scal, [&]{ BLAS<T>::F(clblasSscal, clblasDscal)(N, a, cl(y), off(y), inc(y), 1, &clqueue, 0, NULL, NULL); }, y, cy, nfail, npass);
      test<T>(DT+"DOT"+ST, dot, [&]{ ds = BLAS<T>::F(clblasSdot, clblasDdot)(N, cl(ds), 0, cl(x), off(x), inc(x), cl(y), off(y), inc(y),  cl(scratch), 1, &clqueue, 0, NULL, NULL);}, y, cy, nfail, npass);
      test<T>(DT+"ASUM"+ST, asum, [&]{  ds = BLAS<T>::F(clblasSasum, clblasDasum)(N, cl(ds), 0, cl(x), off(x), inc(x), cl(scratch), 1, &clqueue, 0, NULL, NULL); }, y, cy, nfail, npass);
      test<T>(DT+"IAMAX"+ST, iamax, [&]{  di = BLAS<T>::F(clblasiSamax, clblasiDamax)(N, cl(di), 0, cl(x), off(x), inc(x), cl(scratch), 1, &clqueue, 0, NULL, NULL); }, y, cy, nfail, npass);
  }
  if(queue.device().backend()==sc::driver::CUDA)
  {
    test<T>(DT+"AXPY"+ST, axpy, [&]{ BLAS<T>::F(cublasSaxpy, cublasDaxpy)(N, a, (T*)cu(x) + off(x), inc(x), (T*)cu(y) + off(y), inc(y)); }, y, cy, nfail, npass);
    test<T>(DT+"COPY"+ST, copy, [&]{ BLAS<T>::F(cublasScopy, cublasDcopy)(N, (T*)cu(x) + off(x), inc(x), (T*)cu(y) + off(y), inc(y)); }, y, cy, nfail, npass);
    test<T>(DT+"SCAL"+ST, scal, [&]{ BLAS<T>::F(cublasSscal, cublasDscal)(N, a, (T*)cu(y) + off(y), inc(y)); }, y, cy, nfail, npass);
    test<T>(DT+"DOT"+ST, dot, [&]{ ds = BLAS<T>::F(cublasSdot, cublasDdot)(N, (T*)cu(x) + off(x), inc(x), (T*)cu(y) + off(y), inc(y)); }, y, cy, nfail, npass);
    test<T>(DT+"ASUM"+ST, asum, [&]{ ds = BLAS<T>::F(cublasSasum, cublasDasum)(N, (T*)cu(x) + off(x), inc(x)); }, y, cy, nfail, npass);
    test<T>(DT+"IAMAX"+ST, iamax, [&]{  di =  BLAS<T>::F(cublasIsamax, cublasIdamax)(N, (T*)cu(x) + off(x), inc(x)); }, di, ci, nfail, npass);
    test<T>(DT+"IAMIN"+ST, iamin, [&]{  di =  BLAS<T>::F(cublasIsamin, cublasIdamin)(N, (T*)cu(x) + off(x), inc(x)); }, di, ci, nfail, npass);
  }

}

template<typename T>
void test(sc::driver::Context const & ctx, int& nfail, int& npass)
{
  int_t N = 10007;
  int_t SUBN = 7;
  INIT_VECTOR(N, SUBN, 3, 11, cx, x, ctx);
  INIT_VECTOR(N, SUBN, 3, 11, cy, y, ctx);
  test_impl("-FULL", cx, cy, x, y, nfail, npass);
  test_impl("-SUB", cx_s, cy_s, x_s, y_s, nfail, npass);
}

int main()
{
  clblasSetup();
  int err = run_test(test<float>, test<double>);
  clblasTeardown();
  return err;
}
