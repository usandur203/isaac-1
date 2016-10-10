/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

#include "isaac/array.h"
#include "isaac/runtime/execute.h"
#include "cublas.h"

namespace sc = isaac;
using sc::driver::Buffer;
using sc::assign;

//Opaque context structure
class cublasContext
{
public:
  cublasContext(sc::driver::Context const & ctx);
  sc::driver::CommandQueue const & queue() const;
  sc::driver::Context const & context() const;
  void update(CUstream stream);
private:
  sc::driver::CommandQueue init_queue_; //Keeps ownership of allocated handle even when inactive
  sc::driver::CommandQueue active_queue_;
};

cublasContext::cublasContext(sc::driver::Context const & ctx): init_queue_(ctx), active_queue_(init_queue_)
{ }

sc::driver::CommandQueue const & cublasContext::queue() const
{ return active_queue_; }

sc::driver::Context const & cublasContext::context() const
{ return active_queue_.context(); }

void cublasContext::update(CUstream stream)
{ active_queue_ = sc::driver::CommandQueue(context(), stream, false); }



//Actual functions implementation
extern "C"
{

inline cublasOperation_t cvt_trans(char c)
{
  if(c=='n' || c=='N') return CUBLAS_OP_N;
  if(c=='t' || c=='T') return CUBLAS_OP_T;
  return CUBLAS_OP_C;
}

inline cublasHandle_t alloc_default_handle()
{ return new cublasContext(sc::driver::backend::contexts::get_default()); }

static cublasHandle_t dft_handle = alloc_default_handle();

cublasStatus cublasInit()
{
  if(!dft_handle)
    dft_handle = alloc_default_handle();
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus cublasShutdown()
{
  cublasDestroy_v2(dft_handle);
  isaac::runtime::profiles::release();
  isaac::driver::backend::release();
  return CUBLAS_STATUS_SUCCESS;
}

//*****************
//Context
//*****************

cublasStatus_t cublasCreate_v2 (cublasHandle_t *handle)
{
  CUcontext ctx;
  sc::driver::dispatch::cuCtxGetCurrent(&ctx);
  *handle = new cublasContext(sc::driver::Context(ctx, false));
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2 (cublasHandle_t handle)
{
  if(handle){
    delete handle;
    handle = 0;
  }
  return CUBLAS_STATUS_SUCCESS;
}

//*****************
//Stream
//*****************

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId)
{
  handle->update((CUstream)streamId);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *streamId)
{
  *streamId = (cudaStream_t)handle->queue().handle().cu();
  return CUBLAS_STATUS_SUCCESS;
}

static cublasStatus_t execute(cublasHandle_t handle, sc::expression_tree const & operation)
{
  sc::runtime::execution_options_type options(handle->queue());
  sc::runtime::execute(sc::runtime::execution_handler(operation, options));
  return CUBLAS_STATUS_SUCCESS;
}

//*****************
//BLAS1
//*****************

//AXPY
#define MAKE_AXPY(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
  cublasStatus_t cublas ## TYPE_CHAR ## axpy_v2 (cublasHandle_t handle, int n, const TYPE_CU *alpha,\
  const TYPE_CU *x, int incx, TYPE_CU *y, int incy)\
{\
  sc::array dx(n, TYPE_ISAAC, Buffer((CUdeviceptr)x,false), 0, incx); \
  sc::array dy(n, TYPE_ISAAC, Buffer((CUdeviceptr)y,false), 0, incy); \
  return execute(handle, assign(dy, *alpha*dx + dy));\
}\
  \
  void cublas ## TYPE_CHAR ## axpy (int n, TYPE_CU alpha, const TYPE_CU *x, int incx, TYPE_CU *y, int incy)\
{ cublas ## TYPE_CHAR ## axpy_v2(dft_handle, n, &alpha, x, incx, y, incy); }

MAKE_AXPY(S, sc::FLOAT_TYPE, float)
MAKE_AXPY(D, sc::DOUBLE_TYPE, double)

//COPY
#define MAKE_COPY(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
  cublasStatus_t cublas ## TYPE_CHAR ## copy_v2 (cublasHandle_t handle, int n, const TYPE_CU *x, int incx, TYPE_CU *y, int incy)\
{\
  sc::array dx(n, TYPE_ISAAC, Buffer((CUdeviceptr)x,false), 0, incx); \
  sc::array dy(n, TYPE_ISAAC, Buffer((CUdeviceptr)y,false), 0, incy); \
  return execute(handle, assign(dy,dx));\
}\
\
void cublas ## TYPE_CHAR ## copy (int n, const TYPE_CU *x, int incx, TYPE_CU *y, int incy)\
{ cublas ## TYPE_CHAR ## copy_v2(dft_handle, n, x, incx, y, incy); }

MAKE_COPY(S, sc::FLOAT_TYPE, float)
MAKE_COPY(D, sc::DOUBLE_TYPE, double)

//SCAL
#define MAKE_SCAL(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
  cublasStatus_t cublas ## TYPE_CHAR ## scal_v2 (cublasHandle_t handle, int n, const TYPE_CU * alpha, TYPE_CU *x, int incx)\
{\
  sc::array dx(n, TYPE_ISAAC, Buffer((CUdeviceptr)x,false), 0, incx); \
  return execute(handle, assign(dx,*alpha*dx));\
}\
\
void cublas ## TYPE_CHAR ## scal (int n, TYPE_CU alpha, TYPE_CU *x, int incx)\
{   cublas ## TYPE_CHAR ## scal_v2(dft_handle, n, &alpha, x, incx); }\

MAKE_SCAL(S, sc::FLOAT_TYPE, float)
MAKE_SCAL(D, sc::DOUBLE_TYPE, double)

//DOT
#define MAKE_DOT(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
  cublasStatus_t cublas ## TYPE_CHAR ## dot_v2 (cublasHandle_t handle, int n, const TYPE_CU *x, int incx, const TYPE_CU *y, int incy, TYPE_CU* result)\
{\
  sc::array dx(n, TYPE_ISAAC, Buffer((CUdeviceptr)x,false), 0, incx); \
  sc::array dy(n, TYPE_ISAAC, Buffer((CUdeviceptr)y,false), 0, incy); \
  sc::scalar scr(TYPE_ISAAC);\
  cublasStatus_t status = execute(handle, assign(scr, sc::dot(dx,dy)));\
  *result = scr;\
  return status;\
}\
\
TYPE_CU cublas ## TYPE_CHAR ## dot (int n, const TYPE_CU *x, int incx, const TYPE_CU *y, int incy)\
{\
  TYPE_CU result;\
  cublas ## TYPE_CHAR ## dot_v2(dft_handle, n, x, incx, y, incy, &result);\
  return result;\
}\

MAKE_DOT(S, sc::FLOAT_TYPE, float)
MAKE_DOT(D, sc::DOUBLE_TYPE, double)

//ASUM
#define MAKE_ASUM(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
cublasStatus_t cublas ## TYPE_CHAR ## asum_v2 (cublasHandle_t handle, int n, const TYPE_CU *x, int incx, TYPE_CU* result)\
{\
  sc::array dx(n, TYPE_ISAAC, Buffer((CUdeviceptr)x,false), 0, incx); \
  sc::scalar scr(TYPE_ISAAC);\
  cublasStatus_t status = execute(handle, assign(scr, sum(abs(dx))));\
  *result = scr;\
  return status;\
}\
\
TYPE_CU cublas ## TYPE_CHAR ## asum (int n, const TYPE_CU *x, int incx)\
{\
  TYPE_CU result;\
  cublas ## TYPE_CHAR ## asum_v2(dft_handle, n, x, incx, &result);\
  return result;\
}\

MAKE_ASUM(S, sc::FLOAT_TYPE, float)
MAKE_ASUM(D, sc::DOUBLE_TYPE, double)

//*****************
//BLAS2
//*****************

#define MAKE_GEMV(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
  cublasStatus_t cublas ## TYPE_CHAR ## gemv_v2 (cublasHandle_t handle, cublasOperation_t trans, int m,  int n, const TYPE_CU *alpha,\
  const TYPE_CU *A, int lda, const TYPE_CU *x, int incx, const TYPE_CU *beta, TYPE_CU *y, int incy)\
{\
  if(trans==CUBLAS_OP_C)\
    return CUBLAS_STATUS_NOT_SUPPORTED;\
  bool AT = trans==CUBLAS_OP_T;\
  sc::array dA(m, n, TYPE_ISAAC, Buffer((CUdeviceptr)A, false), 0, lda);\
  sc::int_t sx = n;\
  sc::int_t sy = m;\
  if(AT)\
    std::swap(sx, sy);\
  sc::array dx(sx, TYPE_ISAAC, Buffer((CUdeviceptr)x, false), 0, incx);\
  sc::array dy(sy, TYPE_ISAAC, Buffer((CUdeviceptr)y, false), 0, incy);\
  \
  if(AT)\
    return execute(handle, assign(dy, *alpha*dot(dA.T, dx) + *beta*dy));\
  else\
    return execute(handle, assign(dy, *alpha*dot(dA, dx) + *beta*dy));\
}\
\
void cublas ## TYPE_CHAR ## gemv (char trans, int m, int n, TYPE_CU alpha,\
const TYPE_CU *A, int lda, const TYPE_CU *x, int incx,\
TYPE_CU beta, TYPE_CU *y, int incy)\
{ cublas ## TYPE_CHAR ## gemv_v2(dft_handle, cvt_trans(trans), m, n, &alpha, A, lda, x, incx, &beta, y, incy); }

MAKE_GEMV(S, sc::FLOAT_TYPE, float)
MAKE_GEMV(D, sc::DOUBLE_TYPE, double)


#define MAKE_GER(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
  cublasStatus_t cublas ## TYPE_CHAR ## ger_v2 (cublasHandle_t handle, int m, int n, const TYPE_CU * alpha, const TYPE_CU *x, int incx,\
  const TYPE_CU *y, int incy, TYPE_CU *A, int lda)\
{\
  sc::array dx(n, TYPE_ISAAC, Buffer((CUdeviceptr)x,false), 0, incx); \
  sc::array dy(n, TYPE_ISAAC, Buffer((CUdeviceptr)y,false), 0, incy); \
  sc::array dA(m, n, TYPE_ISAAC, Buffer((CUdeviceptr)A, false), 0, lda);\
  return execute(handle, assign(dA, *alpha*outer(dx, dy) + dA));\
}\
\
  void cublas ## TYPE_CHAR ## ger (int m, int n, TYPE_CU alpha, const TYPE_CU *x, int incx,\
  const TYPE_CU *y, int incy, TYPE_CU *A, int lda)\
{ cublas ## TYPE_CHAR ## ger_v2(dft_handle, m, n, &alpha, x, incx, y, incy, A, lda); }\

MAKE_GER(S, sc::FLOAT_TYPE, float)
MAKE_GER(D, sc::DOUBLE_TYPE, double)


//*****************
//BLAS3
//*****************

#define MAKE_GEMM(TYPE_CHAR, TYPE_ISAAC, TYPE_CU) \
cublasStatus_t cublas ## TYPE_CHAR ## gemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,\
  int m, int n, int k, const TYPE_CU *alpha, const TYPE_CU *A,\
  int lda, const TYPE_CU *B, int ldb,const TYPE_CU *beta, TYPE_CU *C, int ldc)\
{\
  if(transa==CUBLAS_OP_C || transb==CUBLAS_OP_C)\
    return CUBLAS_STATUS_NOT_SUPPORTED;\
  bool AT = transa==CUBLAS_OP_T;\
  bool BT = transb==CUBLAS_OP_T;\
  TYPE_CU a = *alpha;\
  TYPE_CU b = *beta;\
  if(k==1 && m>1 && n>1){\
    sc::array dA(m, TYPE_ISAAC, Buffer((CUdeviceptr)A, false), 0, AT?lda:1);\
    sc::array dB(n, TYPE_ISAAC, Buffer((CUdeviceptr)B, false), 0, BT?1:ldb);\
    sc::array dC(m, n, TYPE_ISAAC, Buffer((CUdeviceptr)C, false), 0, ldc);\
    return execute(handle, assign(dC, a*sc::outer(dA, dB) + b*dC));\
  }\
  sc::int_t As1 = m, As2 = k;\
  sc::int_t Bs1 = k, Bs2 = n;\
  if(AT)\
    std::swap(As1, As2);\
  if(BT)\
    std::swap(Bs1, Bs2);\
  /*Struct*/\
  sc::array dA(As1, As2, TYPE_ISAAC, Buffer((CUdeviceptr)A, false), 0, lda);\
  sc::array dB(Bs1, Bs2, TYPE_ISAAC, Buffer((CUdeviceptr)B, false), 0, ldb);\
  sc::array dC(m, n, TYPE_ISAAC, Buffer((CUdeviceptr)C, false), 0, ldc);\
  /*Operation*/\
  if(AT && BT)\
    return execute(handle, assign(dC, a*dot(dA.T, dB.T) + b*dC));\
  else if(AT && !BT)\
    return execute(handle, assign(dC, a*dot(dA.T, dB) + b*dC));\
  else if(!AT && BT)\
    return execute(handle, assign(dC, a*dot(dA, dB.T) + b*dC));\
  else\
    return execute(handle, assign(dC, a*dot(dA, dB) + b*dC));\
}\
\
void cublas ## TYPE_CHAR ## gemm (char transa, char transb, int m, int n, int k,\
  TYPE_CU alpha, const TYPE_CU *A, int lda,\
  const TYPE_CU *B, int ldb, TYPE_CU beta, TYPE_CU *C,\
  int ldc)\
{ cublas ## TYPE_CHAR ## gemm_v2(dft_handle, cvt_trans(transa), cvt_trans(transb), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc); }\

MAKE_GEMM(S, sc::FLOAT_TYPE, cl_float)
MAKE_GEMM(D, sc::DOUBLE_TYPE, cl_double)
}
