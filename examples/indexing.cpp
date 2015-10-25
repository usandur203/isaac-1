#include "isaac/symbolic/execute.h"
#include "isaac/array.h"

namespace sc = isaac;

void test(sc::view A)
{
  A(0,0) = (float)123;
}

int main()
{
    static const char * sline = "--------------------";
    static const char * dline = "====================";

    std::cout << dline << std::endl;
    std::cout << "Tutorial: Indexing " << std::endl;
    std::cout << dline << std::endl;

    sc::int_t M = 10, N = 10;

    std::vector<float> data(M*N);
    for(unsigned int i = 0 ; i < data.size(); ++i)
      data[i] = i;
    sc::array A(M, N, data);

    std::cout << "A:" << std::endl;
    std::cout << sline << std::endl;
    std::cout << A << std::endl;
    std::cout << std::endl;

//    std::cout << "A[3, 2:end]:" << std::endl;
//    std::cout << sline << std::endl;
//    std::cout << A(3, {2,sc::end}) << std::endl;
//    std::cout << std::endl;

//    std::cout << "A[2:end, 4]:" << std::endl;
//    std::cout << sline << std::endl;
//    std::cout << A({2,sc::end}, 4) << std::endl;
//    std::cout << std::endl;

//    std::cout << "diag(A,  1): " << std::endl;
//    std::cout << sline << std::endl;
//    std::cout << sc::diag(A, 1) << std::endl;
//    std::cout << std::endl;

//    std::cout << "diag(A, -7): " << std::endl;
//    std::cout << sline << std::endl;
//    std::cout << sc::diag(A, -7) << std::endl;
//    std::cout << std::endl;

    using sc::_i0;
    sc::execute(sc::assign(row(A, 1),row(A, 0)));
//    sc::execute(sfor(_i0 = 8, _i0 >= 0, _i0-=1, sc::assign(row(A, _i0 + 1),row(A, _i0))));
    std::cout << A << std::endl;
}
