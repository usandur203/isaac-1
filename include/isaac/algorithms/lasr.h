#ifndef ISAAC_ALGORTITHMS_LASR_H
#define ISAAC_ALGORTITHMS_LASR_H

#include "isaac/array.h"

namespace isaac
{

void lasr(char side, char pivot, char direct,  array_base const & cos, array_base const & sin, view A);

}

#endif
