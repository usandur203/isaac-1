#ifndef ISAAC_BACKEND_BINDER_H
#define ISAAC_BACKEND_BINDER_H

#include <map>
#include "isaac/driver/buffer.h"

namespace isaac
{

enum binding_policy_t
{
  BIND_INDEPENDENT,
  BIND_SEQUENTIAL
};

class array;

class symbolic_binder
{
public:
  symbolic_binder();
  virtual ~symbolic_binder();
  virtual bool bind(array const * a, bool) = 0;
  virtual unsigned int get(array const * a, bool) = 0;
  unsigned int get();
protected:
  unsigned int current_arg_;
  std::map<array const *,unsigned int> memory;
};


class bind_sequential : public symbolic_binder
{
public:
  bind_sequential();
  bool bind(array const * a, bool);
  unsigned int get(array const * a, bool);
};

class bind_independent : public symbolic_binder
{
public:
  bind_independent();
  bool bind(array const * a, bool);
  unsigned int get(array const * a, bool);
};

}

#endif
