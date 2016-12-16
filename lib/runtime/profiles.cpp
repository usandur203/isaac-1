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

#include <fstream>
#include <algorithm>
#include <memory>
#include <numeric>

#include "rapidjson/document.h"
#include "rapidjson/to_array.hpp"

#include "isaac/driver/program_cache.h"
#include "isaac/runtime/profiles.h"
#include "isaac/jit/generation/elementwise_1d.h"
#include "isaac/jit/generation/reduce_1d.h"
#include "isaac/jit/generation/elementwise_2d.h"
#include "isaac/jit/generation/reduce_2d.h"
#include "isaac/jit/generation/gemm.h"
#include "isaac/exception/api.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/tools/sys/getenv.hpp"
#include "isaac/tools/cpp/string.hpp"
#include "isaac/tools/cpp/timer.hpp"
#include "isaac/tools/sys/mkdir.hpp"

namespace isaac
{
namespace runtime
{

template <typename Word>
std::ostream& write_word(std::ostream& outs, Word value ){
  for (unsigned size = sizeof( Word ); size; --size, value >>= 8)
    outs.put( static_cast <char> (value & 0xFF) );
  return outs;
}

template <typename Word>
std::istream& read_word( std::istream& ins, Word& value){
  value = 0;
  for (unsigned size = 0; size < sizeof( Word ); ++size)
    value |= ins.get() << (8 * size);
  return ins;
}

driver::Program const & profiles::value_type::init(runtime::execution_handler const & expression)
{
  driver::Context & context = (driver::Context&)expression.x().context();
  std::string pname;
  runtime::compilation_options_type const & opt = expression.compilation_options();
  pname = opt.program_name;
  if(pname.empty())
    pname = symbolic::hash(expression.x());
  driver::Program const * program = programs_.find(pname);
  if(program)
      return *program;
  std::string srcs;
   for(unsigned int i = 0 ; i < templates_.size() ; ++i)
     srcs += templates_[i]->generate(tools::to_string(i), expression.x(), context.device());
   return programs_.add(context, pname, srcs);
}

profiles::value_type::value_type(numeric_type dtype, std::vector< std::shared_ptr<templates::base> > const & templates, driver::CommandQueue const & queue) :
  templates_(templates), queue_(queue), programs_(driver::backend::programs::get(queue,templates[0]->type(),dtype)), labels_cache_(new std::fstream(),
                                                                                                                                   [](std::fstream* ptr){ptr->close();})
{
  expression_type type = templates[0]->type();
  programs_.clear();
  std::string labels_path = queue_.context().cache_prefix() + "/labels/";
  labels_path += std::to_string(type) + "_" + to_string(dtype);
  tools::mkpath(labels_path);
  labels_cache_->open(labels_path.c_str(), std::ios::in | std::ios::binary);
  if(labels_cache_){
    std::vector<int_t> shapes(nshapes(type));
    uint8_t k;
    while (labels_cache_->peek()!=std::fstream::traits_type::eof()) {
      for(auto& s: shapes)
        read_word(*labels_cache_, s);
      read_word(*labels_cache_, k);
      labels_.insert({shapes, k});
    }
    labels_cache_->close();
  }
  labels_cache_->open(labels_path.c_str(), std::ios::out | std::ios::app | std::ios::binary);
}

void profiles::value_type::execute(runtime::execution_handler const & expr)
{
  static const int MAX_TEMPORARY_WORKSPACE = 1e6;
  expression_tree const & tree = expr.x();
  driver::Program const & program = init(expr);
  std::vector<int_t> shapes = templates_[0]->input_sizes(tree);

  size_t label = 0;
  auto it = labels_.find(shapes);
  //Cached
  if(it!=labels_.end())
    label = it->second;
  //Not cached
  else
  {
    expression_tree::node const & root = tree[tree.root()];
    expression_tree::node const & left = tree[root.binary_operator.lhs];
    array_base* out = left.array.base;
    auto read_out = [&](expression_tree::node const & x){
      return x.type == DENSE_ARRAY_TYPE && (&x != &left) && x.array.base == out;
    };
    bool modify_output = std::find_if(tree.data().begin(), tree.data().end(), read_out) != tree.data().end();
    std::unique_ptr<array> bkp;
    if(modify_output){
      bkp.reset(new array(out->shape(), out->dtype(), queue_.context()));
      *bkp = execution_handler(-(-*out), execution_options_type(queue_));
    }
    tools::Timer tmr;
    std::vector<double> times;
    for(size_t i = 0 ; i < templates_.size() ; i++){
      if(templates_[i]->temporary_workspace(tree) > MAX_TEMPORARY_WORKSPACE){
        times.push_back(INFINITY);
        continue;
      }
      try{
        double total_time = 0;
        std::vector<double> ctimes;
        while(total_time < 1e-2){
          tmr.start();
          templates_[i]->enqueue(queue_, program, tools::to_string(i), runtime::execution_handler(tree));
          queue_.synchronize();
          ctimes.push_back(1e-9*tmr.get().count());
          total_time += ctimes.back();
        }
        times.push_back( *std::min_element(ctimes.begin(), ctimes.end()));
      }catch(...){
        times.push_back(INFINITY);
      }
    }
    label = std::distance(times.begin(),std::min_element(times.begin(), times.end()));
    if(modify_output)
      *out = execution_handler(-(-*bkp), execution_options_type(queue_));
    labels_.insert({shapes, label});
    for(auto s: shapes)
      write_word(*labels_cache_, s);
    write_word(*labels_cache_, (uint8_t)label);
    labels_cache_->flush();
  }
  if(templates_[label]->temporary_workspace(expr.x()) > MAX_TEMPORARY_WORKSPACE)
    throw operation_not_supported_exception("Running this operation would require an overly large temporary.");
  templates_[label]->enqueue(queue_, program, tools::to_string(label), expr);
}

profiles::value_type::templates_container const & profiles::value_type::templates() const
{
    return templates_;
}

std::shared_ptr<templates::base> profiles::create(std::string const & op, std::string const & str)
{
    if(str=="cublas_gemm"){
        if(op=="gemm_nn") return std::shared_ptr<templates::base>(new templates::cublas_gemm('N', 'N'));
        if(op=="gemm_nt") return std::shared_ptr<templates::base>(new templates::cublas_gemm('N', 'T'));
        if(op=="gemm_tn") return std::shared_ptr<templates::base>(new templates::cublas_gemm('T', 'N'));
        if(op=="gemm_tt") return std::shared_ptr<templates::base>(new templates::cublas_gemm('T', 'T'));
    }
    throw;
}

std::shared_ptr<templates::base> profiles::create(std::string const & template_name, std::vector<int> const & x)
{
  if(template_name=="elementwise_1d")
    return std::shared_ptr<templates::base>(new templates::elementwise_1d(x[0], x[1], x[2]));
  else if(template_name=="reduce_1d")
    return std::shared_ptr<templates::base>(new templates::reduce_1d(x[0], x[1], x[2]));
  else if(template_name=="elementwise_2d")
    return std::shared_ptr<templates::base>(new templates::elementwise_2d(x[0], x[1], x[2], x[3], x[4]));
  else if(template_name.find("reduce_2d_rows")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::reduce_2d_rows(x[0], x[1], x[2], x[3], x[4]));
  else if(template_name.find("reduce_2d_cols")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::reduce_2d_cols(x[0], x[1], x[2], x[3], x[4]));
  else if(template_name.find("gemm_nn")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::gemm_nn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]));
  else if(template_name.find("gemm_tn")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::gemm_tn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]));
  else if(template_name.find("gemm_nt")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::gemm_nt(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]));
  else if(template_name.find("gemm_tt")!=std::string::npos)
    return std::shared_ptr<templates::base>(new templates::gemm_tt(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]));
  else
    throw std::invalid_argument("Invalid expression: " + template_name);
}

void profiles::import(std::string const & str, driver::CommandQueue const & queue)
{
  map_type & result = cache_[queue];
  //Parse the JSON document
  rapidjson::Document document;
  document.Parse<0>(str.c_str());
  //Deserialize
  std::vector<std::string> operations = {"elementwise_1d", "reduce_1d", "elementwise_2d", "reduce_2d_rows", "reduce_2d_cols", "gemm_nn", "gemm_tn", "gemm_nt", "gemm_tt"};
  std::vector<std::string> dtype = {"float32", "float64"};
  for(auto & operation : operations)
  {
    const char * opcstr = operation.c_str();
    if(document.HasMember(opcstr))
    {
      expression_type etype = expression_type_from_string(operation);
      for(auto & elem : dtype)
      {
        const char * dtcstr = elem.c_str();
        if(document[opcstr].HasMember(dtcstr))
        {
          numeric_type dtype = numeric_type_from_string(elem);
          // Get profiles
          std::vector<std::shared_ptr<templates::base> > templates;
          rapidjson::Value const & profiles = document[opcstr][dtcstr]["profiles"];
          for (rapidjson::SizeType i = 0 ; i < profiles.Size() ; ++i){
            if(profiles[i].IsString())
                 templates.push_back(create(operation, profiles[i].GetString()));
            else
                templates.push_back(create(operation, rapidjson::to_int_array<int>(profiles[i])));
          }
          result[{etype, dtype}] = std::make_shared<value_type>(dtype, templates, queue);
        }
      }
    }
  }
}

profiles::map_type& profiles::init(driver::CommandQueue const & queue)
{
  map_type & map = cache_[queue];
  driver::Device const & device = queue.device();
  //Default
  import(presets_.at(std::make_tuple(driver::Device::Type::UNKNOWN, driver::Device::Vendor::UNKNOWN, driver::Device::Architecture::UNKNOWN)), queue);
  //Database profile
  presets_type::const_iterator it = presets_.find(std::make_tuple(device.type(), device.vendor(), device.architecture()));
  if(it!=presets_.end())
      import(it->second, queue);
  //User-provided profile
  std::string homepath = tools::getenv("HOME");
  if(homepath.size())
  {
    std::string json_path = homepath + "/.isaac/devices/device0.json";
    std::ifstream ifs(json_path);
    if(!ifs)
        return map;
    std::string str;
    ifs.seekg(0, std::ios::end);
    str.reserve(ifs.tellg());
    ifs.seekg(0, std::ios::beg);
    str.assign((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    import(str, queue);
  }

  return map;
}

profiles::map_type& profiles::get(driver::CommandQueue const & queue)
{
  std::map<driver::CommandQueue, map_type>::iterator it = cache_.find(queue);
  if(it == cache_.end())
    return init(queue);
  return it->second;
}

void profiles::set(driver::CommandQueue const & queue, expression_type operation, numeric_type dtype, std::shared_ptr<value_type> const & profile)
{ cache_[queue][std::make_pair(operation,dtype)] = profile; }

void profiles::release()
{ cache_.clear(); }

std::map<driver::CommandQueue, profiles::map_type> profiles::cache_;

}
}
