#ifndef CAFFE2_OPERATORS_ROI_ENTROPY_OP_H_
#define CAFFE2_OPERATORS_ROI_ENTROPY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class StatOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  StatOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        cur_iter_(0),
        init_(true),
        display_(this->template GetSingleArgument<int32_t>("display", 1280)),
        prefix_(this->template GetSingleArgument<string>("prefix", "")),
        debug_info_(
            this->template GetSingleArgument<bool>("debug_info", false)) {}
  ~StatOp() {}

  bool RunOnDevice() override;

 protected:
  bool debug_info_;
  string prefix_;

  int display_;
  int cur_iter_;
  bool init_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_ROI_ENTROPY_OP_H_
