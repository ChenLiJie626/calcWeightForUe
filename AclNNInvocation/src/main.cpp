#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "common.h"
#include "op_runner.h"

namespace {
constexpr int64_t FEATURES = 256;
} // namespace

bool g_isDevice = false;
int deviceId = 0;

void ResolveDeviceId()
{
    const char *deviceIdEnv = std::getenv("DEVICE_ID");
    if (deviceIdEnv == nullptr || deviceIdEnv[0] == '\0') {
        deviceIdEnv = std::getenv("ASCEND_DEVICE_ID");
    }
    if (deviceIdEnv != nullptr && deviceIdEnv[0] != '\0') {
        deviceId = std::atoi(deviceIdEnv);
    }
}

int64_t CountUInt32Elements(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0 || st.st_size % static_cast<off_t>(sizeof(uint32_t)) != 0) {
        ERROR_LOG("Invalid uint32 input file: %s", path);
        return 0;
    }
    return static_cast<int64_t>(st.st_size / static_cast<off_t>(sizeof(uint32_t)));
}

int64_t CountFloatElements(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0 || st.st_size % static_cast<off_t>(sizeof(float)) != 0) {
        ERROR_LOG("Invalid float input file: %s", path);
        return 0;
    }
    return static_cast<int64_t>(st.st_size / static_cast<off_t>(sizeof(float)));
}

OperatorDesc CreateOpDesc()
{
    const int64_t indexCount = CountUInt32Elements("../input/input_lens.bin");
    const int64_t flagCount = CountUInt32Elements("../input/input_flag.bin");
    const int64_t weightElems = CountFloatElements("../input/input_weight_r.bin");
    const int64_t totalRankEntries = weightElems > 0 ? weightElems / FEATURES : 0;
    if (indexCount != flagCount) {
        ERROR_LOG("lens count %ld does not match flag count %ld", indexCount, flagCount);
    }
    if (weightElems % FEATURES != 0) {
        ERROR_LOG("input_weight_r element count %ld is not divisible by %ld", weightElems, FEATURES);
    }
    std::vector<int64_t> weightShape{totalRankEntries, FEATURES};
    std::vector<int64_t> lensShape{indexCount};
    std::vector<int64_t> flagShape{indexCount};
    std::vector<int64_t> outputShape{totalRankEntries, FEATURES};
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(ACL_FLOAT, weightShape.size(), weightShape.data(), format);
    opDesc.AddInputTensorDesc(ACL_FLOAT, weightShape.size(), weightShape.data(), format);
    opDesc.AddInputTensorDesc(ACL_UINT32, lensShape.size(), lensShape.data(), format);
    opDesc.AddInputTensorDesc(ACL_UINT32, flagShape.size(), flagShape.data(), format);
    opDesc.AddOutputTensorDesc(ACL_FLOAT, outputShape.size(), outputShape.data(), format);
    opDesc.AddOutputTensorDesc(ACL_FLOAT, outputShape.size(), outputShape.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    if (!ReadFile("../input/input_weight_r.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0)) ||
        !ReadFile("../input/input_weight_i.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1)) ||
        !ReadFile("../input/input_lens.bin", fileSize, runner.GetInputBuffer<void>(2), runner.GetInputSize(2)) ||
        !ReadFile("../input/input_flag.bin", fileSize, runner.GetInputBuffer<void>(3), runner.GetInputSize(3))) {
        return false;
    }
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    if (!WriteFile("../output/output_weightout_r.bin", runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0)) ||
        !WriteFile("../output/output_weightout_i.bin", runner.GetOutputBuffer<void>(1), runner.GetOutputSize(1))) {
        return false;
    }
    INFO_LOG("Write output success");
    return true;
}

void DestroyResource()
{
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destroy resource failed");
    } else {
        INFO_LOG("Destroy resource success");
    }
}

bool InitResource()
{
    ResolveDeviceId();
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        } else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    if (aclInit(nullptr) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestroyResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOp()
{
    OperatorDesc opDesc = CreateOpDesc();

    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    if (!SetInputData(opRunner)) {
        ERROR_LOG("Set input data failed");
        return false;
    }

    if (!opRunner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    if (!ProcessOutputData(opRunner)) {
        ERROR_LOG("Process output data failed");
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

int main()
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp()) {
        DestroyResource();
        return FAILED;
    }

    DestroyResource();
    return SUCCESS;
}
