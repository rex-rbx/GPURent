#include <ixwebsocket/IXWebSocket.h>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

using json = nlohmann::json;

struct Options {
    std::string url;
    std::string model;
    std::string file;
    int partition = -1;
    bool cuda = false;
};

Options parseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; i += 1) {
        const std::string key = argv[i];
        if (key == "--cuda") {
            options.cuda = true;
            continue;
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        const std::string value = argv[i + 1];
        if (key == "--url") options.url = value;
        else if (key == "--model") options.model = value;
        else if (key == "--partition") options.partition = std::stoi(value);
        else if (key == "--file") options.file = value;
        else throw std::runtime_error("unknown argument: " + key);
        ++i;
    }
    if (options.url.empty() || options.model.empty() || options.file.empty() || options.partition < 0) {
        throw std::runtime_error("usage: gpurent-worker --url URL --model ID --partition N --file PATH");
    }
    return options;
}

template <>
json outputValues<int64_t>(const Ort::Value& value) {
    const auto info = value.GetTensorTypeAndShapeInfo();
    const auto* data = value.GetTensorData<int64_t>();
    json values = json::array();
    for (size_t i = 0; i < info.GetElementCount(); ++i) {
        values.push_back(std::to_string(data[i]));
    }
    return values;
}

template <typename T>
std::vector<T> valuesAs(const json& values) {
    std::vector<T> result;
    result.reserve(values.size());
    for (const auto& value : values) {
        if constexpr (std::is_same_v<T, int64_t>) {
            result.push_back(value.is_string() ? std::stoll(value.get<std::string>()) : value.get<T>());
        } else {
            result.push_back(value.get<T>());
        }
    }
    return result;
}

template <typename T>
json outputValues(const Ort::Value& value) {
    const auto info = value.GetTensorTypeAndShapeInfo();
    const auto* data = value.GetTensorData<T>();
    return json(data, data + info.GetElementCount());
}

json tensorOutput(const std::string& name, const Ort::Value& value) {
    const auto info = value.GetTensorTypeAndShapeInfo();
    json values;
    std::string dtype;
    switch (info.GetElementType()) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            dtype = "float32"; values = outputValues<float>(value); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            dtype = "float64"; values = outputValues<double>(value); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            dtype = "int64"; values = outputValues<int64_t>(value); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            dtype = "int32"; values = outputValues<int32_t>(value); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            dtype = "uint8"; values = outputValues<uint8_t>(value); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            dtype = "bool"; values = outputValues<bool>(value); break;
        default: throw std::runtime_error("unsupported ONNX output tensor type");
    }
    return {{"name", name}, {"dtype", dtype}, {"shape", info.GetShape()}, {"values", values}};
}

using TensorValues = std::variant<
    std::vector<float>,
    std::vector<double>,
    std::vector<int64_t>,
    std::vector<int32_t>,
    std::vector<uint8_t>
>;

struct InputTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    TensorValues values;
};

InputTensor parseInput(const json& input) {
    InputTensor tensor{
        input.at("name").get<std::string>(),
        input.at("dtype").get<std::string>(),
        input.at("shape").get<std::vector<int64_t>>(),
        std::vector<float>{}
    };
    const auto& values = input.at("values");
    if (tensor.dtype == "float32") tensor.values = valuesAs<float>(values);
    else if (tensor.dtype == "float64") tensor.values = valuesAs<double>(values);
    else if (tensor.dtype == "int64") tensor.values = valuesAs<int64_t>(values);
    else if (tensor.dtype == "int32") tensor.values = valuesAs<int32_t>(values);
    else if (tensor.dtype == "uint8" || tensor.dtype == "bool") tensor.values = valuesAs<uint8_t>(values);
    else throw std::runtime_error("unsupported tensor dtype: " + tensor.dtype);
    size_t expected = 1;
    for (const auto dimension : tensor.shape) {
        if (dimension < 0) throw std::runtime_error("input tensor shape must be concrete");
        expected *= static_cast<size_t>(dimension);
    }
    const size_t actual = std::visit([](const auto& data) { return data.size(); }, tensor.values);
    if (actual != expected) throw std::runtime_error("tensor value count does not match shape");
    return tensor;
}

Ort::Value tensorInput(InputTensor& input, const Ort::MemoryInfo& memory) {
    return std::visit([&](auto& values) -> Ort::Value {
        using ValueType = typename std::decay_t<decltype(values)>::value_type;
        if (input.dtype == "bool") {
            return Ort::Value::CreateTensor(memory, values.data(), values.size(),
                input.shape.data(), input.shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
        }
        return Ort::Value::CreateTensor<ValueType>(memory, values.data(), values.size(),
            input.shape.data(), input.shape.size());
    }, input.values);
}

int main(int argc, char** argv) {
    try {
        const auto options = parseArgs(argc, argv);
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "gpurent");
        Ort::SessionOptions sessionOptions;
#ifdef GPURENT_CUDA
        if (options.cuda) {
            OrtCUDAProviderOptions cudaOptions{};
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        }
#else
        if (options.cuda) throw std::runtime_error("rebuild with -DGPURENT_CUDA=ON to enable CUDA");
#endif
        Ort::Session session(environment, options.file.c_str(), sessionOptions);
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> sessionInputNames;
        std::vector<std::string> sessionOutputNames;
        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            sessionInputNames.emplace_back(name.get());
        }
        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            sessionOutputNames.emplace_back(name.get());
        }
        ix::WebSocket socket;
        socket.setUrl(options.url);
        const std::string workerId = "cpp-" + std::to_string(reinterpret_cast<uintptr_t>(&socket));

        socket.setOnMessageCallback([&](const ix::WebSocketMessagePtr& message) {
            if (message->type == ix::WebSocketMessageType::Open) {
                socket.send(json{
                    {"type", "worker:register"},
                    {"workerId", workerId},
                    {"modelId", options.model},
                    {"partition", options.partition},
                    {"file", options.file},
                    {"inputs", sessionInputNames},
                    {"outputs", sessionOutputNames}
                }.dump());
                return;
            }
            if (message->type == ix::WebSocketMessageType::Message) {
                try {
                    const auto request = json::parse(message->str);
                    if (request.value("type", "") == "worker:registered") {
                        std::cout << "Registered C++ worker for " << options.model
                                  << " partition " << options.partition << std::endl;
                    } else if (request.value("type", "") == "inference:execute") {
                        std::vector<InputTensor> ownedInputs;
                        std::vector<std::string> inputNames;
                        std::vector<Ort::Value> inputValues;
                        for (const auto& input : request.at("inputs")) {
                            ownedInputs.push_back(parseInput(input));
                        }
                        for (auto& input : ownedInputs) {
                            inputNames.push_back(input.name);
                            inputValues.push_back(tensorInput(input, memory));
                        }
                        std::vector<const char*> inputNamePointers;
                        for (const auto& name : inputNames) inputNamePointers.push_back(name.c_str());

                        std::vector<std::string> outputNames;
                        std::vector<const char*> outputNamePointers;
                        outputNames = sessionOutputNames;
                        for (const auto& name : outputNames) outputNamePointers.push_back(name.c_str());

                        auto results = session.Run(Ort::RunOptions{nullptr},
                            inputNamePointers.data(), inputValues.data(), inputValues.size(),
                            outputNamePointers.data(), outputNamePointers.size());
                        json outputs = json::array();
                        for (size_t i = 0; i < results.size(); ++i) {
                            outputs.push_back(tensorOutput(outputNames[i], results[i]));
                        }
                        socket.send(json{
                            {"type", "inference:result"},
                            {"requestId", request.at("requestId")},
                            {"outputs", outputs}
                        }.dump());
                    }
                } catch (const std::exception& error) {
                    try {
                        const auto request = json::parse(message->str);
                        socket.send(json{
                            {"type", "inference:result"},
                            {"requestId", request.at("requestId")},
                            {"error", error.what()}
                        }.dump());
                    } catch (...) {
                        std::cerr << error.what() << std::endl;
                    }
                }
            }
        });
        const auto connection = socket.connect(60);
        if (!connection.success) throw std::runtime_error(connection.errorStr);
        socket.run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
