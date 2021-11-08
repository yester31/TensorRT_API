#pragma once

using Severity = nvinfer1::ILogger::Severity;

class Logger : public ILogger
{
	void log(Severity severity, const char* msg) noexcept override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
};

// CUDA RUNTIME API 에러 체크를 위한 매크로 함수 정의
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
