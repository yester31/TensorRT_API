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
