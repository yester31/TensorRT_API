#include <io.h>
#include <iostream>
#include <string>
#include <vector>

//파일 이름 가져오기(DFS) window용
int SearchFile(const std::string& folder_path, std::vector<std::string> &file_names, bool recursive = false)
{
	_finddata_t file_info;
	std::string any_file_pattern = folder_path + "\\*";
	intptr_t handle = _findfirst(any_file_pattern.c_str(), &file_info);

	if (handle == -1)
	{
		std::cerr << "folder path not exist: " << folder_path << std::endl;
		return -1;
	}

	do
	{
		std::string file_name = file_info.name;
		if (recursive) {
			if (file_info.attrib & _A_SUBDIR)//check whtether it is a sub direcotry or a file
			{
				if (file_name != "." && file_name != "..")
				{
					std::string sub_folder_path = folder_path + "//" + file_name;
					SearchFile(sub_folder_path, file_names);
					std::cout << "a sub_folder path: " << sub_folder_path << std::endl;
				}
			}
			else
			{
				std::string file_path = folder_path + "/" + file_name;
				file_names.push_back(file_path);
			}
		}
		else {
			if (!(file_info.attrib & _A_SUBDIR))//check whtether it is a sub direcotry or a file
			{
				std::string file_path = folder_path + "/" + file_name;
				file_names.push_back(file_path);
			}
		}
	} while (_findnext(handle, &file_info) == 0);
	_findclose(handle);
	return 0;
}
