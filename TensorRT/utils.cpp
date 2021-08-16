#include "utils.h"

//파일 이름 가져오기(DFS) window용
int SearchFile(const std::string& folder_path, std::vector<std::string> &file_names, bool recursive = false)
{
	_finddata_t file_info;
	string any_file_pattern = folder_path + "\\*";
	intptr_t handle = _findfirst(any_file_pattern.c_str(), &file_info);

	if (handle == -1)
	{
		cerr << "folder path not exist: " << folder_path << endl;
		return -1;
	}

	do
	{
		string file_name = file_info.name;
		if (recursive) {
			if (file_info.attrib & _A_SUBDIR)//check whtether it is a sub direcotry or a file
			{
				if (file_name != "." && file_name != "..")
				{
					string sub_folder_path = folder_path + "//" + file_name;
					SearchFile(sub_folder_path, file_names);
					cout << "a sub_folder path: " << sub_folder_path << endl;
				}
			}
			else
			{
				string file_path = folder_path + "/" + file_name;
				file_names.push_back(file_path);
			}
		}
		else {
			if (!(file_info.attrib & _A_SUBDIR))//check whtether it is a sub direcotry or a file
			{
				string file_path = folder_path + "/" + file_name;
				file_names.push_back(file_path);
			}
		}
	} while (_findnext(handle, &file_info) == 0);
	_findclose(handle);
	return 0;
}
