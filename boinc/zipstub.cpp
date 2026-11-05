// Stub: the Skia wrapper does not use <zip_output>/<unzip_input>; archive is unpacked by tar in job.xml.
#include <string>
#include <vector>
#include "boinc_zip.h"
int boinc_zip(int, const std::string&, const ZipFileList*) { return -1; }
int boinc_zip(int, const std::string&, const std::string&) { return -1; }
