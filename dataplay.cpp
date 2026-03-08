#include <zlib.h>
#include <iostream>

int main() {
    gzFile paths = gzopen("/Users/alexanderayvazyan/Documents/cpplearning/project/crawl-data/wet.paths.gz", "rb");

    char buffer[10000];

    char* link = gzgets(paths, buffer, sizeof(buffer));
    
    char* newline = strchr(buffer, '\n');
    if (newline) *newline = '\0';
    gzclose(paths);

    gzFile textfile = gzopen("/Users/alexanderayvazyan/Documents/cpplearning/project/crawl-data/CC-MAIN-20260112161239-20260112191239-00000.warc.wet.gz", "rb");

    char buffer2[1000];
    char* text = gzgets(textfile, buffer2, sizeof(buffer2));
    std::cout << text;

    while (gzgets(textfile, buffer2, sizeof(buffer2))) {
        std::cout << buffer2;
    };
    gzclose(textfile);
    return 0;
}
