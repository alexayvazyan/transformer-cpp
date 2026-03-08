print("dog")

import gzip
import re


wet_path = "/Users/alexanderayvazyan/Documents/cpplearning/project/crawl-data/unprocessed_files/CC-MAIN-20260112161239-20260112191239-00000.warc.wet.gz"
dictionary = "/usr/share/dict/words"
open("samples.txt", "w").close()
def main():
    with open(dictionary, "r", encoding="utf-8") as f:
        text = f.read()
    words = text.lower().split()
    print(len(words))
    currpage = []
    recording_mode = 0
    with gzip.open(wet_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if "WARC-Identified-Content-Language" in line:
                recording_mode = 0
                if line.strip() == "WARC-Identified-Content-Language: eng":
                    if len(currpage) > 0:
                        dump_samples(currpage)
                    recording_mode = 1
                    currpage = []
            
            if recording_mode == 1:
                line_words = re.findall(r'\b\w+\b', line.lower())
                info = [word for word in line_words if word in words]
                if len(info) > 0:
                    currpage.extend([word for word in line_words if word in words])
                    currpage.append("~") #LETS USE ~ AS ENDLINE CHAR
            if i >= 50000:
                break

def dump_samples(page, context_budget = 10, stride = 2):
    with open("samples.txt", "a") as f:
        idx = 0
        sep = " "
        while idx + context_budget < len(page):
            f.write(sep.join(map(str, page[idx: idx+context_budget])) + "\n")
            idx += stride
    




main()