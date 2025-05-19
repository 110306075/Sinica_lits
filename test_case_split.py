import re

def extract_case_ids(txt_path):
    """
    Reads a file where each line looks like 'volume-100_slice_427',
    extracts the '100' part, and returns a set of all unique case IDs.
    """
    case_ids = set()
    pattern = re.compile(r'volume-(\d+)_slice')

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            m = pattern.search(line)
            if m:
                case_ids.add(m.group(1))  # or int(m.group(1)) if you want ints
    case_ids = list(case_ids)
    cv_data, test_data = case_ids[0:-10], case_ids[-10:] 
    return cv_data, test_data

if __name__ == "__main__":
    txt_file = "large_tumor_slices.txt"
    cv_data, test_data = extract_case_ids(txt_file)
    print(cv_data)
    print("cv length: ",len(cv_data))
    print(test_data)
    print("test length: ",len(test_data))
