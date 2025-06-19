import re

def extract_case_ids(txt_path,is_slice_file=None,test_set=None):
    """
    Reads a file where each line looks like 'volume-100_slice_427',
    extracts the '100' part, and returns a set of all unique case IDs.
    """
    case_ids = set()
    if is_slice_file:
        pattern = re.compile(r'volume-(\d+)_slice')
    else:
        pattern = re.compile(r'volume-(\d+)\.nii')

    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            m = pattern.search(line)
            if m:
                case_ids.add(m.group(1))  # or int(m.group(1)) if you want ints
    case_ids = list(case_ids)
    if test_set:
        test_data = [cid for cid in case_ids if cid in test_set]
        cv_data   = [cid for cid in case_ids if cid not in test_set]
    else:  
        cv_data, test_data = case_ids[0:-10], case_ids[-10:] 
    return cv_data, test_data

if __name__ == "__main__":
    txt_file = "large_tumor_case_new.txt"
    test_set = ['100','76','72','75','69','40','111','95','27','79']
    cv_data, test_data = extract_case_ids(txt_file,test_set=test_set)
    print(cv_data)
    print("cv length: ",len(cv_data))
    print(test_data)
    print("test length: ",len(test_data))
