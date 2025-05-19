filter_tumor_size_path='small_tumor_slices.txt'
with open(filter_tumor_size_path, 'r') as f:
            valid_keywords = set(line.split('_slice')[0] for line in f)
print(valid_keywords)
print(len(valid_keywords))