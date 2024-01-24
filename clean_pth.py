from fileinput import filename
import os
import sys
base_path = 'output'
g = os.walk(base_path)
file_path_to_delete = []
# candidates = []
# for i in range(50):
#     candidates.append(f'ckpt_epoch_{i}.pth')
# for i in range(51, 100):
#     candidates.append(f'ckpt_epoch_{i}.pth')
# for i in range(101, 150):
#     candidates.append(f'ckpt_epoch_{i}.pth')
# for i in range(151, 200):
#     candidates.append(f'ckpt_epoch_{i}.pth')
# for i in range(201, 250):
#     candidates.append(f'ckpt_epoch_{i}.pth')
# for i in range(251, 290):
#     candidates.append(f'ckpt_epoch_{i}.pth')
for path, dir_list, file_list in g:
    for file_name in file_list:
        _path = os.path.join(path, file_name)
        # for cand in candidates:
        #     if _path.find(cand) != -1:
        #         os.remove(_path)
        if _path.endswith('.pth') and _path.find('important') == -1:
            file_path_to_delete.append(_path)
for name in file_path_to_delete:
    os.remove(name)
print('done')