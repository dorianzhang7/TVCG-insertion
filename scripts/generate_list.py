import os

path = "/home/zq/data2/zq/code/fusion-diffusion3.0/results/cat_test_bench/input/"
filelist = os.listdir(path)
filelist = sorted(filelist, key=lambda x: os.path.getmtime(os.path.join(path, x)))

# mask_path = "/home/zq/data2/zq/code/fusion-diffusion3.0/results/cat_test_bench/mask/"
# mask_filelist = os.listdir(mask_path)

count = 0
for file in filelist:
    print(file)
for file in filelist:
    Olddir = os.path.join(path, file)
    if os.path.isdir(Olddir):
        continue
    filename = os.path.splitext(file)[0]
    filetype = os.path.splitext(file)[1]
    Newdir = os.path.join(path, str(count).zfill(4) + '_input' + filetype)
    os.rename(Olddir, Newdir)

    count += 1
