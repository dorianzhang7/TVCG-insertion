# import os
# import shutil
#
# list_path = r"/workspace/data/InsertSet/list"
#
# root_path = r"/workspace/data/InsertSet/test_bench_all/"
# # root_path_new = root_path[:-1] + "our_new"
# root_path_new = "/workspace/data/InsertSet/test_bench_final/"
#
# if os.path.exists(root_path_new) == 0:
#     os.mkdir(root_path_new)
#
# folders = os.listdir(root_path)
# for id in folders:
#     txt_path = os.path.join(list_path, id+".txt")
#     img_path = os.path.join(root_path, id, "image")
#     mask_path = os.path.join(root_path, id, "mask")
#
#     root_path_new_id = os.path.join(root_path_new, id)
#     if os.path.exists(root_path_new_id) == 0:
#         os.mkdir(root_path_new_id)
#     img_path_new = os.path.join(root_path_new_id, "image")
#     if os.path.exists(img_path_new) == 0:
#         os.mkdir(img_path_new)
#     mask_path_new = os.path.join(root_path_new_id, "mask")
#     if os.path.exists(mask_path_new) == 0:
#         os.mkdir(mask_path_new)
#     i = 0
#     with open(txt_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             i = i + 1
#             data = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
#             img = os.path.join(img_path, data[0].zfill(4)+"_img.png")
#             mask = os.path.join(mask_path, data[0].zfill(4)+"_mask.png")
#
#             img_new = os.path.join(img_path_new, data[0].zfill(4)+"_img.png")
#             mask_new = os.path.join(mask_path_new, data[0].zfill(4)+"_mask.png")
#
#             shutil.copy(img, img_new)
#             shutil.copy(mask, mask_new)
#
# print(i)



import os
import shutil

# list_path = r"/workspace/data/InsertSet/list"
#
# root_path = r"/workspace/exp/shared_feature/"
# root_path_new = r"/workspace/exp_our/shared_feature/"
# if os.path.exists(root_path_new) == 0:
#     os.mkdir(root_path_new)
#
# folders = os.listdir(root_path)
#
# # for id in folders:
# id = 'bear'
# txt_path = os.path.join(list_path, id + ".txt")
# img_path = os.path.join(root_path, id, "results")
# root_path_new_id = os.path.join(root_path_new, id)
# if os.path.exists(root_path_new_id) == 0:
#     os.mkdir(root_path_new_id)
# img_path_new = os.path.join(root_path_new_id, "results")
# if os.path.exists(img_path_new) == 0:
#     os.mkdir(img_path_new)
#
# with open(txt_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         data = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
#         img = os.path.join(img_path, data[0].zfill(4) + ".png")
#         img_new = os.path.join(img_path_new, data[0].zfill(4) + ".png")
#         shutil.copy(img, img_new)


import os
import shutil

list_path = r"/workspace/data/InsertSet/list"
exp_path = r"/workspace/exp"

for method in os.listdir(exp_path):

    root_path = os.path.join("/workspace/exp", method)
    root_path_new = os.path.join("/workspace/exp_final", method + "_final")
    if os.path.exists(root_path_new) == 0:
        os.mkdir(root_path_new)

    folders = os.listdir(root_path)

    for object in folders:
        img_path = os.path.join(root_path, object, "results")
        if os.path.exists(os.path.join(root_path_new, object)) == 0:
            os.mkdir(os.path.join(root_path_new, object))

        if os.path.exists(os.path.join(root_path_new, object, "results")) == 0:
            os.mkdir(os.path.join(root_path_new, object, "results"))

        img_path_new = os.path.join(root_path_new, object, "results")

        txt_path = os.path.join(list_path, object+".txt")

        with open(txt_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
                #
                if method == 'anydoor':
                    img = os.path.join(img_path, data[0].zfill(4) + "_img.png")
                else:
                    img = os.path.join(img_path, data[0].zfill(4) + ".png")

                # img = os.path.join(img_path, data[0].zfill(4) + ".png")

                img_new = os.path.join(img_path_new, data[0].zfill(4)+".png")

                shutil.copy(img, img_new)


# list_path = r"/workspace/data/InsertSet/list_new/monkey.txt"
#
# root_path = r"/workspace/monkey_noise/results/"
# # root_path_new = root_path[:-1] + "our_new"
# root_path_new = "/workspace/monkey_noise/results_new/"
#
# if os.path.exists(root_path_new) == 0:
#     os.mkdir(root_path_new)
#
# with open(list_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         data = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
#
#         img = os.path.join(root_path, data[0].zfill(4) + ".png")
#
#         img_new = os.path.join(root_path_new, data[0].zfill(4)+".png")
#
#         shutil.copy(img, img_new)