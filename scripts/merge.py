import os
import shutil

root_path = "/workspace/exp_final"

for method in os.listdir(root_path):

    folder_path = os.path.join(root_path, method)
    folders = os.listdir(folder_path)

    target_folder = folder_path + "/merged_folder_real"
    if os.path.exists(target_folder) == 0:
        os.mkdir(target_folder)

    for folder in folders:
        if folder == 'bear' or folder == 'castle' or folder == 'cat' or folder == 'dog' or folder == 'foxman' or folder == 'guitar' or folder == 'light_tower' or folder == 'teddybear' or folder == 'wooden_pot' or folder == 'penguin':
            # path = folder_path + "/" + folder + "/image"
            path = folder_path + "/" + folder + "/results"
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    shutil.copy(file_path, target_folder)
                    ori_path = os.path.join(target_folder, file)
                    new_path = os.path.join(target_folder, folder + "_" + file)
                    os.rename(ori_path, new_path)
            # shutil.rmtree(folder_path)

#
#
# import os
# import shutil
#
# folder_path = "/workspace/data/InsertSet/id_image/real_object/cat/reg_data"
# folders = os.listdir(folder_path)
#
# for folder in folders:
#     ori_path = os.path.join(folder_path, folder)
#     folder_new = folder[:1] + '1' + folder[2:]
#     new_path = os.path.join(folder_path, folder_new)
#     os.rename(ori_path, new_path)
#     # shutil.rmtree(folder_path)



# import os
# import shutil
#
# name = 'snowman'
#
# folder_path = "/workspace/data/InsertSet/id_image/virtual_object/" + name + "/reg_data"
# save_path = "/workspace/code/diffusers/examples/custom_diffusion/reg_data/" + name
#
# if os.path.exists(save_path) == 0:
#     os.mkdir(save_path)
#
# list_img = os.listdir(folder_path)
#
# list_caption = open(os.path.join(save_path, 'caption.txt'), 'a')
# list_path = open(os.path.join(save_path, 'images.txt'), 'a')
#
# for i in range(len(list_img)):
#     if i < 200:
#         if os.path.exists(os.path.join(save_path, "images")) == 0:
#             os.mkdir(os.path.join(save_path, "images"))
#
#         ori_path = os.path.join(folder_path, list_img[i])
#         new_path = os.path.join(save_path, 'images', str(i+1).zfill(3) + '.jpg')
#
#         shutil.copy(ori_path, new_path)
#         # os.rename(os.path.join(save_path, 'images', list_img[i]), new_path)
#
#         list_caption.write("a photo of " + name + "\n")
#         list_path.write(new_path + '\n')
#     else:
#         break
#
# list_caption.close()
# list_path.close()