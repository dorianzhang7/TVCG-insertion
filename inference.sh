python scripts/inference.py \
--plms \
--config /workspace/code/fusion-diffusion3.0/configs/v.yaml \
--dir /workspace/code/fusion-diffusion3.0/ \
--id_dir /workspace/data/InsertSet/id_image/ \
--model_name the name of obtained model(2024-11-16T10-13-33_v2) \
--example_img 0001 \
--example_roi example_101_ROI.txt \
--id_class rabbit \
--seed 321 \
--scale 5 \
--gpu 0 \

#--plms --outdir results \
#--config configs/v1.yaml \
#--ckpt checkpoints/model.ckpt \
#--image_path examples/image/example_1.png \
#--mask_path examples/mask/example_1.png \
#--reference_path examples/reference/example_1.jpg \
#--seed 321 \
#--scale 5


#python scripts/inference.py \
#--plms --outdir results \
#--config configs/v1.yaml \
#--ckpt checkpoints/model.ckpt \
#--image_path examples/image/example_2.png \
#--mask_path examples/mask/example_2.png \
#--reference_path examples/reference/example_2.jpg \
#--seed 5876 \
#--scale 5
#
#python scripts/inference.py \
#--plms --outdir results \
#--config configs/v1.yaml \
#--ckpt checkpoints/model.ckpt \
#--image_path examples/image/example_3.png \
#--mask_path examples/mask/example_3.png \
#--reference_path examples/reference/example_3.jpg \
#--seed 5065 \
#--scale 5
