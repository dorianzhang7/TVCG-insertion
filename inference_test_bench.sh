python scripts/inference_test_bench.py \
--plms \
--id_dir /workspace/data/InsertSet/id_image/real_object/ \
--outdir /workspace/exp_final/test/cat_new/ \
--test_bench_dir /workspace/data/InsertSet/test_bench_final/cat/ \
--config /workspace/code/fusion-diffusion3.0/configs/v2.yaml \
--ckpt /workspace/code/fusion-diffusion3.0/checkpoints/pbe_model.ckpt \
--delta_ckpt /workspace/code/fusion-diffusion3.0/models/fusion-diffusion3.0/cat_new_gen_test/checkpoints/delta_last.ckpt \
--list_dir /workspace/data/InsertSet/list/cat.txt \
--seed 321 \
--scale 5 \
--gpu 0 \
--id_class cat \
--fixed_code \

#
#--plms \
#--outdir results/test_bench \
#--config configs/v1.yaml \
#--ckpt checkpoints/model.ckpt \
#--scale 5