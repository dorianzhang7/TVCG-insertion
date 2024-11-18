python scripts/inference_test_bench.py \
--plms \
--id_dir /workspace/data/InsertSet/id_image \
--outdir /workspace/exp/test/rabbit/ \
--test_bench_dir /workspace/data/InsertSet/test_bench/rabbit/ \
--config /workspace/code/fusion-diffusion3.0/configs/v.yaml \
--ckpt /workspace/code/fusion-diffusion3.0/checkpoints/model.ckpt \
--delta_ckpt /workspace/code/fusion-diffusion3.0/models/fusion-diffusion3.0/the name of obtained model/checkpoints/delta_last.ckpt \
--list_dir /workspace/data/InsertSet/list/rabbit.txt \
--seed 321 \
--scale 5 \
--gpu 0 \
--id_class rabbit \
--fixed_code \

#
#--plms \
#--outdir results/test_bench \
#--config configs/v1.yaml \
#--ckpt checkpoints/model.ckpt \
#--scale 5
