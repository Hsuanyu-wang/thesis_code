# # 測試不同的 num_workers 值 (CPU 並行度)
# for workers in 4 2 0; do
#     echo "Testing num_workers=$workers"
#     python train.py -d webqsp --num_workers $workers --batch_size 1 --samples_per_epoch 1000 -id_sup "workers_${workers}"
# done
# #  workers_4 -> CPU MEM剩1.6, 1.7G (epoch = 20左右穩定)

# # 測試不同的 batch_size 值 (GPU 利用率)
# for batch_size in 512 256 128; do
#     echo "Testing batch_size=$batch_size"
#     python train.py -d webqsp --batch_size $batch_size --num_workers 4 --samples_per_epoch 1000 -id_sup "batch_${batch_size}"
# done
# # 越大train越快

# 測試不同的 samples_per_epoch 值(記憶體使用)
for samples in 50000 25000 10000 5000 1000; do
    echo "Testing samples_per_epoch=$samples"
    python train.py -d webqsp --samples_per_epoch $samples --batch_size 4 --num_workers 4 -id_sup "samples_${samples}"
done

# 測試不同的 samples_per_batch_load 值(磁盤 I/O)
for batch_load in 512 256 128 64 32 16; do
    echo "Testing samples_per_batch_load=$batch_load"
    python train.py -d webqsp --samples_per_batch_load $batch_load --batch_size 4 --num_workers 4 -id_sup "batch_load_${batch_load}"
done