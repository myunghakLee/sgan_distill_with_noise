python Baseline.py --dataset_name 'hotel' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode lrp --feat_distill_loss_weight 0 --checkpoint_save_path lrp_teacher --checkpoint_load_path models/sgan-models/hotel_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 0 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'hotel' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode lrp --feat_distill_loss_weight 1 --checkpoint_save_path lrp_teacher --checkpoint_load_path models/sgan-models/hotel_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 1 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'hotel' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode random_noise --feat_distill_loss_weight 0 --checkpoint_save_path random_noise_teacher --checkpoint_load_path models/sgan-models/hotel_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 2 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'hotel' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode random_noise --feat_distill_loss_weight 1 --checkpoint_save_path random_noise_teacher --checkpoint_load_path models/sgan-models/hotel_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 3 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'zara1' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode lrp --feat_distill_loss_weight 0 --checkpoint_save_path lrp_teacher --checkpoint_load_path models/sgan-models/zara1_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 0 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'zara1' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode lrp --feat_distill_loss_weight 1 --checkpoint_save_path lrp_teacher --checkpoint_load_path models/sgan-models/zara1_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 1 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'zara1' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode random_noise --feat_distill_loss_weight 0 --checkpoint_save_path random_noise_teacher --checkpoint_load_path models/sgan-models/zara1_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 2 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'zara1' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode random_noise --feat_distill_loss_weight 1 --checkpoint_save_path random_noise_teacher --checkpoint_load_path models/sgan-models/zara1_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 3 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'zara2' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode lrp --feat_distill_loss_weight 0 --checkpoint_save_path lrp_teacher --checkpoint_load_path models/sgan-models/zara2_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 0 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'zara2' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode lrp --feat_distill_loss_weight 1 --checkpoint_save_path lrp_teacher --checkpoint_load_path models/sgan-models/zara2_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 1 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'zara2' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode random_noise --feat_distill_loss_weight 0 --checkpoint_save_path random_noise_teacher --checkpoint_load_path models/sgan-models/zara2_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 2 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'zara2' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode random_noise --feat_distill_loss_weight 1 --checkpoint_save_path random_noise_teacher --checkpoint_load_path models/sgan-models/zara2_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 3 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'eth' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode lrp --feat_distill_loss_weight 0 --checkpoint_save_path lrp_teacher --checkpoint_load_path models/sgan-models/eth_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 0 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'eth' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode lrp --feat_distill_loss_weight 1 --checkpoint_save_path lrp_teacher --checkpoint_load_path models/sgan-models/eth_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 1 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'eth' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode random_noise --feat_distill_loss_weight 0 --checkpoint_save_path random_noise_teacher --checkpoint_load_path models/sgan-models/eth_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 2 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'eth' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode random_noise --feat_distill_loss_weight 1 --checkpoint_save_path random_noise_teacher --checkpoint_load_path models/sgan-models/eth_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 3 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'univ' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode lrp --feat_distill_loss_weight 0 --checkpoint_save_path lrp_teacher --checkpoint_load_path models/sgan-models/univ_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 0 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'univ' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode lrp --feat_distill_loss_weight 1 --checkpoint_save_path lrp_teacher --checkpoint_load_path models/sgan-models/univ_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 1 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'univ' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode random_noise --feat_distill_loss_weight 0 --checkpoint_save_path random_noise_teacher --checkpoint_load_path models/sgan-models/univ_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 2 --response_distill_loss --feat_distill&
sleep 2


python Baseline.py --dataset_name 'univ' --delim tab --pred_len 8 --encoder_h_dim_g 32 --encoder_h_dim_d 48 --decoder_h_dim 32 --embedding_dim 16 --bottleneck_dim 8 --mlp_dim 64 --num_layers 1 --noise_dim 8 --noise_mix_type global --pool_every_timestep 0 --l2_loss_weight 1 --batch_size 32 --g_learning_rate 1e-3 --d_learning_rate 1e-3 --checkpoint_every 10 --print_every 50 --num_iterations 20000 --num_epochs 500 --pooling_type 'none' --clipping_threshold_g 1.5 --best_k 10 --restore_from_checkpoint 0 --mode random_noise --feat_distill_loss_weight 1 --checkpoint_save_path random_noise_teacher --checkpoint_load_path models/sgan-models/univ_8_model.pt --decoder_h_dim_g 32 --checkpoint_every 300 --gpu_num 3 --response_distill_loss --feat_distill&
sleep 2
