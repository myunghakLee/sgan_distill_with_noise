# +
echo "de"
python evaluate_model.py --model_path de
echo "\n"

echo "lrp_teacher"
python evaluate_model.py --model_path lrp_teacher
echo "\n"

echo "negative_lrp_teacher"
python evaluate_model.py --model_path negative_lrp_teacher
echo "\n"

echo "negative_lrp_teacher_feat"
python evaluate_model.py --model_path negative_lrp_teacher_feat
echo "\n"

echo "no_noise"
python evaluate_model.py --model_path no_noise
echo "\n"

echo "no_noise_teacher"
python evaluate_model.py --model_path no_noise_teacher
echo "\n"

echo "no_noise_teacher_feat"
python evaluate_model.py --model_path no_noise_teacher_feat
echo "\n"

echo "random_noise_teacher"
python evaluate_model.py --model_path random_noise_teacher
echo "\n"
