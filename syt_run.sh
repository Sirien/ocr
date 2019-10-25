source ocr/bin/activate
python gen_printed_char.py --out_dir ./dataset  --width 19 --height 19 --image_num 500

deactivate

source ../torch/bin/activate

python train.py
