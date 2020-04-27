# Smear-Detection
## run code:
#### run test: 

`bash runtest.bash`

#### run your own pictures:
`python3 code/contrast.py`

`python3 code/grad_and_avg.py`

`python3 code/smear_seg.py -i <your_avg_img_path> -g <your_avg_gradient_path> -t <threshold_for_binary_mask>` 

`python3 code/dilation.py -i <your_img_path> -k1 <kernel_size_for_erosion> -k2 <kernel_size_for_dilation>`

_example:_

`python3 code/smear_seg.py -i 'results/average_image/test.png' -g 'results/average_grad/test.png' -t 0.5`

`python3 code/dilation.py -i 'results/average_image/test.png' -k1 2 -k2 10`

## file structure:
code:
- smear_seg.py: the iterative fitting method implementing the paper.

data/sample_drive: _*(git ignored)_
- cam_i: source data from ith camera

results:
- average_grad: average of image gradient Avg(|grad(I)|)
- average_image: average of images
- intermediate: attenuation and scattering maps
- final_output: final binary masks for each camera
