# globals.py
RGB_img_res = [3, 64, 64] #[channels, height, width]

# --- parametri augmentation ---
augmentation_parameters = {
    'flip': 0.5,
    'mirror': 0.5,
    'c_swap': 0.5,
    'random_crop': 0.5,
    'shifting_strategy': 0.5
}

# --- tipo di dataset, da usare per eventuali distinzioni ---
dts_type = 'nyu'   # oppure 'diml', a seconda di cosa stai usando

global_var = {

    'imagenet_w_init': True,   # attiva init da ImageNet

}
