# Seg-InvRender: Fusing Semantic Segmentation Based on NeRF for Inverse Rendering Considering Shadows

## Create environment

```
conda env create --file environment.yml
conda activate shadow
```

## Data

Our datasets and preprocessed synthetic dataset from can be found [here](https://drive.google.com/drive/folders/1hzpvl8i15hUJ5gNvULlxrTKvKDGqB5Kp?usp=sharing). Put the downloaded folders into "data/" sub-folder in the code directory.

## Training

Taking the scene `hotdog` as an example, the training process is as follows.

1. Implicit neural surface reconstruction. (Same as [Neus](https://github.com/Jangmin-Lee/NeuS))

   ```sh
   python exp_runner_sdf.py --conf ./confs/sdf.conf \
                                 --case hotdog
   ```

2. Semantic segmentation.

   ```sh
   python exp_runner_cato.py --conf ./confs/sdf.conf \
                                 --case hotdog
   ```
   
3. Jointly optimize diffuse albedo, roughness, indirect illumination and direct illumination.

   ```sh
   python exp_runner_shadow.py --conf ./confs/invrender.conf \
                                 --case hotdog
   ```
   
## Acknowledgement

We have used codes from the following repositories, and we thank the authors for sharing their codes.

- NeuS: [https://github.com/Jangmin-Lee/NeuS](https://github.com/Jangmin-Lee/NeuS)

- InvRender: [https://github.com/zju3dv/InvRender](https://github.com/zju3dv/InvRender)

- NeFII: [https://github.com/FuxiComputerVision/Nefii](https://github.com/FuxiComputerVision/Nefii)


