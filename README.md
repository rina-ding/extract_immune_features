# extract_immune_features
Nuclei detection/classification and extraction of immune-tumor colocalization features

This code achieves nuclei detection/classification using the TIAToolBox first and then extract immune-tumor colocalization features for one image tile. You can modify the code to extract features from all provided tiles from a whole slide image in the future.

See this documentation for details on the 27 immune-tumor colocalization features: https://docs.google.com/document/d/1rn1eI37WYUh5Kgx_OwyMVFFxXpfjKVxJ-Qg7lUQ-m3o/edit?tab=t.0

### Required packages
First, create a pytorch docker container using:
```
docker run  --shm-size=2g --gpus all -it --rm -v /:/workspace -v /etc/localtime:/etc/localtime:ro nvcr.io/nvidia/pytorch:24.02-py3
```
Then run the following commands:

```
chmod +x pip_commands.sh
```
```
./pip_commands.sh
```

More information on the pytorch docker container `nvcr.io/nvidia/pytorch:24.02-py3` can be found here(https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags).


### Steps to run the code

1. `cd extract_immune_features`
2. Put your input tile named `image.png` into the folder `example_input`
2. Run `CUDA_VISIBLE_DEVICES=0 python main.py`

### Expected outputs

The outputs are saved in the `./example_input/results` folder
1. Some intermediate files which you don't need to care about.
2. `nuclei_classification_results.csv` contains the x, y coordinates, nuclei type, and nuclei area (in pixels) for each detected nucleus in that image tile.
3. `tile_features.csv` contains the extracted 27 immune-tumor colocalization features.
4. `overlaid_predictions.png` visualizes the nuclei classification results on your original image tile.

### References

- https://github.com/TissueImageAnalytics/tiatoolbox
- Shaban, Muhammad, et al. "A novel digital score for abundance of tumour infiltrating lymphocytes predicts disease free survival in oral squamous cell carcinoma." Scientific reports 9.1 (2019): 13341.
- Ding, Ruiwen, et al. "Image analysis reveals molecularly distinct patterns of TILs in NSCLC associated with treatment outcome." NPJ precision oncology 6.1 (2022): 33.

