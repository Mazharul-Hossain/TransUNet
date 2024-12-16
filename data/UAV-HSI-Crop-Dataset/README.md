# UAV HSI Crop Dataset

Ref: <https://www.scidb.cn/en/detail?dataSetId=6de15e4ec9b74dacab12e29cb557f041>

The study area locates in Shenzhou City, Hebei Province, China, and it consists of two sub-regions, Majiakou Village's plots(MJK_N, MJK_S) and Xijingmeng Valliage's plots(XJM). The hyperspectral data obtained on September 18th, 2019, with an electric hexacopter. The UAV carried the sensor Pika L hyperspectral imager made by Resonon company, which has a spectral range of 385nm to 1024nm with a total of 200 bands. And the flight height was set to be 100 m, resulting in a spatial resolution of 0.1 m per pixel. Spectronon and ENVI software was utilized for the post-processing of the hyperspectral data, including radiometric calibration, geometric correction, image stitching, and atmospheric correction. The images cropped to the 96*96*200 patches, then split into the Training set and Test set with a ratio of 8:2. The patches in the dataset are all Numpy data type.

## Folder Structure

Download and unzip the data to /project/mhssain9/data/UAV-HSI-Crop-Dataset. The folder structure will look like this:

    UAV-HSI-Crop-Dataset
        |- Train
            |- Training
                |- rs
                |- gt        
            |- Validation
                |- rs
                |- gt
        |- Test
            |- rs
            |- gt        

## References

    1. Niu, B., Feng, Q., Chen, B., Ou, C., Liu, Y. and Yang, J., 2022. HSI-TransUNet: A transformer based semantic segmentation model for crop mapping from UAV hyperspectral imagery. Computers and Electronics in Agriculture, 201, p.107297.
