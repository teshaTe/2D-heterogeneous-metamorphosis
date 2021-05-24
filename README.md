

# Automatically Controlled Morphing of 2D Shapes with Textures

## published on January 2020, SIAM Journal on Imaging Sciences 13(1):78-107
## SIAM SIIMS: https://epubs.siam.org/doi/10.1137/19M1241581
## ResearchGate: https://www.researchgate.net/publication/339050410_Automatically_Controlled_Morphing_of_2D_Shapes_with_Textures

### Status: *Finished*

**A short description of the project**

This repository presents an implementation of a novel method for morphing between two topologically arbitrary 2D shapes converted to Signed Distance Fields (SDFs) with sophisticated textures (raster colour attributes) using a metamorphosis technique called Space-Time Blending (STB) coupled with Space-Time Transfinite Interpolation (STTI). The method allows for a smooth transition between source and target objects with generating in-between shapes and associated textures without setting any correspondences between boundary points or features.

**Requirements**
- Windows / Linux / MacOS (not tested)
- Qt Creator & Cmake
- OpenCV: https://opencv.org/
- GLM: https://glm.g-truc.net/0.9.9/index.html
- Geometric tools [for convenience headers are included in the repository]: https://www.geometrictools.com/
- SFML: https://www.sfml-dev.org/
- OpenMP [optional fot better performance]: https://www.openmp.org/

*Other libraries essential to build these libraries should be manually resolved* 

**Input for the sample**
- '-im1'     - specify input image \n";
- '-im2'     - specify target image \n";
- '-d'       - specify draw mode [white, blend, closest] \n";
- '-c'       - specify colour mode [rgb, hsv, lab] \n";
- '--bsolid' - specify bounding solid type [halfPlanes, trPyramid, trCone]\n";
- '-s'       - specify the scale for the picture [optional] \n";
- '-n'       - specify the number of output frames \n";
           	   recomended: if bounding solid = 'truncated cone', n = 150+.\n\n";
- '-o'       - specify output path [optional] \n";
- '--affine' - set to enable affine transform to the target image 'im2; \n";
- '--back'   - specify image for background [optional] \n";
- '--help'   - output help\n" << std::endl;

The presented code is free to use. 

*@CopyRight by Alexander Tereshin*
