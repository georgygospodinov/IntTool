# IntTool
[Jupyter notebook](https://jupyter.org/) based Laser Interferometry Toolbox

This module is based on the articles:
* [Mitsuo Takeda, Hideki Ina and Seiji Kobayashi, “Fourier-transform method of fringe-pattern analysis for computer-based topography and interferometry”](https://doi.org/10.1364/JOSA.72.000156)
* [Georg Pretzier, "A New Method for Numerical Abel-Inversion"](https://doi.org/10.1515/zna-1991-0715)

## Structure
* [IntTool](./IntTool) - Core python module with functions for data preprocessing, fourier trasforming, abel inversion
* [requirements.txt](./requirements.txt) - Project dependencies
* [demo.ipynb](./demo.ipynb) - Demonstration of fourier filtration technique and abel inversion on simulated interferograms
* [idea_helper.ipynb](./idea_helper.ipynb) - Auxiliary symmetrization pipeline for interferogram processing using **[IDEA](http://www.optics.tugraz.at/idea/idea.html)**
* [solid_target](./solid_target) - Pipeline to process data from laser-solid target interaction interferometry
* [filamentation](./filamentation) - Pipeline to process data from filament plasma channel interferometry
