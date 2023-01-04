
## OVERVIEW

This directory contains analysis code and data to generate Figure 2 in Ryan, C., Santangelo, M., Stephenson, B., Branch, T., Wilson, E. and Savoca, M. (2023): Commercial krill fishing within a foraging supergroup of fin whales in the Southern Ocean. Ecology, in press.

A streamlined version of our analysis and plotting code is provided in the jupyter notebook `analysis_code-clean.ipynb`, which calls functions stored in `helper_functions.py`. Data files and plots are stored in `data/` and `plots/` respectively.


## DATA SOURCES

Links to the original data sources are provided below. Since we do not manage these databases, we cannot guarantee their availability. Links to alternative databases are provided but these data may be in different formats.

**sea ice (NOAA/NSIDC CDR v4)**: 
- retrieved from https://nsidc.org/data/g02202 
- alternate source: https://polarwatch.noaa.gov/erddap/files/nsidcG02202v4shmday/

**SST (NOAA OI v2)**: 
- retrieved from https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html

**Chlorophyll-a (NOAA VIIRS via polarwatch):** 
- retrieved using: https://polarwatch.noaa.gov/catalog/chl-viirs-noaa-sq/preview/?dataset=monthly&var=chlor_a&time_min=2022-02-01T12:00:00Z&time_max=2022-02-01T12:00:00Z&proj=epsg3031&colorBar=KT_algae,,Log,0.001,10,
- alternate source (daily): https://polarwatch.noaa.gov/erddap/files/nesdisVHNSQchlaDaily/


**SOCCOM float 5906226**:
- retrieved from https://www.mbari.org/science/upper-ocean-systems/chemical-sensor-group/soccom-float-visualization/
- Alternate source: https://fleetmonitoring.euro-argo.eu/float/5906226

Spatial ranges:
+ SST and Chl: 100W-0W, 75S-40S (for chl data, a stride of 5 was used for lat/lon)
+ sea ice: entire SO (no option to subselect pre-download)
+ SOCCOM float 5906226 (all available data)

Time ranges:
+ Chl: Jan 2012-Feb 2022, monthly
+ SST: Jan 1982-Feb 2022, monthly
+ SIE: 1978-May 2022, monthly 
+ SOCCOM float 5906226 (all available data)



## Python configuration
The analysis code is written in Python (version 3.9) as distributed by Anaconda (version  4.13.0). The code uses several packages that are not pre-installed in most Python distributions. Details of the specific packages and dependencies are contained in the `environment.yml` file. Once `conda` is installed, one can recreate this environment using the command: `conda env create -f environment.yml`. Though `conda` should resolve any cross-platform compatibility issues, this has not been tested extensively. Please refer to the Conda documentation (https://conda.io/projects/conda/en/latest/index.html) for further details.

