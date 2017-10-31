# Maximum Likelihood ratio matching for the LOFAR Tier1

Multiwavelength cross-match and maximum likelihood for the LOFAR 
Surveys Tier 1.

The software requires Python 3.4 or higher to run the ML estimation in 
parallel. Please notice that running the code in big datasets requires a
considerable amount of memory.

The dependencies are:
* astropy
* numpy
* wquantiles
* pandas (used to estimate the i-band magnitude)
* tqdm
* scipy (optional; used in some likelihood ratio threshold methods)

## Matching between PanSTARRS and WISE data

### WISE data

We selected the AllWISE Source Catalog 
[catalogue](http://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd?catalog=allwise_p3as_psd&mode=html)

All sky query is selected and the constrains in __ra__ and __dec__ are included.

### PanSTARRS data

We use the DR1 of PanSTARRS and download the data using the CasJobs interface.

The query used for the Tier 1 region is in: [query](https://github.com/nudomarinero/mltier1/blob/master/panstarrs_query.md)

### ML matching

The data PanSTARRS and WISE are matched using a Maximum Likelihood (ML) method.

The steps are the following:
- Filtering of the catalogues: [notebook](https://github.com/nudomarinero/mltier1/blob/master/PanSTARRS_WISE_catalogues.ipynb)
- Correct the reddening: [notebook](https://github.com/nudomarinero/mltier1/blob/master/PanSTARRS_WISE_reddening.ipynb)
- Computing of Q0: [notebook](https://github.com/nudomarinero/mltier1/blob/master/PanSTARRS_WISE_Q0.ipynb)
- Estimation of the ML parameters: [notebook](https://github.com/nudomarinero/mltier1/blob/master/PanSTARRS_WISE_pre_ml.ipynb)
- Application of the ML matching: [notebook](https://github.com/nudomarinero/mltier1/blob/master/PanSTARRS_WISE_ML.ipynb)

A final catalogue with all the sources (matched and non-matched) called ```pw.fits``` 
is produced at the end. 

## Matching between LOFAR sources and the combined catalogue

A Maximum Likelihood method is also applied to LOFAR sources and sources
in the combined WISE-PanSTARRS catalogue.

Before applying the ML matching we corrected an error in the format of the pw 
catalogue with this [notebook](https://github.com/nudomarinero/mltier1/blob/master/Correct_pw_catalogue.ipynb) 

### Compute Q_0 and intermediate parameters

The $Q_0$ for the catalogue is computed in this 
[notebook](https://github.com/nudomarinero/mltier1/blob/master/Match_LOFAR_Q0.ipynb).

The output of the notebook 
[lofar_params.pckl](https://github.com/nudomarinero/mltier1/blob/master/lofar_params.pckl)
is used as the initial input for the notebook [Match_LOFAR_combined-iteration.ipynb](https://github.com/nudomarinero/mltier1/blob/master/Match_LOFAR_combined-iteration.ipynb). 
In this last notebook the ML paremeters 
are adjusted iteratively until they converge (currently in 5 iterations). 
The output of this notebook is located in the directory idata/main. The 
lofar_params_<n>.pckl (being "n" the number of the last iteration and the 
bigger number found in the directory) should be manually moved to the main 
directory and renamed to "lofar_params.pckl". This will be the parameters 
used for the ML matching notebooks. 

We discarded the extended sources with a major axis bigger than 30 arcseconds to 
compute the parameters.

### ML matching

The ML matching is shown in:
[notebook](https://github.com/nudomarinero/mltier1/blob/master/Match_LOFAR_combined_final.ipynb).

The matching can be applied to any input catalogue. We applied it to the Gaussians catalogue in:
[notebook](https://github.com/nudomarinero/mltier1/blob/master/Match_LOFAR_combined_generic.ipynb).
This notebook can be used to match to other catalogues as well.

There is also a version that saves all the matches avobe the selected ML threshold.
In this case, if a source is matched by two or more WISE-PanSTARRS sources with 
a ML above the threshold, all the matches are saved:
[notebook](https://github.com/nudomarinero/mltier1/blob/master/Match_LOFAR_combined_above-threshold.ipynb).

#### Auxiliary code

Due to an error we run the ML matching notebooks only for non-extended sources.
For completitude we also run the matching in extended sources in the corresponding
notebooks:
* [final](https://github.com/nudomarinero/mltier1/blob/master/Match_LOFAR_combined_final-extended.ipynb)
* [generic (gaus catalogue)](https://github.com/nudomarinero/mltier1/blob/master/Match_LOFAR_combined_generic-extended.ipynb)
* [avobe threshold](https://github.com/nudomarinero/mltier1/blob/master/Match_LOFAR_combined_above-threshold-extended.ipynb)

The output catalogues can then be combined toghether.

There are a couple of obsolete notebooks that are no longer used:
* [Save_main.ipynb](https://github.com/nudomarinero/mltier1/blob/master/Save_main.ipynb)
* [Save_gaus.ipynb](https://github.com/nudomarinero/mltier1/blob/master/Save_gaus.ipynb)

The main auxiliary code used by the ML estimators is in ```mltier1.py```

## Automatic execution

It is possible to execute automatically the notebooks without the need of
a browser. This is very useful for the automated execution of pipelines or
for the running of heavy computations (some of the notebooks take several 
hours or even days to run). To run automatically a notebook you can use:

```

```



