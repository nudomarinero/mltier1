# Maximum Likelihood ratio matching for the LOFAR Tier1

Multiwavelength cross-match and maximum likelihood for the LOFAR 
Surveys Tier 1.

**The software is currently under heavy development**

The dependencies are:
    * astropy
    * numpy
    * wquantiles
    * pandas (used to estimate the i-band magnitude)

## Matching between PanSTARRS and WISE data

### WISE data

We selected the AllWISE Source Catalog 
[catalogue](http://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd?catalog=allwise_p3as_psd&mode=html&passproj&)

All sky query is selected and the constrains in __ra__ and __dec__ are included.

### PanSTARRS data

We use the DR1 of PanSTARRS and download the data using the CasJobs interface.

### ML matching

The data PanSTARRS and WISE are matched using a Maximum Likelihood (ML) method.

The steps are the following:
- Filtering of the catalogues: [notebook](https://github.com/nudomarinero/mltier1/blob/master/PanSTARRS_WISE_catalogues.ipynb)
- Computing of Q0: [notebook](https://github.com/nudomarinero/mltier1/blob/master/PanSTARRS_WISE_Q0.ipynb)
- Estimation of the ML parameters: [notebook](https://github.com/nudomarinero/mltier1/blob/master/PanSTARRS_WISE_pre_ml.ipynb)
- Application of the ML matching: [notebook](https://github.com/nudomarinero/mltier1/blob/master/PanSTARRS_WISE_ML.ipynb)

A final catalogue with all the sources (matched and non-matched) called pw.fits 
is produced at the end. 

## Matching between LOFAR sources and the combined catalogue

A Maximum Likelihood method is applied to LOFAR sources and sources
in the combined WISE PanSTARRS catalogue.

### Compute Q_0

The Q_0 for the individual Gaussian catalogue are computed in this 
[notebook](https://github.com/nudomarinero/mltier1/blob/master/Match_LOFAR_Q0_gaus.ipynb).
