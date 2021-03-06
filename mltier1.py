import numpy as np
from numpy.linalg import det
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from tqdm import tqdm, tnrange, tqdm_notebook
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.neighbors import KernelDensity


## general functions
def describe(var, decimals=3, nullvalue=-999):
    """Describe one variable
    """
    format_string_base = ("{{:.{d}f}} +/- {{:.{d}f}}; "
                          "median: {{:.{d}f}}; "
                          "limits: [{{:.{d}f}}, {{:.{d}f}}]; "
                          "N={{}} ({{}} NaN; {{}} masked)")
    format_string = format_string_base.format(d=decimals)
    print(format_string.format(np.nanmean(var), np.nanstd(var), np.nanmedian(var), 
                               np.nanmin(var), np.nanmax(var), len(var), 
                               np.sum(np.isnan(var)), len(var[var == nullvalue])))

## Sky functions
def ru2ra(x, ra1=0., ra2=360.):
    """Transform a random uniform number to a RA between
    the RA specified as an input"""
    return x*(ra2-ra1)+ra1

def ru2dec(y, dec1=-90, dec2=90.):
    """Transform a random uniform number to a Dec between
    the decs. specified as an input"""
    sin_dec1rad = np.sin(np.deg2rad(dec1))
    sin_dec2rad = np.sin(np.deg2rad(dec2))
    inner = y*(sin_dec2rad - sin_dec1rad) + sin_dec1rad
    return np.rad2deg(np.arcsin(inner))

def generate_random_catalogue(n, ra1=0., ra2=360., dec1=-90, dec2=90.):
    """Generate a random catalogue in the zone deffined by the 
    input coordinates
    """
    x = np.random.rand(n)
    y = np.random.rand(n)
    ra = ru2ra(x, ra1=ra1, ra2=ra2)
    dec = ru2dec(y, dec1=dec1, dec2=dec2)
    return SkyCoord(ra, dec,
                    unit=(u.deg, u.deg), 
                    frame='icrs')

def area_ra_dec(ra_down, ra_up, dec_down, dec_up):
    """Compute the area in a region between two right ascentions and
    two declinations.
    The unit of the output is squared arsecs.
    """
    return ((np.deg2rad(ra_up) - np.deg2rad(ra_down)) *
            (np.sin(np.deg2rad(dec_up)) - np.sin(np.deg2rad(dec_down))) *
            np.rad2deg(3600.)**2)

## Data holding classes
class Field(object):
    """
    Class to represent a region of the sky between two right ascensions 
    and two declinations.
    """
    def __init__(self, ra_down, ra_up, dec_down, dec_up):
        self.ra_down = ra_down
        self.ra_up = ra_up
        self.dec_down = dec_down
        self.dec_up = dec_up
        self.area = area_ra_dec(self.ra_down, self.ra_up, 
                                self.dec_down, self.dec_up)
        
    def filter_catalogue(self, catalogue, colnames=("ra", "dec")):
        """
        Filter a catalogue to the 
        """
        # TODO: Check if colnames in the catalogue
        return catalogue[
                ((catalogue[colnames[0]] >= self.ra_down)   & 
                 (catalogue[colnames[0]] <= self.ra_up)     & 
                 (catalogue[colnames[1]] >= self.dec_down) & 
                 (catalogue[colnames[1]] <= self.dec_up))]
    
    def random_catalogue(self, n):
        """
        Generate a random catalogue in the area with n sources
        """
        return generate_random_catalogue(
            n, ra1=self.ra_down, ra2=self.ra_up, 
            dec1=self.dec_down, dec2=self.dec_up)

class Q_0(object):
    """
    Compute the Q_0 given a set of catalogues and a field
    """
    def __init__(self, coords_small, coords_big, field, radius=5.):
        self.coords_small = coords_small
        self.coords_big = coords_big
        self.field = field
        self.radius = radius
        self.n_small = len(self.coords_small)
        self.n_big = len(self.coords_big)
    
    def __call__(self, radius=None):
        """Compute the Q_0 for a given radius (in arcsecs)"""
        if radius is None:
            radius = self.radius
        
        # Generate random catalogue with n sources as the small one
        random_small = self.field.random_catalogue(self.n_small)
        idx_random_small, idx_big, d2d, d3d = search_around_sky(
            random_small, self.coords_big, radius*u.arcsec)
        
        nomatch_random = self.n_small - len(np.unique(idx_random_small))
        
        # Compute match in radius
        idx_small, idx_big, d2d, d3d = search_around_sky(
            self.coords_small, self.coords_big, radius*u.arcsec)
        nomatch_small = self.n_small - len(np.unique(idx_small))
                                          
        return (1. - float(nomatch_small)/float(nomatch_random))
    
## Error functions
def R(theta):
    """Rotation matrix.
    Input:
      - theta: angle in degrees
    """
    theta_rad = np.deg2rad(theta)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s], [s, c]])

def get_sigma(maj_error, min_error, pos_angle, 
              radio_ra, radio_dec, 
              opt_ra, opt_dec, opt_ra_err, opt_dec_err, 
              additional_error=0.6):
    """
    Get the covariance matrix between an elongated 
    radio source and an optical source.
    
    Input:
    * maj_error: error in the major axis of the radio Gaussian in arsecs
    * min_error: error in the minor axis of the radio Gaussian in arsecs
    * pos_angle: position angle of the radio Gaussian in degrees
    * radio_ra: Right ascension of the radio source in degrees
    * radio_dec: Declination of the radio source in degrees
    * opt_ra: Right ascension of the optical source in degrees
    * opt_dec: Declination of the optical source in degrees
    * opt_ra_err: Error in right ascension of the optical source in degrees
    * opt_dec_err: Error in declination of the optical source in degrees
    * additonal_error: Additonal term to add to the error. By default
        it adds an astrometic error of 0.6 arcsecs.
    
    Output:
    * sigma: Combined covariance matrix
    """
    factor = 0.60056120439322491 # sqrt(2.0) / sqrt(8.0 * log(2)); see Condon(1997) for derivation of adjustment factor
    majerr = factor * maj_error
    minerr = factor * min_error
    # angle between the radio and the optical sources
    cosadj = np.cos(np.deg2rad(0.5*(radio_dec + opt_dec)))
    phi = np.arctan2((opt_dec - radio_dec), ((opt_ra - radio_ra)*cosadj))
    # angle from direction of major axis to vector joining LOFAR source and optical source
    alpha = np.pi/2.0 - phi - np.deg2rad(pos_angle) 
    # Covariance matrices
    sigma_radio_nr = np.array([[majerr**2, 0], [0, minerr**2]])
    sigma_optical_nr = np.array([[opt_ra_err**2, 0], [0, opt_dec_err**2]])
    # Rotate the covariance matrices
    R_radio = R(alpha)
    sigma_radio = R_radio @ sigma_radio_nr @ R_radio.T
    R_optical = R(-phi)
    sigma_optical = R_optical @ sigma_optical_nr @ R_optical.T
    # Additional error
    sigma_additonal_error = np.array([[additional_error**2, 0], [0, additional_error**2]])
    sigma = sigma_radio + sigma_optical + sigma_additonal_error
    return sigma

def get_sigma_all_old(maj_error, min_error, pos_angle, 
              radio_ra, radio_dec, 
              opt_ra, opt_dec, opt_ra_err, opt_dec_err, 
              additonal_error=0.6):
    """
    Get the combined error and the axes components between an elongated 
    radio source and an optical source.
    
    Input:
    * maj_error: error in the major axis of the radio Gaussian in arsecs
    * min_error: error in the minor axis of the radio Gaussian in arsecs
    * pos_angle: position angle of the radio Gaussian in degrees
    * radio_ra: Right ascension of the radio source in degrees
    * radio_dec: Declination of the radio source in degrees
    * opt_ra: Right ascension of the optical source in degrees
    * opt_dec: Declination of the optical source in degrees
    * opt_ra_err: Error in right ascension of the optical source in degrees
    * opt_dec_err: Error in declination of the optical source in degrees
    * additonal_error: Additonal term to add to the error. By default
        it adds an astrometic error of 0.6 arcsecs.
    
    Output:
    * sigma: Combined error
    * sigma_maj: Error in the major axis direction
    * sigma_min: Error in the minor axis direction
    """
    factor = 0.60056120439322491 # sqrt(2.0) / sqrt(8.0 * log(2)); see Condon(1997) for derivation of adjustment factor
    majerr = factor * maj_error
    minerr = factor * min_error
    cosadj = np.cos(np.deg2rad(0.5*(radio_dec + opt_dec)))
    phi = np.arctan2((opt_dec - radio_dec), ((opt_ra - radio_ra)*cosadj))
    # angle from direction of major axis to vector joining LOFAR source and optical source
    sigma = np.pi/2.0 - phi - np.deg2rad(pos_angle) 
    
    maj_squared = ((majerr * np.cos(sigma))**2 + 
                   (opt_ra_err * np.cos(phi))**2 +
                   additonal_error**2/2.
                   )
    min_squared = ((minerr * np.sin(sigma))**2 + 
                   (opt_dec_err * np.sin(phi))**2 +
                   additonal_error**2/2.
                   )
    return np.sqrt(maj_squared + min_squared), np.sqrt(maj_squared), np.sqrt(min_squared)


def get_sigma_all(maj_error, min_error, pos_angle, 
              radio_ra, radio_dec, 
              opt_ra, opt_dec, opt_ra_err, opt_dec_err, 
              additional_error=0.6):
    """Apply the get_sigma function in parallel and return the determinant of 
    the covariance matrix and its [1,1] term (or [0,0] in Python)
    """
    n = len(opt_ra)
    det_sigma = np.empty(n)
    sigma_0_0 = np.empty(n)
    for i in range(n):
        sigma = get_sigma(maj_error, min_error, pos_angle, 
              radio_ra, radio_dec, 
              opt_ra[i], opt_dec[i], opt_ra_err[i], opt_dec_err[i], 
              additional_error=additional_error)
        det_sigma[i] = det(sigma)
        sigma_0_0[i] = sigma[0,0]
    return sigma_0_0, det_sigma

## ML functions
def get_center(bins):
    """
    Get the central positions for an array defining bins
    """
    return (bins[:-1] + bins[1:]) / 2

def get_n_m(magnitude, bin_list, area):
    """Compute n(m)
    Density of sources per unit of area
    **Note that the output is cumulative**
    """
    n_hist, _ = np.histogram(magnitude, bin_list)
    return np.cumsum(n_hist)/area

def get_n_m_kde(magnitude, bin_centre, area, bandwidth=0.2):
    """Compute n(m)
    Density of sources per unit of area in a non-cumulative
    fashion using a KDE.
    For this function we need the centre of the bins instead 
    of the edges.
    **Note that the output is non-cumulative**
    """
    kde_skl = KernelDensity(bandwidth=bandwidth)
    kde_skl.fit(magnitude[:, np.newaxis])
    pdf = np.exp(kde_skl.score_samples(bin_centre[:, np.newaxis]))
    return pdf/area*len(magnitude)/np.sum(pdf)

def get_q_m(magnitude, bin_list, n, n_m, area, radius=5):
    """Compute q(m)
    Normalized probability of a real match
    **Note that the output is cumulative**
    """
    n_hist_total, _ = np.histogram(magnitude, bin_list)
    # Correct probability if there are no sources
    if len(magnitude) == 0:
        n_hist_total = np.ones_like(n_hist_total)*0.5
    # Estimate real(m)
    real_m = n_hist_total
    # Remove small negative numbers
    real_m[real_m <= 0.] = 0.
    real_m_cumsum = np.cumsum(real_m)
    return real_m_cumsum/real_m_cumsum[-1]

def get_q_m_kde(magnitude, bin_centre, radius=5, bandwidth=0.2):
    """Compute q(m)
    Normalized probability of a real match in a 
    non-cumulative fashion using a KDE.
    For this function we need the centre of the bins instead 
    of the edges.
    **Note that the output is non-cumulative**
    """
    # Get real(m)
    kde_skl = KernelDensity(bandwidth=bandwidth)
    kde_skl.fit(magnitude[:, np.newaxis])
    pdf_q_m = np.exp(kde_skl.score_samples(bin_centre[:, np.newaxis]))
    real_m = pdf_q_m*len(magnitude)/np.sum(pdf_q_m)
    # Correct probability if there are no sources
    if len(magnitude) == 0:
        real_m = np.ones_like(n_hist_total)*0.5
    # Remove small negative numbers
    real_m[real_m <= 0.] = 0.
    return real_m/np.sum(real_m)


def estimate_q_m(magnitude, bin_list, n_m, coords_small, coords_big, radius=5):
    """Compute q(m)
    Estimation of the distribution of real matched sources with respect 
    to a magnitude (normalized to 1). As explained in Fleuren et al.
    """
    assert len(magnitude) == len(coords_big)
    # Cross match
    idx_small, idx_big, d2d, d3d = search_around_sky(
        coords_small, coords_big, radius*u.arcsec)
    n_small = len(coords_small)
    idx = np.unique(idx_big)
    # Get the distribution of matched sources
    n_hist_total, _ = np.histogram(magnitude[idx], bin_list)
    # Correct probability if there are no sources
    if len(magnitude[idx]) == 0:
        n_hist_total = np.ones_like(n_hist_total)*0.5
    # Estimate real(m)
    n_m_nocumsum = np.ediff1d(n_m, to_begin=n_m[0])
    real_m = n_hist_total - n_small*n_m_nocumsum*np.pi*radius**2
    # Remove small negative numbers
    real_m[real_m <= 0.] = 0.
    real_m_cumsum = np.cumsum(real_m)
    return real_m_cumsum/real_m_cumsum[-1]

def estimate_q_m_kde(magnitude, bin_centre, n_m, coords_small, coords_big, radius=5, bandwidth=0.2):
    """Compute q(m)
    Estimation of the distribution of real matched sources with respect 
    to a magnitude (normalized to 1). As explained in Fleuren et al. in a 
    non-cumulative fashion using a KDE.
    For this function we need the centre of the bins instead 
    of the edges.
    """
    assert len(magnitude) == len(coords_big)
    # Cross match
    idx_small, idx_big, d2d, d3d = search_around_sky(
        coords_small, coords_big, radius*u.arcsec)
    n_small = len(coords_small)
    idx = np.unique(idx_big)
    # Get the distribution of matched sources
    kde_skl_q_m = KernelDensity(bandwidth=bandwidth)
    kde_skl_q_m.fit(magnitude[idx][:, np.newaxis])
    pdf_q_m = np.exp(kde_skl_q_m.score_samples(bin_centre[:, np.newaxis]))
    n_hist_total = pdf_q_m*len(magnitude[idx])/np.sum(pdf_q_m)
    # Correct probability if there are no sources ## CHECK
    if len(magnitude[idx]) == 0:
        n_hist_total = np.ones_like(n_hist_total)*0.5
    # Estimate real(m)
    real_m = n_hist_total - n_small*n_m*np.pi*radius**2
    # Remove small negative numbers ## CHECK
    real_m[real_m <= 0.] = 0.
    return real_m/np.sum(real_m)

def fr(r, sigma):
    """Get the probability related to the spatial distribution"""
    return 0.5/np.pi/det(sigma)*np.exp(-0.5*r**2/sigma[0,0])

def fr_u(r, sigma_0_0, det_sigma):
    """Get the probability related to the spatial distribution"""
    return 0.5/np.pi/det_sigma*np.exp(-0.5*r**2/sigma_0_0)

def fr_u_old(r, sigma, sigma_maj, sigma_min):
    """Get the probability related to the spatial distribution.
    UPDATED TO NEW FORMULA. MERGE LATER."""
    return 0.5/np.pi/sigma_maj/sigma_min*np.exp(-0.5*r**2/(sigma**2))

class SingleMLEstimator(object):
    """
    Class to estimate the Maximum Likelihood ratio
    """
    def __init__(self, q0, n_m, q_m, center):
        self.q0 = q0
        self.n_m = n_m
        self.q_m = q_m
        self.center = center
    
    def get_qm(self, m):
        """Get q(m)
        """
        return np.interp(m, self.center, self.q_m*self.q0)

    def get_nm(self, m):
        """Get n(m)"""
        return np.interp(m, self.center, self.n_m)
    
    def __call__(self, m, r, sigma_0_0, det_sigma):
        """Get the likelihood ratio"""
        return fr_u(r, sigma_0_0, det_sigma) * self.get_qm(m) / self.get_nm(m)

class SingleMLEstimatorOld(object):
    """
    Class to estimate the Maximum Likelihood ratio
    """
    def __init__(self, q0, n_m, q_m, center):
        self.q0 = q0
        self.n_m = n_m
        self.q_m = q_m
        self.center = center
    
    def get_qm(self, m):
        """Get q(m)
        """
        return np.interp(m, self.center, self.q_m*self.q0)

    def get_nm(self, m):
        """Get n(m)"""
        return np.interp(m, self.center, self.n_m)
    
    def __call__(self, m, r, sigma, sigma_maj, sigma_min):
        """Get the likelihood ratio"""
        return fr_u_old(r, sigma, sigma_maj, sigma_min) * self.get_qm(m) / self.get_nm(m)
    
class MultiMLEstimator(object):
    """
    Class to estimate the Maximum Likelihood ratio in a vectorized 
    fashion.
    """
    def __init__(self, q0, n_m, q_m, center):
        self.q0 = q0
        self.n_m = n_m
        self.q_m = q_m
        self.center = center
    
    def get_qm(self, m, k):
        """Get q(m)
        """
        return np.interp(m, self.center[k], self.q_m[k]*self.q0[k])

    def get_nm(self, m, k):
        """Get n(m)"""
        return np.interp(m, self.center[k], self.n_m[k])
    
    def get_qm_vect(self, m, k):
        """Get q(m) for a given category
        """
        return np.vectorize(self.get_qm)(m, k)

    def get_nm_vect(self, m, k):
        """Get n(m) for a given category
        """
        return np.vectorize(self.get_nm)(m, k)
    
    def __call__(self, m, r, sigma_0_0, det_sigma, k):
        """Get the likelihood ratio"""
        return fr_u(r, sigma_0_0, det_sigma) * self.get_qm_vect(m, k) / self.get_nm_vect(m, k)

class MultiMLEstimatorOld(object):
    """
    Class to estimate the Maximum Likelihood ratio in a vectorized 
    fashion.
    """
    def __init__(self, q0, n_m, q_m, center):
        self.q0 = q0
        self.n_m = n_m
        self.q_m = q_m
        self.center = center
    
    def get_qm(self, m, k):
        """Get q(m)
        """
        return np.interp(m, self.center[k], self.q_m[k]*self.q0[k])

    def get_nm(self, m, k):
        """Get n(m)"""
        return np.interp(m, self.center[k], self.n_m[k])
    
    def get_qm_vect(self, m, k):
        """Get q(m) for a given category
        """
        return np.vectorize(self.get_qm)(m, k)

    def get_nm_vect(self, m, k):
        """Get n(m) for a given category
        """
        return np.vectorize(self.get_nm)(m, k)
    
    def __call__(self, m, r, sigma, sigma_maj, sigma_min, k):
        """Get the likelihood ratio"""
        return fr_u_old(r, sigma, sigma_maj, sigma_min) * self.get_qm_vect(m, k) / self.get_nm_vect(m, k)
    
def q0_min_level(q_0_list, min_level=0.001):
    """Ensures that the minimum value of the Q_0 for each bin is always 
    above a minimum threshold
    """
    q_0 = np.array(q_0_list)
    q_0[q_0 < min_level] = min_level
    return q_0

def q0_min_numbers(q_0_list, numbers_combined_bins):
    """Ensures that the minimum value of the Q_0 for each bin is always 
    above a minimum threshold that depends on the number of sources
    in each bin.
    """
    q_0 = np.array(q_0_list)
    thresholds = 1./numbers_combined_bins
    q_0[q_0 < thresholds] = thresholds[q_0 < thresholds]
    return q_0

def get_threshold(lr_dist, n_bins=200, n_gal_cut=1000):
    """Get the threshold as the position of the first minima in
    the LR distribution in log space"""
    from scipy.signal import savgol_filter
    val, bins = np.histogram(np.log10(lr_dist + 1), bins=200)
    if n_gal_cut is not None:
        val[val >= n_gal_cut] = n_gal_cut
    v1 = savgol_filter(val, 31, 3)
    g1 = np.gradient(v1) # Firt derivative
    g2 = np.gradient(g1) # Second derivative
    center = get_center(bins)
    t_value = 10**(center[np.argmax(g2 < 0)])-1
    return t_value

## Multiprocessing functions
def parallel_process(array, function, n_jobs=3, use_kwargs=False, front_num=3, notebook=False):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
        see: http://danshiebler.com/2016-09-14-parallel-progress-bar/
    """
    if notebook:
        tqdm_f = tqdm_notebook
    else:
        tqdm_f = tqdm
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm_f(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm_f(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in enumerate(tqdm_f(futures)):
        out.append(future.result())
    return front + out
