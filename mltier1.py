import numpy as np
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky

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
    

## ML functions
def get_center(bins):
    """
    Get the central positions for an array defining bins
    """
    return (bins[:-1] + bins[1:]) / 2

def get_n_m(magnitude, bin_list, area):
    """Compute n(m)
    Density of sources per unit of area
    """
    n_hist, _ = np.histogram(magnitude, bin_list)
    return np.cumsum(n_hist)/area

def estimate_q_m(magnitude, bin_list, n_m, coords_small, coords_big, radius=5):
    """Compute q(m)
    Estimation of the distribution of real matched sources with respect 
    to a magnitude (normalized to 1). As explained in Fleuren et al.
    """
    assert len(magnitude) == len(coords_big)
    # Cross match
    idx_small, idx_big, d2d, d3d = search_around_sky(
        coords_small, coords_big, radius*u.arcsec)
    n_xm_small = len(np.unique(idx_small))
    idx = np.unique(idx_big)
    # Get the distribution of matched sources
    n_hist_total, _ = np.histogram(magnitude[idx], bin_list)
    # Estimate real(m)
    real_m = n_hist_total - n_xm_small*n_m*np.pi*(radius/3600.)**2
    # Remove small negative numbers
    real_m[real_m <= 0.] = 0.
    real_m_cumsum = np.cumsum(real_m)
    return real_m_cumsum/real_m_cumsum[-1]
    

#idx_lofar_i, idx_combined_i, d2d_i, d3d_i = search_around_sky(
    #coords_lofar, 
    #coords_combined[combined_i], 
    #5*u.arcsec)
#n_xm_lofar_i = len(np.unique(idx_lofar_i))
#n_hist_total_i, bins_hist_total_i = np.histogram(combined[combined_i][idx_i]["i"], bin_list_i)
