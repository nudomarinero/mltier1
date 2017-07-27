from __future__ import print_function
import json
import requests
import numpy as np
from astropy.utils.console import ProgressBar
from scipy.interpolate import CubicSpline
from astropy.table import Table
from astropy import units as u
from astropy.io.votable import parse_single_table


FILTER_URIS = {
    "W1": "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID=WISE/WISE.W1",
    "W2": "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID=WISE/WISE.W2",
    "W3": "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID=WISE/WISE.W3",
    "W4": "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID=WISE/WISE.W4",
    "g": "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID=PAN-STARRS/PS1.g",
    "r": "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID=PAN-STARRS/PS1.r",
    "i": "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID=PAN-STARRS/PS1.i",
    "z": "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID=PAN-STARRS/PS1.z",
    "y": "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID=PAN-STARRS/PS1.y",
}

FILTER_EXT = {
    "W1": 0.19562893570345422,
    "W2": 0.13438419437135862,
    "W3": 0.046003159224496736,
    "W4": 0.024851094687942197,
    "g": 3.6121011749827514,
    "r": 2.5687511251039137,
    "i": 1.897167710862949,
    "z": 1.4948335405125801,
    "y": 1.2478667172854474
}

FILTERS = ["g", "r", "i", "z", "y", "W1", "W2", "W3", "W4"]

def query(lon, lat, coordsys='equ', mode='sfd', verbose=True):
    '''
    Send a line-of-sight reddening query to the Argonaut web server.

    Inputs:
      lon, lat: longitude and latitude, in degrees.
      coordsys: 'gal' for Galactic, 'equ' for Equatorial (J2000).
      mode: 'full', 'lite' or 'sfd'

    In 'full' mode, outputs a dictionary containing, among other things:
      'distmod':    The distance moduli that define the distance bins.
      'best':       The best-fit (maximum proability density)
                    line-of-sight reddening, in units of SFD-equivalent
                    E(B-V), to each distance modulus in 'distmod.' See
                    Schlafly & Finkbeiner (2011) for a definition of the
                    reddening vector (use R_V = 3.1).
      'samples':    Samples of the line-of-sight reddening, drawn from
                    the probability density on reddening profiles.
      'success':    1 if the query succeeded, and 0 otherwise.
      'converged':  1 if the line-of-sight reddening fit converged, and
                    0 otherwise.
      'n_stars':    # of stars used to fit the line-of-sight reddening.
      'DM_reliable_min':  Minimum reliable distance modulus in pixel.
      'DM_reliable_max':  Maximum reliable distance modulus in pixel.

    Less information is returned in 'lite' mode, while in 'sfd' mode,
    the Schlegel, Finkbeiner & Davis (1998) E(B-V) is returned.
    '''

    url = 'http://argonaut.skymaps.info/gal-lb-query-light'

    payload = {'mode': mode}

    if coordsys.lower() in ['gal', 'g']:
        payload['l'] = lon
        payload['b'] = lat
    elif coordsys.lower() in ['equ', 'e']:
        payload['ra'] = lon
        payload['dec'] = lat
    else:
        raise ValueError("coordsys '{0}' not understood.".format(coordsys))

    headers = {'content-type': 'application/json'}

    response = requests.post(url, data=json.dumps(payload), headers=headers)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if verbose:
            print('Response received from Argonaut:')
            print(response.text)
        raise e

    return json.loads(response.text)

def f99_extinction(wave):
    """
    Return Fitzpatrick 99 galactic extinction curve as a function of wavelength
    """
    anchors_x = [0., 0.377, 0.820, 1.667, 1.828, 2.141, 2.433, 3.704, 3.846]
    anchors_y = [0., 0.265, 0.829, 2.688, 3.055, 3.806, 4.315, 6.265, 6.591]

    f99 = CubicSpline(anchors_x, anchors_y)

    output_x = (1 / wave.to(u.micron)).value

    return f99(output_x)

def get_eb_v(ra, dec, subsize=450000):
    """
    Get E(B-V) values for all sources based on their position
    """
    #### #### #### 
    ra = list(ra)
    dec = list(dec)
    
    eb_v = np.zeros(len(ra))
    nsteps = int(len(ra)/subsize)+1

    with ProgressBar(nsteps, ipython_widget=True) as bar:
        for i in range(nsteps):
            ra_sub = ra[i*subsize:(i+1)*subsize]
            dec_sub = dec[i*subsize:(i+1)*subsize]

            qresult2 = query(ra_sub, dec_sub, coordsys='equ', mode='sfd')

            eb_v[i*subsize:(i+1)*subsize] = qresult2['EBV_SFD']
            bar.update()
    return eb_v

def get_filter_extinction(filter_URI):
    """
    Get the extinction values for a given filter
    """
    filter_vo_table = parse_single_table(filter_URI)
    filter_array = filter_vo_table.array
    wavelenghts = filter_array["Wavelength"].data * u.AA
    transmissions = filter_array["Transmission"].data
    f99_ext = f99_extinction(wavelenghts)
    return (np.trapz(transmissions * f99_ext, wavelenghts.value) / 
            np.trapz(transmissions, wavelenghts.value))

if __name__ == "__main__":
    pass
    # ## For each filter correct the extinction
    # extinctions = f99_means[ifx]*input_data['EBV']
    # # Apply extinction correction to magnitudes
    # input_data['{0}Mag'.format(filt)][det] -= extinctions[det]
    # # Multiply both flux and fluxerr to maintain same S/N
    # flux_correction = 10**(extinctions/2.5)