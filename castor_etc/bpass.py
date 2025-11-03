#         GNU General Public License v3 (GNU GPLv3)
#
# (c) 2022.                            (c) 2022.
# Government of Canada                 Gouvernement du Canada
# National Research Council            Conseil national de recherches
# Ottawa, Canada, K1A 0R6              Ottawa, Canada, K1A 0R6
# All rights reserved                  Tous droits réservés
#
# NRC disclaims any warranties,        Le CNRC dénie toute garantie
# expressed, implied, or               énoncée, implicite ou légale,
# statutory, of any kind with          de quelque nature que ce
# respect to the software,             soit, concernant le logiciel,
# including without limitation         y compris sans restriction
# any warranty of merchantability      toute garantie de valeur
# or fitness for a particular          marchande ou de pertinence
# purpose. NRC shall not be            pour un usage particulier.
# liable in any event for any          Le CNRC ne pourra en aucun cas
# damages, whether direct or           être tenu responsable de tout
# indirect, special or general,        dommage, direct ou indirect,
# consequential or incidental,         particulier ou général,
# arising from the use of the          accessoire ou fortuit, résultant
# software. Neither the name           de l'utilisation du logiciel. Ni
# of the National Research             le nom du Conseil National de
# Council of Canada nor the            Recherches du Canada ni les noms
# names of its contributors may        de ses  participants ne peuvent
# be used to endorse or promote        être utilisés pour approuver ou
# products derived from this           promouvoir les produits dérivés
# software without specific prior      de ce logiciel sans autorisation
# written permission.                  préalable et particulière
#                                      par écrit.
#
# This file is part of the             Ce fichier fait partie du projet
# FORECASTOR ETC project.              FORECASTOR ETC.
#
# FORECASTOR ETC is free software:     FORECASTOR ETC est un logiciel
# you can redistribute it and/or       libre ; vous pouvez le redistribuer
# modify it under the terms of         ou le modifier suivant les termes de
# the GNU General Public               la "GNU General Public
# License as published by the          License" telle que publiée
# Free Software Foundation,            par la Free Software Foundation :
# either version 3 of the              soit la version 3 de cette
# License, or (at your option)         licence, soit (à votre gré)
# any later version.                   toute version ultérieure.
#
# FORECASTOR ETC is distributed        FORECASTOR ETC est distribué
# in the hope that it will be          dans l'espoir qu'il vous
# useful, but WITHOUT ANY WARRANTY;    sera utile, mais SANS AUCUNE
# without even the implied warranty    GARANTIE : sans même la garantie
# of MERCHANTABILITY or FITNESS FOR    implicite de COMMERCIALISABILITÉ
# A PARTICULAR PURPOSE. See the        ni d'ADÉQUATION À UN OBJECTIF
# GNU General Public License for       PARTICULIER. Consultez la Licence
# more details.                        Générale Publique GNU pour plus
#                                      de détails.
#
# You should have received             Vous devriez avoir reçu une
# a copy of the GNU General            copie de la Licence Générale
# Public License along with            Publique GNU avec FORECASTOR ETC ;
# FORECASTOR ETC. If not, see          si ce n'est pas le cas, consultez :
# <http://www.gnu.org/licenses/>.      <http://www.gnu.org/licenses/>.

"""
BPASS
=====

`castor_etc.bpass` provides the methods to incorporate the BPASS models into the calculator.
It inherits one of the classes from the `castor_etc.sources`, and then calculates its own
spectrum using the BPASS models. There are also classes to hold the functions required to
calculate stellar histories and attenuations for modification to the BPASS spectra.

This includes the following classes and functions:
  - star_formation_history
  - spec_attenuation
  - make_bpass_source
  - BPASS_spec
  - make_bpass_stellar_file
  - make_bpass_nebular_file
"""

import os
import warnings
from numbers import Number
from os.path import join

import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18
from astropy.io import fits

from .sources import Profiles, PointSource, ExtendedSource, GalaxySource
from .filepaths import DATAPATH


class star_formation_history:
  """ 
  Class to hold all the different star-formation history models. To be used with
  the BPASSSource class. Includes the function to combine the star formation
  history with the chemical evolution history to generate weights for both the
  metallicity files and the SSP BPASS models.
  """
  def make_sfh_profile(self):
    """
    Generates the star-formation history profile for the galaxy. This calls one
    of the star-formation history profiles (burst, constant, delayed, custom)
    to calculate the star-formation rate at each time step. This is used to
    generate the weights for the BPASS SSP models.

    Attributes
    ----------
      sfr :: array
        The star formation rate in at each time in the star formation
        history profile, in Solar_mass per yr

      sfr_weights :: array
        The weighting for each BPASS SSP model to give the desired star-
        formation history profile (calculated as the star formation rate 
        in each time bin times the width of the time bin)

    Returns
    -------
      None
    """
    # Creating a finely sampled age array. This is done over the
    # same range as the BPASS grid (1 Myr to 100 Gyr), but with 
    # a much finer log sampling (0.001 instead of 0.1)
    sfh_ages_log = np.arange(6., 11., 0.001)

    # Converting from log space
    sfh_ages = 10**sfh_ages_log

    # Initialising width of time bin array
    time_widths = np.zeros_like(sfh_ages)

    # Calculating the total age width of each SSP model
    time_widths[0] = sfh_ages[0] + (sfh_ages[1] - sfh_ages[0])/2.
    time_widths[1:-1] = (sfh_ages[2:] - sfh_ages[:-2])/2.
    time_widths[-1] = sfh_ages[-1] - sfh_ages[-2]

    # Initialising star formation rate array for the SFH
    self.sfr = np.zeros_like(sfh_ages)

    # Getting the inputted SFH model chosen
    try:
      sfh_model = self.pars["sfh_model"]
    except:
      raise ValueError("The model parameters dictionary needs " +\
                         "`sfh_model` to be a definied key.")

    # Permitted SFH models
    sfh_dir = np.array(["burst", "constant", "delayed", "custom"])

    # Checking that the chosen SFH is valid
    if not sfh_model in sfh_dir:
      raise ValueError(f"The chosen SFH model of {sfh_model} is " +\
                       "not permitted. Please choose one of the " +\
                       f"following SFH models: {sfh_dir}")
    
    # Calling function to generated chosen SFH profile
    getattr(self, sfh_model)(self.sfr, sfh_ages)

    # Getting the mass normalisation
    mass_norm = np.sum(self.sfr * time_widths)

    # Checking whether the mass was inputted in the dictionary
    if "mass" in list(self.pars):
      mass = self.pars["mass"]
    else:
      # Default value for the mass is 1 Solar_mass
      mass = 1.
      # Notifying that the user has not set `mass` and the default
      # value is being used
      print("No value set for the `mass`. Using default value of 1 Solar_mass.")
    
    # Normalising the SFH is the correct total mass
    self.sfr *= mass/mass_norm

    # Creating the weights for the simple stellar population models
    # by summing up the contributions within each bin
    wei = self.sfr * time_widths

    # Calculating the bins for the SPS model grid
    sps_bins = np.zeros(len(self.bpass_ages) + 1)
    sps_bins[0] = 0.
    sps_bins[1:-1] = (self.bpass_ages[1:] + self.bpass_ages[:-1])/2.
    sps_bins[-1] = self.bpass_ages[-1]

    # Mapping the weights from the finely sample array to the SPS
    # model grid by adding using a histogram
    self.sfr_weights = np.histogram(sfh_ages, bins=sps_bins, 
                                weights=wei)[0]


  def burst(self, sfr, sfh_ages):
    """
    A burst of star formation at one specific age, defined by a delta function.

    Parameters
    ----------
      sfr :: array of scalars
        The star-formation rate at each time step.

      pars :: dict
        Dictionary containing the model parameters to generate a BPASS spectrum.
        Keys are strings and values are floats.

      sfh_ages :: array
        The ages of the star formation history profile to calculate the star
        formation rate at

    Return
    ------
      None
    """
    # Getting the chosen input age
    burst_age = self.pars["age"] * 1.e9

    # Finding the age bin which is closest to the inputted age
    # and putting all star-formation into that one bin
    sfr[np.argmin(np.abs(sfh_ages - burst_age))] += 1
  

  def constant(self, sfr, sfh_ages):
    """
    A constant star formation between the two age limits, with age
    representing the oldest stellar population created and age_min
    representing the youngest stellar population created.

    Parameters
    ----------
      sfr :: array of scalars
        The star-formation rate at each time step.

      pars :: dict
        Dictionary containing the model parameters to generate a BPASS spectrum.
        Keys are strings and values are floats.
      
      sfh_ages :: array
        The ages of the star formation history profile to calculate the star
        formation rate at

    Return
    ------
      None
    """
    # Getting the chosen input age
    age_max = self.pars["age"] * 1.e9

    # Setting the minimum age, checking if input else using default value
    if "age_min" in list(self.pars):
      age_min = self.pars["age_min"] * 1.e9
    else:
      print("No age set for `age_min`. Using default value of 0 Gyrs.")
      age_min = 0.

    # Ensure that the minimum age is less than the maximum age
    if age_min >= age_max:
      raise ValueError(f"The input value for `age_min` {age_min/1.e9} " +\
                       "Gyr needs to be less than the input for age " +\
                       f"{age_max/1.e9} Gyr.")

    # Creating a mask to only include ages betwen the max and min
    mask = (sfh_ages > age_min) & (sfh_ages < age_max)

    # Setting the SFR at ages between the max and min age as equal
    sfr[mask] +=1


  def delayed(self, sfr, sfh_ages):
    """
    A delayed-tau star formation history profile following the equation
    SFR ~ t*e^(-t/tau), where t is the time since star formation started,
    and tau is the characteristic timescale of decrease in the SFR.

    Parameters
    ----------
      sfr :: array of scalars
        The star-formation rate at each time step.

      pars :: dict
        Dictionary containing the model parameters to generate a BPASS spectrum.
        Keys are strings and values are floats.
      
      sfh_ages :: array
        The ages of the star formation history profile to calculate the star
        formation rate at

    Return
    ------
      None
    """
    # Getting the chosen input age
    age_max = self.pars["age"] * 1.e9

    # Setting the tau value, checking if input else using default value
    if "tau" in list(self.pars):
      tau = self.pars["tau"] * 1.e9
    else:
      print("No tau value set for delayed SFH. Using default value of 1 Gyrs.")
      tau = 1.e9

    # Calculating the time since star formation started
    time = age_max - sfh_ages[sfh_ages < age_max]

    # Calculating the delayed SFH profile SFR in each time bin
    sfr[sfh_ages < age_max] = time * np.exp(-time/tau)


  def custom(self, sfr, sfh_ages):
    """
    Custom star formation history profile inputted by the user.

    Parameters
    ----------
      sfr :: array of scalars
        The star-formation rate at each time step.

      pars :: dict
        Dictionary containing the model parameters to generate a BPASS spectrum.
        Keys are strings and values are floats.

      sfh_ages :: array
        The ages of the star formation history profile to calculate the star
        formation rate at

    Return
    ------
      None
    """
    # Checking that a custom SFH array has been included
    # This is an array which defines the SFR as a function of time, with
    # the axis 0 being the time in yr and axis 1 being the SFR at each
    # time in solar_mass per yr.
    try:
      history = self.pars['sf_history']
    except:
      raise ValueError("The model parameters dictionary needs " +\
                         "`sf_history` to be a definied key for the " +\
                          "custom SFH profile.")
    
    # Checking if any SFH profile ages are greater than the input age or the
    # age of the Universe
    if any(history[:,0] > self.pars["age"] * 1.e9):
      print("Note: Some of the ages in the custom SFH are greater than " +\
             "the age set in the dictionary under `age`.")
    if any(history[:,0] > self.uni_age * 1.e9):
      print("Warning: Some of the ages in the custom SFH are greater than " +\
             "the age of the Universe. These will have SFRs set to zero.")
    
    # Interpolating onto the spectrum grid
    sfr[:] = np.interp(sfh_ages, history[:,0], history[:,1], left=0, right=0)

    # Ensuring that anything greater than the age of the Universe has
    # zero star formation
    sfr[sfh_ages > self.uni_age * 1.e9] = 0.
  

  def make_ceh_profile(self):
    """
    Generates the chemical-evolution history profile for the galaxy. This calls
    one of the two functions `fixed_met` or `evolving_met` which calculates the
    profile.

    Attributes
    ----------
      sfh_ceh_grid :: array
        A 2D array containing the weightings for the metallicity models and the
        SSP BPASS models. It takes the SFR weightings for each SSP BPASS model
        and splits this between the metallicity models in BPASS.

    Returns
    -------
      None
    """
    # Making a 2D array to store the SFH weights as a function of metallicity,
    # where axis zero runs along metallicity, and axis one runs along the ages
    # of the SSP model grid
    self.sfh_ceh_grid = np.zeros((len(self.mets), len(self.bpass_ages)))

    # Checking whether to run the fixed or evolving metallicity function
    if "met_mode" in list (self.pars):
      if self.pars["met_mode"] == "fixed":
        self.fixed_met()
      elif self.pars["met_mode"] == "evolving":
        self.evolving_met()
      else:
        raise ValueError(f"The input metallicity mode of {self.pars['met_mode']} " +\
                         "is invalid. Please use either `fixed` or `evolving`.")
    else:
      print("Input `met_mode` has not been included. Using fixed metallicity assumption.")
      self.fixed_met()
  

  def fixed_met(self):
    """
    Function to distribute the SFR between the two adjacent metallicity SSP
    models when a fixed metallicity is inputted.

    Returns
    -------
      None
    """
    # Getting the chosen metallicity and checking that it is valid
    try:
      input_met = self.pars["metallicity"]
    except:
      raise ValueError("The model parameters dictionary needs " +\
                         "`metallicity` to be a definied key.")
    if not isinstance(input_met, (int, float)):
      raise ValueError("The fixed metallicity assumption requires " +\
                       "the inputted metallicity to be a single " +\
                       "float value.")
    if input_met < self.mets[0] or input_met > self.mets[-1]:
      raise ValueError("The inputted metallicity is outside the grid. " +\
                       "Please ensure the metallicity is between " +\
                       f"{self.mets[0]} and {self.mets[-1]}.")
    
    # Determining the upper grid metallicity points
    up_ind = self.mets[self.mets < input_met].shape[0]

    # If lowest metallicity is chosen:
    if up_ind == 0:
      # Set all weights in the lowest metallicity bin
      self.sfh_ceh_grid[0] = self.sfr_weights
    else:
      # Split the weighting between the two nearest bins
      
      # Calculating the metallicity bin width
      met_width = self.mets[up_ind] - self.mets[up_ind - 1]

      # Calculating the weighting for the upper metallicity
      up_wei = (input_met - self.mets[up_ind - 1])/met_width

      # Combining metallicity weighting with the SFH weight to determine
      # the overall weighting for each time bin in each SSP model
      self.sfh_ceh_grid[up_ind] = self.sfr_weights * up_wei
      self.sfh_ceh_grid[up_ind - 1] = self.sfr_weights * (1 - up_wei)


  def evolving_met(self):
    """
    Function to distribute the SFR between the metallicity models based
    on a chemical evolution history profile, allowing the metallicity to
    evolve with age. This requires a 2D input array of [[t,Z]], where
    axis zero is the age in yr and axis one is the metallicity at each age.
    """
    # Getting the input metallicity array and checking it is valid
    try:
      met_array = self.pars["metallicity"]
    except:
      raise ValueError("The model parameters dictionary needs " +\
                         "`metallicity` to be a definied key.")
    
    if isinstance(met_array, (int, float)):
      raise ValueError("The evolving metallicity assumption requires " +\
                       "the inputted metallicity to be an array of " +\
                       "[[t (in yr), Z]].")
    if any(met_array[:,1] < self.mets[0]) or any(met_array[:,1] > self.mets[-1]):
      raise ValueError("An input metallicity value is outside the grid. " +\
                       "Please ensure all metallicity values are between " +\
                       f"{self.mets[0]} and {self.mets[-1]}.")
    
    # Interpolating onto the BPASS model ages
    ceh_met = np.interp(self.bpass_ages, met_array[:,0], met_array[:,1])

    # Determining the upper grid metallicity points
    up_inds = [self.mets[self.mets < x].shape[0] for x in ceh_met]

    # Looping over the time bins
    for i, ind in enumerate(up_inds):
      # If lowest metallicity is chosen:
      if ind == 0:
        # Set all weight in the lowest metallicity bin
        self.sfh_ceh_grid[0,i] = self.sfr_weights[i]
      else:
        # Split the weighting between the two nearest bins
      
        # Calculating the metallicity bin width
        met_width = self.mets[ind] - self.mets[ind - 1]

        # Calculating the weighting for the upper metallicity
        up_wei = (ceh_met[i] - self.mets[ind - 1])/met_width

        # Combining metallicity weighting with the SFH weight to determine
        # the overall weighting for each time bin in each SSP model
        self.sfh_ceh_grid[ind,i] = self.sfr_weights[i] * up_wei
        self.sfh_ceh_grid[ind - 1,i] = self.sfr_weights[i] * (1 - up_wei)
  

class spec_attenuation:
  """
  Class to hold all the attenuation functions required to modify the spectra.
  To be used with the BPASSSource class. This includes dust attenuation, nebular
  attenuation, and intergalactic medium attenuation and their associated
  functions (i.e. different dust attenuation models).

  Returns
  -------
    None
  """
  def make_dust_attenuation(self):
    """
    Function use to calculate the dust attenuation. The function calls the desired
    attenuation/extinction law based on the choosen model provided in the model
    parameter input dictionary.

    The dust attenuation/extinction law, k(lambda) where lambda is the wavelength,
    is defined in separate functions for different laws (i.e. types of dust or
    environments). These are calculated and then coverted to A_lambda/Av, where
    A_lambda is the extinction as a function of wavelength and Av is the total 
    extinction in the V band in magnitudes, by dividing by Rv, the total-to-selective
    extinction ratio (describes shape of extinction curve). This allows for the
    dust attenuation to be calculated as 10**(-0.4 * A_lambda * Av).

    Attributes
    ----------
      Alam :: array
        Extinction value as a function of wavelength.
    
    Returns
    -------
      None
    TODO Add other dust attenuation laws
    """
    # Getting the inputted dust attenuation model chosen
    try:
      dust_model = self.pars["dust_model"]
    except:
      raise ValueError("The model parameters dictionary needs " +\
                         "`dust_model` to be a definied key.")

    # Permitted dust attenuation models
    dust_dir = np.array(["calzetti", "salim_sf", "salim_qui", "salim_custom"])

    # Checking that the chosen SFH is valid
    if not dust_model in dust_dir:
      raise ValueError(f"The chosen dust attenuation model of {dust_model} " +\
                       "is not permitted. Please choose one of the " +\
                       f"following dust attenuation models: {dust_dir}")
    
    # Calling function to generated chosen dust attenuation law
    getattr(self, dust_model)()


  def calzetti(self):
    """
    Dust extinction law of Calzetti et al. (2000). The Rv value is set
    as 4.05 as defined in the publication.

    Returns
    -------
      None
    """
    # Converting the wavelength to microns
    mu_wave = self.wavelengths * 1.e-4

    # Creating masks for each regime of the law
    mask1 = (mu_wave < 0.12)
    mask2 = (mu_wave >= 0.12) & (mu_wave < 0.63)
    mask3 = (mu_wave >= 0.63) & (mu_wave <= 2.20)

    # Creating masked wavelength regimes
    wave1 = mu_wave[mask1]
    wave2 = mu_wave[mask2]
    wave3 = mu_wave[mask3]

    # Creating storage array for klam
    klam = np.zeros_like(mu_wave)

    # Calculating the extinction law in the lowest regime
    klam[mask1] = ((wave1/0.12)**-0.77 * (4.05 
                        + 2.659 * (-2.156 + 1.509/0.12 -
                                   0.198/0.12**2 + 0.011/0.12**3)))

    # Calculating the extinction law in the middle regime
    klam[mask2] = (4.05 + 2.659 * (-2.156 + 1.509/wave2
                                - 0.198/wave2**2 + 0.011/wave2**3))
    
    # Calculating the extinction law in the highest regime
    klam[mask3] = 4.05 + 2.659 * (-1.857 + 1.040/wave3)

    # Converting from klam to Alam by dividing by Rv
    self.Alam = klam/4.05


  def salim_sf(self):
    """
    Dust attenuation law model parameters for Salim et al. (2018) for
    the average curve of all star-forming galaxies, as given in Table 1.

    Returns
    -------
      None
    """
    # Extinction law parameter values
    B, a0, a1, a2, a3, Rv = 1.57, -4.30, 2.71, -0.191, 0.0121, 3.15

    # Passing parameter values to extinction law function
    self._salim_law(B, a0, a1, a2, a3, Rv)


  def salim_qui(self):
    """
    Dust attenuation law model parameters for Salim et al. (2018) for
    the average curve of all quiescent galaxies, as given in Table 1.

    Returns
    -------
      None
    """
    # Extinction law parameter values
    B, a0, a1, a2, a3, Rv = 2.21, -3.72, 2.20, -0.062, 0.0080, 2.61

    # Passing parameter values to extinction law function
    self._salim_law(B, a0, a1, a2, a3, Rv)


  def salim_custom(self):
    """
    Dust attenuation law model parameters for Salim et al. (2018).
    This allows for a custom set of model parameters to be provided
    for the law via the model parameter input dictionary.

    Returns
    -------
      None
    """
    # Unpacking extinction law parameter values from input dictionary
    # B
    try:
      B = self.pars['salim_B']
    except:
      raise ValueError("To use the custom Salim et al. (2018) law " +\
                        "the model parameters dictionary needs " +\
                         "`salim_B` to be a definied key.")
    
    # a0
    try:
      a0 = self.pars['salim_a0']
    except:
      raise ValueError("To use the custom Salim et al. (2018) law " +\
                        "the model parameters dictionary needs " +\
                         "`salim_a0` to be a definied key.")
    
    # a1
    try:
      a1 = self.pars['salim_a1']
    except:
      raise ValueError("To use the custom Salim et al. (2018) law " +\
                        "the model parameters dictionary needs " +\
                         "`salim_a1` to be a definied key.")
    
    # a2
    try:
      a2 = self.pars['salim_a2']
    except:
      raise ValueError("To use the custom Salim et al. (2018) law " +\
                        "the model parameters dictionary needs " +\
                         "`salim_a2` to be a definied key.")
    
    # a3
    try:
      a3 = self.pars['salim_a3']
    except:
      raise ValueError("To use the custom Salim et al. (2018) law " +\
                        "the model parameters dictionary needs " +\
                         "`salim_a3` to be a definied key.")
    
    # Rv
    try:
      Rv = self.pars['salim_Rv']
    except:
      raise ValueError("To use the custom Salim et al. (2018) law " +\
                        "the model parameters dictionary needs " +\
                         "`salim_Rv` to be a definied key.")
    
    # Passing parameter values to extinction law function
    self._salim_law(B, a0, a1, a2, a3, Rv)
  

  def _salim_law(self, B, a0, a1, a2, a3, Rv):
    """
    Dust attenuation law of Salim et al. (2018), as given by Equation 8.
    The drude profile is given in Equation 9. Functional fits for the
    model and derived parameter values are given in Table 1, which are
    passed to this function as arugments. There are defined functions
    above for the overall star-forming and quiescent functional fits,
    and a function to input custome values for the attenuation law.

    Parameters
    ----------
      B :: float
        The amplitude of the ultraviolet bump.

      a0 :: float
        The zeroth order coefficient of the polynomial fit to the total
        attenuation curve.
      
      a1 :: float
        The first order coefficient of the polynomial fit to the total
        attenuation curve.
      
      a2 :: float
        The second order coefficient of the polynomial fit to the total
        attenuation curve.
      
      a3 :: float
        The third order coefficient of the polynomial fit to the total
        attenuation curve.
      
      Rv :: float
        The total-to-selective extinction ratio which describes the shape 
        of the attenuation curve.

    Returns
    -------
      None
    """
    # Converting the wavelength to microns
    mu_wave = self.wavelengths * 1.e-4

    # Calculating the Drude profile
    drude = (B * (mu_wave**2) * (0.035**2))
    drude /= (((mu_wave**2) - (0.2175**2))**2 + (mu_wave**2) * (0.035**2))

    # Creating a storage array for klam
    klam = np.zeros_like(mu_wave)
    
    # Calculating the extinction profile
    klam = a0 + a1/mu_wave + a2/(mu_wave**2) + a3/(mu_wave**3) + drude + Rv

    # Fit produces negative values above a certain regime (lambda_max), so
    # set all negative values to zero
    klam[klam < 0.] = 0.

    # Converting from klam to Alam by dividing by Rv
    self.Alam = klam/Rv


  def calc_igm_trans(self, z, wavs):
    """
    Calculates the intergalactic medium transmission function following the
    analytic funtions provided in Inoue et al. (2014). The mean optical
    depth of the IGM, tau_IGM, is calculated as

    tau_IGM = tau_LAF_LS + tau_DLA_LS + tau_LAF_LC + tau_DLA_LC,

    where tau_i_j are the optical depths for the Lyman-alpha forest (LAF)
    and damped Lyman-alpha system (DLA) components for the Lyman series
    (LS) and Lyman continuum (LC).

    Parameters
    ----------
      z :: float
        Redshift of the source to calculate the transmission function for

      wavs :: array
        The rest-frame wavelengths of the model at which to calculate the
        transmission function for
    
    Returns
    -------
      tau :: array
        Intergalacitc medium optical depth as a function of wavelength
    """
    # File containing the wavelengths and coefficients for Lyman-series
    # absorption as given in Table 2 of Inoue et al. (2018).
    filepath = join(DATAPATH, "bpass_files", "lyman_series_coefs_inoue_2014_table2.txt")

    # Opening file for Lyman-series absorption coefficients
    LS_tab = np.loadtxt(filepath)

    # Lyman-limit wavelength of 911.8 Angstroms
    lamL = 911.8
      
    # Observed wavelengths once redshifted by input redshift value
    ob_wavs = wavs * (1. + z)

    # Creating an array of the observed wavelength divided by the 
    # Lyman-limit wavelength
    ld_wavs = ob_wavs/lamL

    # Storage values for the different optical depth components
    LAF_LS = np.zeros_like(wavs)
    DLA_LS = np.zeros_like(wavs)
    LAF_LC = np.zeros_like(wavs)
    DLA_LC = np.zeros_like(wavs)

    # Looping over the Inoue et al. (2018) table wavlengths
    for j in range(39):

    ### Calculating the LS optical depth for the LAF component ###

      # For the regime where the target object is at z<1.2
      if z < 1.2:
        # Creating wavelength masks
        mask1 = (ob_wavs > LS_tab[j,1]) & (ob_wavs < LS_tab[j,1] * (1. + z))

        # Calculating optical depth
        LAF_LS[mask1] += LS_tab[j,2] * (ob_wavs[mask1]/LS_tab[j,1])**1.2
      
      # For the regime where the target object is at 1.2<=z<4.7
      elif z < 4.7:
        # Creating wavelength masks
        mask1 = (ob_wavs > LS_tab[j,1]) & (ob_wavs < 2.2 * LS_tab[j,1])
        mask2 = (ob_wavs >= 2.2 * LS_tab[j,1]) & (ob_wavs < LS_tab[j,1] * (1. + z))

        # Calculating optical depth
        LAF_LS[mask1] += LS_tab[j,2] * (ob_wavs[mask1]/LS_tab[j,1])**1.2
        LAF_LS[mask2] += LS_tab[j,3] * (ob_wavs[mask2]/LS_tab[j,1])**3.7

      # For the regime where the target object is at z>=4.7
      else:
        # Creating wavelength masks
        mask1 = (ob_wavs > LS_tab[j,1]) & (ob_wavs < 2.2 * LS_tab[j,1])
        mask2 = (ob_wavs >= 2.2 * LS_tab[j,1]) & (ob_wavs < 5.7 * LS_tab[j,1])
        mask3 = (ob_wavs >= 5.7 * LS_tab[j,1]) & (ob_wavs < LS_tab[j,1] * (1. + z))

        # Calculating optical depth
        LAF_LS[mask1] += LS_tab[j,2] * (ob_wavs[mask1]/LS_tab[j,1])**1.2
        LAF_LS[mask2] += LS_tab[j,3] * (ob_wavs[mask2]/LS_tab[j,1])**3.7
        LAF_LS[mask3] += LS_tab[j,4] * (ob_wavs[mask3]/LS_tab[j,1])**5.5

    ### Calcualting the LS optical depth for the DLA component ###

      # For the regime where the target object is at z<2.0
      if z < 2.0:
        # Creating wavelength masks
        mask1 = (ob_wavs > LS_tab[j,1]) & (ob_wavs < LS_tab[j,1] * (1. + z))

        # Calculating optical depth
        DLA_LS[mask1] += LS_tab[j,5] * (ob_wavs[mask1]/LS_tab[j,1])**2.0

      # For the regime where the taget object is at z>=2.0
      else:
        # Creating wavelength masks
        mask1 = (ob_wavs > LS_tab[j,1]) & (ob_wavs < 3.0 * LS_tab[j,1])
        mask2 = (ob_wavs >= 3.0 * LS_tab[j,1]) & (ob_wavs < LS_tab[j,1] * (1. + z))

        # Calculating optical depth
        DLA_LS[mask1] += LS_tab[j,5] * (ob_wavs[mask1]/LS_tab[j,1])**2.0
        DLA_LS[mask2] += LS_tab[j,6] * (ob_wavs[mask2]/LS_tab[j,1])**3.0

    ### Calculating the LC optical depth for the LAF component ###

    # For the regime where the target object is at z<1.2
    if z < 1.2:
      # Creating wavelength masks
      mask1 = (ob_wavs > lamL) & (ob_wavs < lamL * (1. + z))

      # Calculating optical depth
      LAF_LC[mask1] = (0.325 * (ld_wavs[mask1]**1.2 
                        - (1. + z)**-0.9 * ld_wavs[mask1]**2.1))
      
    # For the regime where the target object is at 1.2<=z<4.7
    elif z < 4.7:
      # Creating wavelength masks
      mask1 = (ob_wavs > lamL) & (ob_wavs < 2.2 * lamL)
      mask2 = (ob_wavs >= 2.2 * lamL) & (ob_wavs < lamL * (1. + z))

      # Calculating optical depth
      LAF_LC[mask1] = (2.55e-2 * (1. + z)**1.6 * ld_wavs[mask1]**2.1
                        + 0.325 * ld_wavs[mask1]**1.2
                          - 0.250 * ld_wavs[mask1]**2.1)
      
      LAF_LC[mask2] = (2.55e-2 * ((1. + z)**1.6 * ld_wavs[mask2]**2.1
                                  - ld_wavs[mask2]**3.7))
    
    # For the regime where the target object is at z>=4.7
    else:
      # Creating wavelength masks
      mask1 = (ob_wavs > lamL) & (ob_wavs < 2.2 * lamL)
      mask2 = (ob_wavs >= 2.2 * lamL) & (ob_wavs < 5.7 * lamL)
      mask3 = (ob_wavs >= 5.7 * lamL) & (ob_wavs < lamL * (1. + z))

      # Calculating optical depth
      LAF_LC[mask1] = (5.22e-4 * (1. + z)**3.4 * ld_wavs[mask1]**2.1
                        + 0.325 * ld_wavs[mask1]**1.2
                          - 3.14e-2 * ld_wavs[mask1]**2.1)
      
      LAF_LC[mask2] = (5.22e-4 * (1. + z)**3.4 * ld_wavs[mask2]**2.1
                        + 0.218 * ld_wavs[mask2]**2.1
                          - 2.55e-2 * ld_wavs[mask2]**3.7)
      
      LAF_LC[mask3] = (5.22e-4 * ((1. + z)**3.4 * ld_wavs[mask3]**2.1
                                  - ld_wavs[mask3]**5.5))

    ### Calculating the LC optical depth for the DLA component ###

    # For the regime where the target object is at z<2.0
    if z < 2.0:
      # Creating wavelength masks
      mask1 = (ob_wavs > lamL) & (ob_wavs < lamL * (1. + z))

      # Calculating optical depth
      DLA_LC[mask1] = (0.211 * (1. + z)**2.0
                        - 7.66e-2 * (1. + z)**2.3 * ld_wavs[mask1]**-0.3
                          - 0.135 * ld_wavs[mask1]**2.0)
      
    # For the regime where the target object is at z>=2.0
    else:
      # Creating wavelength masks
      mask1 = (ob_wavs > lamL) & (ob_wavs < 3.0 * lamL)
      mask2 = (ob_wavs >= 3.0 * lamL) & (ob_wavs < lamL * (1. + z))

      # Calculating optical depth
      DLA_LC[mask1] = (0.634 + 4.70e-2 * (1. + z)**3.0
                        - 1.78e-2 * (1. + z)**3.3 * ld_wavs[mask1]**-0.3
                          - 0.135 * ld_wavs[mask1]**2.0 
                            - 0.291 * ld_wavs[mask1]**-0.3)
      
      DLA_LC[mask2] = (4.70e-2 * (1. + z)**3.0
                        - 1.78e-2 * (1. + z)**3.3 * ld_wavs[mask2]**-0.3
                          - 2.92e-2 * ld_wavs[mask2]**3.0)

    # Combining all individual component optical depths to get the
    # total IGM optical depth
    tau = LAF_LS + DLA_LS + LAF_LC + DLA_LC

    return tau


def make_bpass_source(base_source, model_parameters):
  """
  Generates a BPASS stellar population model spectrum based upon the parameters
  provided in the model component dictionary. This function inherits its class
  from one of the source classes in the file `castor_etc.sources`.

  Parameters
  ----------
    base_source :: PointSource, ExtendedSource, GalaxySource
      The source class which the BPASS class will inherit (i.e. whether you
      want to spectrum represented as a point, extended or galaxy source)
        
    model_parameters :: dict
      Dictionary containing the model parameters to generate a BPASS spectrum.
      Keys are strings and values are floats.
  
  Returns
  -------
    None
  """
  class BPASSSource(star_formation_history, spec_attenuation, base_source):
    """
    Creates a class containing all the attributes to generate a BPASS spectrum.
    During initiation, generates a BPASS emission spectrum from a desired set
    of inputs. Inherits class properties from one of the classes in the 
    `source` program.

    Parameters
    ----------
      base_soruce :: class
        A class from the program `sources` from which this class inherits
    
    Returns
    -------
      None
    """
    def __init__(self, *args, **kwargs):
      """
      Generates the class, its inheritance from a `source` class, and the
      BPASS emission spectrum.

      Parameters
      ----------
        *args :: tuple
          Parameters required to initialise the source class

        **kwargs :: dict
          Parameters required to initialise the source class

      Attributes
      ----------
        pars :: dict
          Dictionary containing the model parameters to generate a BPASS spectrum.
          Keys are strings and values are floats.

        redshift :: float
          The redshift of the object for which an emission spectrum is being generated
        
        uni_age :: float
          Age of the Universe at the redshift of the object, in units of Gyr

        ldist :: float
          Luminosity distance to the object at the inputted redshift, in Mpc
      
      Returns
      -------
        None
      """
      # Storing the model parameter dictionary as part of the class
      self.pars = model_parameters

      # Checking that a redshift has been included as one of the parameters
      # and is a float greater than one
      try:
        self.redshift = self.pars['redshift']
      except:
        raise ValueError("The model parameters dictionary needs " +\
                         "`redshift` to be a definied key.")
      
      if not (self.redshift >= 0.):
        raise ValueError("The model parameter `redshift` needs to be " +\
                          "a float value greater than or equal to 0.")
      
      # Initialising the main source
      super().__init__(*args, **kwargs)

      # Using the Planck18 comoslogy from astropy to calculate properties
      # at the inputted redshift
      # Age of the Universe at the given redshift in Gyr
      self.uni_age = Planck18.age(self.redshift).value

      # Luminoisty distance at the given redshift in Mpc
      self.ldist = Planck18.luminosity_distance(self.redshift)

      # Generating the spectrum from the function in the spectrum.py program
      self.gen_bpass_spec()
    
  return BPASSSource


def BPASS_spec(base_source, model_parameters, *args, **kwargs):
  """
  Makes and initialises the BPASS source class and the class it inherits from, 
  generated in the function `make_bpass_source` above.

  Parameters
  ----------
    base_source :: PointSource, ExtendedSource, GalaxySource
      The source class which the BPASS class will inherit (i.e. whether you
      want to spectrum represented as a point, extended or galaxy source)
        
    model_parameters :: dict
      Dictionary containing the model parameters to generate a BPASS spectrum.
      Keys are strings and values are floats.
        
    *args :: tuple
      Parameters required to initialise the source class

    **kwargs :: dict
      Parameters required to initialise the source class
    
  Returns
  -------
    bpass_class :: class
      An initialised source class which includes a BPASS spectrum profile to
      model the emission profile
  """
  # Creating the BPASS source class and BPASS spectrum profile
  bpass_class = make_bpass_source(base_source, model_parameters)

  # Initialising the inherited class and returning the result
  return bpass_class(*args, **kwargs)


def make_bpass_stellar_file(filepath, name_comp={}):
  """
  Makes the BPASS fits file required to run the BPASS spectrum generation
  code in FORECASTOR, storing the generated fits file in the directory
  `/castor_etc/data/bpass_files/`. The BPASS files need to be downloaded
  from the BPASS website and stored in a directory, which is passed as
  an argument to this function.

  Parameters
  ----------
    filepath :: str
      Path to the directory where the BPASS files are stored

    name_comp :: dict
      Dictionary containing the file name component parts (i.e. IMF variant)

  Returns
  -------
    None
  """
  # Metallicity value names used in the BPASS file names
  met_nams = ["zem5", "zem4", "z001", "z002", "z003", "z004",
              "z006", "z008", "z010", "z014", "z020", "z030", "z040"]
  
  # The stellar evolution type, binary or single (bin or sin)
  if "evo_type" in list(name_comp):
    evo_nam = name_comp["evo_type"]
  else:
    evo_nam = "bin"

  # The IMF variant
  if "imf_var" in list(name_comp):
    imf_nam = name_comp["imf_var"]
  else:
    imf_nam = "imf135_300"
  
  # The stellar spectral library used
  # Note: This is only required for v2.3.1 models
  if "stel_lib" in list(name_comp):
    stel_lib = name_comp["stel_lib"]
  else:
    stel_lib = None

  # The alpha enhancement being used
  # Note: This is only require for v2.3 models and later
  if "alpha" in list(name_comp):
    alpha_val = name_comp["alpha"]
  else:
    alpha_val = None

  # Creating the base name for each file, which will start with `spectra`
  base_nam = "spectra-" + evo_nam + "-" + imf_nam + "."

  # The name for saving the file
  sav_pri = "bpass_" + evo_nam + "-" + imf_nam

  # Modifying base name and save file name if using BPASS v2.3.1
  if stel_lib != None:
    base_nam = base_nam + stel_lib + "."
    sav_pri = sav_pri + "_" + stel_lib + "_" + alpha_val
    # Modifying the metallicity names to include the alpha enhancement
    met_nams = [x + '.' + alpha_val for x in met_nams]

  # Modifying base name and save file name if using BPASS v2.3
  elif alpha_val != None:
    base_nam = base_nam + alpha_val + "."
    sav_pri = sav_pri + "_" + alpha_val
  
  # Joining base file name with path to directory
  base = join(filepath, base_nam)
  
  # Ages at which SSP models are generated in BPASS
  ages = 10**(6 + 0.1*(np.arange(0, 51)))

  # Wavelength array for the BPASS models
  wav = np.loadtxt(base + met_nams[0] + ".dat", usecols=0)

  # Creating a storage HDU list for the different grids
  list_of_hdus = [fits.PrimaryHDU()]

  # Looping over the metallicity files
  for i in range(len(met_nams)):
    # Loading in the BPASS SSP models
    # Note: BPASS stores these at 10^6 Solar masses, so the divide by
    # 1.e6 is to convert to units of L_sun/Angstrom/M_sun
    grid = np.loadtxt(base + met_nams[i] + ".dat", 
                       usecols=np.arange(1, 52)).T/1.e6

    gridname = "met_" + met_nams[i]
    list_of_hdus.append(fits.ImageHDU(name=gridname, data=grid))
  
  # Appending the age and wavelength parameter grids at the end
  list_of_hdus.append(fits.ImageHDU(name="Stellar_age_yr", data=ages))
  list_of_hdus.append(fits.ImageHDU(name="Wavelength_Angstroms", data=wav))

  # Creating and saving the fits file
  sav_nam = join(DATAPATH, "bpass_files", sav_pri)
  hdulist = fits.HDUList(hdus=list_of_hdus)
  hdulist.writeto(sav_nam + "_stellar_grids.fits", overwrite=True)

  # Printing the file has been saved along with its name
  print(f"File {sav_pri + '_stellar_grids.fits'} has been saved " +
         f"in directory {join(DATAPATH, 'bpass_files')}")
  

def make_bpass_nebular_file(filepath):
  """
  Makes the BPASS nebular fits file required to generate nebular emission
  when running the BPASS spectrum generation code in FORECASTOR, storing
  the generated fits file in the directory `/castor_etc/data/bpass_files/`.
  This uses the BPASS generated nebular emission files, which are only
  currently available on the BPASS website for v2.2.1. The BPASS nebular
  files need to be downloaded from the BPASS website and stored in a
  directory, which is passes as an argument to this function.

  Parameters
  ----------
    filepath :: str
      Path to the directory where the BPASS nebular files are stored
  
  Returns
  -------
    None
  """
  #TODO

