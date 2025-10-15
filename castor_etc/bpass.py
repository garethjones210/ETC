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
spectrum using the BPASS models.

This includes two function
  - gen_bpass_spec
  - make_bpass_source
  - BPASS_spec
"""

import os
import warnings
from numbers import Number

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import numpy as np

from .sources import Profiles, PointSource, ExtendedSource, GalaxySource


class star_formation_history:
  """ 
  Class to hold all the different star-formation history models. To be used with
  the BPASSSource class.
  """



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
  """
  class BPASSSource(base_source, star_formation_history):
    """
    TODO
    """
    def __init__(self, *args, **kwargs):
      """
      TODO
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

      # Initialising the cosmology of the Universe, which will be used for
      # calculating the age of the Universe and distance luminoisty
      self.cosmo = FlatLambdaCDM(H0=70, Tcmb0=2.725, Om0=0.3)

      # Age of the Universe at the given redshift
      self.uni_age = self.cosmo.age(self.redshift).value

      # Luminoisty distance at the given redshift
      self.ldist = self.cosmo.luminosity_distance(self.redshift).value

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
