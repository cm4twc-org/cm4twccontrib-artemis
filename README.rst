A unifhy-compliant version of the distributed hydrological model Artemis
------------------------------------------------------------------------

.. image:: https://img.shields.io/pypi/v/unifhycontrib-artemis?style=flat-square&color=00b0f0
   :target: https://pypi.python.org/pypi/unifhycontrib-artemis
   :alt: PyPI Version
.. image:: https://img.shields.io/badge/dynamic/json?url=https://zenodo.org/api/records/5779946&label=doi&query=doi&style=flat-square&color=00b0f0
   :target: https://zenodo.org/badge/latestdoi/365224813
   :alt: DOI

Artemis provides a simple runoff production model designed to be
comparable with the runoff-production models typically embedded
within climate models.

It is driven with precipitation, radiation,
temperature, humidity and wind speed on a daily time step and
calculates canopy interception, evaporation, snowmelt, infiltration,
and runoff. It uses a Rutter–Gash canopy formulation (`Gash, 1979`_)
to represent interception, together with Penman–Monteith evaporation
calculated using available radiation data (`Monteith, 1965`_). Soil
moisture is accounted for using a two-layer model with
saturation-excess runoff computed using a generalized TOPMODEL
(`Clark and Gedney, 2008`_). The snowpack is represented using a
temperature-based model of accumulation and melt (`Moore et al.,
1999`_, `Hock, 2003`_, `Beven, 2011`_). Snow accumulates when
precipitation falls while temperature is below a threshold
temperature. When temperature is above a threshold for melt, melting
occurs at a rate proportional to the difference between the current
temperature and the melting temperature. This conceptual model is
widely used (`Hock, 2003`_, `Zhang et al., 2006`_, `Rango and
Martinec, 1995`_, `Beven, 2011`_) and gives performance comparable
with that of more parameter rich energy balance models, despite
their greater complexity (`Parajka et al., 2010`_).

The surface layer component of Artemis comprises canopy interception,
evaporation, and snowmelt.

The sub-surface component of Artemis comprises infiltration and runoff.

.. _`Gash, 1979`: https://doi.org/10.1002/qj.49710544304
.. _`Monteith, 1965`: https://repository.rothamsted.ac.uk/item/8v5v7
.. _`Clark and Gedney, 2008`: https://doi.org/10.1029/2007JD008940
.. _`Moore et al., 1999`: https://doi.org/10.5194/hess-3-233-1999
.. _`Hock, 2003`: https://doi.org/10.1016/S0022-1694(03)00257-9
.. _`Beven, 2011`: http://doi.org/10.1002/9781119951001
.. _`Rango and Martinec, 1995`: https://doi.org/10.1111/j.1752-1688.1995.tb03392.x
.. _`Zhang et al., 2006`: https://doi.org/10.3189/172756406781811952
.. _`Parajka et al., 2010`: https://doi.org/10.1029/2010JD014086

:contributors: Simon Dadson [1,2], Thibault Hallouin [3], Rich Ellis [1]
:affiliations:
    1. UK Centre for Ecology and Hydrology
    2. School of Geography and the Environment, University of Oxford
    3. Department of Meteorology, University of Reading
:licence: BSD-3-Clause
:copyright: 2020, University of Oxford
:codebase: https://github.com/unifhy-org/unifhycontrib-artemis


How to install
~~~~~~~~~~~~~~

This package is available on PyPI, so you can simply run:

.. code-block:: bash

   pip install unifhycontrib-artemis
