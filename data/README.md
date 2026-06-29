# Data Sources

This directory contains the processed hourly data used by the public experiment.
The original large raw files are not included; the URLs below identify the public
sources used to construct the load and solar components.

## Load Data

- Source: Ausgrid Distribution Zone Substation Data
- Official URL: https://www.ausgrid.com.au/about-us/about-ausgrid/research-data-sets/distribution-zone-substation-data
- Use in this repository: processed and mapped into the `load_bus_*` columns in
  `processed_1h/aligned_dataset.csv`.

## Solar Data

- Generation time series source: AEMO NEMWeb Dispatch SCADA
- Current Dispatch SCADA directory: https://www.nemweb.com.au/REPORTS/CURRENT/Dispatch_SCADA/
- Archive Dispatch SCADA directory: https://www.nemweb.com.au/REPORTS/ARCHIVE/Dispatch_SCADA/
- Solar unit metadata source: AEMO CDEII available-generator metadata
- Metadata CSV URL: https://www.nemweb.com.au/Reports/Current/CDEII/CO2EII_AVAILABLE_GENERATORS.CSV
- Use in this repository: NSW1 solar DUIDs are identified from the CDEII metadata
  and their Dispatch SCADA generation profiles are processed and mapped into the
  `pv_bus_*` columns in `processed_1h/aligned_dataset.csv`.

## Important Notes

- The AEMO CDEII metadata are used to identify solar units; they do not provide
  plant-level latitude/longitude or meteorological variables for this experiment.
- The current experiments do not use weather, irradiance, temperature, cloud-cover,
  or plant-coordinate features.
- The PV-to-bus mapping in `processed_1h/solar_to_bus_mapping.csv` is synthetic
  for the test feeder and should not be interpreted as geographic matching.
