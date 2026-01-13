#%%
import numpy as np
import pandas as pd

from astropy.io import fits,ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy import constants as const
from astropy.table import Table, join, vstack, hstack, unique
import requests
import io
from tqdm import tqdm
import pyvo
from pathlib import Path

def add_suffix_to_all(table, suffix, keys_to_exclude):
    for col in table.colnames:
        if col not in keys_to_exclude:
            table.rename_column(col, f"{col}{suffix}")


gaia_version = {"rel_version": "dr3"}
lamost_version = {"rel_version": "dr11", "subversion": "v2.0"}
sdss_version = {"rel_version": "dr19"}
desi_version = {"rel_version": "dr1"}
base_dir = '../data/SNAQS'
sep_threshold = 1.0  # arcsec

Path(f'{base_dir}/cache').mkdir(parents=True, exist_ok=True)
#%%
'''
survey_list = Table.read(f'{base_dir}/SNAQS_source_ids.csv.gz', format='csv')
print(f"Total SNAQS sources: {len(survey_list)}")

from astroquery.gaia import Gaia
print(f"Querying Gaia {gaia_version['rel_version'].upper()} catalog...")
query = f"""
SELECT g.source_id, g.ra, g.dec, g.parallax, g.phot_g_mean_mag
FROM gaia{gaia_version["rel_version"]}.gaia_source AS g
JOIN tap_upload.my_list AS u
ON g.source_id = u.source_id
"""
job = Gaia.launch_job_async(
    query=query, 
    upload_resource=survey_list, 
    upload_table_name='my_list'
)
gaia_table = job.get_results()
gaia_table.write(f'{base_dir}/cache/GAIA_{gaia_version["rel_version"].upper()}.xml.gz', format='votable', overwrite=True)
print(f"Total Gaia matches: {len(gaia_table)}")
survey_list = gaia_table[['source_id','ra','dec']]
survey_list.write(f'{base_dir}/SNAQS_List.csv.gz', format='csv', overwrite=True)
'''

survey_list = Table.read(f'{base_dir}/SNAQS_List.csv.gz', format='csv')
print(f"Total SNAQS sources: {len(survey_list)}")

# %%
# Get SDSS DR19 spectra
print(f"Querying SDSS {sdss_version['rel_version'].upper()} catalog...")

try:
    sdss_table = Table.read(f'{base_dir}/cache/SDSS_{sdss_version["rel_version"].upper()}_full.xml.gz', format='votable')
    print(f"Loaded cached SDSS DR19 data with {len(sdss_table)} spectra.")
except:
    print("No cached SDSS data found. Downloading from SDSS server...")
    url = f"http://skyserver.sdss.org/{sdss_version['rel_version']}/SkyServerWS/SearchTools/SqlSearch"
    query = """SELECT s.ra,s.dec,s.specobjid, s.class, s.z as redshift,
        s.plate, s.mjd, s.fiberid, s.bestobjid 
        FROM SpecObj AS s
        WHERE 
            s.ra between 189.5 and 210.5
        and s.dec between 21.5 and 36.5"""
        
    print("SDSS data download started...")
    sdss_search = requests.get(url, params={'cmd': query,'format':'fits'},stream=True)
    total_size = int(sdss_search.headers.get('content-length', 0))
    buffer = io.BytesIO()
    block_size = 1024 

    with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading SDSS Data') as pbar:
        for chunk in sdss_search.iter_content(block_size):
            if chunk:
                buffer.write(chunk)
                pbar.update(len(chunk))

    buffer.seek(0)
    print("SDSS data download complete. Reading data...")

    sdss_table = Table.read(buffer, format='fits')
    sdss_table.write(f'{base_dir}/cache/SDSS_{sdss_version["rel_version"].upper()}_full.xml.gz', format='votable', overwrite=True)
    print(f"Total SDSS spectra retrieved: {len(sdss_table)}")

# %%
c_snaqs = SkyCoord(ra=survey_list['ra']*u.degree, dec=survey_list['dec']*u.degree, frame='icrs')
c_sdss = SkyCoord(ra=sdss_table['ra']*u.degree, dec=sdss_table['dec']*u.degree, frame='icrs')
idx, d2d, d3d = c_snaqs.match_to_catalog_sky(c_sdss)
max_sep = sep_threshold * u.arcsec
sep_constraint = d2d < max_sep

add_suffix_to_all(sdss_table, '_sdss', keys_to_exclude=[])
sdss_matched = hstack([survey_list[sep_constraint], sdss_table[idx[sep_constraint]]])
print(f"Total SDSS spectra matched to SNAQS sources: {len(sdss_matched)}")
sdss_matched.write(f'{base_dir}/cache/SDSS_SNAQS_matched.xml.gz', format='votable', overwrite=True)

# %%
# Get LAMOST DR11 spectra
print(f"Querying LAMOST {lamost_version['rel_version'].upper()} catalog...")

try:
    lamost_table = Table.read(f'{base_dir}/cache/LAMOST_{lamost_version["rel_version"].upper()}_full.xml.gz', format='votable')
    print(f"Loaded cached LAMOST DR11 data with {len(lamost_table)} spectra.")
except:
    print("No cached LAMOST data found. Downloading from LAMOST server...")
    url = f"https://www.lamost.org/openapi/{lamost_version['rel_version']}/{lamost_version['subversion']}/sql"
    query = """select c.obsid,c.obsdate, c.ra, c.dec, c.z, c.mjd, c.class, c.gaia_source_id
        from catalogue c 
        where c.ra between 189.5 and 210.5
        and c.dec between 21.5 and 36.5"""
        
    lamost_search = requests.get(url, params={'sql': query,'output.fmt':'votable'},stream=True)
    #print(lamost_search.url)
    print("LAMOST data download started...")

    total_size = int(lamost_search.headers.get('content-length', 0))
    buffer = io.BytesIO()
    block_size = 1024 

    with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading LAMOST Data') as pbar:
        for chunk in lamost_search.iter_content(block_size):
            if chunk:
                buffer.write(chunk)
                pbar.update(len(chunk))

    buffer.seek(0)
    print("LAMOST data download complete. Reading data...")
    lamost_table = Table.read(buffer, format='votable')
    lamost_table.write(f'{base_dir}/cache/LAMOST_{lamost_version["rel_version"].upper()}_full.xml.gz', format='votable', overwrite=True)
    print(f"Total LAMOST spectra retrieved: {len(lamost_table)}")

# %%
lamost_table.rename_column('gaia_source_id','source_id')
add_suffix_to_all(lamost_table, '_lamost', keys_to_exclude=['source_id'])

lamost_table['source_id'] = lamost_table['source_id'].filled('-9999').astype(np.int64)
lamost_table['ra_lamost'] = lamost_table['ra_lamost'].filled('-9999').astype(np.float64)
lamost_table['dec_lamost'] = lamost_table['dec_lamost'].filled('-9999').astype(np.float64)
lamost_table['z_lamost'] = lamost_table['z_lamost'].filled('-9999').astype(np.float32)
lamost_table['mjd_lamost'] = lamost_table['mjd_lamost'].filled('-9999').astype(np.int32)
lamost_table['obsid_lamost'] = lamost_table['obsid_lamost'].filled('-9999').astype(np.int64)


mask_no_source_id = lamost_table['source_id'] == -9999
lamost_with_gaia_id = lamost_table[~mask_no_source_id]
lamost_without_gaia_id = lamost_table[mask_no_source_id]
lamost_without_gaia_id.remove_column('source_id')

# %%
lamost_match = join(survey_list, lamost_with_gaia_id, keys='source_id', join_type='left',table_names=['', '_lamost'])

# %%
lamost_matched_gaia_id = lamost_match[~lamost_match['obsid_lamost'].mask]
not_matched = lamost_match[lamost_match['obsid_lamost'].mask]
not_matched = not_matched['source_id','ra','dec']

c_snaqs = SkyCoord(ra=not_matched['ra']*u.degree, dec=not_matched['dec']*u.degree, frame='icrs')
c_lamost = SkyCoord(ra=lamost_without_gaia_id['ra_lamost']*u.degree, dec=lamost_without_gaia_id['dec_lamost']*u.degree, frame='icrs')
idx, d2d, d3d = c_snaqs.match_to_catalog_sky(c_lamost)
max_sep = sep_threshold * u.arcsec
sep_constraint = d2d < max_sep

lamost_matched_pos = hstack([not_matched[sep_constraint], lamost_without_gaia_id[idx[sep_constraint]]])
lamost_SNAQS_all = vstack([lamost_matched_gaia_id, lamost_matched_pos])
lamost_SNAQS_all.write(f'{base_dir}/cache/LAMOST_SNAQS_matched.xml.gz', format='votable', overwrite=True)
print(f"Total LAMOST spectra matched to SNAQS sources: {len(lamost_SNAQS_all)}")

#%%
# Query DESI DR1 spectra
print(f"Querying DESI {desi_version['rel_version'].upper()} catalog...")
try:
    desi_table = Table.read(f'{base_dir}/cache/DESI_{desi_version["rel_version"].upper()}_full.xml.gz', format='votable')
    print(f"Loaded cached DESI DR1 data with {len(desi_table)} spectra.")
except:
    print("No cached DESI data found. Downloading from DESI server...")
    desi_service = pyvo.dal.TAPService(f"https://datalab.noirlab.edu/tap")
    desi_query = f"""SELECT d.targetid, p.ra, p.dec, d.z, d.survey, d.program, d.healpix, d.spectype
        FROM desi_{desi_version['rel_version']}.zpix AS d
        JOIN desi_{desi_version['rel_version']}.photometry AS p ON d.targetid = p.targetid
        WHERE p.ra between 189.5 and 210.5
        and p.dec between 21.5 and 36.5"""
    desi_job = desi_service.submit_job(desi_query)
    desi_job.run()
    desi_job.wait()
    desi_table = desi_job.fetch_result().to_table()
    desi_table.write(f'{base_dir}/cache/DESI_{desi_version["rel_version"].upper()}_full.xml.gz', format='votable', overwrite=True)
    print(f"Total DESI spectra retrieved: {len(desi_table)}")
    
c_snaqs = SkyCoord(ra=survey_list['ra']*u.degree, dec=survey_list['dec']*u.degree, frame='icrs')
c_desi = SkyCoord(ra=desi_table['ra'], dec=desi_table['dec'], frame='icrs')
idx, d2d, d3d = c_snaqs.match_to_catalog_sky(c_desi)
max_sep = sep_threshold * u.arcsec
sep_constraint = d2d < max_sep

add_suffix_to_all(desi_table, '_desi', keys_to_exclude=[])
desi_matched = hstack([survey_list[sep_constraint], desi_table[idx[sep_constraint]]])
print(f"Total DESI spectra matched to SNAQS sources: {len(desi_matched)}")
desi_matched.write(f'{base_dir}/cache/DESI_SNAQS_matched.xml.gz', format='votable', overwrite=True)

# %%
snaqs_sdss = join(survey_list, sdss_matched, keys=['source_id','ra','dec'], join_type='left',table_names=['', '_sdss'])
snaqs_sdss_lamost = join(snaqs_sdss, lamost_SNAQS_all, keys=['source_id','ra','dec'], join_type='left',table_names=['', '_lamost'])
snaqs_with_match = join(snaqs_sdss_lamost, desi_matched, keys=['source_id','ra','dec'], join_type='left',table_names=['', '_desi'])
snaqs_with_match.write(f'{base_dir}/SNAQS_SDSS_LAMOST_DESI_matched.xml.gz', format='votable', overwrite=True)
snaqs_all_matched = snaqs_with_match[~(snaqs_with_match['specobjid_sdss'].mask & snaqs_with_match['obsid_lamost'].mask)]
snaqs_all_matched = unique(snaqs_all_matched, keys='source_id')
snaqs_no_spectra = snaqs_with_match[snaqs_with_match['specobjid_sdss'].mask & snaqs_with_match['obsid_lamost'].mask & snaqs_with_match['targetid_desi'].mask]
snaqs_no_spectra = snaqs_no_spectra['source_id','ra','dec']
snaqs_no_spectra.write(f'{base_dir}/SNAQS_no_spectra.csv', format='csv', overwrite=True)
print(f"Total SNAQS sources with LAMOST and/or SDSS and/or DESI spectra: {len(snaqs_all_matched)}")
print(f"Total SNAQS sources without LAMOST or SDSS or DESI spectra: {len(snaqs_no_spectra)}")

spec_avail = Table([snaqs_with_match['source_id'],snaqs_with_match['ra'],snaqs_with_match['dec'],
                    ~snaqs_with_match['specobjid_sdss'].mask,~snaqs_with_match['obsid_lamost'].mask, ~snaqs_with_match['targetid_desi'].mask], 
                   names=('source_id','ra','dec','SDSS_spec','LAMOST_spec','DESI_spec'))
spec_avail = unique(spec_avail, keys='source_id')
spec_avail.write(f'{base_dir}/SNAQS_spectra_availability.csv', format='csv', overwrite=True)
