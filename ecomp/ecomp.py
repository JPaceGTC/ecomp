#!/data/studies/popeb/miniconda3/envs/uafinfra/bin/python
#!/home/wcasey/miniconda3/envs/aimlv2/bin/python # also works

# Versions -> miniconda3/envs/aimlv2:
# python: 3.9.20
# pandas: 2.2.3
# numpy: 1.26.4
# obspy: 1.4.1
# scipy: 1.13.1
# tqdm: 4.66.5

# Imports
import math
import os
import pandas as pd
import numpy as np
from obspy.geodetics import gps2dist_azimuth
from scipy.stats import norm
from tqdm import tqdm 

"""
Assumptions
1. The first bulletin is the automated bulletin
2. The second bulletin is the reference bulletin
3. Stations mostly overlap, with the stations in the second account a superset of those in the first
4. Most associations in bulletins are time-defining (used in location); events are scored only on time-defining associations

As implemented, non-defining associations do not significantly affect scoring (with one exception; time-defining associations in the automated bulletin 
are considered valid if they are associated but non-defining in the reference bulletin). 

Author: Brian Pope, GG-14, Geophysicist, Geophysical Capabilities and Assessments, AFTAC 709th SAG, 23rd ANS/ANX | brian.pope.4@us.af.mil | 321-494-3067
Editor: Harman Casey, AI/ML Engineer, Leidos | william.h.k.casey@leidos.com | 321-494-1830

2024-10-01 BMP: Initial Author
2024-10-15 BMP: Updated to include other match types (merge, mixed)
2024-10-16 WHC: Reformatted and edited 
"""

script_dir = os.path.abspath(os.path.dirname(__file__)) # directory of script

########### REQUIRED PARAMETERS ###########
# Bulletins consist of event (lat, lon, depth, time) and association/arrival (sta, phase, time) information
autobul_csvfile = script_dir+'/ugeb_catalog.csv'
refbul_csvfile = script_dir+'/ugeb_catalog.csv'
# Output file
outcsvfile = script_dir+'/ecomp_ugeb_catalog.csv'
# Output verbosity bool; True to print all stats and output, False to just run script
verbose = True
###########################################


########### OPTIONAL PARAMETERS ###########
# arrival time match tolerance, s
ArrTimeTol = 5.0
# location difference tolerance, km
LocDiffMean = 250.0
# location difference stdev, log10(km)
LocDiffSig = 0.7
# event commonality weights
distwgt = 0.5
prewgt = 0.25
recwgt = 0.25
# BQ Pd-to-Pf weight
BQPdWgt = 0.75
###########################################

def read_input_bulletins(autobul_csvfile: str, 
                         refbul_csvfile: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    autobul_csvfile : str
        file path to automated bulletin
    refbul_csvfile : str
        file path to reference bulletin

    Raises
    ------
    ValueError
        Checks all necessary columns are present in the automated and reference bulletins

    Returns
    -------
    autobul : pd.DataFrame
        automated bulletin as a pandas DataFrame with necessary columns set to specified dtypes; sorted by ATIME
    refbul : pd.DataFrame
        reference bulletin as a pandas DataFrame with necessary columns set to specified dtypes; sorted by ATIME
    """
    BULLETIN_DTYPES = {
      'ARID'    : int,
      'PHASE'   : str,
      'STA'     : str,
      'ATIME'   : float,
      'TIMEDEF' : str,
      'ORID'    : int,
      'LAT'     : float,
      'LON'     : float,
      'DEPTH'   : float,
      'OTIME'   : float
    }
    
    autobul = pd.read_csv(autobul_csvfile, dtype=BULLETIN_DTYPES)
    refbul = pd.read_csv(refbul_csvfile, dtype=BULLETIN_DTYPES)
    
    # check necessary columns are present
    for col in BULLETIN_DTYPES.keys():
        if col not in list(autobul.columns):
            raise ValueError(f"{col} not present in Automated bulletin columns, require {list(BULLETIN_DTYPES.keys())}")
        if col not in list(refbul.columns):
            raise ValueError(f"{col} not present in Reference bulletin columns, require {list(BULLETIN_DTYPES.keys())}")
    
    autobul = autobul.sort_values('ATIME')
    refbul = refbul.sort_values('ATIME')
    
    return autobul, refbul

def merge_bulletins(autobul: pd.DataFrame, 
                    refbul: pd.DataFrame, 
                    ArrTimeTol: float = 500.0,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Parameters
    ----------
    autobul : pd.DataFrame
        automated bulletin
    refbul : pd.DataFrame
        reference bulletin
    ArrTimeTol : float, optional
        Arrival time match tolerance, seconds. The default is 5.0.
    verbose: bool, optional
        binary variable to control printing station processing. Default is True 
        
    Returns
    -------
    bul_match : pd.DataFrame
        pandas DataFrame of matched arrivals between the automated and reference bulletins

    Notes
    -----
    use merge_asof to link arrivals between bulletins based on time within ArrTimeTol; second call to merge_asof is needed to produce full outer join
    
    first merge gets ALL automated bulletin arrivals and nearest (in time) corresponding reference bulletin arrivals at the same sta (if within the time tolerance)
    second merge gets the remaining reference bulletin arrivals that did not have a match in the automated bulletin
    
    merge on TIME for the first station
    note: all records within the time tolerance will be matched; this is not the 'best' match
    """
    
    # loop over stations in the reference bulletin and get the matching arrivals between the automated and the reference bulletins
    stalist = sorted(set(refbul['STA']))
    
    #if verbose: print("\tStations:")

    bul_match = []
    
    for sta in stalist:
        #if verbose: print('\t\t'+sta)
        bul_match.append(pd.merge_asof(autobul[autobul['STA']==sta], refbul[refbul['STA']==sta], on="ATIME", direction="nearest", tolerance=ArrTimeTol))
        refbul_remaining = pd.merge_asof(refbul[refbul['STA']==sta], autobul[autobul['STA']==sta], on="ATIME", direction="nearest", tolerance=ArrTimeTol,suffixes=('_y','_x'))
        bul_match.append(refbul_remaining[refbul_remaining['STA_x'].isnull()])
    
    return pd.concat(bul_match).sort_index()

def compute_similarity_statistics(autobul: pd.DataFrame, 
                                  refbul: pd.DataFrame, 
                                  bul_match: pd.DataFrame, 
                                  LocDiffMean: float = 250.0,
                                  LocDiffSig: float = 0.7, 
                                  distwgt: float = 0.5, 
                                  prewgt: float = 0.25, 
                                  recwgt: float = 0.25,
                                  verbose: bool = True) -> (pd.DataFrame, list, list):
    """
    Parameters
    ----------
    autobul : pd.DataFrame
        automated bulletin
    refbul : pd.DataFrame
        reference bulletin
    bul_match : pd.DataFrame
        bulletin of matching arrivals
    LocDiffMean: float, optional
        location difference tolerance, km. Default is 250.0
    LocDiffSig: float, optional
        location difference standard deviation, log10(km). Default is 0.7.
    distwgt : float, optional.
        distance weight used in event commonality calculation. Default is 0.5
    prewgt : float, optional
        precision weight used in event commonality calculation. Default is 0.25
    recwgt : float, optional
        recall weight used in event commonality calculation. Default is 0.25
    verbose: bool, optional
        binary variable to control printing station processing. Default is True
        
    Returns
    -------
    ecomp : pd.DataFrame
        pandas DataFrame of automated and reference bulletins match statistics by ORID
    autobul_orid_list: list
        list of automated bulletin ORIDs
    refbul_orid_list: list
        list of reference bulletin ORIDs
    """

    autobul_orid_list = sorted(set(autobul['ORID']))
    refbul_orid_list = sorted(set(refbul['ORID']))

    
    # get stats for the reference bulletin
    # define ecomp data frame, lists to hold columns
    ecomp = pd.DataFrame()
    autobul_orids = []
    refbul_orids = []
    autobul_ndefs = []
    refbul_ndefs = []
    nmatch_defs = []
    nmatch_alls = []
    nmatch_refall_autodefs = []
    ddist = []
    dtime = []
    ddepth = []
    distfactor = []
    precision = []
    recall = []
    evcom = []
    
    # loop through each automated bulletin orid
    for auto_orid in tqdm(autobul_orid_list, disable=not(verbose)):
        match_count = 0
        # Use set to limit counts to unique arids; the merge process may produce duplicate rows
        
        # get the number of time defining automated bulletin arrivals
        autobul_ndef = len(set(bul_match[(bul_match['ORID_x']==auto_orid) & (bul_match['ARID_x'].notna()) & (bul_match['TIMEDEF_x']=='d')]['ARID_x']))


        # loop through the matching reference ORIDs for the automated bulletin ORID
        for ref_orid in set(pd.to_numeric(bul_match[(bul_match['ORID_x']==auto_orid) & (bul_match['ORID_y'].notna())]['ORID_y'],downcast='integer')):
            autobul_orids.append(auto_orid)
            autobul_ndefs.append(autobul_ndef)
            # get the number of time defining reference bulletin arrivals
            refbul_ndef = len(set(bul_match[(bul_match['ORID_y']==ref_orid) & (bul_match['ARID_y'].notna()) & (bul_match['TIMEDEF_y']=='d')]['ARID_y']))
            # get the number of reference bulletin arrivals that have a matching automated bulletin arrival
            nmatch_all = len(set(bul_match[(bul_match['ORID_x']==auto_orid) & (bul_match['ORID_y']==ref_orid) & (bul_match['ARID_x'].notna()) & (bul_match['ARID_y'].notna())]['ARID_y']))
            # get the number of time-defining reference bulletin arrivals that have a have a matching time-defining automated bulletin arrival
            nmatch_def = len(set(bul_match[(bul_match['ORID_x']==auto_orid) & (bul_match['ORID_y']==ref_orid) & (bul_match['TIMEDEF_x']=='d') & (bul_match['TIMEDEF_y']=='d') & (bul_match['ARID_x'].notna()) & (bul_match['ARID_y'].notna())]['ARID_y']))
            # get the number of reference bulletin arrivals that have a matching time-defining automated bulletin arrival
            nmatch_refall_autodef = len(set(bul_match[(bul_match['ORID_x']==auto_orid) & (bul_match['ORID_y']==ref_orid) & (bul_match['TIMEDEF_x']=='d') & (bul_match['ARID_x'].notna()) & (bul_match['ARID_y'].notna())]['ARID_y']))
            
            # if there are matching arrivals between the reference bulletin and automated bulletin
            if nmatch_all > 0:
                refbul_orids.append(ref_orid)
                refbul_ndefs.append(refbul_ndef)
                nmatch_defs.append(nmatch_def)
                nmatch_alls.append(nmatch_all)
                match_count = 1
                nmatch_refall_autodefs.append(nmatch_refall_autodef)
                # get the automated bulletin event location(s)
                autobul_lat = pd.unique(bul_match[(bul_match['ORID_x']==auto_orid) & (bul_match['ORID_y']==ref_orid)]['LAT_x'])[0]
                autobul_lon = pd.unique(bul_match[(bul_match['ORID_x']==auto_orid) & (bul_match['ORID_y']==ref_orid)]['LON_x'])[0]
                # get the reference bulletin event location(s)
                refbul_lat = pd.unique(bul_match[(bul_match['ORID_x']==auto_orid) & (bul_match['ORID_y']==ref_orid)]['LAT_y'])[0]
                refbul_lon = pd.unique(bul_match[(bul_match['ORID_x']==auto_orid) & (bul_match['ORID_y']==ref_orid)]['LON_y'])[0]


                mdist, seaz, esaz = gps2dist_azimuth(autobul_lat, autobul_lon, refbul_lat, refbul_lon)
                ddist.append(mdist/1000) # distance error, km
                ddepth.append(pd.unique(bul_match[(bul_match['ORID_x']==auto_orid) & (bul_match['ORID_y']==ref_orid)]['DEPTH_x'])[0]-pd.unique(bul_match[(bul_match['ORID_x']==auto_orid) & (bul_match['ORID_y']==ref_orid)]['DEPTH_y'])[0])
                dtime.append(pd.unique(bul_match[(bul_match['ORID_x']==auto_orid) & (bul_match['ORID_y']==ref_orid)]['OTIME_x'])[0]-pd.unique(bul_match[(bul_match['ORID_x']==auto_orid) & (bul_match['ORID_y']==ref_orid)]['OTIME_y'])[0])
                # compute precision, recall, f1

                if autobul_ndef > 0 and refbul_ndef > 0 :
                    pre = nmatch_refall_autodef / autobul_ndef
                    precision.append(pre)
                    rec = nmatch_def/refbul_ndef
                    recall.append(rec)
                else:
                    precision.append(0) 
                    recall.append(0)
                    # need to make 'pre' variable equal something here. Likely either 1 or len(matching orids)
                    pre = 0
                if mdist < 0.001:
                    distf = 1.0
                else:
                    distf = 1 - norm.cdf(math.log10(mdist/1000),math.log10(LocDiffMean),LocDiffSig)
                distfactor.append(distf)
                evcom.append((distwgt*distf + prewgt*pre + recwgt*rec)/(distwgt + prewgt + recwgt))
                
        # no matching reference bulletin arrivals
        if match_count == 0:
            autobul_orids.append(auto_orid)
            autobul_ndefs.append(autobul_ndef)
            refbul_orids.append(-1)
            refbul_ndefs.append(-1)
            nmatch_defs.append(-1)
            nmatch_alls.append(-1)
            nmatch_refall_autodefs.append(-1)
            ddist.append(-1)
            dtime.append(-999)
            ddepth.append(-999)
            distfactor.append(-1)
            precision.append(-1)
            recall.append(-1)
            evcom.append(0)
    
    # populate lists for missing entries from the reference bulletin (refbul_orid_list)
    
    for ref_orid in (set(refbul_orid_list).difference(refbul_orids)):
        refbul_orids.append(ref_orid)
        refbul_ndef = len(bul_match[(bul_match['ORID_y']==ref_orid) & (bul_match['ARID_y'].notna()) & (bul_match['TIMEDEF_y']=='d')])
        refbul_ndefs.append(refbul_ndef)
        autobul_orids.append(-1)
        autobul_ndefs.append(-1)
        nmatch_defs.append(-1)
        nmatch_alls.append(-1)
        nmatch_refall_autodefs.append(-1)
        ddist.append(-1)
        dtime.append(-999)
        ddepth.append(-999)
        distfactor.append(-1)
        precision.append(-1)
        recall.append(-1)
        evcom.append(0)
    
    # populate ecomp
    ecomp['AutoBul_ORID'] = autobul_orids
    ecomp['RefBul_ORID'] = refbul_orids
    ecomp['AutoBul_NDEF'] = autobul_ndefs
    ecomp['RefBul_NDEF'] = refbul_ndefs
    ecomp['NumMatch_DEF'] = nmatch_defs 
    ecomp['NumMatch_ALL'] = nmatch_alls
    ecomp['NumMatch_RefDEF_AutoALL'] = nmatch_refall_autodefs
    ecomp['Distance_Error_km'] = ddist
    ecomp['Time_Error_s'] = dtime
    ecomp['Depth_Error_km'] = ddepth
    ecomp['Distance_Factor'] = distfactor
    ecomp['Precision'] = precision
    ecomp['Recall'] = recall 
    ecomp['Event_Commonality'] = evcom
    
    # based on evcom, assign match type...
    missed = (ecomp['AutoBul_ORID'] == -1) & (ecomp['RefBul_ORID'] > 0)
    false = (ecomp['RefBul_ORID'] == -1) & (ecomp['AutoBul_ORID'] > 0)
    
    # only consider events with nmatch_def>0 as possible 'best' matches
    autobul_def = ecomp.loc[ecomp['NumMatch_DEF']>0].sort_values('Event_Commonality').groupby('AutoBul_ORID').tail(1)
    refbul_def = ecomp.loc[ecomp['NumMatch_DEF']>0].sort_values('Event_Commonality').groupby('RefBul_ORID').tail(1)
    
    # find indices common to autobul_def and refbul_def
    best = ecomp.index.isin(np.intersect1d(autobul_def.index,refbul_def.index))
    best_autobul_orids = ecomp['AutoBul_ORID'].value_counts()
    best_refbul_orids = ecomp['RefBul_ORID'].value_counts()
    
    # split: multiple auto bulletin orids mapped to a single reference bulletin orid
    split = (ecomp['RefBul_ORID'].map(best_refbul_orids)>1) & (ecomp['AutoBul_ORID'].map(best_autobul_orids)==1) & (~best) & (ecomp['NumMatch_DEF']>0)
    
    # merge: multiple reference bulletin orids mapped to a single auto bulletin orid
    merge = (ecomp['RefBul_ORID'].map(best_refbul_orids)==1) & (ecomp['AutoBul_ORID'].map(best_autobul_orids)>1) & (~best) & (ecomp['NumMatch_DEF']>0)
    
    # mixed: multiple auto bulletin orids mapped to multiple reference bulletin orids
    mixed = (ecomp['RefBul_ORID'].map(best_refbul_orids)>1) & (ecomp['AutoBul_ORID'].map(best_autobul_orids)>1) & (~best) & (ecomp['NumMatch_DEF']>0)
    
    # nodef: remaining matching events that don't share any time defining
    nodef = (ecomp['RefBul_ORID']>0) & (ecomp['AutoBul_ORID']>0) & (ecomp['NumMatch_DEF']<1) & (~best)
    
    # populate matchtype
    ecomp['MatchType'] = ['-']*len(ecomp['AutoBul_ORID'])
    ecomp.loc[missed,'MatchType'] = 'AutoBul_Missed_Event'
    ecomp.loc[false,'MatchType'] = 'AutoBul_False_Event'
    ecomp.loc[best,'MatchType'] = 'AutoBul_Best_Match'
    ecomp.loc[nodef,'MatchType'] = 'AutoBul_NoDEF_Event'
    ecomp.loc[split,'MatchType'] = 'AutoBul_Split_Event'
    ecomp.loc[merge,'MatchType'] = 'AutoBul_Merge_Event'
    ecomp.loc[mixed,'MatchType'] = 'AutoBul_Mixed_Event'
    
    return ecomp, autobul_orid_list, refbul_orid_list

def calculate_bulletin_quality(ecomp: pd.DataFrame, 
                               BQPdWgt: float = 0.75) -> np.float64:
    """
    Parameters
    ----------
    ecomp : pd.DataFrame
        pandas DataFrame of automated and reference bulletins match statistics by ORID.
    BQPdWgt : float, optional
        Bulletin Quality Probability of Detect to Probability False weighting. The default is 0.75.

    Returns
    -------
    BQ : np.float64
        Bulletin Quality score
    """
    # original formulation of BQ only included best/missed/false matchtypes; 
    #Pd = np.mean(ecomp['evcom'][best | missed])
    #Pf = 1 - np.mean(ecomp['evcom'][best | false])
    # 
    # Updated formulation includes all matchtypes, only considers the highest evcom for each auto (orid1) and ref (orid2)
    # Probability False
    Pf =  1 - np.mean(ecomp.loc[ecomp['AutoBul_ORID']>0].sort_values('Event_Commonality').groupby('AutoBul_ORID').tail(1)['Event_Commonality'])
    # Probability Detect
    Pd = np.mean(ecomp.loc[ecomp['RefBul_ORID']>0].sort_values('Event_Commonality').groupby('RefBul_ORID').tail(1)['Event_Commonality'])
    # Bulletin Quality Score
    BQ = np.power(Pd,BQPdWgt)*np.power((1-Pf),(1-BQPdWgt))
    return BQ

def ecomp_write_and_stats(ecomp: pd.DataFrame, 
                          outcsvfile: str, 
                          autobul_orid_list: list, 
                          refbul_orid_list: list, 
                          verbose: bool = True):
    """
    Parameters
    ----------
    ecomp : pd.DataFrame
        pandas DataFrame of automated and reference bulletins match statistics by ORID
    outcsvfile : str
        file path to save ecomp data
    autobul_orid_list : list
        list of automated bulletin ORIDs.
    refbul_orid_list : list
        list of reference bulletin ORIDs
    verbose: bool, optional
        binary variable to control printing station processing. Default is True 

    Returns
    -------
    None.
    """
    # write out ecomp
    ecomp.to_csv(outcsvfile,index=False)
    
    # below here, evaluate results
    if verbose:
        # Compare total reference bulletin origins to the number of best matches and automated bulletin missed events
        #print(len(refbul_orid_list))
        #print(len(ecomp.loc[ (ecomp['MatchType'] == 'AutoBul_Best_Match') | (ecomp['MatchType'] =='AutoBul_Missed_Event') ]))
        
        # Compare total automated bulletin origins to the number of best matches and automated bulletin missed events
        #print(len(autobul_orid_list))
        #print(len(ecomp.loc[ (ecomp['MatchType'] == 'AutoBul_Best_Match') | (ecomp['MatchType'] =='AutoBul_False_Event') ]))
    
        print("\tAutomated Bulletin Origins:",len(autobul_orid_list))
        for mtype in ['AutoBul_Best_Match','AutoBul_False_Event','AutoBul_NoDEF_Event','AutoBul_Split_Event','AutoBul_Missed_Event','AutoBul_Mixed_Event','AutoBul_Merge_Event']:
            # Note: This will not have any "missed" events since all automated events are either accounted for in the reference bulletin, or are "false" events
            N = len(ecomp.loc[(ecomp.groupby(['AutoBul_ORID'])['Event_Commonality'].transform('max') == ecomp['Event_Commonality']) & (ecomp['MatchType']==mtype) & (ecomp['AutoBul_ORID']!=-1)])
            print('\t'+mtype, N)
          
        print("\tReference Bulletin Origins",len(refbul_orid_list))
        for mtype in ['AutoBul_Best_Match','AutoBul_False_Event','AutoBul_NoDEF_Event','AutoBul_Split_Event','AutoBul_Missed_Event','AutoBul_Mixed_Event','AutoBul_Merge_Event']:
            # Note: This will not habe any "false" events since all reference events are either accounted for in the automated bulletin, or are "missed" events
            N = len(ecomp.loc[(ecomp.groupby(['RefBul_ORID'])['Event_Commonality'].transform('max') == ecomp['Event_Commonality']) & (ecomp['MatchType']==mtype) & (ecomp['RefBul_ORID']!=-1)])
            print('\t'+mtype, N)
        
        #myorids =  ecomp.loc[(ecomp['MatchType']=='AutoBul_Best_Match') & (ecomp['NumMatch_DEF']==0),'RefBul_ORID']
        #ecomp[ecomp['RefBul_ORID'].isin(myorids)].sort_values(by=['RefBul_ORID','Event_Commonality'])



def ecomp(autocsv, refcsv, outputcsv, arrival_time_tol=5, location_tol=250.0, loc_diff=0.7, verbose=True):
    print("Start")
    ArrTimeTol = arrival_time_tol
    LocDiffMean = location_tol
    LocDiffSig = loc_diff
    print("Arrival time tolerance:", ArrTimeTol)
    print("Location dif tolerance:", LocDiffMean)
    if len(autocsv.split('/')) < 1:
        print("No path given")
        script_dir = os.path.abspath(os.path.dirname(__file__))  # directory of script
        ########### REQUIRED PARAMETERS ###########
        # Bulletins consist of event (lat, lon, depth, time) and association/arrival (sta, phase, time) information
        autobul_csvfile = script_dir + '/' + autocsv
        refbul_csvfile = script_dir + '/' + refcsv
        # Output file
        outcsvfile = script_dir + '/' + outputcsv
    # Output verbosity bool; True to print all stats and output, False to just run script
    else:
        print("Path provided")
        autobul_csvfile = autocsv
        refbul_csvfile = refcsv
        # Output file
        outcsvfile = outputcsv

    print("Outfile:", outcsvfile)
    if outcsvfile[-3:] != 'csv':
        outcsvfile = outcsvfile + '.csv'



    # Step 1: Read input bulletins
    if verbose: print('Step 1: Read input bulletins')
    autobul, refbul = read_input_bulletins(autobul_csvfile, refbul_csvfile)
    print(autobul['ARID'])

    # Step 2: Join bulletins on arrival time
    if verbose: print('\nStep 2: Join bulletins on arrival time')
    bul_match = merge_bulletins(autobul, refbul, ArrTimeTol=ArrTimeTol, verbose=verbose)

    # Step 3: Match events based on shared associations, compute similarity statistics
    if verbose: print('\nStep 3: Match events based on shared associations, compute similarity statistics')
    ecomp, autobul_orid_list, refbul_orid_list = compute_similarity_statistics(autobul, refbul, bul_match,
                                                                           LocDiffMean=LocDiffMean,
                                                                           LocDiffSig=LocDiffSig, distwgt=distwgt,
                                                                           prewgt=prewgt, recwgt=recwgt,
                                                                           verbose=verbose)

    # Step 4: compute Bulletin Quality
    if verbose: print('\nStep 4: compute Bulletin Quality')
    BQ = calculate_bulletin_quality(ecomp, BQPdWgt=BQPdWgt)
    if verbose: print("\tBulletin Quality:", BQ)

    # Step 5: write out ecomp table
    if verbose: print('\nStep 5: write out ecomp table')
    ecomp_write_and_stats(ecomp, outcsvfile, autobul_orid_list, refbul_orid_list, verbose=verbose)

'''
# run the script
if __name__ == '__main__':
    # Step 1: Read input bulletins
    if verbose: print('Step 1: Read input bulletins')
    autobul, refbul = read_input_bulletins(autobul_csvfile, refbul_csvfile)
    
    # Step 2: Join bulletins on arrival time
    if verbose: print('\nStep 2: Join bulletins on arrival time')
    bul_match = merge_bulletins(autobul, refbul, ArrTimeTol=ArrTimeTol, verbose=verbose)
    
    # Step 3: Match events based on shared associations, compute similarity statistics
    if verbose: print('\nStep 3: Match events based on shared associations, compute similarity statistics')
    ecomp, autobul_orid_list, refbul_orid_list = compute_similarity_statistics(autobul, refbul, bul_match, LocDiffMean=LocDiffMean, LocDiffSig=LocDiffSig, distwgt=distwgt, prewgt=prewgt, recwgt=recwgt, verbose=verbose)
    
    # Step 4: compute Bulletin Quality
    if verbose: print('\nStep 4: compute Bulletin Quality')
    BQ = calculate_bulletin_quality(ecomp, BQPdWgt=BQPdWgt)
    if verbose: print("\tBulletin Quality:",BQ)
    
    # Step 5: write out ecomp table
    if verbose: print('\nStep 5: write out ecomp table')
    ecomp_write_and_stats(ecomp, outcsvfile, autobul_orid_list, refbul_orid_list, verbose=verbose)
    '''
