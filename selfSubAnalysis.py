#!/usr/bin/env python3
'''Simple script to rotate and subtract images as a test for asymmetry.

Give FITS files to do this on as command line argument, e.g.
./rot_sub_cont.py my1.fits
or e.g.:
run rot_sub_cont.py '{path_to_data}/{file_name.fits}
or e.g. call a list:
run rot_sub_cont.py /arc/projects/ARKS/data/products/*/images/*.combined.*.briggs.0.5*.fits

Originally published by: GMK 9 Feb 2023
-- Code to produce rotation and minor-axis self-subtraction
Updated by JBL: Nov 2025 
-- Provided param_csv and jsonfile lookup for source vals, x0, y0, PA, image_rms, robust values; x0,y0 selection can be made with reference to either ARKS I or ARKS III (different fitting routines lead to systematic difference in offsets); inclusion of major-axis self-subtraction; inclusion of pb-correction to contours on top of non-pb corrected maps
'''

## standard imports
import sys, os
import numpy as np
import pandas as pd
from scipy.ndimage import rotate, shift
from scipy.optimize import minimize
import json

## astronomy-specific packages
from astropy import units as u
from astropy.io import fits
from gofish import imagecube
from scipy.optimize import minimize
from astropy.convolution import Gaussian2DKernel, convolve

## plotting
import cmocean
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable

### STELLAR OFFSET ON/OFF -- set by target in a dictionary; only setting for those with best-fit results consistent with offsets

### By definition, we will only call .image files for nice plotting, but we will call up the .pb files for uncertainties. See later.
print('#####')
print('#####')
print('Make sure you are using .image.fits files, and .pb.fits is correctly pointed at...')
print('#####')
print('#####')

locsummary = '/arc/projects/ARKS/website/ARKS/dust_products/' ## standard location within ARKS project -- UPDATE IF RUNNING ELSEWHERE!
param_csv  = 'summary_disc_parameters.csv' ## name of summary parameters csv file
locjsonpar = '/arc/projects/ARKS/website/ARKS/dust_products/' ## standard location within ARKS project -- UPDATE IF RUNNING ELSEWHERE!
jsonfile   = 'pars_source.json' ## name of source parameters json file

## Here we define a series of data/plotting metrics for each source. One could output these to an external file, if inclined...
### note if crop too large, can result in empty residual images! If this happens: trial a smaller crop

# first image/plotting-specific stuff
crops = {'61005':  500, '15257':  200, '76582':  110, '84870':  150, '95086':  250,
         '109573': 450, '121617': 200, '131835': 225, '145560': 200, '161868': 100,
         '170773': 250, '9340':   180, '131488': 500, '10647':  425, '32297':  400,
         '107146': 300, '197481': 300, '206893': 400, '218396': 150, '39060':  350, 
         '92945':  400, '9672':   550, '15115':  325, '14055':  425}

# set to False for all sources with insignificant offsets. See paper for details on HD15115, HD32297 and HD109573
offFlag= {'61005':  False, '15257':  False, '76582':  False, '84870':  False, '95086':  False,
          '109573': True, '121617': False, '131835': False, '145560': False, '161868': False,
          '170773': False, '9340':   False, '131488': False, '10647':  False, '32297':  True,
          '107146': False, '197481': False, '206893': False, '218396': False, '39060':  False,  
          '92945':  False, '9672':   False, '15115':  True, '14055':  False}

titleLeft= {'61005':  False, '15257':  False, '76582':  True,  '84870':  False, '95086':  False,
            '109573': False, '121617': False, '131835': False, '145560': False, '161868': False,
            '170773': True,  '9340':   True,  '131488': False, '10647':  False, '32297':  False,
            '107146': True,  '197481': True,  '206893': False, '218396': False, '39060':  False,  
            '92945':  True,  '9672':   True,  '15115':  True,  '14055':  True}

imcropExtra = {'61005':  1.35, '15257':  1.25, '76582':  1.25, '84870':  1.25, '95086':  1.75,
               '109573': 1.5,  '121617': 1.75, '131835': 2.0,  '145560': 1.5,  '161868': 1.25,
               '170773': 1.25, '9340':   1.25, '131488': 1.5,  '10647':  1.5,  '32297':  1.75,
               '107146': 1.5,  '197481': 1.5,  '206893': 2.0,  '218396': 1.5,  '39060':  1.25,  
               '92945':  2.0,  '9672':   1.25, '15115':  1.35, '14055':  1.25}

ellX = {'61005':  1.35,  '15257':  1.35, '76582':  1.35, '84870':  1.35, '95086':  1.35,
        '109573': 1.35,  '121617': 1.35, '131835': 1.35, '145560': 1.35, '161868': 1.35,
        '170773': 1.35,  '9340':   1.35, '131488': 1.35, '10647':  1.35, '32297':  1.35,
        '107146': 1.35,  '197481': 1.35, '206893': 1.35, '218396': 1.35, '39060':  1.35,  
        '92945':  1.175, '9672':   1.35, '15115':  1.35, '14055':  1.35}

ellY = {'61005':  1.9,   '15257':  1.9, '76582':  1.9, '84870':  1.9, '95086':  1.9,
        '109573': 1.9,   '121617': 1.9, '131835': 1.9, '145560': 1.9, '161868': 1.9,
        '170773': 1.9,   '9340':   1.9, '131488': 1.9, '10647':  1.9, '32297':  1.9,
        '107146': 1.9,   '197481': 1.9, '206893': 1.9, '218396': 1.9, '39060':  1.9, 
        '92945':  1.250, '9672':   1.9, '15115':  1.9, '14055':  1.9}

imConts = {'61005':  [3, 9, 15],     '15257':  [3, 6, 9],      '76582':  [3, 6, 9], 
           '84870':  [3, 6, 9],      '95086':  [3, 6, 9],      '109573': [3, 7, 11], 
           '121617': [3, 6, 9],      '131835': [3, 6, 9],      '145560': [3, 7, 11], 
           '161868': [3, 6, 9],      '170773': [3, 6, 9],      '9340':   [3, 6, 9], 
           '131488': [3, 7, 11],     '10647':  [3, 9, 15, 21], '32297':  [3, 9, 15, 21],
           '107146': [3, 9, 15, 21], '197481': [3, 7, 11],     '206893': [3, 6, 9], 
           '218396': [3, 6, 9],      '39060':  [3, 6, 9],      '92945':  [3, 6, 9], 
           '9672':   [3, 7, 11, 15], '15115':  [3, 7, 11, 15], '14055':  [3, 6, 9]}

# next, source parameter stuff, and bonus checks for other ARKS results. 
# Note: these are only provided for the systems where a UVT value is given in the paper.
smoothed_rms = {'61005':  '',      '15257':  '',  '76582':  '',      '84870':  '',      '95086':  2.8e-5,
                '109573': '',      '121617': '',  '131835': '',      '145560': '',      '161868': '',
                '170773': '',      '9340':   '',  '131488': 1.04e-5, '10647':  '',      '32297':  '',
                '107146': '',      '197481': '',  '206893': '',      '218396': 1.07e-5, '39060':  2e-5,  
                '92945':  2.10e-5, '9672':   '',  '15115':  '',      '14055':  ''}
    
### these are from Table A1 in ARKS I paper
offXY   = {'61005':  [0.0,1.0],  '15257':  [-3.,-3.],   '76582':  [0.0,0.0],  '84870':  [0.0,0.0], '95086':  [0.0,0.0],
           '109573': [0.0,-1.0], '121617': [0.0,0.0],   '131835': [0.0,0.0],  '145560': [0.0,0.0], '161868': [27.,4.],
           '170773': [0.0,0.0],  '9340':   [0.0,0.0],   '131488': [0.0,0.0],  '10647':  [1.0,0.0], '32297':  [-1.0,2.0],
           '107146': [-1.,-1.],  '197481': [5.0,-19.0], '206893': [15.,-6.0], '218396': [15.,-2.], '39060':  [67.,46.],  
           '92945':  [-8.,5.],   '9672':   [1.0,0.0],   '15115':  [-1.,1.],   '14055':  [17.,10.]}

### these are from Table 3 in ARKS I paper (regardless of significance
offXYsig_Disk= {'61005':  [8,22,-4,8],    '15257':  [86,134,14,180], '76582':  [100,46,54,33],  '84870':  [81,57,-103,63], '95086':  [-91,45,127,34],
                '109573': [11,2,-32,2],   '121617': [8,8,14,7],      '131835': [6,6,3,6],       '145560': [15,12,-9,11],   '161868': [49,40,-131,42],
                '170773': [-77,40,26,35], '9340':   [81,65,-98,64],  '131488': [0,2,1,1],       '10647':  [90,56,28,51],   '32297':  [19,4,18,4],
                '107146': [36,35,-48,35], '197481': [32,15,6,13],    '206893': [40,132,53,106], '218396': [-41,65,-57,69], '39060':  [-54,14,-2,20],  
                '92945':  [-84,55,12,35], '9672':   [68,76,-33,30],  '15115':  [44,12,-8,5],    '14055':  [-57,59,-84,155]} #milli-arcseconds
    
### these are from Paper III -- vertical structure -- values used verbatim
offXY_ARKSIII = {'61005':  [-1,-17],     '15257':  [],        '76582':  [-40,40],   '84870':  [],       '95086':  [],
                 '109573': [13.7,-39.7], '121617': [],        '131835': [1.8,-5.6], '145560': [],       '161868': [-10,-80],
                 '170773': [],           '9340':   [],        '131488': [1.1,-7],   '10647':  [92,11],  '32297':  [20.2,15.1],
                 '107146': [],           '197481': [-21,-9],  '206893': [],         '218396': [],       '39060':  [-17,-1],
                 '92945':  [],           '9672':   [48, -37], '15115':  [36,-13.8], '14055':  [-62,-67]}

### these are from Paper III -- vertical structure -- values used if >3sigma sig. otherwise set to 0
offXY_ARKSIII_sig = {'61005':  [0,-17],      '15257':  [],        '76582':  [0,0],      '84870':  [],       '95086':  [],
                     '109573': [13.7,-39.7], '121617': [],        '131835': [0,-5.6],   '145560': [],       '161868': [0,-80],
                     '170773': [],           '9340':   [],        '131488': [0,-7],     '10647':  [92,0],   '32297':  [20.2,15.1],
                     '107146': [],           '197481': [-21,0],   '206893': [],         '218396': [],       '39060':  [0,0],  
                     '92945':  [],           '9672':   [48, -37], '15115':  [36,-13.8], '14055':  [-62,-67]}
 
PA_ARKSIII   = {'61005':  [70.279], '15257':  [],       '76582':  [103.9],  '84870':  [],      '95086':  [],
                '109573': [26.52],  '121617': [],       '131835': [58.96],  '145560': [],      '161868': [57.3],
                '170773': [],       '9340':   [],       '131488': [97.298], '10647':  [56.8],  '32297':  [47.5],
                '107146': [],       '197481': [128.73], '206893': [],       '218396': [],      '39060':  [29.74],  
                '92945':  [],       '9672':   [107.74], '15115':  [98.463], '14055':  [161.59]}


def sourcePars(locsummary, ID, param_csv):
  # return summary data to use in sample scripts
  dats   = pd.read_csv(locsummary+param_csv)
  datsID = dats[dats['name']==ID]
  inc, PA, x0, y0 = float(datsID['i']), float(datsID['PA']), float(datsID['deltaRA']), float(datsID['deltaDec'])
  return PA, x0, y0

def sourceParsJSON(jsonfile_loc, ID, robust):
  # return rms value from image depending on robust param
  jsondat = open(jsonfile_loc) #json file dictionary
  dataList = json.load(jsondat)
  try:
    robusts = dataList[ID]['clean']['image_robust']
    for i in range(len(robusts)):
      if float(robusts[i])==robust:
        rmsval = dataList[ID]['clean']['image_rms'][i]
        jsondat.close() #close the file
        return rmsval
  except:
    print('error finding rms in json file')
    return -np.inf

fs = sys.argv[1:]

for f in fs:

    print(f)
    im = fits.getdata(f).squeeze()
    hd = fits.getheader(f)
    pixelL = hd['CDELT2']*3600. # arcsec/pix
    beamMaj,beamMin,beamPA = hd['BMAJ']*3600., hd['BMIN']*3600., hd['BPA']
    print(beamMaj,beamMin, ': beam major and minor axis FWHM in arcsec')
    NpixPerBeam = 1.1*beamMaj*beamMin/(pixelL**2.0)
    
    # crop nan around edges
    ok = np.where(np.isfinite(np.nanmax(im, axis=0, )))[0]
    im = im[:, ok]
    ok = np.where(np.isfinite(np.nanmax(im, axis=1)))[0]
    im = im[ok]
    im[np.invert(np.isfinite(im))] = 0.0
    sz = im.shape

    def opt(par, fliplr=False, flipud=False, image=False, resid=False):
        im_sh = shift(im, [par[2], par[1]])
        rot = rotate(im_sh, par[0]-90, reshape=False)
        rot_fliplr = np.fliplr(rot)
        rot_flipud = np.flipud(rot)
        rot_rot = np.rot90(np.rot90(rot))
        if image:
            if fliplr:
                return rot_fliplr
            elif flipud:
                return rot_flipud
            else:
                return rot
        if resid:
            if fliplr:
                return rot - rot_fliplr
            elif flipud:
                return rot - rot_flipud
            else:
                return rot - rot_rot
        return np.nansum((rot-rot_fliplr)**2 + (rot-rot_flipud)**2 + (rot-rot_rot)**2)

    ### get robust parameter from file name; string search on 'briggs.X.Y', extract final 3
    ### call up the json dictionary, get the index at which the robust param appears, 
    ### then get the image_rms with that same rms
    ### See below for HD107146 which had rms selected via hard-coded value
    
    def getRobVal(fileID):
        robval = fileID[fileID.find('briggs')+7:fileID.find('briggs')+10]
        return float(robval)
    
    rob_val = getRobVal(f)
    print('robust: ',rob_val)
    
    for k in crops.keys():
        if k in f:
            print(k)
            if '9340' in k:
                sourceID   = 'TYC 9340-437-1'
                sourceIDns = 'TYC9340-437-1'
            else:
                sourceID   = 'HD '+str(k)
                sourceIDns = 'HD'+str(k)

            crop = crops[k]
            diskphaseoffset = offFlag[k]
            init, x0, y0 = sourcePars(locsummary, sourceID, param_csv)
            imcropExtraS = imcropExtra[k]
            ellXS, ellYS = ellX[k],ellY[k]
            imInConts    = imConts[k]
            titleSide    = titleLeft[k]

            # Get image RMS 
            # Typically stored in the JSON file -- assumption is ARKS-standard naming convention; 
            # 'arcsec' in name indicates a uv-taper was used, so json file is then not right
            # some special cases not in json files; HD10647 archival & ARKS, band 6 reimaged HD107146. So hard-coded rms values here for those
            if 'arcsec' in f:
              rms = smoothed_rms[k] 
            elif 'HD107146.b6.12m.SMGsub.corrected.briggs.2.0.1024.0.04.fits' in f:
              rms = 5.6e-6 ## hard-coded for HD107146, since the JSON file is for ARKS-standard products
            elif 'HD10647.combined.arks.corrected.briggs.2.0.1024.0.03' in f:
              rms = 13.1e-6 ## hard-coded for HD107146, since the JSON file is for ARKS-standard products
            elif 'HD10647.combined.archival.corrected.briggs.2.0.1024.0.03.fits' in f:
              rms = 13.6e-6 ## hard-coded for HD107146, since the JSON file is for ARKS-standard products
            else:
              rms = sourceParsJSON(locjsonpar+jsonfile, sourceIDns, rob_val)

            # Translate image offsets to scipy-readable values, and/or alter these to values derived in ARKS III with 'ARKSIIIOffsets=True' (Zawadzki+)
            x0, y0 = x0/pixelL, y0/pixelL #since image rot requires pixel units, and params csv is in arcsecs
            if diskphaseoffset==False: ## this is in general set to FALSE for ARKS
                x0, y0 = offXY[k][0]/(pixelL*1000.),offXY[k][1]/(pixelL*1000.) #Table A.1: stellar position
            ARKSIIIoffsets = False ## This can be turned on to use the centroid values from ARKS Paper III
            if ARKSIIIoffsets == True:
                x0, y0 = offXY_ARKSIII[k][0]/(pixelL*1000.),offXY_ARKSIII[k][1]/(pixelL*1000.) #ARKS PAPER 3
                init = PA_ARKSIII[k][0]

    
    print('PA: ',init, 'X0, Y0: ',x0, y0, '. Crop: ',crop, 'Disk phase offset set to: ', diskphaseoffset)
    print('rms: ', rms, ' Jy/bm [. Note: phase centre noise]')
    if rms == -np.inf:
        print('source missing from json: skipping!')
        break

    ## image cropping before subtraction analysis conducted
    cropIm = True
    if cropIm == True:
      im = im[sz[0]//2-crop:sz[0]//2+crop, sz[1]//2-crop:sz[1]//2+crop]
    else:
      im = im[:,:]
    sz = im.shape
    print(sz)

    ## Here is the main aspect of the script where residuals are produced
    out = minimize(opt, [init, x0, y0], method='Nelder-Mead', options={'maxiter': 1000})
    xy_rotsub    = opt(out['x'], fliplr=False, flipud=False, resid=True)
    xy_flipsublr = opt(out['x'], fliplr=True,  flipud=False, resid=True)
    xy_flipsubud = opt(out['x'], fliplr=False, flipud=True,  resid=True)
    
    def rotBack(rotIm, PA, x0, y0): #function to rotate images back to original axes
        rot = rotate(rotIm, 90-PA, reshape=False)
        rot_sh = shift(rot, [y0, x0])
        return rot_sh

    PA_rotsub    = rotBack(xy_rotsub, init, x0, y0)
    PA_flipsublr = rotBack(xy_flipsublr, init, x0, y0)
    PA_flipsubud = rotBack(xy_flipsubud, init, x0, y0)
    
    def getPBmap(f):
        # want to estimate rms from the residuals with primary beam uncorrection,
        # otherwise images skewed high by outermost noisy regions of rms map 
        ### Script assumes the .pb map has specific naming convention, and located in same directory as .image file
        try:
          pbmap = f[:-8]+'.fits'
          pbm = fits.getdata(pbmap).squeeze()
        except:
          try:
            pbmap = f[:-4]+'pb.fits'
            pbm = fits.getdata(pbmap).squeeze()
          except:
            pbm = np.ones(np.shape(im))
            print('### !!!! ###')
            print('NO .pbm file provided -- ! Errors not accounting for primary beam degradation.')
            print('### !!!! ###')
        # crop nan around edges
        ok = np.where(np.isfinite(np.nanmax(pbm, axis=0, )))[0]
        pbm = pbm[:, ok]
        ok = np.where(np.isfinite(np.nanmax(pbm, axis=1)))[0]
        pbm = pbm[ok]
        pbm[np.invert(np.isfinite(pbm))] = 0.0
        sz = pbm.shape
        if cropIm == True:
          pbm = pbm[sz[0]//2-crop:sz[0]//2+crop, sz[1]//2-crop:sz[1]//2+crop]
        else:
          pbm = pbm[:,:]
        return pbm

    
    ## Error analysis of maps.
    pbm = getPBmap(f) ## get the .pb file for each .image to correctly measure the residual significance
    rms_resid     = rms*np.sqrt(2) ## since self-subtraction combines two equally uncertain regions of a map -- valid only for uncorrelated regions of a map
    levelsImgIn   = np.array(imInConts)*rms
    levelsImg     = [3*rms] #this is just the contours of the data, so do not need scaling
    levelsRes     = [-5*rms_resid, -3*rms_resid, 3*rms_resid, 5*rms_resid] ## choose what you like here
  
    imupb, PA_rotsubupb, PA_flipsublrupb, PA_flipsubudupb = im, PA_rotsub, PA_flipsublr, PA_flipsubud ## not necessary; could clean this up

    # Since the .images are provided, to correctly estimate significance of contours, we need to scale these by the primary beam.
    # This is implemented with an SNR map. Here, the residuals are divided a pb-scaled RMS map, that has the rms_resid value at the phase centre, and larger values radially outwards
    # as per the primary beam correction. This correctly scales the significance of contours towards the edges of maps. Important for large disks with non-optimal mapping.
    rms_map = np.divide(rms_resid, pbm)
    resid_SNRmap_rotsub, resid_SNRmap_flipsub_lr, resid_SNRmap_flipsub_ud = np.divide(PA_rotsub, rms_map), np.divide(PA_flipsublr, rms_map), np.divide(PA_flipsubud, rms_map)
    levelsRes_SNR = [-5,-3, 3, 5] # for pb-corrected noise maps, to get contour lines of constant SNR

    
    ## Cropping aides. Also used to preferentially locate titles, beams
    cropmin, cropmax       = crop-(crop/1.25), crop+(crop/1.25)
    if cropIm == False:
      cropminIm, cropmaxIm   = sz[0]//2-(crop/1.25), sz[0]//2+(crop/1.25)
    else:
      cropmin, cropmax       = crop-(crop/imcropExtraS), crop+(crop/imcropExtraS)
      cropminIm, cropmaxIm   = sz[0]//2-(crop/imcropExtraS), sz[0]//2+(crop/imcropExtraS)
      cropminIm, cropmaxIm   = cropmin, cropmax
    if titleSide == True:
      titlex, titley         = cropmin+((cropmax-cropmin)*0.025), cropmin+((cropmax-cropmin)*0.915)
    else:
      titlex, titley         = cropmin+((cropmax-cropmin)*0.525), cropmin+((cropmax-cropmin)*0.915)
    type_x, type_y         = cropmin+((cropmax-cropmin)*0.025), cropmin+((cropmax-cropmin)*0.915)
    ell_x,  ell_y          = cropmin+((cropmax-cropmin)*0.075), cropmin+((cropmax-cropmin)*0.075)
    rob_x, rob_y           = titlex, cropmin*1.175
    off_xlab, off_ylab     = crop*1.05, ell_y
    

    ### PLOT!
    fig, ax   = plt.subplots(1, 4, figsize=(20,7))
    divpal=sns.diverging_palette(220, 20, as_cmap=True)

    ## Unaltered input image plotting
    cbim=ax[0].imshow(imupb*1000, origin='lower', cmap=sns.color_palette("rocket", as_cmap=True))
    ax[0].imshow(imupb, origin='lower', cmap=sns.color_palette("rocket", as_cmap=True))
    ax[0].contour(im, levels=levelsImgIn, origin='lower',colors='k')
    ax[0].text(s=f'{sourceID}',           x=titlex, y=titley, fontsize=28, color='w', path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    ax[0].plot([crop-x0],[crop+y0],marker='+',color='w') #negative x0 needed as we're plotting RA & Decl.

    ## Bespoke contours for some systems to help illustrate the presence of asymmetries
    if '39060' in f or '61005' in f or '131488' in f or '131835' in f:
      ax[0].contour(PA_flipsublr, origin='lower', levels=levelsRes, cmap='RdGy_r')    
    elif '218396' in f:
      ax[0].contour(PA_flipsubud, origin='lower', levels=levelsRes, cmap='RdGy_r')    
    else:
      ax[0].contour(PA_rotsub, origin='lower', levels=levelsRes, cmap='RdGy_r')    

    smoothIm = False
    if smoothIm == True: #### WARNING -- USE AT OWN RISK -- THIS FUNCTIONALITY HAS NOT YET BEEN TESTED -- WAS NOT USED IN ARKS
      smoothscale = XXX ## SOME SCALE
      kernel = Gaussian2DKernel(x_stddev = smoothscale, y_stddev = smoothscale)
      PA_flipsublrupb = convolve(PA_flipsublrupb, kernel)
      PA_rotsubupb = convolve(PA_rotsubupb, kernel)
      PA_flipsubudupb = convolve(PA_flipsubudupb, kernel)
      
      rms_resid = rms_resid/np.sqrt( (smoothscale+NpixPerBeam)/NpixPerBeam) ### this needs verifying...
      levelsImgIn = np.array(imInConts)*rms
      levelsImg   = [3*rms]
      levelsRes   = [-5*rms_resid, -3*rms_resid, 3*rms_resid, 5*rms_resid]

    ## Rotation-subtraction plotting
    ax[1].imshow(PA_rotsubupb, origin='lower', cmap='RdGy_r', vmin=-5*rms_resid, vmax=5*rms_resid)
    #ax[1].contour(PA_rotsub, origin='lower', levels=levelsRes, cmap='RdGy_r')    ## these contours for pb-uncorrected residuals
    ax[1].contour(resid_SNRmap_rotsub, origin='lower', levels=levelsRes_SNR, cmap='RdGy_r')  ## these are SNR contours, accounting for pb-correction
    ax[1].contour(im, levels=levelsImg, origin='lower', cmap='binary_r', alpha=0.8)
    ax[1].text(s=r'$180^\circ$-Rotation res.', x=type_x, y=type_y, fontsize=26.5, color='k', path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    
    ## Mirror-subtraction 1 (minor axis) plotting
    ax[2].imshow(PA_flipsublrupb, origin='lower', cmap='RdGy_r', vmin=-5*rms_resid, vmax=5*rms_resid)
    #ax[2].contour(PA_flipsublr, origin='lower', levels=levelsRes, cmap='RdGy_r')    
    ax[2].contour(resid_SNRmap_flipsub_lr, origin='lower', levels=levelsRes_SNR, cmap='RdGy_r')  ## these are SNR contours, accounting for pb-correction
    ax[2].contour(im, levels=levelsImg, origin='lower', cmap='binary_r', alpha=0.8)
    ax[2].text(s=r'Mirror res: Minor ax.', x=type_x, y=type_y, fontsize=26.5, color='k', path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    
    ## Mirror-subtraction 2 (major axis) plotting
    cbres = ax[3].imshow(PA_flipsubudupb*1000, origin='lower', cmap='RdGy_r', vmin=-5*rms_resid*1000, vmax=5*rms_resid*1000)
    ax[3].imshow(PA_flipsubudupb, origin='lower', cmap='RdGy_r', vmin=-5*rms_resid, vmax=5*rms_resid)
    #ax[3].contour(PA_flipsubud, origin='lower', levels=levelsRes, cmap='RdGy_r')    
    ax[3].contour(resid_SNRmap_flipsub_ud, origin='lower', levels=levelsRes_SNR, cmap='RdGy_r')  ## these are SNR contours, accounting for pb-correction
    ax[3].contour(im, levels=levelsImg, origin='lower', cmap='binary_r', alpha=0.8)
    ax[3].text(s=r'Mirror res: Major ax.', x=type_x, y=type_y, fontsize=26.5, color='k', path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    
    ## Make axes style consistent
    ax[0].tick_params(which='both',color='w',direction='in',top=True, right=True,labelsize=12,length=5,width=1.5)
    ax[1].tick_params(which='both',color='k',direction='in',top=True, right=True,labelsize=12,length=5,width=1.5)
    ax[2].tick_params(which='both',color='k',direction='in',top=True, right=True,labelsize=12,length=5,width=1.5)
    ax[3].tick_params(which='both',color='k',direction='in',top=True, right=True,labelsize=12,length=5,width=1.5)
    ax[0].get_yaxis().set_ticklabels([])
    ax[1].get_yaxis().set_ticklabels([])
    ax[2].get_yaxis().set_ticklabels([])
    ax[3].get_yaxis().set_ticklabels([])
    ax[0].get_xaxis().set_ticklabels([])
    ax[1].get_xaxis().set_ticklabels([])
    ax[2].get_xaxis().set_ticklabels([])
    ax[3].get_xaxis().set_ticklabels([])
    ax[0].set_ylabel(r'Decl. offset [arcsec]', fontsize=24)
    ax[0].set_xlabel(r'RA offset [arcsec]', fontsize=24)
    ax[1].set_xlabel(r'RA offset [arcsec]', fontsize=24)
    ax[2].set_xlabel(r'RA offset [arcsec]', fontsize=24)
    ax[3].set_xlabel(r'RA offset [arcsec]', fontsize=24)

    
    ## Additional image cropping
    ax[0].set_xlim(cropminIm, cropmaxIm)
    ax[1].set_xlim(cropminIm, cropmaxIm)
    ax[2].set_xlim(cropminIm, cropmaxIm)
    ax[3].set_xlim(cropminIm, cropmaxIm)
    ax[0].set_ylim(cropminIm, cropmaxIm)
    ax[1].set_ylim(cropminIm, cropmaxIm)
    ax[2].set_ylim(cropminIm, cropmaxIm)
    ax[3].set_ylim(cropminIm, cropmaxIm)

    
    ## add beams to each sub-panel
    ellipse0 = Ellipse((ell_x,  ell_y), width=beamMin/pixelL, height=beamMaj/pixelL, angle=beamPA, facecolor='none', edgecolor='w', hatch='//')
    ellipse1 = Ellipse((ell_x,  ell_y), width=beamMin/pixelL, height=beamMaj/pixelL, angle=beamPA, facecolor='none', edgecolor='black', hatch='//')
    ellipse2 = Ellipse((ell_x,  ell_y), width=beamMin/pixelL, height=beamMaj/pixelL, angle=beamPA, facecolor='none', edgecolor='black', hatch='//')
    ellipse3 = Ellipse((ell_x,  ell_y), width=beamMin/pixelL, height=beamMaj/pixelL, angle=beamPA, facecolor='none', edgecolor='black', hatch='//')
    ax[0].add_patch(ellipse0)
    ax[1].add_patch(ellipse1)
    ax[2].add_patch(ellipse2)
    ax[3].add_patch(ellipse3)

    
    # Make colorbars across top of the image panels. OFF as standard in ARKS VI. Note this is fiddly and requires trial-and-error per system
    turnOnColorBars=False
    if turnOnColorBars==True:
      # Setup is one colorbar for the image, and one for the residuals
      cb = fig.colorbar(cbres, ax=ax.ravel().tolist(), orientation='horizontal', location='top',shrink=0.92, extend='both',anchor=(2.52,0.65),pad=0.05, aspect=55)
      cb.set_label(r'Residual intensity [mJy$\,$beam$^{-1}$]', rotation=0, labelpad=13, fontsize=22)
      cb.ax.tick_params(labelsize=16)
      cb1 = fig.colorbar(cbim, ax=ax.ravel().tolist(), orientation='horizontal', location='top',shrink=0.31, extend='both',anchor=(-0.168,3.14),pad=0.05, aspect=20)
      cb1.set_label(r'Intensity [mJy$\,$beam$^{-1}$]', rotation=0, labelpad=13, fontsize=22)
      cb1.ax.tick_params(labelsize=16)
    
    for a in ax.flatten():
        a.plot(sz[1]/2, sz[0]/2, 'w+', alpha=0.2)
        a.plot([0,sz[1]/2], [0, 0], 'w-', alpha=0.2)
        a.plot([0, sz[1]], [sz[0]/2, sz[0]/2], 'w-', alpha=0.2)
        a.plot([sz[1]/2, sz[1]/2], [0, sz[0]], 'w-', alpha=0.2)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.01)

    
    ## Save your image -- note this goes to some arbitrary new directory one level below where the script is run.
    print(f'saving: {f}')
    outdir = f'./selfSubMaps/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fig.savefig(f'{outdir}/{os.path.basename(f)}_diskphaseoffset{diskphaseoffset}._ARKSIIIOffsets_{ARKSIIIoffsets}.rotsubtract.pdf')
