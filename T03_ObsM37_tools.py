"""
VERSION 10 January 2018

Used for datareduction PS1
"""

from astropy.io import fits
import math
import numpy
import sys
import scipy.ndimage
import re


def regextract(filename, comments=False):
    """
Converts ds9 region files to become usable by the aper function. 

INPUTS:
    filename --  input ds9 regions file array.
    The ds9 file must be saved in physical coordinates. In DS9: 
        Region->Save Regions
            [Choose destination/filename.reg and press OK]
        Format=ds9
        Coordinate System=physical
            [OK]
    

OPTIONAL INPUTS:
    comments -- if comments=True then all circles must have comments. (Default = False)

OUTPUTS:
    The output is an array of strings containing the values as shown below. This is done to enable the use of string names in comments.
    Even when comments are turned off, the format is kept to keep the format consistent.

    The format is 3xn if comments=False and 4xn if comments=True

    Array -- ['x','y','radius','comment'] 

EXAMPLE:
    Convert the following region file into python format 

        reg.ds9 contains: 
        
        ================
        # Region file format: DS9 version 4.1
        global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
        physical
        circle(2763.4747,3175.7129,29.0882) # text={1}
        circle(2860.7076,3094.7166,25.0868) # text={text}
        ================
        
    Then calling: 
        regions = regextract('reg.ds9', comment=True)
        
        regions then gives:
        array([['2763.4747','3175.7129','29.0882', '1'],
             ['860.7076','3094.7166','25.0868', 'text'],], 
            dtype='|S32')

        If the array does not contain text, setting it to be a float array is done by simply saying 
        array.dtype = float     

REVISION HISTORY:
    Created by A.G.M. Pietrow   22 Apr 2015
    Changed to give an array of
    floats - O. Burggraaff      6 May 2015
    astropy instead of pyfits
    - D.S Doelman           22 March 2018
    """
    array = numpy.array([])
    array2 = numpy.array([])
    regions = numpy.genfromtxt(filename, skip_header=3, comments='@', delimiter='\n' ,dtype='str')
    #print(regions)
    for line in regions: #for line in regions.split("\n"):
        array = numpy.append(array, numpy.array([str(x) for x in re.findall(r"\d+(?:\.\d+)?(?=[^()\n]*\))", line)]))
        
        if comments == True:
            array2 = numpy.array([str(x) for x in re.findall(r"(?<=\{)[^}]+(?=\})", line)])

        array = numpy.append(array, array2)
    if comments == True:
        array = array.reshape(len(array)/4,4)
        x = array[:,0].astype(numpy.float)
        y = array[:,1].astype(numpy.float)
        r = array[:,2].astype(numpy.float)
        comments = array[:,3]
        return x,y,r,comments
    else:
        return array.reshape(int(len(array)/3),3).astype(numpy.float64)
    



def meanclip(image,mean,sigma,clipsig=3.,maxiter=5.,converge_num=0.02,verbose=False):

    """

 NAME:
       MEANCLIP

 PURPOSE:
       Computes an iteratively sigma-clipped mean on a data set
 EXPLANATION:
       Clipping is done about median, but mean is returned.
 CATEGORY:
       Statistics

 CALLING SEQUENCE:
       [mean,sigma]=MEANCLIP( image,mean,sigma, SUBS=
          CLIPSIG=, MAXITER=, CONVERGE_NUM=, VERBOSE=False, DOUBLE=False )

 INPUT POSITIONAL PARAMETERS:
       image:    Input data, any numeric array
       
 OUTPUT POSITIONAL PARAMETERS:
       Mean:     N-sigma clipped mean.
       Sigma:   Standard deviation of remaining pixels.

 INPUT KEYWORD PARAMETERS:
       CLIPSIG=3:  Number of sigma at which to clip.  Default=3
       MAXITER=5:  Ceiling on number of clipping iterations.  Default=5
       CONVERGE_NUM=0.02:  If the proportion of rejected pixels is less
       than this fraction, the iterations stop.  Default=0.02, i.e.,
       iteration stops if fewer than 2% of pixels excluded.
       VERBOSE=False:  Set this flag to get messages.
       DOUBLE=False - if set then perform all computations in double precision.
         Otherwise double precision is used only if the input
         data is double
 OUTPUT KEYWORD PARAMETER:
       SUBS:     Subscript array for pixels finally used. [not functional]


 MODIFICATION HISTORY:
       Written by:   RSH, RITSS, 21 Oct 98
       20 Jan 99 - Added SUBS, fixed misplaced paren on float call, 
           improved doc.  RSH
       Nov 2005   Added /DOUBLE keyword, check if all pixels are removed  
          by clipping W. Landsman 
       Nov 2012   converted to python G.P.P.L. Otten
       Feb 2015   removed by reference last=ct G.P.P.L. Otten
    """
    
    image=numpy.ravel(image)
    imagenumbers=numpy.arange(numpy.size(image))
    subs=imagenumbers[numpy.isfinite(image)]
    ct=numpy.sum(numpy.isfinite(image))
    iternr=0
    for iternr2 in numpy.arange(maxiter+1):
    #while iternr <= maxiter:
        skpix=image[subs]
        #print( numpy.sum(skpix)
        #print( numpy.size(skpix)
        iternr=iternr+1
        lastct=ct*1.
        medval=numpy.median(skpix)
        mean=numpy.mean(skpix,dtype=numpy.float64)
        sig=numpy.std(skpix,ddof=1,dtype=numpy.float64)
        wsm = (abs(skpix-medval) < clipsig*sig)
        ct=numpy.sum(wsm)
        if ct > 0:
            subs=subs[wsm]
        #print( iternr
        if (iternr > maxiter) | (ct == 0) | (((abs(ct-lastct))/lastct) <= converge_num):
            break
    skpix=image[subs]
    mean=numpy.mean(skpix,dtype=numpy.float64)
    sig=numpy.std(skpix,ddof=1,dtype=numpy.float64)
    return [mean,sig]

def aper(image,xc,yc,phpadu,apr,skyrad,badpix=[0,0],prnt=False, silent=False, flux=False, exact = False, nan = False, setskyval = [], readnoise = [], meanback = False, clipsig=3., maxiter=5.,converge_num=0.02, minsky = 20.):

    """Performs aperture photometry on stars
INPUTS:
     image --  input image array
     xc  -- vector of x coordinates. 
     yc  -- vector of y coordinates.
     phpadu -- Photons per Analog Digital Units, numeric scalar.  Converts
           the data numbers in IMAGE to photon units.  (APER assumes
           Poisson statistics.)
           COMMENT BY GILLES: phpadu seems to do very little and only scales the error on the flux
     apr    -- Vector of up to 12 REAL photometry aperture radii.
     skyrad -- Two element vector giving the inner and outer radii
           to be used for the sky annulus.   Ignored if the SETSKYVAL
           keyword is set.
     badpix -- Two element vector giving the minimum and maximum value
           of a good pixel.   If badpix is not supplied or if BADPIX[0] is
           equal to BADPIX[1] then it is assumed that there are no bad
           pixels.   Note that fluxes will not be computed for any star
           with a bad pixel within the aperture area, but that bad pixels
           will be simply ignored for the sky computation.  The BADPIX
           parameter is ignored if the /NAN keyword is set.

 OPTIONAL INPUTS:
     clipsig -- if meanback == True, then this is the number of sigma at which 
           to clip the background. (default=3)
     converge_num --  if meanback == True then if the proportion of 
       rejected pixels is less than this fraction, the iterations stop.  
       (default=0.02, i.e., iteration stops if fewer than 2% of pixels 
       excluded.)
     exact --  By default, APER counts subpixels, but uses a polygon 
         approximation for the intersection of a circular aperture with
         a square pixel (and normalizes the total area of the sum of the
         pixels to exactly match the circular area).   If the /EXACT 
         keyword, then the intersection of the circular aperture with a
         square pixel is computed exactly.  The /EXACT keyword is much
         slower and is only needed when small (~2 pixels) apertures are
         used with very undersampled data. (default = False)
     flux -- By default, APER uses a magnitude system where a magnitude of
           25 corresponds to 1 flux unit.   If set, then APER will keep
          results in flux units instead of magnitudes. (default = False)
     maxiter -- if meanback == True then this is the ceiling on number of 
         clipping iterations of the background.  (default=5)
     meanback -- if set, then the background is computed using the 3 sigma 
         clipped mean (using meanclip.pro) rather than using the mode 
         computed with mmm.pro. This keyword is useful for the Poisson 
         count regime or where contamination is known  to be minimal. (default False)
     minsky -- Integer giving mininum number of sky values to be used with MMM
         APER will not compute a flux if fewer valid sky elements are 
           within the sky annulus.  (default = 20)
     nan -- If set then APER will check for NAN values in the image.   /NAN
         takes precedence over the BADPIX parameter.   Note that fluxes 
         will not be computed for any star with a NAN pixel within the 
         aperture area, but that NAN pixels will be simply ignored for 
         the sky computation. (default = False)
     prnt -- if set and non-zero then APER will also write its results to
           a file aper.prt.   One can specify the output file name by
           setting PRNT = 'filename'. (default = False) [DOES NOT FUNCTION - Gilles]
     readnoise -- Scalar giving the read noise (or minimum noise for any
          pixel.   This value is passed to the procedure mmm.pro when
          computing the sky, and is only need for images where
          the noise is low, and pixel values are quantized.   
     silent -  If supplied and non-zero then no output is displayed to the
           terminal. (default = False)
     setskyval -- Use this keyword to force the sky to a specified value 
           rather than have APER compute a sky value.   SETSKYVAL 
           can either be a scalar specifying the sky value to use for 
           all sources, or a 3 element vector specifying the sky value, 
           the sigma of the sky value, and the number of elements used 
           to compute a sky value.   The 3 element form of SETSKYVAL
           is needed for accurate error budgeting.

 OUTPUTS:
     mags   -  NAPER by NSTAR array giving the magnitude for each star in
           each aperture.  (NAPER is the number of apertures, and NSTAR
           is the number of stars).   If flux == False, then
           a flux of 1 digital unit is assigned a zero point magnitude of 
           25.
     errap  -  NAPER by NSTAR array giving error for each star.  If a 
           magnitude could not be determined then  ERRAP = 9.99 (if in 
        magnitudes) or ERRAP = nan (if /FLUX is set).
     sky  - NSTAR element vector giving sky value for each star in 
           flux units
     skyerr -  NSTAR element vector giving error in sky values

 EXAMPLE:
       Determine the flux and error for photometry radii of 3 and 5 pixels
       surrounding the position x,y=234.2,344.3 on an image array, im.   Compute
       the partial pixel area exactly. Assume that the flux units are in
       Poisson counts, so that PHPADU = 1, and the sky value is already known
       to be 1.3, and that the range [-32767,80000] for bad low and bad high
       pixels + output to terminal
      

       [flux, eflux, sky,skyerr]=aper( im, xc=[234.2], yc=[344.3], phpadu=1., apr=[3,5], skyrad=[-1], badpix=[-32767,80000], exact=True, flux=True, setskyval = [1.3])
       
 
 NOTES:
       Reasons that a valid magnitude cannot be computed include the following:
      (1) Star position is too close (within 0.5 pixels) to edge of the frame
      (2) Less than 20 valid pixels available for computing sky
      (3) Modal value of sky could not be computed by the procedure MMM
      (4) *Any* pixel within the aperture radius is a "bad" pixel
      (5) The total computed flux is negative.   In this case the negative
      flux and error are returned.


       For the case where the source is fainter than the background, APER will
       return negative fluxes if /FLUX is set, but will otherwise give 
       invalid data (since negative fluxes can't be converted to magnitudes) 
 
       APER was modified in June 2000 in two ways: (1) the EXACT keyword was
       added (2) the approximation of the intersection of a circular aperture
       with square pixels was improved (i.e. when EXACT is not used) 
 REVISON HISTORY:
       Adapted to IDL from DAOPHOT June, 1989   B. Pfarr, STX
       FLUX keyword added              J. E. Hollis, February, 1996
       SETSKYVAL keyword, increase maxsky      W. Landsman, May 1997
       Work for more than 32767 stars      W. Landsman, August 1997
       Don't abort for insufficient sky pixels  W. Landsman  May 2000
       Added /EXACT keyword          W. Landsman  June 2000 
       Allow SETSKYVAL = 0            W. Landsman  December 2000 
       Set BADPIX[0] = BADPIX[1] to ignore bad pixels W. L.  January 2001    
       Fix chk_badpixel problem introduced Jan 01 C. Ishida/W.L. February 2001
       Set bad fluxes and error to NAN if /FLUX is set  W. Landsman Oct. 2001 
       Remove restrictions on maximum sky radius W. Landsman  July 2003
       Added /NAN keyword  W. Landsman November 2004
       Set badflux=0 if neither /NAN nor badpix is set  M. Perrin December 2004
       Added READNOISE keyword   W. Landsman January 2005
       Added MEANBACK keyword   W. Landsman October 2005
       Correct typo when /EXACT and multiple apertures used.  W.L. Dec 2005
       Remove VMS-specific code W.L. Sep 2006
       Add additional keywords if /MEANBACK is set W.L  Nov 2006
       Allow negative fluxes if /FLUX is set  W.L.  Mar 2008
       Previous update would crash if first star was out of range  W.L. Mar 2008
       Fix floating equality test for bad magnitudes W.L./J.van Eyken Jul 2009
       Added MINSKY keyword W.L. Dec 2011
       Converted to python G.P.P.L. Otten Nov 2012
       fixed row-column problem G.P.P.L. Otten April 2013
       fixed setskyval option returning NaN fluxes G.P.P.L. Otten February 2015
    """
    xc=numpy.array(xc)
    yc=numpy.array(yc)
    badpix=numpy.array(badpix)
    apr=numpy.array(apr)
    skyrad=numpy.array(skyrad)
    maxsky = 10000.
    pi=numpy.pi
    s=numpy.ndim(image)
    if (numpy.sum((skyrad[0]< apr))>0) & (skyrad[0] >=0.):
        sys.exit("sky should be measured in annulus outside of aperture")
    if s != 2:
        sys.exit("image must be 2 dimensions")
    [nrow,ncol]=numpy.shape(image)

    if nan == False:
        if (numpy.size(badpix) != 2):
            sys.exit('Expecting 2 badpixel values')
        chk_badpix = (badpix[0] < badpix[1])
    if ( numpy.size(apr) < 1 ):
        sys.exit("Expecting at least 1 aper value")

    if numpy.size(setskyval) > 0:
        #print( numpy.size(setskyval)
        if numpy.size(setskyval) == 1:
            setskyval = numpy.array([setskyval,0.,1.])
        if numpy.size(setskyval) != 3: 
            sys.exit('ERROR - Keyword SETSKYVAL must contain 1 or 3 elements')
        skyrad = numpy.array([ 0., numpy.max(apr) + 1.])
    else:
        if numpy.size(skyrad) != 2: 
            sys.exit("Expecting 2 sky annulus radii")
        else:
            skyrad = numpy.ndarray.astype(skyrad,"float")

    if ( numpy.size(phpadu) < 1 ):
        sys.exit('Expecting Photons per Analog per Digital Unit')

    Naper=numpy.size(apr)
    Nstars=numpy.min(numpy.array([numpy.size(xc),numpy.size(yc)]))
    mags = numpy.zeros([Nstars, Naper])
    errap = numpy.zeros([Nstars, Naper])
    sky = numpy.zeros(Nstars)  
    skyerr = numpy.zeros(Nstars)
    area = pi*apr**2.          #Area of each aperture

    if exact == True:
        bigrad = apr + 0.5
        smallrad = apr/numpy.sqrt(2) - 0.5 
    
    if numpy.size(setskyval) == 0:
        rinsq =  numpy.max(numpy.array([skyrad[0],0.]))**2. 
        routsq = skyrad[1]**2.

#  Compute the limits of the submatrix.   Do all stars in vector notation.
    lx = numpy.max(numpy.array([numpy.trunc(xc-skyrad[1]), numpy.zeros(Nstars)]),axis=0)
    #Lower limit X direction
    ux = numpy.min(numpy.array([numpy.trunc(xc+skyrad[1]), (ncol-1.)*numpy.ones(Nstars)]),axis=0)   #Upper limit X direction
    nx = ux-lx+1.            #Number of pixels X direction
    ly = numpy.max(numpy.array([numpy.trunc(yc-skyrad[1]), numpy.zeros(Nstars)]),axis=0)       #Lower limit Y direction
    uy = numpy.min(numpy.array([numpy.trunc(yc+skyrad[1]), (nrow-1.)*numpy.ones(Nstars)]),axis=0)   #Upper limit Y direction
    ny = uy-ly+1.              #Number of pixels Y direction
    dx = xc-lx           #X coordinate of star's centroid in subarray
    dy = yc-ly           #Y coordinate of star's centroid in subarray#
    edge = numpy.min(numpy.array([[dx-0.5],[nx+0.5-dx],[dy-0.5],[ny+0.5-dy]]) ,axis=0)
    badstar=numpy.zeros(Nstars,dtype=bool)
    for ii in numpy.arange(Nstars):
        badstar[ii] = ((xc[ii] < 0.5) | (xc[ii] > ncol-1.5) | (yc[ii] < 0.5) | (yc[ii] > nrow-1.5))

    Nbad = numpy.sum(badstar)
    if ( Nbad < 0 ):
        print('WARNING - ' + repr(Nbad) + ' star positions outside image')
    if flux == True:
        badval = float('nan')
        baderr = float('nan')
    if flux == False: 
        badval = 99.999
        baderr = 9.999
    for i in numpy.arange(0,Nstars):
        noskyval=0
        skipstar = 0
        apmag = badval*numpy.ones(Naper)
        magerr = baderr*numpy.ones(Naper)
        skymod = 0.
        skysig = 0.
        skyskw = 0.
        if badstar[i] == True:
            print( "warning: badstar")
            skipstar = 1
        if skipstar == 0:
            error1=badval*numpy.ones(Naper)
            error2=badval*numpy.ones(Naper)
            error3=badval*numpy.ones(Naper)
            error1 = 1.*apmag
            error2 = 1.*apmag
            error3 = 1.*apmag
            rotbuf = image[int(ly[i]):int(uy[i]+1),int(lx[i]):int(ux[i]+1)]
            #print( rotbuf
            dxsq = ( numpy.arange( nx[i] ) - dx[i] )**2
            rsq = numpy.zeros( [int(ny[i]), int(nx[i])])
            for ii in numpy.arange( ny[i]):
                rsq[int(ii),:] = dxsq + (ii-dy[int(i)])**2
                        
            if exact == True:
                nbox = numpy.arange(nx[i]*ny[i])
                XX = (numpy.reshape(numpy.mod(nbox,nx[i]),[ny[i],nx[i]]))
                YY = numpy.trunc(numpy.reshape((nbox/nx[i]),[ny[i],nx[i]]))
                x1 = numpy.abs(XX-dx[i])
                y1 = numpy.abs(YY-dy[i])
            if exact == False:
                r = numpy.sqrt(rsq) - 0.5
            if numpy.size(setskyval) == 0:
                noskyval=1
                skypix=numpy.zeros([ny[i],nx[i]],dtype=bool)
                for ii in numpy.arange(nx[i]):
                    for jj in numpy.arange(ny[i]):
                        skypix[jj,ii] = (rsq[jj,ii] >= rinsq) & (rsq[jj,ii] <= routsq)
                
                
                #print( skypix
                
                if nan == True:
                    for ii in numpy.arange(nx[i]):
                        for jj in numpy.arange(ny[i]):
                            skypix[jj,ii] = skypix[jj,ii] & (numpy.isfinite(rotbuf[jj,ii]))
                if nan == False:
                    if chk_badpix == True:
                        for ii in numpy.arange(nx[i]):
                            for jj in numpy.arange(ny[i]):
                                skypix[jj,ii] = skypix[jj,ii] & (rotbuf[jj,ii] > badpix[0]) & (rotbuf[jj,ii] < badpix[1])
                Nsky = numpy.sum(skypix)
                #print( Nsky
                Nsky = numpy.min(numpy.array([Nsky, maxsky]))
                if ( Nsky < minsky ):
                    if silent == False:
                        print( 'There aren''t enough valid pixels in the sky annulus.')
                        skipstar = 1
                if skipstar == 0:
                    skybuf = numpy.ravel(rotbuf[skypix])
                    skybuf=skybuf[0:Nsky]
                    if meanback == True:
                        [skymod,skysig]=meanclip(skybuf,skymod,skysig,clipsig,maxiter,converge_num)
                    if meanback == False:
                        [skymod,skysig,skyskw]=mmm(skybuf, skymod, skysig, skyskw, readnoise=readnoise,minsky=minsky)
                    skyvar = skysig**2
                    sigsq = skyvar/Nsky
                    if ( skysig < 0.0 ):
                        skipstar = 1
                    if skipstar == 0:
                        skysig = numpy.min(numpy.array([skysig,999.99]))
                        skyskw = numpy.max(numpy.array([skyskw,-99.]))
                        skyskw = numpy.min(numpy.array([skyskw,999.99]))
                    #setskyval = 0
            elif (numpy.size(setskyval) != 0) & (skipstar == 0):
                skymod = setskyval[0]   
                skysig = setskyval[1]
                Nsky = setskyval[2]
                skyvar = skysig**2
                sigsq = skyvar/Nsky
                skyskw = 0.
            if skipstar == 0:
                for k in numpy.arange(Naper):
                    if ( edge[0,i] >= apr[k] ):
                        if exact == True:
                            mask = numpy.zeros([ny[i],nx[i]])
                            good=numpy.zeros([ny[i],nx[i]],dtype=bool)
                            bad=numpy.zeros([ny[i],nx[i]],dtype=bool)
                            for ii in numpy.arange(nx[i]):
                                for jj in numpy.arange(ny[i]):
                    
                                    good[jj,ii] = ((x1[jj,ii] < smallrad[k]) & (y1[jj,ii] < smallrad[k]))
                                    bad[jj,ii] = ((x1[jj,ii] > bigrad[k]) | (y1[jj,ii] > bigrad[k] ))
                            Ngood=numpy.sum(good)
                            if Ngood > 0:
                                mask[good] = 1.0
                            mask[bad] = -1
                            gfract = (mask == 0.0)
                            Nfract=numpy.sum(gfract)
                            if Nfract > 0:
                                mask[gfract] = numpy.max(numpy.array([pixwt(dx[i],dy[i],apr[k],XX[gfract],YY[gfract]),numpy.zeros(Nfract)]),axis=0)
                                mask=numpy.reshape(mask,[ny[i],nx[i]])
                            thisap = (mask > 0.0)
                            thisapd = rotbuf[thisap]
                            fractn = mask[thisap]
                        if exact == False:
                            thisap = (r < apr[k])
                            thisapd = rotbuf[thisap]
                            thisapr = r[thisap]
                            fractn = numpy.max(numpy.array([[apr[k]-thisapr],[numpy.zeros(numpy.size(thisapr))]]),axis=0)
                            fractn = numpy.min(numpy.array([[fractn[0,:]],[numpy.ones(numpy.size(fractn))]]),axis=0)
                            full = (fractn == 1.0)
                            Nfull= numpy.sum(full)
                            gfract = (fractn != 1.0)
                            factor = (area[k] - Nfull ) / numpy.sum(fractn[gfract])
                            fractn[gfract] = fractn[gfract]*factor
                            
                        #end exact == false
                        if nan == True:
                            badflux =  (numpy.min(thisapd[numpy.isfinite(thisapd)]) == 0)
                        if nan == False:
                            if chk_badpix == True:
                                minthisapd = numpy.min(thisapd)
                                maxthisapd = numpy.max(thisapd)
                                badflux = (minthisapd <= badpix[0] ) | ( maxthisapd >= badpix[1])
                            if chk_badpix == False:
                                badflux=0
                        if badflux == 0:
                            apmag[k] = numpy.sum(thisapd*fractn) #Total over irregular aperture
                    #end edge-if
                #end k-loop over apertures
            if flux == True:
                g = numpy.isfinite(apmag)
                Ng=numpy.sum(g)
            if flux == False:
                g = (numpy.abs(apmag - badval) > 0.01)
                Ng=numpy.sum(g)
            if Ng > 0:
                apmag[g] = apmag[g] - skymod*area[g]
                error1[g] = area[g]*skyvar   #Scatter in sky values
                error2[g] = numpy.max(numpy.array([[apmag[g]],[numpy.zeros(Ng)]]))/phpadu  #Random photon noise 
                error3[g] = sigsq*area[g]**2  #Uncertainty in mean sky brightness
                magerr[g] = numpy.sqrt(error1[g] + error2[g] + error3[g])
                if flux == False:
                    good = (apmag > 0.0) #Are there any valid integrated fluxes?
                    Ngood=numpy.sum(good)
                    if ( Ngood > 0 ):
                        magerr[good] = 2.5/numpy.log(10)*magerr[good]/apmag[good]
                        apmag[good] =  25.-2.5*numpy.log10([good])
                            
    #for ii in numpy.arange(Naper):
   #         ms[ii] = string( apmag[ii],'+-',magerr[ii], FORM = fmt)
   #if print( then  print(f,lun, $    ;Write results to file?
   #   form = fmt3,  i, xc[i], yc[i], skymod, skysig, skyskw, ms
   #if ~SILENT then print(,form = fmt2, $      ;Write results to terminal?
   #       i,xc[i],yc[i],skymod,ms
        sky[i] = skymod
        skyerr[i] = skysig
        mags[i,:] = apmag
        errap[i,:]= magerr

    if silent == False:
        print( "x, y in pixels, flux and sky in ADU")
        for i in numpy.arange(Naper):
            print( "=======================================")
            print( "aperture radius is "+str(apr[i])+" pixels")
            print( "=======================================")
            print( "x\t\t","y\t\t","flux","+-","fluxerr","\t\tsky","+-","skyerr")
            for j in numpy.arange(Nstars):
                print( xc[j],"\t",yc[j],"\t",mags[j,i],"+-",errap[j,i],"\t",sky[j],"+-",skyerr[j])
    return [mags,errap,sky,skyerr]


def mmm(sky_vector,skymod,sigma,skew,highbad=[],debug=False,readnoise=[],maxiter=50.,minsky=20.,integer=False,silent=False):
    #print( sky_vector
    sky_vector=numpy.ravel(sky_vector)
    sky = numpy.sort(sky_vector)
    Nsky=numpy.size(sky_vector)
    Nlast=int(Nsky-1.)
    if Nsky < minsky:
        sigma=-1.0
        skew = 0.0
        print( 'ERROR -Input vector must contain at least the minimal amount of elements')
        return [skymod,sigma,skew]
    skymid=numpy.median(sky)
    cut1=numpy.min(numpy.array([skymid-numpy.min(sky),numpy.max(sky)-skymid]))
    if numpy.size(highbad) == 1:
        cut1=numpy.min(numpy.array( [cut1,highbad - skymid]))

    cut2=skymid+cut1
    cut1=skymid-cut1

    good = ( (sky <= cut2) & (sky >= cut1))
    Ngood=numpy.sum(good)
    good=(numpy.arange(numpy.size(sky)))[good]
    delta = sky[good] - skymid
    tot = numpy.sum(delta,dtype='float64')           
    totsq = numpy.sum(delta**2,dtype='float64')
    if ( Ngood == 0 ):
        sigma=-1.0
        skew = 0.0   
        print( 'ERROR - No sky values fall within cuts')
        return [skymod,sigma,skew]
    
    minimm=int(numpy.min(good)-1)
    maximm=int(numpy.max(good))
    
    skymed = numpy.median(sky[good])
    skymn = tot/(maximm-minimm)
    sigma = numpy.std(sky[good])
    skymn = skymn + skymid
    
    if (skymed < skymn):
        skymod = (3.*skymed)-(2.*skymn)
    else:
        skymod=skymn*1.
    
    
    clamp=1.
    old=0.
    niter=0
    #redo=True
    for niter1 in numpy.arange(maxiter+1):
        #while(redo == 1):
        niter=niter+1
        if niter > maxiter:
            sigma=-1.
            skew=0.
            print( 'Too many iterations')
            return [skymod,sigma,skew]
        if maximm-minimm < minsky:
            sigma=-1.
            skew=0.
            print( 'Too few valid sky elements')
            return [skymod,sigma,skew]
        

        r=numpy.log10(maximm-minimm)
        r=numpy.max(numpy.array([2.,( -0.1042*r + 1.1695)*r + 0.8895 ]))
        cut=r*sigma+0.5*numpy.abs(skymn-skymod)
        if integer == True:
            cut=numpy.max(numpy.array([cut,1.5]))
        cut1=skymod-cut
        cut2=skymod+cut
        
        redo=False
        newmin=int(minimm*1.)
        tst_min=sky[newmin+1] >= cut1
        done = (newmin == -1) & (tst_min)
        if done == False:
            done = (sky[numpy.max(numpy.array([newmin,0.]))] < cut1) & (tst_min)
        if done == False:
            
            if tst_min == True:
                istep = -1
            else:
                istep=1
            
            for niter2 in numpy.arange(Nsky):
                #while(done == False):
                newmin=newmin+istep
                done= (newmin == Nlast) | (newmin == -1)
                if done == False:
                    done = (sky[newmin] <= cut1) & (sky[newmin+1] >= cut1)
                if done == True:
                    break
            if tst_min == True:
                delta = sky[(newmin+1):(minimm+1)] - skymid
            else:
                delta= sky[(minimm+1):(newmin+1)] - skymid
            tot=tot-istep*numpy.sum(delta,dtype="float64")
            totsq=totsq-istep*numpy.sum(delta**2,dtype="float64")
            redo=True
            minimm = int(newmin*1.)
            
        newmax=int(maximm*1.)
        tst_max = (sky[maximm] <= cut2)
        done = (maximm == Nlast) & tst_max
        if done == False:
            done=tst_max & (sky[numpy.min(numpy.array([maximm+1,Nlast]))] > cut2)
        if done == False:
            if tst_max == False:
                istep = -1
            else:
                istep=1

            for niter3 in numpy.arange(Nsky):
                newmax=newmax+istep
                done= (newmax == Nlast) | (newmax == -1)
                if done == False:
                    done = (sky[newmax] <= cut2) & (sky[newmax+1] >= cut2)
                if done == True:
                    break
            if tst_max == True:
                delta=sky[(maximm+1):(newmax+1)]-skymid
            else:
                delta=sky[(newmax+1):(maximm+1)]-skymid
            tot=tot+istep*numpy.sum(delta,dtype="float64")
            totsq=totsq+istep*numpy.sum(delta**2,dtype="float64")
            redo=True
            maximm=int(newmax*1.)


        Nsky = maximm - minimm
        if ( Nsky < minsky ):
            sigma = -1.0
            skew = 0.0
            print( 'ERROR - Outlier rejection left too few sky elements')
            return [skymod,sigma,skew]

        skymn = tot/Nsky
        sigma = numpy.sqrt( numpy.max( numpy.array([(totsq/Nsky - skymn**2),0.]) ))
        skymn = skymn + skymid
        
        
        CENTER = (minimm + 1 + maximm)/2.
        SIDE = int(numpy.round(0.2*(maximm-minimm)))/2.  + 0.25
        j = int(numpy.round(CENTER-SIDE))
        k = int(numpy.round(CENTER+SIDE))
        
        if numpy.size(readnoise) > 0:
            L = int(numpy.round(CENTER-0.25))
            M = int(numpy.round(CENTER+0.25))
            R = 0.25*readnoise
            while ((j > 0) & (k < Nsky-1) & ( ((sky[L] - sky[j]) < R) | ((sky[k] - sky[M]) < R))):
                j=j-1
                k=k+1
        
        skymed = numpy.sum(sky[j:(k+1)],dtype="float64")/(k-j+1)
        if (skymed < skymn):
            dmod = 3.*skymed-2.*skymn-skymod
        else:
            dmod = skymn-skymod
        if dmod*old < 0.:
            clamp = 0.5*clamp
        skymod=skymod+clamp*dmod
        old=dmod*1.
        if redo == False:
            break

    skew=(skymn-skymod)/numpy.max([1.,sigma])
    Nsky=maximm-minimm
    if (debug==True):
        print( '% MMM: Number of unrejected sky elements: ',Nsky)
        print( '% MMM: Number of iterations: ',niter)
        print( '% MMM: Mode, Sigma, Skew of sky vector:', skymod, sigma, skew   )
 
        
    return [skymod,sigma,skew]

def pixwt(xc, yc, r, x, y):
    return intarea(xc, yc, r, x-0.5, x+0.5, y-0.5, y+0.5)
def arc(x,y0,y1,r):
    return 0.5 * (r**2) * ( numpy.arctan((y1)/(x)) - numpy.arctan((y0)/(x)))
def chord( x, y0, y1):
    return 0.5 * x * ( y1 - y0 )
def intarea( xc, yc, r, x0, x1, y0, y1):
    x0 = x0 - xc
    y0 = y0 - yc
    x1 = x1 - xc
    y1 = y1 - yc
    return Oneside( x1, y0, y1, r ) + Oneside( y1, -x1, -x0, r )+  Oneside( -x0, -y1, -y0, r ) + Oneside( -y0, x0, x1, r )

def Oneside(x, y0, y1, r):
    true = 1
    size_x  = numpy.size( x )

    if size_x <= 1:
        if x == 0:
            return x
        if numpy.abs(x) >= r:
            return arc( x, y0, y1, r )
        yh = numpy.sqrt( r**2 - x**2 )
        if (y0 <=-yh):
            if y1 <= -yh:
                return arc( x, y0, y1, r )
            elif y1 <= yh:
                return arc( x, y0, -yh, r ) + chord( x, -yh, y1 )
            else:
                return arc( x, y0, -yh, r ) + chord( x, -yh, yh ) + arc( x, yh, y1, r )
        elif ( y0 <  yh ):
            if y1 <= -yh:
                return chord( x, y0, -yh ) + arc( x, -yh, y1, r )
            elif y1 <= yh:
                return chord( x, y0, y1 )
            else:
                return chord( x, y0, yh ) + arc( x, yh, y1, r )
        else:
            if y1 <= -yh:
                return arc( x, y0, yh, r ) + chord( x, yh, -yh ) + arc( x, -yh, y1, r )
            elif y1 <= yh:
                return arc( x, y0, yh, r ) + chord( x, yh, y1 )
            else:
                return arc( x, y0, y1, r )
    else:
        ans = x*1.
        t0 = ( x == 0)
        count = numpy.sum(t0)
        if count == numpy.size( x ):
            return ans
        ans = x * 0.
        yh = ans*1.
        to = ( numpy.abs( x ) >= r)
        tocount=numpy.sum(to)
        to = numpy.arange(size_x)[( numpy.abs( x ) >= r)]
        ti = ( numpy.abs( x ) < r)
        ticount=numpy.sum(ti)
        ti = numpy.arange(size_x)[( numpy.abs( x ) < r)]
        if tocount != 0:
            ans[ to ] = arc( x[to], y0[to], y1[to], r )
        if ticount == 0:
            return ans
        yh[ ti ] = numpy.sqrt( r**2 - x[ti]**2 )
        t1 = (y0[ti] <= -yh[ti])
        count=numpy.sum(t1)
        t1 = numpy.arange(size_x)[(y0[ti] <= -yh[ti])]
        if count != 0:
            i = ti[t1]
            
            t2=(y1[i] <= -yh[i])
            count=numpy.sum(t2)
            t2= numpy.arange(size_x)[(y1[i] <= -yh[i])]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] =  arc( x[j], y0[j], y1[j], r )
            t2=( y1[i] > -yh[i] ) & ( y1[i] <=  yh[i] )
            count = numpy.sum(t2)
            t2 =  numpy.arange(size_x)[( y1[i] > -yh[i] ) & ( y1[i] <=  yh[i] )]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = arc( x[j], y0[j], -yh[j], r )+ chord( x[j], -yh[j], y1[j] )
            t2 = (y1[i] > yh[i])
            count=numpy.sum(t2)
            t2=numpy.arange(size_x)[(y1[i] > yh[i])]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = arc( x[j], y0[j], -yh[j], r )+ chord( x[j], -yh[j], yh[j] ) + arc( x[j], yh[j], y1[j], r )
        t1 =  ( y0[ti] > -yh[ti] ) & ( y0[ti] < yh[ti] )
        count=numpy.sum(t1)
        t1=numpy.arange(size_x)[t1]
        
        if count != 0:
            i = ti[ t1 ]
            t2 = ( y1[i] <= -yh[i])
            count=numpy.sum(t2)
            t2=numpy.arange(size_x)[t2]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = chord( x[j], y0[j], -yh[j] ) + arc( x[j], -yh[j], y1[j], r )
            t2 = ( ( y1[i] > -yh[i] ) & ( y1[i] <=  yh[i] ))
            count=numpy.sum(t2)
            t2=numpy.arange(size_x)[t2]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = chord( x[j], y0[j], y1[j] )
         

            t2 = ( y1[i] > yh[i])
            count=numpy.sum(t2)
            t2=numpy.arange(size_x)[t2]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = chord( x[j], y0[j], yh[j] ) + arc( x[j], yh[j], y1[j], r )
         
        t1 = ( y0[ti] >= yh[ti])
        count = numpy.sum(t1)
        t1=numpy.arange(size_x)[t1]
        if count != 0:
            i = ti[ t1 ]
            t2 = ( y1[i] <= -yh[i])
            count = numpy.sum(t2)
            t2=numpy.arange(size_x)[t2]
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = arc( x[j], y0[j], yh[j], r ) + chord( x[j], yh[j], -yh[j] ) + arc( x[j], -yh[j], y1[j], r )
         

            t2 = ( ( y1[i] > -yh[i] )& ( y1[i] <=  yh[i] ))
            count=numpy.sum(t2)
            t2=numpy.arange(size_x)[t2]
            if count != 0:
                    j = ti[ t1[ t2 ] ]
                    ans[j] = arc( x[j], y0[j], yh[j], r )+chord( x[j], yh[j], y1[j] )
            t2 = ( y1[i] > yh[i])
            count=numpy.sum(t2)
            t2=numpy.arange(size_x)[t2]
            if count != 0:
                    j = ti[ t1[ t2 ] ]
                    ans[j] = arc( x[j], y0[j], y1[j], r )
    return ans

def find(image, hmin, fwhm, nsigma=1.5, roundlim=[-1.,1.], sharplim=[0.2,1.]):
    """

    ASTROLIB-routine
    
    Identifies stars in an image.
    Returns a list [x, y, flux, sharpness, roundness].
    
    image: 2D array containing the image
    hmin: Minimum threshold for detection. Should be 3-4 sigma above background RMS.
    fwhm: FWHM to be used for the convolution filter. Should be the same as the PSF FWHM.
    nsigma (1.5): radius of the convolution kernel.
    roundlim ([-1.,1.]): Threshold for the roundness criterion.
    sharplim ([0.2,1.]): Threshold for the sharpness criterion.
    
    Note: Pyfits imports images with x and y inverted with respect to IDL's convention.
    (see below)
    Note 2: We use the IDL/C/Python convention, with pixel indexation starting at 0.
    Therefore, a +1 offset is required to things the same as DS9, IRAF, etc.
    
    >>> import pyfits, pyastrolib
    >>> image = pyfits.getdata('test.fits')
    >>> dim_y, dim_x = image.shape
    >>> [x, y, flux, sharpness, roundness] = find(image, 15, 5.)
    """
    ###
    # Setting the convolution kernel
    ###
    n_x = image.shape[1] # x dimension
    n_y = image.shape[0] # y dimension
    
    sigmatofwhm = 2*numpy.sqrt(2*numpy.log(2))
    radius = nsigma * fwhm / sigmatofwhm # Radius is 1.5 sigma
    if radius < 1.0:
        radius = 1.0
        fwhm = sigmatofwhm/nsigma
        print( "WARNING!!! Radius of convolution box smaller than one." )
        print( "Setting the 'fwhm' to minimum value, %f, given the provided nsigma." %fwhm )
    sigsq = (fwhm/sigmatofwhm)**2 # sigma squared
    nhalf = int(radius) # Center of the kernel
    nbox = 2*nhalf+1 # Number of pixels inside of convolution box
    middle = nhalf # Index of central pixel
    
    kern_y, kern_x = numpy.ix_(numpy.arange(nbox),numpy.arange(nbox)) # x,y coordinates of the kernel
    g = (kern_x-nhalf)**2+(kern_y-nhalf)**2 # Compute the square of the distance to the center
    mask = g <= radius**2 # We make a mask to select the inner circle of radius "radius"
    nmask = mask.sum() # The number of pixels in the mask within the inner circle.
    g = numpy.exp(-0.5*g/sigsq) # We make the 2D gaussian profile
    
    ###
    # Convolving the image with a kernel representing a gaussian (which is assumed to be the psf)
    ###
    c = g*mask # For the kernel, values further than "radius" are equal to zero
    c[mask] = (c[mask] - c[mask].mean())/(c[mask].var() * nmask) # We normalize the gaussian kernel
    
    c1 = g[nhalf] # c1 will be used to the test the roundness
    sumc1 = c1.mean()
    sumc1sq = (c1**2).sum() - sumc1
    c1 = (c1-c1.mean())/((c1**2).sum() - c1.mean())
    
    h = scipy.ndimage.convolve(image,c,mode='constant',cval=0.0) # Convolve image with kernel "c"
    h[:nhalf,:] = 0 # Set the sides to zero in order to avoid border effects
    h[-nhalf:,:] = 0
    h[:,:nhalf] = 0
    h[:,-nhalf:] = 0
    
    mask[middle,middle] = False # From now on we exclude the central pixel
    nmask = mask.sum() # so the number of valid pixels is reduced by 1
    goody,goodx = mask.nonzero() # "good" identifies position of valid pixels
    
    ###
    # Identifying the point source candidates that stand above the background
    ###
    indy,indx = (h >= hmin).nonzero() # we identify point that are above the threshold, image coordinate
    nfound = indx.size # nfound is the number of candidates
    if nfound <= 0:
        print( "WARNING!!! There is no source meeting the 'hmin' criterion." )
        print( "Aborting the 'find' function." )
        return None
    offsetsx = numpy.resize(goodx-middle,(nfound,nmask)) # a (nfound, nmask) array of good positions in the mask, mask coordinate
    offsetsx = indx + offsetsx.T # a (nmask, nfound) array of positions in the mask for each candidate, image coordinate
    offsetsy = numpy.resize(goody-middle,(nfound,nmask)) # a (nfound, nmask) array of good positions in the mask, mask coordinate
    offsetsy = indy + offsetsy.T # a (nmask, nfound) array of positions in the mask for each candidate, image coordinate
    offsets_vals = h[offsetsy,offsetsx] # a (nmask, nfound) array of mask values roundness each candidate
    vals = h[indy,indx] # a (nfound) array of the intensity of each candidate
    
    ###
    # Identifying the candidates that are local maxima
    ###
    ind_goodcandidates = ((vals - offsets_vals) > 0).all(axis=0) # a (nfound) array identifying the candidates whose values are above the mask (i.e. neighboring) pixels, candidate coordinate
    nfound = ind_goodcandidates.sum() # update the number of candidates
    if nfound <= 0:
        print( "WARNING!!! There is no source meeting the 'hmin' criterion that is a local maximum." )
        print( "Aborting the 'find' function." )
        return None
    indx = indx[ind_goodcandidates] # a (nfound) array of x indices of good candidates, image coordinate
    indy = indy[ind_goodcandidates] # a (nfound) array of y indices of good candidates, image coordinate
    
    ###
    # Identifying the candidates that meet the sharpness criterion
    ###
    d = h[indy,indx] # a (nfound) array of the intensity of good candidates
    d_image = image[indy,indx] # a (nfound) array of the intensity of good candidates in the original image (before convolution)
    offsetsx = offsetsx[:,ind_goodcandidates] # a (nmask, nfound) array of positions in the mask for each candidate, image coordinate
    offsetsy = offsetsy[:,ind_goodcandidates] # a (nmask, nfound) array of positions in the mask for each candidate, image coordinate
    offsets_vals = image[offsetsy,offsetsx]
    sharpness = (d_image - offsets_vals.sum(0)/nmask) / d
    ind_goodcandidates = (sharpness > sharplim[0]) * (sharpness < sharplim[1]) # a (nfound) array of candidates that meet the sharpness criterion
    nfound = ind_goodcandidates.sum() # update the number of candidates
    if nfound <= 0:
        print( "WARNING!!! There is no source meeting the 'sharpness' criterion." )

        print( "Aborting the 'find' function." )
        return None
    indx = indx[ind_goodcandidates] # a (nfound) array of x indices of good candidates, image coordinate
    indy = indy[ind_goodcandidates] # a (nfound) array of y indices of good candidates, image coordinate
    sharpness = sharpness[ind_goodcandidates] # update sharpness with the good candidates
    
    ###
    # Identifying the candidates that meet the roundness criterion
    ###
    temp = numpy.arange(nbox)-middle # make 1D indices of the kernel box
    temp = numpy.resize(temp, (nbox,nbox)) # make 2D indices of the kernel box (for x or y)
    offsetsx = numpy.resize(temp, (nfound,nbox,nbox)) # make 2D indices of the kernel box for x, repeated nfound times
    offsetsy = numpy.resize(temp.T, (nfound,nbox,nbox)) # make 2D indices of the kernel box for y, repeated nfound times
    offsetsx = (indx + offsetsx.swapaxes(0,-1)).swapaxes(0,-1) # make it relative to image coordinate
    offsetsy = (indy + offsetsy.swapaxes(0,-1)).swapaxes(0,-1) # make it relative to image coordinate
    offsets_vals = image[offsetsy,offsetsx] # a (nfound, nbox, nbox) array of values (i.e. the kernel box values for each nfound candidate)
    dx = (offsets_vals.sum(2)*c1).sum(1)
    dy = (offsets_vals.sum(1)*c1).sum(1)
    roundness = 2*(dx-dy)/(dx+dy)
    ind_goodcandidates = (roundness > roundlim[0]) * (roundness < roundlim[1]) * (dx >= 0.) * (dy >= 0.) # a (nfound) array of candidates that meet the roundness criterion
    nfound = ind_goodcandidates.sum() # update the number of candidates
    if nfound <= 0:
        print("WARNING!!! There is no source meeting the 'roundness' criterion." )
        print( "Aborting the 'find' function." )
        return None
    indx = indx[ind_goodcandidates] # a (nfound) array of x indices of good candidates, image coordinate
    indy = indy[ind_goodcandidates] # a (nfound) array of y indices of good candidates, image coordinate
    sharpness = sharpness[ind_goodcandidates] # update sharpness with the good candidates
    roundness = roundness[ind_goodcandidates] # update roundness with the good candidates
    offsets_vals = offsets_vals[ind_goodcandidates] # update offsets_vals with good candidates
    offsetsx = offsetsx[ind_goodcandidates]
    offsetsy = offsetsy[ind_goodcandidates]
    
    ###
    # Recenter the source position and compute the approximate flux
    ###
    c = numpy.empty((nfound,2), dtype=float)
    for i in xrange(nfound):
        c[i] = scipy.ndimage.center_of_mass(offsets_vals[i])
    x = c[:,1]+indx-middle
    y = c[:,0]+indy-middle
    flux = h[indy,indx]
    """ # This is the way IDL was used to do it
    wt = nhalf - abs(numpy.arange(nbox, dtype=float)-nhalf) + 1
    xwt = numpy.resize(wt, (nbox,nbox)).T
    ywt = xwt.T
    sgx = (g*xwt).sum(1)
    sgy = (g*ywt).sum(0)
    p = wt.sum()
    sumgx = (wt*sgy).sum()
    sumgy = (wt*sgx).sum()
    sumgsqx = (wt*sgx*sgx).sum()
    sumgsqy = (wt*sgy*sgy).sum()
    vec = nhalf - numpy.arange(nbox)
    dgdx = sgy*vec
    dgdy = sgx*vec
    sdgdxs = (wt*dgdx**2).sum()
    sdgdys = (wt*dgdy**2).sum()
    sdgdx = (wt*dgdx).sum()
    sdgdy = (wt*dgdy).sum()
    sgdgdx = (wt*sgy*dgdx).sum()
    sgdgdy = (wt*sgx*dgdy).sum()
    
    sd = (offsets_vals*ywt).sum(1)
    sumgd = (wt*sgy*sd).sum(1)
    sumd = (wt*sd).sum(1)
    sddgdx = (wt*sd*dgdx).sum(1)
    hx = (sumgd - sumgx*sumd/p) / (sumgsqy - sumgx**2/p)
    skylvl = (sumd - hx*sumgx)/p
    dx = (sgdgdx - (sddgdx-sdgdx*(hx*sumgx + skylvl*p)))/(hx*sdgdxs/sigsq)
    
    sd = (offsets_vals*xwt).sum(2)
    sumgd = (wt*sgx*sd).sum(1)
    sumd = (wt*sd).sum(1)
    sddgdy = (wt*sd*dgdy).sum(1)
    hy = (sumgd - sumgy*sumd/p) / (sumgsqx - sumgy**2/p)
    skylvl = (sumd - hy*sumgy)/p
    dy = (sgdgdy - (sddgdy-sdgdy*(hy*sumgy + skylvl*p)))/(hy*sdgdys/sigsq)
    
    x = indx + dx
    y = indy + dy
    flux = h[indy,indx]
    """
    
    return [x, y, flux, sharpness, roundness]
   
def saveFITS(data, filename, overwrite=False):
    """
    Saves an array of data as a FITS file

    Inputs:
    data - the array with data
    filename - the path where you wish to save the FITS file (e.g. "/home/<yourname>/example.fit")
    overwrite - if True, deletes the file at filename if it already exists
    """
    hdu = fits.PrimaryHDU(data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename, overwrite=True)

