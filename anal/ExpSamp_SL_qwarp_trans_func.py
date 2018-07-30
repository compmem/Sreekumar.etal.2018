# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# These are global imports
import sys
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from numpy.linalg import LinAlgError
import numpy.lib.recfunctions as nprf 
from scipy import ndimage
from scipy import stats
import cPickle as pickle
from joblib import Parallel,delayed

# <codecell>

from mvpa2.suite import *
from scipy.io import loadmat
from scipy.spatial.distance import squareform,pdist
from scipy.stats import rankdata,pearsonr

# <codecell>

debug.active += ["SLC"]

# <codecell>

class CustomCorrMeasure(Measure):
    def __init__(self, ind_data, index, use_rank=True, formula = None, method = 'pearson',
                 ind_rank_method = rankdata,
                 s_transform = None,t_transform = None,e_transform = None,
                 s_bounds = None, t_bounds = None,
                 verbose = False,return_term_array = 0,zdist = False,ndmetric = 'correlation'):
        """ for right now, it's up to you to make sure that the index matches your ind_data, 
        for a glm that means you need a separate index for each subject, for an LMER it'll be 
        1 big block of independent data and 1 big index. 
        
        ind_rank_method lets you define a function for ranking the independent data
        this lets us deal with ranking and sorting and interaction terms and all of that"""
        Measure.__init__(self)
        self._ind_data = ind_data[index]
        self._index = index
        self._use_rank = use_rank
        self._formula = formula
        self._method = method
        self._ndmetric = ndmetric
        self._subj = np.unique(self._ind_data['subject'])
        self._is1=True
        self._s_transform = s_transform
        self._t_transform = t_transform
        self._e_transform = e_transform
        self.verbose = verbose
        self._return_term_array = return_term_array
        self._zdist = zdist


        if s_bounds != None:
            self._s_bounds = s_bounds
        if t_bounds != None:
            self._t_bounds = t_bounds
        
        if s_transform is not None:
            self._ind_data['space'] = s_transform(self._ind_data['space'])
            
        if t_transform is not None:
            self._ind_data['space'] = t_transform(self._ind_data['time'])
        
        if e_transform is not None:
            self._ind_data['space'] = e_transform(self._ind_data['event'])
            
        if self._use_rank == True:
            # rank the ind data
            self._ind_data = ind_rank_method(self._ind_data)
        else:
            idat_df = pd.DataFrame(self._ind_data)
            idat_df['val'] = np.zeros(idat_df.shape[0])
            self._ind_data = idat_df.to_records()
        
        #figure out how long the results array will be by calling the glm on some dummy data
        if self._method=='glm':
            self._ind_data['val'] = np.random.randn(self._ind_data.shape[0])     
            self._res_len = (smf.glm(formula=self._formula, data=self._ind_data).fit().params.shape[0]*2)+4
            #set val back to zeros just in case
            self._ind_data['val'] = np.zeros(self._ind_data.shape[0])     
    

    def __call__(self, dataset):
    
        if ((self._method == 'pearson') or (self._method == 'glm')):
            if dataset.shape[1] == 1:
                metric = 'euclidean'
            else:
                metric = self._ndmetric
            # calc dist for neural data
            ndist = pdist(dataset.samples, metric=metric)[self._index]
            if self._zdist == True:
                ndist = (ndist-np.average(ndist))/np.std(ndist)
            if self._use_rank:
                #rank it
                ndist = rankdata(ndist)
            
        if self._method == 'pearson':
            # compare it
            r,p = pearsonr(ndist, self._ind_data[self._formula])
    
            # convert to z
            return np.arctan(r)
        
        elif self._method == 'glm':
            betas = {}
            
            #norm the neural ranks
            #print ndist.shape
            #print type(ndist)
            #print self._subj
            if self._use_rank == True:
                 ndist = ndist.astype(np.float)/np.max(ndist)
            
            #set val in ind_data equal to dep_data for that feature
            self._ind_data['val']=ndist
            #fit glm based on model given in fe_formula
            try:
                modl= smf.glm(formula=self._formula, data=self._ind_data)
                modl.raise_on_perfect_prediction = False
            except (LinAlgError, ValueError):
                return np.zeros(self._res_len)
            try:
                modl = modl.fit()
            except PerfectSeparationError:
                #print "Perfect Separation Error, returning 0 in blind optimism"
                return np.zeros(self._res_len)
                #save beta's for each factor to an array 

            #for fac in modl.pvalues.keys():              
            #    betas[fac]=modl.params[fac]
            betas = np.array([modl.params[fac] for fac in sorted(modl.pvalues.keys())])
            betas = np.append(betas,np.array([modl.tvalues[fac] for fac in sorted(modl.pvalues.keys())]))            
            betas = np.append(betas,modl.deviance)
            betas = np.append(betas,modl.scale)
            betas = np.append(betas,modl.llf)
            betas = np.append(betas,dataset.nfeatures)
            if self.verbose == True:

                if self._is1 == True:
                    k=0            
                    for fac in sorted(modl.pvalues.keys()):
                        print k,fac,"betas"
                        k+=1
                    for fac in sorted(modl.pvalues.keys()):
                        print k,fac,"tvals"
                        k+=1
                    print k,"deviance"
                    print k+1,"scale"
                    print k+2,"llf"
                    print k+3,"count"
                    self._is1=False
            if self._return_term_array == 2:

                terms = []           
                for fac in sorted(modl.pvalues.keys()):
                    terms.append(fac+'_betas')
                for fac in sorted(modl.pvalues.keys()):
                    terms.append(fac+"_tvals")
                terms.append("deviance")
                terms.append("scale")
                terms.append("llf")
                terms.append("count")
                return terms
            elif self._return_term_array == 1:
                return sorted(modl.pvalues.keys())
                
            return betas
        

            
        
        elif self._method == 'lmer':
            raise error('%s method not implemented'%self._method)
        else:
            raise error('%s method not implemented'%self._method)
         
        

# <codecell>

#

#custom ranking function to handle independent data and correctly rank the interaction term

def rank_norm_data(idat,subf='subject',psf='pair_str',tf='time',sf='space',cf='correl',sxtf='spaceXtime'):
    """"rank data man"""
    #rank new independent data in place
    idat_df = pd.DataFrame(idat)
    #print np.unique(idat[psf])
    #print idat_df.columns
    subj = np.unique(idat[subf])
    #print subj
    idat_df['val'] =np.zeros((idat_df.shape[0],1))
    
    
    tl = []
    sl = []
    sxtl = []
    cl = []
    for s in np.unique(idat[subf]):
        tr = rankdata(idat_df.ix[idat_df[subf]==s][tf])
        sr = rankdata(idat_df.ix[idat_df[subf]==s][sf])
        cr = rankdata(idat_df.ix[idat_df[subf]==s][cf])
        sxtr = rankdata(sr*tr)
    
        #norm ranks to be between 0 an 1
        tl = np.append(tl,tr/np.float(np.max(tr)))
        sl = np.append(sl,sr/np.float(np.max(sr)))
        cl = np.append(cl,cr/np.float(np.max(cr)))
        sxtl = np.append(sxtl,sxtr/np.float(np.max(sxtr)))
            
    idat_df[tf]=tl
    idat_df[sf]=sl
    idat_df[cf]=cl
    idat_df[sxtf]=sxtl
            
    return idat_df.to_records()


def addval(idat):
    #rank new independent data in place
    idat_df = pd.DataFrame(idat)
    idat_df['val'] =np.zeros((idat_df.shape[0],1))
    return idat_df.to_records()
# <codecell>

if __name__ == "__main__":

    #define paths and such
    radius = 3
    fe_formula = 'val~LAH_dist + event' 
    run_name = 'lah_e_r%d'%radius #this one you should probably change for each run

    spacemax = 10.**4.5,
    spacemin = 10.**2,
    timemin = 10**4.75,
    timemax =  10007180




    method = 'glm'
    run_type = '%s_sl'%(method)

    basedir = '/cmlab/data/fmri/expSamp/data/'
    data_dir = os.path.join(basedir,'rsa_new')
    data_file = 'rsa_dataset_gps_time_old_exclude_drop_viv_dif_lah.pickle'
    outdir = os.path.join('/cmlab/data/fmri/expSamp/data/',run_type,run_name)
    ss_out_pat = 'betas.%s.%s.%s.nii.gz'
    combined_out_pat = '%s.%s.%s.nii.gz'
    fmri_pat = '%s/fmri/analysis/afni/%s_stb_qwarp_e2a.results/stb.%s_stb_qwarp_e2a.nii.gz'
    mask_pat = '%s/fmri/analysis/afni/%s_stb_qwarp_e2a.results/mask_GM_full_+tlrc.nii.gz'
    #mask='/cmlab/data/fmri/expSamp/anal/GM_full_mask.nii.gz'
    #mask ='/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'    
    #define formula for glm, this is also a thing that may change run to run

    subjects = ['expSamp01',
    'expSamp02',
    'expSamp03',
    'expSamp04',
    'expSamp05',
    'expSamp07',
    'expSamp08',
    'expSamp09',
    'expSamp10']


    if __name__== "__main__":


        #make output directory if it doesn't exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        #load independent data
        dat = np.load(os.path.join(data_dir,data_file))

        #swap strong and weak so that the main effects relflect pair strength strong
        dat['pair_str'] = ['3weak' if dat['pair_str'][i] == '1weak' else '1strong' if dat['pair_str'][i]=='3strong' else dat['pair_str'][i] for i in range(len(dat))]

        #cheat and grab correl distance from old data
        #data_dir_old = os.path.join(basedir,'rsa/gps_time_correl_all')
        #data_file_old = 'rsa_dataset_gps_time_correl_all.pickle'
        #dat_old = np.load(os.path.join(data_dir_old,data_file_old))

        #cd_new = []
        #for i in range(len(dat)):
        #    ind = ((dat_old['subject'] == dat['subject'][i]) & ( dat_old['s1_trial'] == dat['s1_trial'][i]) & (dat_old['s2_trial'] == dat['s2_trial'][i]))
        #    cd_new.append(dat_old['correl'][ind])

        #dat = nprf.rec_append_fields(dat,'correl',np.array(cd_new).squeeze())

        #Drop 118 from expSamp07
        sind = (dat['subject']=='expSamp07') & ((dat['s1_trial']==118) | (dat['s2_trial']==118))
        for i in range(len(sind)):
            if sind[i]== True:
                dat[i]['pair_str']='0exclude'    # <codecell>
        dat = dat[dat['pair_str']!='0exclude']

        kept_stims = {}
        for subj in subjects:
            s1_exclude = np.array(np.unique(dat[((dat['subject']==subj) & (dat['pair_str'] != '0exclude'))]['s1_trial']))
            s2_exclude = np.array(np.unique(dat[((dat['subject']==subj) & (dat['pair_str'] != '0exclude'))]['s2_trial']))
            kept_stims[subj] = np.unique(np.array([s1_exclude,s2_exclude]))-1

        for subj in subjects:
            mask = os.path.join(basedir,mask_pat%(subj,subj))
            outfile = os.path.join(outdir,ss_out_pat%(subj,run_name,run_type))

            # load subject's fmri data
            ds = fmri_dataset(os.path.join(basedir,fmri_pat%(subj,subj,subj)),mask=mask)[kept_stims[subj]]

            # define index
            #ind = (((dat[dat['subject']==subj]['pair_str']=='1strong') | (dat[dat['subject']==subj]['pair_str']=='3weak')) &
            #ind = ((dat[dat['subject']==subj]['pair_str']=='1strong') &
            ind = ( (dat[dat['subject']==subj]['rem']==True) &
                    (dat[dat['subject']==subj]['space'] < spacemax) & (dat[dat['subject'] == subj]['space'] > spacemin) &
                    (dat[dat['subject']==subj]['time'] > timemin) & (dat[dat['subject']==subj]['time'] < timemax))

            # define the comparison
            dsmetric = CustomCorrMeasure(dat[dat['subject']==subj],ind, 
                                         use_rank=False, formula = fe_formula, 
                                         method = method, ind_rank_method=rank_norm_data,
                                         s_log= True, t_log = True, e_log = True,zdist = True,verbose=True)

            sl = sphere_searchlight(dsmetric, radius=radius)

            sl.nproc = 7

            #print outfileMNI_152_HC.nii.gz
            sl_map = sl(ds)

            # save it out
            map2nifti(sl_map).to_filename(outfile)

    # <codecell>


