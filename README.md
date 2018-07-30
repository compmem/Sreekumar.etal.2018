# Sreekumar.etal.2018
Code used in the analyses for Sreekumar et al. 2018

The Jupyter notebooks and python scripts shared here are written in Python 2.7. They cover analysis of data from single trial betas to the figures included in the publication.

The array of permutations used is saved in multiple parts in `data/input/hame_perms_all.npz_*`. This file can be reassembled with the following command:
```
cat data/input/hame_perms_all.npz_* | gunzip -c > data/input/ham_perms_all.npz
```
Similarly, results from the ROI level analysis of the medial temporal lobe are saved in multiple parts in `data/output/glm_roi/vishu_res/hame_roi_res.csv.gz_*`. This file can be reassembled with the followig command:
```
cat data/output/glm_roi/vishu_res/hame_roi_res.csv.gz_* | gunzip -c > data/output/glm_roi/vishu_res/ham_roi_res.csv
```

We are unable to share all of the raw data in this study due to privacy concerns related to lifelogging data. We have saved the dissimilarity matricies in pickles in `data/input/rsa_dataset_gps_time_old_exclude_drop_viv_bin_scan_time_rem_ham.pickle` and `rsa_dataset_gps_time_old_exclude_viv_bin_scan_time_rem_ham.pickle`. These files were generated by the notebook `Generate_dissimilarity_matrices.ipynb`.

RSA analysis was carried out by for all permutations with `anal\ExpSamp_SL_qwarp_trans_func.py`. Examples of using that script are in `anal\write_swarm_file.ipynb`. The permutations were processed with `anal\Process_permutation_results_all_models.ipynb`. Tables and plots based on the processed results were generated with on of the notebooks `Generate_peak_plots_culster_lists_*` named for each model.
