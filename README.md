# crossvalidated_lasso
The repository contains two Matlab functions: crossvalidated_lasso.m and confusion_matrix.m, and an example script describing 
how to use them: crossvalidated_lasso_example.m
The crossvalidated_lasso function implements LASSO regression based (sparse) classification with hyper-parameter learning 
based on nested cross-validation. The cross-validation schemes and the parameters of the LASSO regression can be defined 
by the user. 
The confusion_matrix function implements function accuracy, averaged F-measure and confusion matrix calculations for given
predictions (0 or 1) and labels (0 or 1).

RESEARCH PAPERS USING THIS SOFTWARE:
- Meszlényi, Regina, Ladislav Peska, Viktor Gál, Zoltán Vidnyánszky, and Krisztian Buza. 2016a. ‘A Model for Classification 
  Based on the Functional Connectivity Pattern Dynamics of the Brain’. Proceedings of the Network Intelligence Conference (ENIC),
  2016 Third European. IEEE, 2016.
- Meszlényi, Regina, Ladislav Peska, Viktor Gál, Zoltán Vidnyánszky, and Krisztian Buza. 2016b. ‘Classification of fMRI Data 
  Using Dynamic Time Warping Based Functional Connectivity Analysis’. Proceedings of the 24th Signal Processing Conference 
  (EUSIPCO),2016.

