%%Example on how to use crossvalidated_lasso function
%Random feature-set and labeling is created,to test how well the LASSO based
%classification works. A nested leave-one-out cross-validation scheme is
%presented.
%
%   Regina Meszlényi 2016.08.12.

%set number of subjects and features
subnum=150;
featurenum=4000;

%generate random labels:
label=round(rand(subnum,1));

%generate random feature set
DATA(1:subnum,1)=1:subnum;    %subject IDs
DATA(:,2:featurenum+1)=rand(subnum,featurenum); %random features

%create IDs where data will be changed
IDs=zeros(subnum,featurenum);
IDs(logical(label),1:featurenum/100:end)=1;

%change DATA accordingly
DATA(:,2:featurenum+1)=DATA(:,2:featurenum+1)+0.15*IDs;

%generate feature names
for i=1:size(DATA,2)-1
    featureNames{i}=num2str(i);
end

%name of the classification target
labelName='labelName';

%standardize data before LASSO regression: stand=1
stand=1;

%hyper-parameter set for the LASSO regression
lambda = exp(log(0.0001):(log(0.5)-log(0.0001))/199:(log(0.5)))';


%e.g. leave-one-out crossvalidation in both outer and inner cross-validations
%first column: subject IDs
%columns 2:end: each column describes which subjects data is retained for testing (1: test, 0:train) 
datasplit_out=[DATA(:,1),eye(size(DATA,1))]; 
datasplit=datasplit_out(2:end,2:end);


    
%fit the crossvalidated LASSO
[DATA_maxfit,DATA_class_test,FeatureNames_DATA]=crossvalidated_lasso(DATA,label,featureNames,lambda,stand,datasplit_out,datasplit);

%evaluate the goodness of the fit
[conf_matrix_DATA,averageF_DATA,ACC_DATA]=confusion_matrix(DATA_class_test,label);

  
