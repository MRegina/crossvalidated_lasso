function [conf_matrix_DATA,averageF_DATA,ACC_DATA]=confusion_matrix(DATA_prediction,label)
%% confusion_matrix - function to compute accuracy, averaged F measure and confusion matrix for given predictions (0 or 1) and labels (0 or 1)
%INPUT:
%   DATA_prediction: nx1 or nxm array of predicted labels (0 or 1), each column contains the predictions of a classifier with a fix hyper-parameter set
%   label: nx1 or nxm array of labels (0 or 1), each column contains the original labels
%OUTPUT:
%   conf_matrix_DATA: 2x2xm array, contains confusion matrices for the m classifiers. 
%                     (1,1) entry is true positive
%                     (1,2) entry is false negative
%                     (2,1) entry is false positive
%                     (2,2) entry is true negative
%   averageF_DATA: 1xm vector, contains F-measure values averaged over the 0 and 1 classes for each classifiers
%   ACC_DATA: 1xm vector, contains accuracy values for each classifier
%
%   Regina Meszlényi 2016.07.27.

%create the confusion matrices
conf_matrix_DATA=zeros(2,2,size(label,2));

conf_matrix_DATA(1,1,:)=sum(DATA_prediction.*label,1)/size(label,1);
conf_matrix_DATA(2,2,:)=sum(DATA_prediction==0 & label==0,1)/size(label,1);
conf_matrix_DATA(2,1,:)=sum(DATA_prediction==1 & label==0,1)/size(label,1);
conf_matrix_DATA(1,2,:)=sum(DATA_prediction==0 & label==1,1)/size(label,1);

%calculate precision and recall for class 1
prec_DATA(1,:)=squeeze(conf_matrix_DATA(1,1,:)./(conf_matrix_DATA(1,1,:)+conf_matrix_DATA(2,1,:)));
rec_DATA(1,:)=squeeze(conf_matrix_DATA(1,1,:)./(conf_matrix_DATA(1,1,:)+conf_matrix_DATA(1,2,:)));

%calculate precision and recall for class 0
prec_DATA(2,:)=squeeze(conf_matrix_DATA(2,2,:)./(conf_matrix_DATA(2,2,:)+conf_matrix_DATA(1,2,:)));
rec_DATA(2,:)=squeeze(conf_matrix_DATA(2,2,:)./(conf_matrix_DATA(2,2,:)+conf_matrix_DATA(2,1,:)));

%calculate F-measure and average F-measure
F_DATA=(prec_DATA.*rec_DATA./(prec_DATA+rec_DATA)); %F_DATA=2*(F-measure)
F_DATA(isnan(F_DATA))=0;
averageF_DATA=sum(F_DATA);

%calculate accuracy
ACC_DATA=squeeze(conf_matrix_DATA(1,1,:)+conf_matrix_DATA(2,2,:))';