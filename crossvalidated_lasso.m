function [DATA_maxfit,DATA_class_test,FeatureNames_DATA]=crossvalidated_lasso(DATA,label,featureNames,lambda,stand,datasplit_out,datasplit)
%%crossvalidated_lasso implements LASSO based classification with hyper-parameter learning based on nested cross-validation 
%INPUT:
%   DATA: nx(m+1) array of instances, every row contains the instance ID and an m dimensional instance
%   label: nx1 vector of original labels (0 or 1)
%   featureNames: 1xm cell-array, contains the names of the feature-dimensions
%   lambda: lx1 vector of hyper-parameters for the LASSO
%   stand: 0 or 1, if 1 the DATA array will be standardized before the LASSO regression calculation
%   datasplit_out: n x(j+1) array of the outer cross-validation cycle, first column contains instance IDs, k column contains the k-fold 
%       cross-validation structure: each entry conains 0 if the instance should be used for training, 1 if it should be used for testing
%   datasplit: kxi+1 array of the inner cross-validation cycle, first column contains instance IDs, j column contains the j-fold
%       cross-validation structure: each entry conains 0 if the instance should be used for training, 1 if it should be used for testing
%
%OUTPUT:
%   DATA_maxfit: 1xj structure, contains LASSO fitting information for the k folds of the outer crossvalidation cycle
%   DATA_class_test: nx1 vector of predicted class labels from the test data of the k folds
%   FeatureNames_DATA: 1xj structure, contains the names of the relevant features for the k folds of the outer crossvalidation cycle
%
%   Regina Meszlényi 2016.07.27.


%outer cross-validation cycle
for j=1:(size(datasplit_out,2)-1)
    
    %split data and labels to train and test data according to the outer cross-validation scheme
    DATA_train=DATA(~datasplit_out(:,1+j),:);
    label_train=label(~datasplit_out(:,1+j));
    datasplit(:,1)=datasplit_out(~datasplit_out(:,1+j),1);
    
    %inner cross-validation cycle
    parfor i=1:(size(datasplit,2)-1)
        %fit LASSO on training data for hyper-parametes lambda
        [B,fitinfo]=lasso(DATA_train(~datasplit(:,1+i),2:end),label_train(~datasplit(:,1+i)),'Weights',1/sum(~datasplit(:,1+i))*ones(sum(~datasplit(:,1+i)),1),'Lambda',lambda,'Standardize',stand,'PredictorNames',featureNames);
        %save fit information
        DATA_fit{i}.B=B;
        DATA_fit{i}.fitinfo=fitinfo;
        for k=1:length(lambda)
            DATA_fit{i}.output(:,k)=DATA_train(~datasplit(:,1+i),2:end)*B(:,k)+fitinfo.Intercept(k);
            DATA_fit{i}.output_test(:,k)=DATA_train(logical(datasplit(:,1+i)),2:end)*B(:,k)+fitinfo.Intercept(k);
        end
    end
    
    %evaluate fitting performance for the hyper-parameters in the inner cross-validation cycle
    parfor i=1:(size(datasplit,2)-1)
        %original labels of the test data
        LABEL_test=repmat(label_train(logical(datasplit(:,1+i))),1,length(lambda));
        
        %calculate predicted labels of the test data
        DATA_class_test=zeros(size(DATA_fit{1,i}.output_test));
        DATA_class_test((DATA_fit{1,i}.output_test)>0.5)=1;
        
        %evaluate performance based on averaged F-measure of the 0 and 1 classes
        [~,averageF_DATA_test(i,:),~]=confusion_matrix(DATA_class_test,LABEL_test);   
    end
    
    %calculate mean averaged F-measure for each hyper-parameter setting over the inner crossvalidation cycle
    averageF_data_test=mean(averageF_DATA_test);    
    
    %find the maximal lambda value (sparsest model) that corresponds to the best mean averaged F-measure
    lambdamax_DATA(j)=max(lambda((averageF_data_test(1,:)==max(averageF_data_test(1,:)))));
    
    %fit LASSO model with the best hyper-parameter to the whole training dataset of the outer cross-validation cycle
    [B,fitinfo]=lasso(DATA_train(:,2:end),label_train,'Weights',1/length(label_train)*ones(length(label_train),1),'Lambda',lambdamax_DATA(j),'Standardize',stand,'PredictorNames',featureNames);
    
    %save fitting data and feature names for the best hyper-parameter
    DATA_maxfit{j}.B=B;
    DATA_maxfit{j}.fitinfo=fitinfo;
    for k=1:sum(datasplit_out(:,1+j))
        DATA_maxfit{j}.output(:,k)=DATA_train(:,2:end)*B(:,k)+fitinfo.Intercept(k);
        DATA_maxfit{j}.output_test(:,k)=DATA(logical(datasplit_out(:,1+j)),2:end)*B(:,k)+fitinfo.Intercept(k);
    
        FeatureNames_DATA{j,k}=featureNames(DATA_maxfit{j}.B(:,k)~=0);
    end
    
end


% summarize predicted class labels in one vector
DATA_class_test=zeros(size(label));

for j=1:(size(datasplit_out,2)-1)
    for m=1:length((DATA_maxfit{1}.output_test))
        if ((DATA_maxfit{j}.output_test(m))>0.5)
            DATA_class_test((j-1)*length((DATA_maxfit{1}.output_test))+m)=1;
        end
    end
end
    
    
    
    
