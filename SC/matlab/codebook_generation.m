clear all; clc;
feature_path='./data/CIFAR-10/train/';
sift=[];
for i=1:10
    f_path=[feature_path,num2str(i-1)];
    dirOutput=dir(fullfile(f_path,'*.mat'));
    fileNames={dirOutput.name}';
    for j=1:length(fileNames)
        data_path=[f_path,'/',fileNames{j}];
        data=load(data_path);
        sift=[sift,data.feaSet.feaArr];
    end
    fprintf('已完成：%d/10 \n',i);
end
%% k-means生成1024维codebook
sift=sift';
[~,sift_dictionary]=kmeans(sift,1024);
B=sift_dictionary';
save('dictionary/CIFAR-10_SIFT_Kmeans_1024.mat','B');

        
    
