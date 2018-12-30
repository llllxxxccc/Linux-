clear all;clc;
feature_path='./features/CIFAR-10/test/';
for i=1:10
    f_path=[feature_path,num2str(i)];
    dirOutput=dir(fullfile(f_path,'*.mat'));
    fileNames={dirOutput.name}';
    for j=1:length(fileNames)
        data_path=[f_path,'/',fileNames{j}];
        data=load(data_path);
        feature=[data.fea;data.label];
        file_name=fileNames{j};
        file_name(end-3:end)=[];
        eval(['writeNPY(feature,''features_npy/test/',file_name,'.npy'');'])
    end
    fprintf('已完成第%d个类别格式转换 \n', i);
end
clear all; clc;
feature_path='./features/CIFAR-10/train/';
for i=1:10
    f_path=[feature_path,num2str(i)];
    dirOutput=dir(fullfile(f_path,'*.mat'));
    fileNames={dirOutput.name}';
    for j=1:length(fileNames)
        data_path=[f_path,'/',fileNames{j}];
        data=load(data_path);
        feature=[data.fea;data.label];
        file_name=fileNames{j};
        file_name(end-3:end)=[];
        eval(['writeNPY(feature,''features_npy/train/',file_name,'.npy'');'])
    end
    fprintf('已完成第%d个类别格式转换 \n', i);
end
        