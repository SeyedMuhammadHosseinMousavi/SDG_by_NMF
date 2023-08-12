%% Synthetic Data Generation by Non-negative Matrix Factorization (NMF)

clear;
load fisheriris
Data=reshape(meas,1,[]); % Preprocessing - convert matrix to vector
Data2=Data;
Target(1:50)=1;Target(51:100)=2;Target(101:150)=3;Target=Target'; % Original labels
mini=min(Data);
maxi=max(Data);

Mass=3; % Amount of synthetic data

for i = 1:Mass
[W{i},NMF{i}] = nnmf(Data2,1);
Temp= NMF{i} .* Data;
NMFSyn{i} = rescale(Temp,mini,maxi);
mini=min(NMFSyn{i});
maxi=max(NMFSyn{i});
Data2=NMFSyn{i};
end

% Converting cell to matrix
Synthetic=NMFSyn';
Synthetic2 = cell2mat(Synthetic);
Synthetic2=Synthetic2';
% Converting matrix to cell
P = size(Data); P = P (1,2);
S = size(Synthetic{i}); SO = size (meas); SF = SO (1,2); SO = SO (1,1); SS = S (1,2); 
for i = 1 : Mass
Generated1{i} = reshape(Synthetic2(:,i),[SO,SF]);
Generated1{i}(:,end+1)=Target;
end
% Converting cell to matrix (the last time)
Synthetic3 = cell2mat(Generated1');
SyntheticData=Synthetic3(:,1:end-1);
SyntheticLbl=Synthetic3(:,end);

%% Plot data and classes
Feature1=2;
Feature2=3;
f1=meas(:,Feature1); % feature1
f2=meas(:,Feature2); % feature 2
ff1=SyntheticData(:,Feature1); % feature1
ff2=SyntheticData(:,Feature2); % feature 2
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
plot(meas, 'linewidth',1); title('Original Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,2)
plot(SyntheticData, 'linewidth',1); title('Synthetic Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,3)
gscatter(f1,f2,Target,'rkgb','.',20); title('Original');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,4)
gscatter(ff1,ff2,SyntheticLbl,'rkgb','.',20); title('Synthetic');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;

%% Train and Test
% Training Synthetic dataset by SVM
Mdlsvm  = fitcecoc(SyntheticData,SyntheticLbl); CVMdlsvm = crossval(Mdlsvm); 
SVMError = kfoldLoss(CVMdlsvm); SVMAccAugTrain = (1 - SVMError)*100;
% Predict new samples (the whole original dataset)
[label5,score5,cost5] = predict(Mdlsvm,meas);
% Test error and accuracy calculations
sizlbl=size(Target); sizlbl=sizlbl(1,1);
countersvm=0; % Misclassifications places
misindexsvm=0; % Misclassifications indexes
for i=1:sizlbl
if Target(i)~=label5(i)
misindex(i)=i; countersvm=countersvm+1; end; end
% Testing the accuracy
TestErrAugsvm = countersvm*100/sizlbl; SVMAccAugTest = 100 - TestErrAugsvm;
% Result SVM
AugResSVM = [' Synthetic Train SVM "',num2str(SVMAccAugTrain),'" Test on Original Dataset"', num2str(SVMAccAugTest),'"'];
disp(AugResSVM);