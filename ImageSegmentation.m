clear all; close all; clc;
set(0,'DefaultFigureVisible','on');
% Eli MacColl          
% 4/14/21

images = {'3096_color.jpg','42049_color.jpg'};
numCheckGMM = 10;
k = 10;

for i=1:length(images)
    switch i
        case 1
            figure(1);
            plane.Data = imread(images{i});
            data =  plane.Data;
        case 2
            figure(2);
            bird.Data = imread(images{i});
            data =  bird.Data;
    end
    subplot(1,3,1);
    imshow(data);
    title('Original');
    [r,c,d] = size(data);
    rows = (1:r)'*ones(1,c);
    cols = ones(r,1)*(1:c);
    featureData = [rows(:)';cols(:)'];
    data = double(data);
    for j=1:d
        dataD = data(:,:,j);
        featureData = [featureData; dataD(:)'];
    end
    minF = min(featureData,[],2);
    maxF = max(featureData,[],2);
    ranges = maxF-minF;
    
    % Normalize and store the data
    x = (featureData-minF)./ranges;
    switch i
        case 1
            plane.x = x;
        case 2
            bird.x = x;
    end
    
    % Fitting GMM with 2-components
    GMM2Comp=fitgmdist(x',2,'RegularizationValue',0.03);
    posteriors2Comp=posterior(GMM2Comp,x')';
    lossMatrix2Comp=ones(2,2)-eye(2);
    expRisk2Comp =lossMatrix2Comp*posteriors2Comp;
    % 0-1 loss (MAP decision rule)
    [~,decisions2Comp] = min(expRisk2Comp,[],1);
    % Display segmented image
    imLabels2Comp=reshape(decisions2Comp-1,r,c);
    dispImLabels2Comp = uint8(imLabels2Comp*255/2);
    subplot(1,3,2);
    imshow(dispImLabels2Comp);
    title('2 Components');

    figure(2);
    % k-fold cross-validation to determine optimal number of GMM components
    N=length(x);
    partSize=floor(N/k);
    partInd=[1:partSize:N N];
    % Using BIC to identify the best number of Gaussian components
    fprintf('Identifying best number of Gaussian components with BIC\n');
    for gmm=1:numCheckGMM
        fprintf('Image #%d GMM Components: %d\n',i,gmm);
        for M=1:k
            fprintf('%d of %d\n',M,k);
            valIndex=partInd(M):partInd(M+1);
            trainIndex=setdiff(1:N,valIndex);
            % GMMk_loop=fitgmdist(x(:,trainIndex)',gmm,'Replicates',5);
            GMMreg = fitgmdist(x(:,trainIndex)',gmm,'RegularizationValue',0.03);
            fprintf('Regularized BIC val: %1.4f\n',GMMreg.BIC);
            fprintf('===========================\n');
            gmmNum(gmm) = {['gmm' num2str(gmm)]}; % BIC
            %if GMMk_loop.Converged
            if GMMkreg.Converged
                regBIC.(gmmNum{gmm}) = GMMreg.BIC; % BIC reg
                %probX(M)=sum(log(pdf(GMMk_loop,x(:,valIndex)')));
            else
                regBIC.(gmmNum{gmm}) = 0; % BIC reg
                %probX(M)=0;
            end
        end
        % Determine average BIC value for a value of M
        %avgProb(i,gmm)=mean(probX);
        avgRegBIC(i,gmm) = mean(regBIC.(gmmNum{gmm})); % BIC reg
        info(i).gmm=1:numCheckGMM;
        info(i).avgRegBIC=avgRegBIC; % BIC reg
        info(i).mRegBIC(:,gmm)=regBIC.(gmmNum{gmm});
    end

    % Select GMM with lowest average BIC value (closer to true model)
    [~,optNumGMM_BIC_reg]=min(avgRegBIC(i,:)); % BIC reg
    %[~,optNumGMM_prob]=max(avgProb(i,:));
    fprintf('optNumGMM_BIC_reg: %d\n',optNumGMM_BIC_reg); % BIC reg
    %fprintf('optNumGMM_BIC_prob: %d\n',optNumGMM_prob);
    %optNumGMM_BIC_reg = optNumGMM_prob;

    % Fit a new GMM using all data with the optimal number of perceptrons
    fprintf('Fitting Final GMM With Optimal Perceptrons\n');
    
    %finalGMM=fitgmdist(x',optNumGMM_prob,'Replicates',10);
    finalGMM=fitgmdist(x',optNumGMM_BIC_reg,'RegularizationValue',0.03);
    
    posteriors=posterior(finalGMM,x')';
    lossMatrix=ones(optNumGMM_BIC_reg,optNumGMM_BIC_reg)-eye(optNumGMM_BIC_reg);
    expRisk =lossMatrix*posteriors; % Expected Risk for each label (rows) for each sample (columns)
    [~,decisions] = min(expRisk,[],1); % Minimum expected risk decision with 0-1 loss is the same as MAP
    % Plot segmented image for Max. Likelihood number of GMMs case
    imLabels=reshape(decisions-1,r,c);
    dispImLabels = uint8(imLabels*255/2);
    switch i
        case 1
            plane.OptNumGMM = optNumGMM_BIC_reg;
            plane.FinalGMM = finalGMM;
            plane.ImageLabels = imLabels;
            plane.DispImageLabels = dispImLabels;
            plane.LossMatrix = lossMatrix;
            plane.ExpRisk = expRisk;
            plane.Decisions = decisions;
        case 2
            bird.OptNumGMM = optNumGMM_BIC_reg;
            bird.FinalGMM = finalGMM;
            bird.ImageLabels = imLabels;
            bird.DispImageLabels = dispImLabels;
            bird.LossMatrix = lossMatrix;
            bird.ExpRisk = expRisk;
            bird.Decisions = decisions;
    end
    subplot(1,3,3);
    imshow(dispImLabels);
    title(sprintf('Optimal: %d',optNumGMM_BIC_reg));
end

% Plotting Average BIC Values vs Number of GMM Components for Plane Image
figure(3);
numComponents=[1,2,3,4,5,6,7,8,9,10];
stem(numComponents,avgRegBIC(1,:),'Color','b');
xlabel('GMM Components'); ylabel('Average BIC Value');
grid on;
subtitle('Airplane Image');
title('Average BIC Value vs Number of GMM Components');
Plotting Average BIC Values vs Number of GMM Components for Bird Image
figure(4);
stem(numComponents,avgRegBIC(2,:),'Color','b');
xlabel('GMM Components'); ylabel('Average BIC Value');
grid on;
subtitle('Bird Image');
title('Average BIC Value vs Number of GMM Components');