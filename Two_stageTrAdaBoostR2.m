% 构建网络,numResponses是输出的sequence的特征个数
numFeatures = 2;
numResponses = 1;
numSequences = 32;
numHiddenUnits = 50; % 隐藏单元的个数
maxEpochs = 150;
layers = [ ...
    sequenceInputLayer(numFeatures,'Name','input')
    lstmLayer(numHiddenUnits,'OutputMode','sequence','Name','LSTM')
    fullyConnectedLayer(50, 'Name','FCL1')
    dropoutLayer(0.5,'Name','drop')
    fullyConnectedLayer(numResponses,'Name','FCL2')];

% 选取训练数据
funcFlag = 1;  % 只取1 
instancesNumSource1 = 300;  % 不超过500,source data for trainning
instancesNumSource2 = 300;  % 不超过500,source data for trainning
instancesNumTarget = 10;  % 不超过500,target data for trainning
n = instancesNumSource1 + instancesNumSource2;
m = instancesNumTarget;
totalInstancesNum = n + m; 
miniBatchSize = totalInstancesNum/5;  
testInstancesNum = floor(totalInstancesNum/5);
load('NPIdata.mat')
% 分别获取source data 和 target data
IdxTrainSource1 = randperm(500,instancesNumSource1); 
XTrainSource1 = NPIdataX{1,funcFlag}(:,IdxTrainSource1,:);
YTrainSource1 = NPIdataY{1,funcFlag}(:,IdxTrainSource1,:);
IdxTrainSource2 = randperm(500,instancesNumSource2); 
XTrainSource2 = NPIdataX{2,funcFlag}(:,IdxTrainSource2,:);
YTrainSource2 = NPIdataY{2,funcFlag}(:,IdxTrainSource2,:);
IdxTrainTarget = randperm(500,instancesNumTarget); 
IdxTestTarget = 1:500;
IdxTestTarget(ismember(IdxTestTarget, IdxTrainTarget)==1)= [];  % 取剩余部分为测试集
XTrainTarget = NPIdataX{3,funcFlag}(:,IdxTrainTarget,:);
YTrainTarget = NPIdataY{3,funcFlag}(:,IdxTrainTarget,:);
XTestTarget =  NPIdataX{3,funcFlag}(:,IdxTestTarget(1:testInstancesNum),:);
YTestTarget = NPIdataY{3,funcFlag}(:,IdxTestTarget(1:testInstancesNum),:);
% 将source data 与target data 合并
XTrain = cat(2,XTrainSource1,XTrainSource2,XTrainTarget);  
YTrain = cat(2,YTrainSource1,YTrainSource2,YTrainTarget);  

% 归一化特征
mu = mean(XTrain,2);
sig = std(XTrain,0,2);

% 下面这步的目的是将标准差为0的部分变为1，这样可以直接使用除法
sig = sig + (sig==0);
for i = 1:totalInstancesNum
    XTrain(:,i,:) = (XTrain(:,i,:) - mu) ./ sig; 
end
% 测试集也得这样操作一下，方便后续预测
for i = 1:testInstancesNum
    XTestTarget(:,i,:) = (XTestTarget(:,i,:) - mu) ./ sig;
end
% 设置网络参数
lgraph = layerGraph(layers);
net = dlnetwork(lgraph);
weights = dlarray(ones(totalInstancesNum,1)/totalInstancesNum,'B');

% two-stage TrAdaBoost R.2
t = 0 ;
maxStep = 10;
% AdaBoost的model error
errorAdaBoost = zeros(maxStep,1);
% instanceWiseErrorLearner
instanceWiseErrorLearner = zeros(totalInstancesNum,maxStep);
DLearner = zeros(maxStep, 1);
maxBoostingIteration = 10;
netsCellsAdaBoost=cell(maxBoostingIteration,maxStep);
BoostingStop = maxBoostingIteration*ones(maxStep,1);% 记录stage-1的终止Iteration
betaAdaBoost = zeros(maxBoostingIteration, maxStep);
while t < maxStep
    t = t+1;
    % 重新设置maxBoostingIteration
    maxBoostingIteration = 10;
    % Stage-2 TradaBoost.R2'
    tempStep = 0;
    % instanceWiseErrorAdaBoost
    instanceWiseErrorAdaBoost = zeros(totalInstancesNum,maxBoostingIteration);
    DAdaBoost = zeros(maxBoostingIteration, 1);
    % adjusted error
    epislonBoost = zeros(maxBoostingIteration, 1);
    % AdaBoost内的weights
    weightsAdaBoost = weights;
    while  tempStep < maxBoostingIteration 
        tempStep = tempStep + 1; 
        % 初始化网络
        net = dlnetwork(lgraph);
        [YTrainPred, net] = ...
            LSTMLearner(net, XTrain, YTrain, maxEpochs, miniBatchSize, weightsAdaBoost); 
        netsCellsAdaBoost{tempStep,t} = net;
        % 范数误差
        errorNorm_temp = sqrt(sum((extractdata(YTrain)-extractdata(YTrainPred)).^2,3));
        DAdaBoost(tempStep) = max(errorNorm_temp);
        % 平方修正
        instanceWiseErrorAdaBoost(:,tempStep) = errorNorm_temp.^2/(DAdaBoost(tempStep)^2);	 
        % 计算 adjusted error
        epislonBoost(tempStep) = sum(instanceWiseErrorAdaBoost(tempStep).*weightsAdaBoost);
        % 计算 betaAdaBoost
        if (epislonBoost(tempStep) > 0.5)
            maxBoostingIteration = tempStep - 1;
            BoostingStop(t,1) = maxBoostingIteration;
            break
        else
            betaAdaBoost(tempStep,t) = epislonBoost(tempStep)...
                                /(1-epislonBoost(tempStep));

            % 更新权重, 前n个样本的权重不变
            weightsAdaBoost(n+1:n+m) = weightsAdaBoost(n+1:n+m).*...
                            (betaAdaBoost(tempStep,t).^...
                            (1-instanceWiseErrorAdaBoost(n+1:n+m,tempStep)));
            normalizationConstant_temp = sum(weightsAdaBoost);
            weightsAdaBoost = weightsAdaBoost/normalizationConstant_temp;
        end
    end

    % 用Target data的误差作为模型误差
    [Ypred_temp,~] = hypothesisAdaBoost(netsCellsAdaBoost(:,t), BoostingStop(t,1),...
                        XTrainTarget, YTrainTarget, betaAdaBoost(:,t));
    errorAdaBoost(t) = l2loss(YTrainTarget, Ypred_temp); 

    % Stage-1 Learner 
    % 用LSTM和weights再得到一个net_Learner 
    % 初始化网络
    net = dlnetwork(lgraph);
    [YTrainPred, net] = ...
            LSTMLearner(net, XTrain, YTrain, maxEpochs, miniBatchSize, weights); 

    % 范数误差
    errorNorm_temp = sqrt(sum((extractdata(YTrain)-extractdata(YTrainPred)).^2,3));
    DLearner(t) = max(errorNorm_temp);
    % 平方修正
    instanceWiseErrorLearner(:,t) = errorNorm_temp.^2/(DLearner(t)^2);
    
    % 更新权重
    beta_temp = updateBeta(n, m, t, maxStep, instanceWiseErrorLearner, weights);
    weights(1:n) = (beta_temp .^instanceWiseErrorLearner(1:n,t)).* weights(1:n);
    normalizationConstant_temp = sum(weights);
    weights = weights/normalizationConstant_temp;
% 
%         如果errorAdaBoost在增加，就跳出循环
    if (t>=2) && (errorAdaBoost(t) >= errorAdaBoost(t-1))
        break
    end
end 
[error,time] = min(errorAdaBoost(1:t));
YTrainTargetPred = hypothesisAdaBoost(netsCellsAdaBoost(:,time), BoostingStop(time,1),...
                        XTrainTarget, YTrainTarget, betaAdaBoost(:,time));
lossTrain = l2loss(YTrainTarget,YTrainTargetPred);
plotfunc(YTrainTarget, YTrainTargetPred);
YTestTargetPred = hypothesisAdaBoost(netsCellsAdaBoost(:,time), BoostingStop(time,1),...
                        XTestTarget, YTestTarget, betaAdaBoost(:,time));
lossTest = l2loss(YTestTarget,YTestTargetPred);
plotfunc(YTestTarget, YTestTargetPred);

function [YTrainPred, net] ...
            = LSTMLearner(net, XTrain, YTrain,...
                          maxEpochs, miniBatchSize, weights)
    % 设置迭代参数
    numEpochs = maxEpochs;
    numObservations = numel(YTrain(1,:,1));
    numIterationsPerEpoch = floor(numObservations./miniBatchSize);
    numIterations = numEpochs * numIterationsPerEpoch;

    % 设置monitor，显示进度
    monitor = trainingProgressMonitor;
    monitor.Info = ["LearningRate","Epoch","Iteration"];
    monitor.Metrics = "TrainingLoss";
    monitor.XLabel = "Iteration";
    monitor.Visible = "off";
    groupSubPlot(monitor,"Loss","TrainingLoss");

    % 使用adam算法更新loss和gradient,此处是初始化一下数据
    averageGrad = [];
    averageSqGrad = [];

    % 训练网络
    iteration = 0;
    epoch = 0;

    while epoch < numEpochs && ~monitor.Stop
        epoch = epoch + 1;
    
        % shuffle data.
        idxTest = randperm(numel(YTrain(1,:,1)));
        XTrain = XTrain(:,idxTest,:);
        YTrain = YTrain(:,idxTest,:);
    
        i = 0;
        while i < numIterationsPerEpoch && ~monitor.Stop
            i = i + 1;
            iteration = iteration + 1;
    
            % 读取miniBatch
            idxTest = (i-1)*miniBatchSize+1:i*miniBatchSize;
            X = XTrain(:,idxTest,:);
            Y = YTrain(:,idxTest,:);
      		% 计算带权重误差，并训练
            weights_temp = dlarray(weights(idxTest),'B');
            [lossTrain,gradients] = dlfeval(@modelLoss,net,X,Y,weights_temp);
            % adam更新
            learnRate = 1e-3;
            [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,...
                                                averageGrad,averageSqGrad,iteration,...
                                                learnRate);
    
            % 更新监控窗格
            recordMetrics(monitor,iteration,...
                          TrainingLoss = lossTrain);
            updateInfo(monitor,Epoch = epoch + " of " + numEpochs);
            monitor.Progress = 100 * iteration/numIterations;
        end
    end
    % 计算结果
    YTrainPred = predict(net, XTrain);
end 

% modelloss 
function [loss,gradients] = modelLoss(net,X,T, weight)

    % 正向传播计算输出.
    Y = forward(net,X);

    % 计算带权重损失.
    loss = l2loss(Y,T,weight);

    % 计算梯度.
    gradients = dlgradient(loss,net.Learnables);

end