% 构建网络,numResponses是输出的sequence的特征个数
numFeatures = 2;
numResponses = 1;
numSequences = 32;
numHiddenUnits = 50; % 隐藏单元的个数
maxEpochs = 100;
plotTrainFlag = 1; % 是否画训练集结果
plotTestFlag = 1; % 是否画测试集结果
layers = [ ...
    sequenceInputLayer(numFeatures,'Name','input')
    lstmLayer(numHiddenUnits,'OutputMode','sequence','Name','LSTM')
    fullyConnectedLayer(50, 'Name','FCL1')
    dropoutLayer(0.5,'Name','drop')
    fullyConnectedLayer(numResponses,'Name','FCL2')];
% 选取训练数据
sourceType = 3;
funcFlag = 1;
instancesNum = 60;  % 不超过500
testInstancesNum = floor(instancesNum/5);
miniBatchSize = instancesNum/5;  % 不超过instancesNum
load('NPIdata.mat')
randIdxTrain = randperm(300,instancesNum); 
randIdxTest = 1:300;
randIdxTest(ismember(randIdxTest,randIdxTrain)==1)= [];  % 取剩余部分的部分为测试集
XTrain = NPIdataX{sourceType,funcFlag}(:,randIdxTrain,:);
YTrain = NPIdataY{sourceType,funcFlag}(:,randIdxTrain,:);
XTest = NPIdataX{sourceType,funcFlag}(:,randIdxTest(1:testInstancesNum),:);
YTest = NPIdataY{sourceType,funcFlag}(:,randIdxTest(1:testInstancesNum),:);
% 归一化特征
mu = mean(XTrain,2);
sig = std(XTrain,0,2);
% 下面这步的目的是将标准差为0的部分变为1，这样可以直接使用除法
sig = sig + (sig==0);
for i = 1:instancesNum
    XTrain(:,i,:) = (XTrain(:,i,:) - mu) ./ sig; 
end
% 测试集也得这样操作一下，方便后续预测
for i = 1:testInstancesNum
    XTest(:,i,:) = (XTest(:,i,:) - mu) ./ sig;
end
% 设置训练参数
lgraph = layerGraph(layers);
net = dlnetwork(lgraph);
numEpochs = maxEpochs;
numObservations = numel(YTrain(1,:,1));
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;
% 显示进度
monitor = trainingProgressMonitor;
monitor.Info = ["LearningRate","Epoch","Iteration"];
monitor.Metrics = "TrainingLoss";
monitor.XLabel = "Iteration";
groupSubPlot(monitor,"Loss","TrainingLoss");
% 使用adam算法更新loss和gradient,此处是初始化一下数据
averageGrad = [];
averageSqGrad = [];
% 训练网络
iteration = 0;
epoch = 0;
weight = dlarray(ones(miniBatchSize,1)/miniBatchSize,'B');  % l2loss中各样本的权重
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Shuffle data.
    idxTest = randperm(numel(YTrain(1,:,1)));
    XTrain = XTrain(:,idxTest,:);
    YTrain = YTrain(:,idxTest,:);

    i = 0;
    while i < numIterationsPerEpoch && ~monitor.Stop
        i = i + 1;
        iteration = iteration + 1;

        % 读取miniBatch.
        idxTest = (i-1)*miniBatchSize+1:i*miniBatchSize;
        X = XTrain(:,idxTest,:);
        Y = YTrain(:,idxTest,:);
        % 计算带权重误差，并训练
        [lossTrain,gradients] = dlfeval(@modelLoss,net,X,Y,weight);
        % adam更新.
        [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration);

        % 更新监控窗格.
        recordMetrics(monitor,iteration,...
            TrainingLoss=lossTrain);
        updateInfo(monitor,Epoch = epoch + " of " + numEpochs);
        monitor.Progress = 100 * iteration/numIterations;
    end
end
YTrainPred = predict(net, XTrain);
Trainloss = l2loss(YTrain,YTrainPred);
plotfunc(YTrain, YTrainPred);
YTestPred = predict(net, XTest);
Testloss = l2loss(YTest,YTestPred);
plotfunc(YTest, YTestPred);
function [loss,gradients] = modelLoss(net,X,T, weight)

    % 正向传播计算输出.
    Y = forward(net,X);

   	% 计算带权重损失.
    loss = l2loss(Y,T,weight);

    % 计算梯度.
    gradients = dlgradient(loss,net.Learnables);

end