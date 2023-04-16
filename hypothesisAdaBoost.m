% 计算加权中位数(向量)
% 或选取最好的网络结果
function [weightsMedian, bestHypothesis] = hypothesisAdaBoost(netsCells, BoostingStop, X, Y, betaAdaBoost)
    sumBetaMedian = 1/2*sum(log(1./betaAdaBoost(1:BoostingStop)));
    % 初始化大小
    weightsMedian = Y;
    bestHypothesis = Y;
    % 每一个样本都得计算
    for i = 1:size(X,2)
        xTemp = X(:,i,:);
        yTemp = Y(:,i,:);
        norm_betaAdaBoost = zeros(BoostingStop, 2);
        norm_betaAdaBoost(:,2) = betaAdaBoost(1:BoostingStop);
        yPredict = cell(BoostingStop, 1);
        errorTemp = zeros(BoostingStop, 1);
        for t=1:BoostingStop
            net = netsCells{t,1};
            yPredict{t,1} = predict(net,xTemp);
            errorTemp(t,1) = l2loss(yTemp,yPredict{t,1});
            % 计算范数
            norm_betaAdaBoost(t,1) = sqrt(sum(extractdata(yPredict{t,1}).^2,3));
        end
        [~,loc] = min(errorTemp);
        bestHypothesis(:,i,:) = yPredict{loc,1};
        % 按范数升序排列
        [~, tempIndex] = sortrows(norm_betaAdaBoost,1);
        tempSum = 0;
        count = 0;
        while (tempSum < sumBetaMedian) && (count < BoostingStop) 
            count = count+1;
            tempSum = tempSum + log(1/betaAdaBoost(count));
        end
        weightsMedian(:,i,:) = yPredict{tempIndex(count),1};
    end 
end 