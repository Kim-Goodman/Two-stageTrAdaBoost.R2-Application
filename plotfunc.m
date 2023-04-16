function plotfunc(Y, YPred)
    % 画图展示部分预测结果 % 训练集
    Idx = 1:numel(Y(1,:,1));
    idxTemp = randperm(numel(Idx),4);
    M=32;
    a=-pi;b=pi;
    h=(b-a)/M;
    x=linspace(a,b-h,M)';
    figure
    for i = 1:numel(idxTemp)
        subplot(2,2,i)
        y1 = Y(1,idxTemp(i),:);
        plot(x, y1(:),'--',LineWidth=1)
        hold on
        y2 = YPred(1,idxTemp(i),:);
        plot(x, y2(:),'.-',LineWidth=1)
        hold off
        
        set (gca, 'FontSize', 14)
        title("Test Sample " + Idx(idxTemp(i)))
        xlabel("x")
        ylabel("\psi(x,T)")
    end
    legend(["Test Data" "Predicted"],'Location','southeast')
end 