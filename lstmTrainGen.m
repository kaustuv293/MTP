% code to generate lstm train data from cnn feature matrix

load Results/Net_cnn_feat
CNNFeatures=x;
lstm_trainData = zeros(size(CNNFeatures,1)-2,3,2048);
for tripletNum = 1:size(CNNFeatures,1)-2    
    temp = reshape(CNNFeatures(tripletNum:tripletNum+2,:),[1 3 2048]);
    lstm_trainData(tripletNum,:,:) = temp;
end
lstm_trainData = permute(lstm_trainData,[2 1 3]);
save 'lstm_trainData' 'lstm_trainData' '-v7.3'

%%
% generate lstm label
load trainLabel

lstm_trainLabel = zeros(size(lstm_trainLabel,1)-2,3,8);
for tripletNum = 1:size(lstm_trainLabel,1)-2
    temp = reshape(lstm_trainLabel(tripletNum:tripletNum+2,:),[1 3 8]);
    lstm_trainLabel(tripletNum,:,:) = temp;
end
lstm_trainLabel = permute(lstm_trainLabel,[2 1 3]);
save 'lstm_trainLabel' 'lstm_trainLabel'

