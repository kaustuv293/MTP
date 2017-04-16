require 'nn'
require 'cudnn'
require 'cunn'
require 'rnn'

matio = require 'matio'

net = torch.load('Net_cnn')
model= net:get(1)
-- for googlenet
model:remove(25)
model:remove(24)

--extract features
print('Loading data')
TestData = torch.div(matio.load('data/trainData.mat','trainData'):float(),255)
print('data loaded!')


local Net_pred = torch.Tensor(TestData:size(1),2048)
batchSize = 32
local SizeData = TestData:size(1)
SizeBatch = math.floor(TestData:size(1)/batchSize)

  local yt,x;
  local num = 1;
  for NumBatches=1,SizeBatch do	    
    print(NumBatches .. '/' .. SizeBatch)
      x = TestData[{{num,num+batchSize-1},{},{},{}}]:cuda()    
      y = model:forward(x)	    
      Net_pred[{{num,num+batchSize-1},{}}] = y:float()
      num = num + batchSize
   end

   if(TestData:size(1)%batchSize ~= 0) then	    
      x = TestData[{{num,TestData:size(1)},{},{},{}}]:cuda()
      y = model:forward(x)
      Net_pred[{{num,TestData:size(1)},{}}] = y:float()
  end

matio.save('Net_cnn_feat.mat',Net_pred); 

