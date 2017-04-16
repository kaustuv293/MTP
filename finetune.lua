require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
require 'nn'
require 'cudnn'
require 'hdf5'
require 'gnuplot'
require 'cunn'
----------------------------------------------------------------------
local cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Finetuning a convolutional network for tool detection')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Training Parameters')
cmd:option('-LR',                 0.001,                    'learning rate')
cmd:option('-LRDecay',            0,                      'learning rate decay (in # samples)')
cmd:option('-weightDecay',        1e-4,                   'L2 penalty on the weights')
cmd:option('-momentum',           0.9,                    'momentum')
cmd:option('-batchSize',          64,                    'batch size')
cmd:option('-optimization',       'sgd',                  'optimization method')
cmd:option('-epoch',              100,                     'number of epochs to train, -1 for unbounded')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                      'number of threads')
cmd:option('-type',               'cuda',                 'cuda/cl/float/double')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                      'num of gpu devices used')
cmd:option('-constBatchSize',     false,                  'do not allow varying batch sizes - e.g for ccn2 kernel')

cmd:text('===>Save Options')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')

cmd:text('===>Misc')
cmd:option('-visualize',          0,                      'visualizing results')

opt = cmd:parse(arg or {})
opt.save = paths.concat('./Results', opt.save)
os.execute('mkdir -p ' .. opt.preProcDir)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
-- Model + Loss:
local model
if paths.filep(opt.load) then
  pcall(require , 'cunn')
  pcall(require , 'cudnn')
end

model=torch.load('inception.t7')

local loss = nn.MultiLabelSoftMarginCriterion()
---------------------------Loading training data------------------------
print('loading training data...')

local trainDatafile = hdf5.open('v3data/traindata/tooltrainData.h5','r')
local TrainData1 = trainDatafile:read('/data',trainData):all()
trainDatafile:close()
TrainData1 = TrainData1:transpose(1,4):transpose(2,3)
shuffleIdx = torch.randperm(TrainData1:size(1))
TrainData = TrainData1[{{shuffleIdx},{},{},{}}]
TrainData1 = nil
print('training data loaded..')
print('loading train label ...')
local trainLabelfile = hdf5.open('v3data/trainlabel/tooltrainLabel.h5','r') 
local TrainLabel1 = trainLabelfile:read('/label',trainLabel):all()
trainLabelfile:close()
TrainLabel1 = TrainLabel1:transpose(1,2)
TrainLabel = TrainLabel1[{{shuffleIdx},{}}]
TrainLabel1 = nil
collectgarbage()
print('Training label loaded!')
-------------Loading validation data-----------------------------------

print('Loading validation data...')
local valDatafile = hdf5.open('v3data/valdata/toolvalData.h5','r')
local TestData = valDatafile:read('/data',valData):all()
valDatafile:close()
TestData = TestData:transpose(1,4):transpose(2,3)
print('validation data loaded!')
print('loading validation label ...')
local valLabelfile = hdf5.open('v3data/vallabel/toolvalLabel.h5','r')
local TestLabel = valLabelfile:read('/label',valLabel):all()
valLabelfile:close()
TestLabel = TestLabel:transpose(1,2)
print('validation label loaded!')
collectgarbage()


classes = {'no_tool','tool1','tool2','tool3','tool4','tool5','tool6','tool7'}
----------------------------------------------------------------------

local confusion = optim.ConfusionMatrix(8,classes)
local AllowVarBatch = not opt.constBatchSize

----------------------------------------------------------------------
os.execute('mkdir -p ' .. opt.save)
local netFilename = paths.concat(opt.save, 'Net_cnn')
----------------------------------------------------------------------

local types = {
  cuda = 'torch.CudaTensor',
  float = 'torch.FloatTensor',
  cl = 'torch.ClTensor',
  double = 'torch.DoubleTensor'
}

local TensorType = types[opt.type] or 'torch.FloatTensor'
if opt.type == 'cuda' then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.devid)
    local cudnnAvailable = pcall(require , 'cudnn')
    print(model)
    if cudnnAvailable then
      model = cudnn.convert(model, cudnn)
    end   
elseif opt.type == 'cl' then
    require 'cltorch'
    require 'clnn'
    cltorch.setDevice(opt.devid)
end
model:type(TensorType)
loss = loss:type(TensorType)

---Support for multiple GPUs 
if opt.nGPU > 1 then
    local net = model
    model = nn.DataParallelTable(1)
    for i = 1, opt.nGPU do
        cutorch.setDevice(i)
        model:add(net:clone():cuda(), i)  -- Use the ith GPU
    end
    cutorch.setDevice(opt.devid)
end

-- Optimization configuration
local Weights,Gradients = model:getParameters()

----------------------------------------------------------------------
print '==> Network'
print('==>' .. Weights:nElement() ..  ' Parameters')

print '==> Loss'
print(loss)

------------------Optimization Configuration--------------------------
local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    dampening = 0,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}
----------------------------------------------------------------------
local function Forward(Data,Label,train)
   
  local SizeData = Data:size(1)
  SizeBatch = math.floor(Data:size(1)/opt.batchSize)

  local yt,x

  local lossVal = 0
  local num = 1;
  for NumBatches=1,SizeBatch do
    yt = Label[{{num,num+opt.batchSize-1}}]:cuda()
    x = torch.div(Data[{{num,num+opt.batchSize-1},{},{},{}}]:float(),255):cuda()      
    local y, currLoss
    y = model:forward(x)   
    currLoss = loss:forward(y,yt)        

    if train then
      local function feval()
        model:zeroGradParameters()
        local dE_dy = loss:backward(y, yt)       
        model:backward(x, dE_dy)        
        return currLoss, Gradients
    end

      _G.optim[opt.optimization](feval, Weights, optimState)
     
      if opt.nGPU > 1 then
        model:syncParameters()
      end
    end

    lossVal = currLoss + lossVal
    
    if type(y) == 'table' then 
      y = y[1]
    end

    confusion:batchAdd(y,yt)
    xlua.progress(NumBatches, SizeBatch)
    if math.fmod(NumBatches,100)==0 then
      collectgarbage()
    end
    num = num + opt.batchSize
  end
  if(Data:size(1)%opt.batchSize ~= 0) then
    yt = Label[{{num,Data:size(1)}}]:cuda()
    x = torch.div(Data[{{num,Data:size(1)},{},{},{}}]:float(),255):cuda()    
    y = model:forward(x)
      currLoss = loss:forward(y,yt)
      if train then
        local function feval()
          model:zeroGradParameters()
          local dE_dy = loss:backward(y, yt)
          model:backward(x, dE_dy)
          return currLoss, Gradients
        end
        _G.optim[opt.optimization](feval, Weights, optimState)
        if opt.nGPU > 1 then
          model:syncParameters()
        end
      end

      lossVal = currLoss + lossVal

      if type(y) == 'table' then 
        y = y[1]
      end

      confusion:batchAdd(y,yt)
    end
    collectgarbage()
  return(lossVal/math.ceil(Data:size(1)/opt.batchSize))
end

------------------------------
local function Train(Data,Label)  
  model:training()
  return Forward(Data,Label, true)
end

local function Test(Data,Label)
  model:evaluate()
  return Forward(Data,Label, false)
end
------------------------------

local epoch = 0
print '\n==> Starting Training\n'

trainLoss = torch.Tensor(1,1)
valLoss = torch.Tensor(1,1)

trainError = torch.Tensor(opt.epoch,1):fill(0)
valError = torch.Tensor(opt.epoch,1):fill(0)


while epoch ~= opt.epoch do   

    print('Epoch ' .. epoch)
    --Train
    confusion:zero()    
    local LossTrain = Train(TrainData,TrainLabel)    
    torch.save(path..'Net', model:clearState())
    confusion:updateValids()
    local ErrTrain = (1-confusion.totalValid)
    if #classes <= 10 then
        print(confusion)
    end
    print('Training Error = ' .. ErrTrain)
    print('Training Loss = ' .. LossTrain)

    --Test
    confusion:zero()
    local LossTest = Test(TestData,TestLabel)
    confusion:updateValids()
    local ErrTest = (1-confusion.totalValid)
    if #classes <= 10 then
        print(confusion)
    end

    print('Validation Error = ' .. ErrTest)
    print('Validation Loss = ' .. LossTest)
    
    trainLoss[{{1},{1}}] = LossTrain
    valLoss[{{1},{1}}] = LossTest
    if epoch == 0 then
      trainLossPlot = trainLoss
      valLossPlot = valLoss
    else
      trainLossPlot = torch.cat(trainLossPlot,trainLoss,2)
      valLossPlot = torch.cat(valLossPlot,valLoss,2)
      torch.save('trainLoss.t7',trainLossPlot)
      torch.save('valLoss.t7',valLossPlot)      
    end

    
    
    epoch = epoch + 1
    trainError[{{epoch},{1}}] = ErrTrain
    valError[{{epoch},{1}}] = ErrTest
   
end

torch.save('trainError.t7',trainError)
torch.save('valError.t7',valError)

gnuplot.pngfigure('ErrorvsEpochs.png')
gnuplot.figure()
gnuplot.plot({'Training error',trainError[1]},{'Validation error',valError[1]})
gnuplot.xlabel('Epochs')
gnuplot.ylabel('Error')
gnuplot.grid(true)
gnuplot.title('Plot of error vs. epochs')
gnuplot.plotflush()
