-- fintuning resnet
require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
require 'nn'
require 'cudnn'
require 'cunn'
local matio = require 'matio'
require 'gnuplot'
require 'loadcaffe'
require 'rnn'

----------------------------------------------------------------------

local cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training an LSTM network for tool detection')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Training Parameters')
cmd:option('-LR',                 0.001,                    'learning rate')
cmd:option('-LRDecay',            0,                      'learning rate decay (in # samples)')
cmd:option('-weightDecay',        1e-4,                   'L2 penalty on the weights')
cmd:option('-momentum',           0.9,                    'momentum')
cmd:option('-batchSize',          256,                    'batch size')
cmd:option('-optimization',       'sgd',                  'optimization method')
cmd:option('-epoch',              2000,                     'number of epochs to train, -1 for unbounded')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            12,                      'number of threads')
cmd:option('-type',               'cuda',                 'cuda/cl/float/double')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                      'num of gpu devices used')
cmd:option('-constBatchSize',     false,                  'do not allow varying batch sizes - e.g for ccn2 kernel')

cmd:text('===>Save Options')
cmd:option('-save',               'Results', 'save directory')

cmd:text('===>Misc')
cmd:option('-visualize',          1,                      'visualizing results')

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
model= nn.Sequential()

model:add(nn.SeqLSTM(1024,1024))
model:add(nn.Sequencer(nn.NormStabilizer()))
model:add(nn.SeqLSTM(1024,256))
model:add(nn.Sequencer(nn.NormStabilizer()))
model:add(nn.SeqLSTM(256,8))
model:add(nn.Sequencer(nn.NormStabilizer()))
model:add(nn.SoftSign())

collectgarbage()

print(model)
local loss = nn.MultiLabelSoftMarginCriterion()

print('loading training data ...')
local TrainData = matio.load('lstm_data/lstm_trainData.mat','lstm_trainData'):float()
local TrainLabel = matio.load('lstm_data/lstm_trainLabel.mat','lstm_trainLabel') 
collectgarbage()

print('Training data loaded!')
print('Loading validation data...')
TestData = matio.load('lstm_data/lstm_valData.mat','lstm_valData'):float()
TestLabel = matio.load('lstm_data/lstm_valLabel.mat','lstm_valLabel') 
print('Validation data loaded!')
collectgarbage()

classes = {'no_tool','tool1','tool2','tool3','tool4','tool5','tool6','tool7'}
----------------------------------------------------------------------
local confusion = optim.ConfusionMatrix(8,classes)
local AllowVarBatch = not opt.constBatchSize
----------------------------------------------------------------------
-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
cmd:log(opt.save .. '/Log.txt', opt)
local netFilename = paths.concat(opt.save, 'Net_lstm')
local logFilename = paths.concat(opt.save,'ErrorRate.log')
local optStateFilename = paths.concat(opt.save,'optState')
local Log = optim.Logger(logFilename)
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
print(model)
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
------------------------------
local function Forward(Data,Label,train)
   
  local SizeData = Data:size(2)
  SizeBatch = math.floor(Data:size(2)/opt.batchSize)

  local yt,x
  --local NumBatches = 1
  local lossVal = 0
  local num = 1;
  for NumBatches=1,SizeBatch do
   	yt = Label[{{},{num,num+opt.batchSize-1},{}}]:cuda()
  	 x = Data[{{},{num,num+opt.batchSize-1},{}}]:cuda()
  	 -- MiniBatch:getNextBatch()
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
    y_temp = torch.Tensor(y:size(1)*y:size(2),y:size(3)):fill(0)
    yt_temp = torch.Tensor(y:size(1)*y:size(2),y:size(3)):fill(0)
    t = 1    
    y = y:float()
    yt = yt:float()
    for i = 1,y:size(1) do
      y_temp[{{t,t+y:size(2)-1},{}}] = torch.reshape(y[{{i},{},{}}],y:size(2),y:size(3))
      yt_temp[{{t,t+yt:size(2)-1},{}}] = torch.reshape(yt[{{i},{},{}}],yt:size(2),yt:size(3))
      t = t+y:size(2)
    end
    confusion:batchAdd(y_temp,yt_temp)
    xlua.progress(NumBatches, SizeBatch)
    if math.fmod(NumBatches,100)==0 then
      collectgarbage()
    end
    num = num + opt.batchSize
  end

  if(Data:size(2)%opt.batchSize ~= 0) then
    yt = Label[{{},{num,Data:size(2)},{}}]:cuda()
    x = Data[{{},{num,Data:size(2)},{}}]:cuda()
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
      y = y:float()
      yt = yt:float()
      y_temp = torch.Tensor(y:size(1)*y:size(2),y:size(3)):fill(0)
      yt_temp = torch.Tensor(y:size(1)*y:size(2),y:size(3)):fill(0)
      t = 1
      for i = 1,y:size(1) do
        y_temp[{{t,t+y:size(2)-1},{}}] = torch.reshape(y[{{i},{},{}}],y:size(2),y:size(3))
        yt_temp[{{t,t+yt:size(2)-1},{}}] = torch.reshape(yt[{{i},{},{}}],yt:size(2),yt:size(3))
        t = t+y:size(2)
      end
      confusion:batchAdd(y_temp,yt_temp)
    end
    collectgarbage()   
  return(lossVal/math.ceil(Data:size(2)/opt.batchSize))
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
LR = opt.LR;
factor = 0.1;
sumError = 0;
count = 0;
ErrTrain1 = torch.Tensor(1,1)
ErrTest1 = torch.Tensor(1,1)
gnuplot.pngfigure(paths.concat(opt.save..'/losscurve.png'))
while epoch ~= opt.epoch do
  timer =torch.Timer()
  print('Epoch ' .. epoch .. '/' .. opt.epoch)
  confusion:zero()  
  local LossTrain = Train(TrainData,TrainLabel)    
  confusion:updateValids()

  torch.save(netFilename, model:clearState())
  local ErrTrain = (1-confusion.totalValid)
  if #classes <= 10 then
    print(confusion)
  end
  print('Training Error = ' .. ErrTrain)
  print('Training Loss = ' .. LossTrain)
  
  confusion:zero()
  local LossTest = Test(TestData,TestLabel)
  confusion:updateValids()
  local ErrTest = (1-confusion.totalValid)
  if #classes <= 10 then
    print(confusion)
  end

  print('Validation Error = ' .. ErrTest)
  print('Validation Loss = ' .. LossTest)
  ErrTrain1[{{1},{1}}] = LossTrain;
  ErrTest1[{{1},{1}}] = LossTest;
  if epoch == 0 then
    trainErrPlot = ErrTrain1;
    valErrPlot = ErrTest1;
  else
    trainErrPlot = torch.cat(trainErrPlot,ErrTrain1,2)
    valErrPlot = torch.cat(valErrPlot,ErrTest1,2)
    
    gnuplot.plot({'Training error',trainErrPlot[1]},{'Validation error',valErrPlot[1]})
    gnuplot.xlabel('Epoch')
    gnuplot.ylabel('Training error')
    gnuplot.grid(true)
    gnuplot.title('Plot of error vs. epochs')
    torch.save(paths.concat(opt.save..'/trainloss.t7'),trainErrPlot)
    torch.save(paths.concat(opt.save..'/valloss.t7'),valErrPlot)
  end

Log:add{['Training Loss']= ErrTrain, ['Validation loss'] = ErrTest}
epoch = epoch + 1
tmp=epoch-1
tm=timer:time().real

print('The time taken for epoch--'..tmp..'--is --'..tm..' seconds')
timer:reset()
end
gnuplot.plotflush()
if opt.visualize == 1 then
        Log:style{['Training loss'] = '-', ['Validation loss'] = '-'}
        Log:plot()
end