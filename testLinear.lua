require 'nn'
require 'optim'

dofile('LinearCentered.lua')
dofile('stats.lua')
torch.manualSeed(123)
torch.setdefaulttensortype('torch.FloatTensor')

--[[
nSamples=2
nDim=2
nOutputs=30
lambda=0.5
nRuns=200

-- check running mean converges to true mean
model = nn.LinearCentered(nDim,nOutputs,lambda)
mean=torch.zeros(nRuns,nDim)
for i=1,nRuns do
	data=torch.randn(nSamples,nDim)
	data:add(5)
	_,mean[i]=model:updateOutput(data)
end
gnuplot.plot(mean[{{},1}])
--]]

-- simple classification example: classify 2 gaussians
--[[
nSamples=500
lambda=0.8
batchSize=10
mean1 = torch.Tensor(1,2):fill(3)
mean2 = torch.Tensor(1,2):fill(-3)
cov1 = torch.Tensor({{1,0},{0,1}})*1
cov2 = torch.Tensor({{1,0},{0,1}})*2
data=torch.Tensor(nSamples,2)
data[{{1,nSamples/2},{}}]=torch.randn(nSamples/2,2)*torch.potrf(cov1)+torch.expand(mean1,nSamples/2,2)
data[{{nSamples/2+1,nSamples},{}}]=torch.randn(nSamples/2,2)*torch.potrf(cov2)+torch.expand(mean2,nSamples/2,2)
targets=torch.zeros(nSamples)
targets[{1,nSamples/2}]=1
targets=targets+1
--]]

dataPath = '/home/mikael/Work/Data/mnist-torch7/train_28x28.th7nn'


-- load data into CPU memory
train_data = torch.load(dataPath)

-- -- hack Reshape
nInputs = train_data['data']:size(2)*train_data['data']:size(3)*train_data['data']:size(4)
nSamples = train_data['data']:size(1)
train_data.data:resize(nSamples,nInputs)
data=train_data.data
targets=train_data['labels']



nHidden=5
lambda=0.8
batchSize=50
nClasses=10
nSamples=5000
data=data[{{1,nSamples},{}}]
targets=targets[{{1,nSamples},{}}]



-- make model
model=nn.Sequential()
model:add(nn.LinearCentered(nInputs,nHidden,lambda))
model:add(nn.Threshold())
model:add(nn.Linear(nHidden,nClasses))
model:add(nn.LogSoftMax())
criterion=nn.ClassNLLCriterion()

-- train 
w,dL_dw=model:getParameters()

feval = function(w_new)
	_nidx_ = (_nidx_ or 0) + 1
	if _nidx_ > nSamples-batchSize then _nidx_ = 1 end
	local inputs=data[{{_nidx_,_nidx_+batchSize-1},{}}]
	--local inputs=data[_nidx_]
	local target=targets[_nidx_]
	local target=targets[{{_nidx_,_nidx_+batchSize-1},{}}]
	dL_dw:zero()
	local loss_w = criterion:forward(model:forward(inputs),target)
	model:backward(inputs,criterion:backward(model.output,target))
	return loss_w,dL_dw
end


sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}

epochs=5

for i=1,epochs do
	current_loss=0
	for j=1,nSamples do
		_,fs=optim.sgd(feval,w,sgdParams)
		current_loss = current_loss + fs[1]
	end
	current_loss=current_loss/nSamples
	print('epoch=' .. i .. ', loss=' .. current_loss .. '\n')
end























--exit()
