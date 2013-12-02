require 'nn'
require 'optim'

dofile('AdaBias.lua')
dofile('stats.lua')
dofile('loadData.lua')
torch.manualSeed(123)
torch.setdefaulttensortype('torch.FloatTensor')

-- model params
nHidden=5
lambda=0.8
batchSize=10
nClasses=10
-- choose 1 for hard centering
centering=1

-- load MNIST
nSamples=1000
data,targets=loadData('MNIST','train',nSamples)

-- make model
nInputs=data:size(2)
nHidden=5
model=nn.Sequential()
model:add(nn.Linear(nInputs,nHidden))
model:add(nn.AdaBias(nHidden,centering,lambda,alpha))
model:add(nn.Threshold())
model:add(nn.Linear(nHidden,nClasses))
model:add(nn.LogSoftMax())
criterion=nn.ClassNLLCriterion()

-- define gradient evaluation 
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

-- optimization parameters
sgd_params = {
   learningRate = 1e-3,
   learningRateDecay = 1e-4,
   weightDecay = 0,
   momentum = 0
}
epochs=20

-- train!
for i=1,epochs do
	current_loss=0
	for j=1,nSamples do
		_,fs=optim.sgd(feval,w,sgdParams)
		current_loss = current_loss + fs[1]
	end
	current_loss=current_loss/nSamples
	print('epoch=' .. i .. ', loss=' .. current_loss .. '\n')
end

-- plot some distributions
m1=nn.Sequential():add(model.modules[1]):add(model.modules[2])
d1=m1:forward(data)
gnuplot.hist(d1[{{},1}])

-- see what proportion of samples is non-zero
m2=nn.Sequential():add(m1):add(model.modules[3])
d2=m2:forward(data)
a=torch.Tensor(nHidden)
for i=1,nHidden do
	a[i]=torch.sum(torch.gt(d2[{{},i}],0))/d2:size(1)
end
print('Proportion of samples set to 0:')
print(a)


























--exit()
