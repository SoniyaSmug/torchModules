require 'optim' 
require 'nn'
dofile('loadData.lua')
dofile('trainTest.lua')



-- process command line options
cmd = torch.CmdLine()
cmd:option('-dataset', 'MNIST', 'dataset to train on: MNIST | CIFAR-10 | SVHN')
cmd:option('-model','linear', 'type of model to construct: linear | mlp | convnet')
cmd:option('-batchSize', 1, 'mini-batch size (1=pure stochastic)')
cmd:option('-epochs', 5, 'number of passes over the dataset')
cmd:option('-plot',false,'plot intermediate results')
cmd:option('-type','double','type: double | float | cuda')

-- parse options
opt = cmd:parse(arg or {})
print(opt)

--------------------------------------------------------------------------------------
-- LOAD DATASET
--------------------------------------------------------------------------------------

if opt.dataset == 'MNIST' then
	trdata,trlabels = loadData('MNIST','train')
	tedata,telabels = loadData('MNIST','test')
	nInputs = trdata:size(2)

elseif opt.dataset == 'CIFAR-10' then
	trdata,trlabels,tedata,telabels = loadData('CIFAR-10')
	nfeats = trdata:size(2)
	height = trdata:size(3)
	width = trdata:size(4)
	nInputs = nFeats*width*height
end

classes = {'1','2','3','4','5','6','7','8','9','10'}
nOutputs = 10
nInputs = trdata:size(2)

--------------------------------------------------------------------------------------
-- MODEL
--------------------------------------------------------------------------------------

if opt.model == 'linear' then
	model = nn.Sequential()
	model:add(nn.Reshape(nInputs))
	model:add(nn.Linear(nInputs,nOutputs))

elseif opt.model == 'mlp' then
	nHidden = 20
	model = nn.Sequential()
	model:add(nn.Reshape(nInputs))
	model:add(nn.Linear(nInputs,nHidden))
	model:add(nn.Tanh())
	model:add(nn.Linear(nHidden,nOutputs))

elseif opt.model == 'convnet' then

	-- number of hidden units at different layers
	nstates = {64,64,128}
	filtsize = 5
	poolsize = 2
	normkernel = image.gaussian1D(7)
   model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
   model:add(nn.Tanh())
   model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
   model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
   model:add(nn.Tanh())
   model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
   model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

   -- stage 3 : standard 2-layer neural network
   model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
   model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
   model:add(nn.Tanh())
   model:add(nn.Linear(nstates[3], nOutputs))

else 
	error('unrecognized model type')
end		
		
-- loss

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

------------------------------------------------------------------------------------------
-- OPTIMIZATION 
------------------------------------------------------------------------------------------

optimMethod = optim.sgd
optimState = {
	learningRate = 0.02,
	weightDecay = 0,
	learningRateDecay = 1e-7
}

opt.optimState = optimState
opt.optimMethod = optimMethod
opt.classes = classes



------------------------------------------------------------------------------------------
-- TRAIN
------------------------------------------------------------------------------------------

-- create logger
trainLogger = optim.Logger('train.log')
testLogger = optim.Logger('test.log')


for i = 1,opt.epochs do
	trainModel(model, criterion, trdata, trlabels, opt, trainLogger)
	testModel(model, tedata, telabels, opt, testLogger)
end



