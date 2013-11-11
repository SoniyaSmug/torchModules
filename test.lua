require 'image'
require 'nn'
require 'cutorch'
require 'cunn'

dofile('FeatureNormalization.lua')
cutorch.setDevice(3)

nSamples=128
nFeatures=300
nRows=20
nCols=20
nTrials=5

X={}
for i=1,nTrials do
	X[i]=torch.randn(nSamples,nFeatures,nRows,nCols)
end
m=nn.FeatureNormalization(X[1]:size())


t1=torch.Timer()
for i=1,nTrials do
	X1=m:forward(X[i])
	--X1=m:updateOutput(X[i])
end
print('Time elapsed for CPU: ' .. t1:time().real/nTrials .. ' seconds')



m=m:cuda()
Xc={}
for i=1,nTrials do
	Xc[i]=X[i]:cuda()
end
cutorch.synchronize()
t2=torch.Timer()
for i=1,nTrials do
	X1=m:forward(Xc[i])
	--X1=m:updateOutput(Xc[i])
end
cutorch.synchronize()
print('Time elapsed for GPU: ' .. t2:time().real/nTrials .. ' seconds')

