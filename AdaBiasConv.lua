dofile('stats.lua')

local AdaBiasConv, parent = torch.class('nn.AdaBiasConv','nn.Module')

function AdaBiasConv:__init(nInputPlanes, iW, iH, lambda, alpha)
	parent:__init(self)
	
	self.nInputPlanes = nInputPlanes
	self.iW = iW
	self.iH = iH
	self.bias = torch.Tensor(nInputPlanes, iW, iH)
	self.mean = torch.Tensor(nInputPlanes)
	self.variance = torch.Tensor(nInputPlanes)
	self.lambda = torch.Tensor(nInputPlanes):fill(lambda)
	self.alpha = alpha or 1e-3
	self.centering = centering
	self:reset()
end

function AdaBiasConv:reset()
	self.bias:zero()
	self.mean:zero()
	self.variance:zero()
end

function AdaBiasConv:updateOutput(input)
	self.output:resize(self.nInputPlanes,self.iW,self.iH)
	self.output:copy(input)
	for i = 1,self.nInputPlanes do
		self.output[{i,{},{}}]:add(self.bias[i])
	end
	return self.output
end

function AdaBiasConv:updateGradInput(input, gradOutput)
	if self.gradInput then 
		self.gradInput:resizeAs(input)
		self.gradInput:copy(gradOutput)
		return self.gradInput
	end
end


function AdaBiasConv:accGradParameters(input, gradOutput, scale)
	scale = scale or 1
	-- update running mean and variance
	self.mean:mul(1-self.alpha)
	self.variance:mul(1-self.alpha)
	for i = 1,self.nInputPlanes do
		local inputPlane = input[{i,{},{}}]
		self.mean[i] = self.mean[i] + self.alpha*torch.mean(inputPlane)
		self.variance[i] = self.variance[i] + self.alpha*torch.mean(torch.pow(inputPlane-self.mean[i],2))
		--self.variance[i]:add(self.alpha,torch.mean(torch.pow(inputPlane - self.mean[i],2)))
	end
	-- set the bias
	self.bias = inormcdf(self.lambda, self.mean, torch.sqrt(self.variance))
end
	

function AdaBiasConv:parameters()
	return 
end































