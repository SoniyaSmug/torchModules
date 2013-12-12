dofile('stats.lua')

local AdaBias, parent = torch.class('nn.AdaBias','nn.Module')

function AdaBias:__init(inputSize, centering, lambda, alpha)
	parent.__init(self)

	self.bias = torch.Tensor(inputSize)
	if centering == 2 then
		self.gradBias = torch.Tensor(inputSize)
	end
	self.sampleMean = torch.Tensor(inputSize)
	self.sampleVariance = torch.Tensor(inputSize)
	self.mean = torch.Tensor(inputSize)
	self.variance = torch.Tensor(inputSize)
	self.lambda = torch.Tensor(inputSize):fill(lambda)
	self.alpha = alpha or 1e-3
	self.centering = centering
	self:reset()
end

function AdaBias:reset()
	self.bias:zero()
	self.mean:zero()
	self.variance:zero()
end

function AdaBias:updateOutput(input)
	if input:dim() == 1 then
		self.output:resize(self.bias:size(1))
		self.output:copy(input)
      self.output:add(self.bias)
	elseif input:dim() == 2 then
		local nframe = input:size(1)
		local nunit = self.bias:size(1)
		self.output:resize(nframe,nunit)
		self.output:copy(input)
		self.output:addr(1, input.new(nframe):fill(1), self.bias)
	else 
		error('input must be vector or matrix')
	end
	return self.output
end


function AdaBias:updateGradInput(input, gradOutput)
	if self.gradInput then 
		self.gradInput:resizeAs(input)
		self.gradInput:copy(gradOutput)
		return self.gradInput
	end
end


function AdaBias:accGradParameters(input, gradOutput, scale)
	scale = scale or 1
	if input:dim() == 1 then
      -- update running average 
      self.mean:mul(1-self.alpha)
      self.mean:add(self.alpha,input)
      -- update running variance
      self.variance:mul(1-self.alpha)
      self.variance:add(self.alpha, torch.pow(input-self.mean,2))
		-- set the bias
      if self.centering == 1 then
         self.bias = inormcdf(self.lambda, self.mean, torch.sqrt(self.variance))
		elseif self.centering == 2 then
			self.gradBias:add(scale, gradOutput)
			self.gradBias:add(scale*self.lambda[1]*2, self.mean - self.bias)
		end
	elseif input:dim() == 2 then
		local nframe = input:size(1)
		local nunit = self.bias:size(1)
      -- update running average
		self.sampleMean = torch.mean(input,1)
      self.mean:mul(1-self.alpha)
      self.mean:add(self.alpha, self.sampleMean)
		-- update running variance
		self.mean:resize(1,nunit)
		self.sampleVariance=torch.mean(torch.pow(input - torch.expand(self.mean,nframe,nunit),2),1)
		self.mean:resize(nunit)
      self.variance:mul(1-self.alpha)
		self.variance:add(self.alpha,self.sampleVariance)
		-- set the bias
		if self.centering == 1 then
			self.bias = inormcdf(self.lambda,self.mean, torch.sqrt(self.variance))
		elseif self.centering == 2 then
			self.gradBias:addmv(scale, gradOutput:t(), input.new(nframe):fill(1))
			--local diff = self.mean - self.bias
			--diff:div(diff:norm())
         self.gradBias:add(scale*self.lambda[1]*2, self.mean - self.bias)
		end
	end
end

-- If we're doing hard centering, specify that the bias is not trainable
function AdaBias:parameters()
	if self.centering == 1 then
		return 
	elseif self.centering == 2 then
		return {self.bias}, {self.gradBias}
	end
end










	
