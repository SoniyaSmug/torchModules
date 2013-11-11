require 'nn'

local FeatureNormalization, parent = torch.class('nn.FeatureNormalization','nn.Module')

-- we are assuming the input is a 4D tensor: [samples x features x height x width]

function FeatureNormalization:__init(inputSize)
	if inputSize:size() ~= 4 then
		error('Input to FeatureNormalization should be have 4 dimensions ([samples x features x height x width])')
	end
	parent.__init(self)
	self.nSamples = inputSize[1]
	self.nFeatures = inputSize[2]
	self.height = inputSize[3]
	self.width = inputSize[4]
	self.featureNorms = torch.zeros(self.nSamples,self.height,self.width)
	self.featureNormsMask = torch.zeros(self.nSamples,self.height,self.width)
end

function FeatureNormalization:updateOutput(input)	
	local eps=0.1
	self.output:resizeAs(input)
	self.output:copy(input)
	-- compute norms across features
	self.featureNorms:copy(input:norm(2,2))
	-- apply a lower threshold
	self.featureNormsMask:copy(self.featureNorms)
	self.featureNormsMask:apply(function (x) return (x>eps) and 1 or 0 end)
	self.featureNorms:apply(function (x) return math.max(x,eps) end)
	self.featureNormsMask:resize(self.nSamples,1,self.height,self.width)
	self.featureNorms:resize(self.nSamples,1,self.height,self.width)
	-- normalize
	self.output:cdiv(self.featureNorms:expandAs(self.output))
	return self.output
end
	

-- the gradient update is given by:
-- G_sfij := G_sfij/S_sij - (X_sfij/S_sij^3) * SUM_f' X_sf'ij G_sf'ij
-- where S = SUM_f' X_{sf'ij}

function FeatureNormalization:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(gradOutput)
	self.gradInput:copy(gradOutput)
	self.gradInput:cdiv(self.featureNorms:expandAs(self.gradInput))
	self.gradInput:add(torch.cmul(self.featureNormsMask:expandAs(self.gradInput),-torch.cmul(torch.cdiv(input,torch.pow(self.featureNorms:expandAs(self.gradInput),3)),torch.cmul(input,gradOutput):sum(2):expandAs(self.gradInput))))
	--self.gradInput:add(torch.cmul(torch.expand(self.featureNormsMask,self.nSamples,self.nFeatures,self.height,self.width),-torch.cmul(torch.cdiv(input,torch.pow(self.featureNorms:expandAs(self.gradInput),3)),torch.expand(torch.cmul(input,gradOutput):sum(2),self.nSamples,self.nFeatures,self.height,self.width))))
	return self.gradInput
end
	


