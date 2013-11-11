local Linear, parent = torch.class('nn.LinearCentered', 'nn.Module')

function Linear:__init(inputSize, outputSize, lambda)
   parent.__init(self)
  
   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   self.outputMean = torch.Tensor(outputSize)
   self.outputVar = torch.Tensor(outputSize)
   self.batchSize = 100
   self.nBatches = 0 
   self.lambda = lambda
   self:reset()
end

-- TODO: reset mean
function Linear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function Linear:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      -- multiply by the weights
      self.output:zero():addmv(1, self.weight, input)
      -- update running average of the unbiased outputs
      --self.outputMean:add((1/(self.nBatches+1)),self.output)
      -- add the bias
      self.output:add(self.bias)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)
      self.output:resize(nframe, nunit)
      self.output:zero():addmm(1, input, self.weight:t())
      -- update running statistics of the unbiased inputs
      self.outputMean = torch.mean(self.output,1)
      self.outputStd = torch.std(self.output,1)
      -- we set the biases so that they set (1-lambda) percent of the samples to 0
      self.bias = inormcdf(self.lambda,self.outputMean, self.outputStd)
      self.output:addr(1, input.new(nframe):fill(1), self.bias)
   else
      error('input must be vector or matrix')
   end
   self.nBatches = self.nBatches + 1

   return self.output
end

function Linear:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function Linear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale*(1-lambda), gradOutput)
	  self.gradBias:add(scale*lambda, self.bias - self.inputMean)      
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      --self.gradBias:addmv(scale, gradOutput:t(), input.new(nframe):fill(1))
	   --self.gradBias:add(self.lambda*2, self.bias - self.outputMean)
   end

end

-- we do not need to accumulate parameters when sharing
Linear.sharedAccUpdateGradParameters = Linear.accUpdateGradParameters
