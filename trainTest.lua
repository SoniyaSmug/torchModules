-- Generic function to train a model for one epoch

function trainModel(model, criterion, data, labels, opt, logger)
	local optimMethod = opt.optimMethod
	local optimState = opt.optimState
	local batchSize = opt.batchSize
	local classes = opt.classes
	local trsize = data:size(1)
	local confusion = optim.ConfusionMatrix(classes)
	local verbose = false

	-- get model parameters
	local w,dL_dw = model:getParameters()
	-- shuffle at each epoch
	local shuffle = torch.randperm(trsize)
	
	local L = 0
	for t = 1,trsize,batchSize do
		xlua.progress(t,trsize)
		-- create minibatch
		local inputs = {}
		local targets = {}
		for i = 1,math.min(t+batchSize-1, trsize) do
			-- load new sample
			local input = data[shuffle[i]]
			local target = labels[shuffle[i]]
			if opt.type == 'double' then 
				input = input:double()
			elseif opt.type == 'cuda' then
				input = input:cuda()
			end
			table.insert(inputs, input)
			table.insert(targets, target)
		end

		
		-- create closure to evaluate L(w) and dL/dw
		local feval = function(w_)
			if w_ ~= w then
				w:copy(w_)
			end

			-- reset gradients
			dL_dw:zero()
			
			-- L is the average loss
			L = 0
			
			-- evaluate function for complete minibatch
			for i=1,#inputs do
				-- estimate L
				local output = model:forward(inputs[i])
				local err = criterion:forward(output, targets[i])
				L = L + err
				
				-- estimate dL/dw
				local dL_do = criterion:backward(output, targets[i])
				model:backward(inputs[i], dL_do)
				
				-- update confusion
				confusion:add(output,targets[i])
			end
			
			-- normalize gradients and L(w)
			dL_dw:div(#inputs)
			L = L/#inputs
			
			-- return L and dL/dw
			return L, dL_dw
		end

		-- optimize on current mini-batch
		optimMethod(feval,w,optimState)
	end
	print('Loss: ' .. L)
	if verbose then
		print(confusion)
	end

	-- record log info
	if logger then
		logger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
		if verbose then
			logger:style{['% mean class accuracy (train set)'] = '-'}
			logger:plot()
		end
	end

   --save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

	
end


			



function testModel(model, data, labels, opt, logger)
	local confusion = optim.ConfusionMatrix(opt.classes)
	for t = 1,data:size(1) do
		-- get new sample
		local input = data[t]
		if opt.type == 'double' then
			input = input:double()
		elseif opt.type == 'cuda' then
			input = input:cuda()
		end
		local target = labels[t]
		local pred = model:forward(input)
		confusion:add(pred,target)
	end
	
	print(confusion)
	
  -- update log/plot
   logger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      logger:style{['% mean class accuracy (test set)'] = '-'}
      logger:plot()
   end

end

































