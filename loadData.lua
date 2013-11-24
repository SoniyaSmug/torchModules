-- General script to load various datasets

function loadData(dataset,set,nSamples)
	if dataset == 'MNIST' then
		local dataPath
		if set == 'train' then
			dataPath = '/home/mikael/Work/Data/mnist-torch7/train_28x28.th7nn'
		elseif set == 'test' then
			dataPath = '/home/mikael/Work/Data/mnist-torch7/test_28x28.th7nn'
		else
			error('set should be train or test')
		end
		-- load data into CPU memory
		local train_data = torch.load(dataPath)
		-- hack Reshape
		local nInputs = train_data['data']:size(2)*train_data['data']:size(3)*train_data['data']:size(4)
		local nSamplesOrig = train_data['data']:size(1)
		train_data.data:resize(nSamplesOrig,nInputs)
		local data=train_data.data
		local targets=train_data['labels']	
		if nSamples then 
			data=data[{{1,nSamples},{}}]
			targets=targets[{{1,nSamples},{}}]
		end
		return data,targets
	end
end
