-- General script to load various datasets
require 'image'
require 'nn'

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

	elseif dataset == 'CIFAR-10' then
		local trainData
		local testData
		local dataPath = '/home/mikael/Work/Data/cifar-10-batches-t7/'
		local fname = dataPath .. '/cifar_10_norm.t7'
		if paths.filep(fname) then
			print('Loading previously normalized data')
			local X=torch.load(fname)
			trainData=X.train
			testData=X.test
		else
			local trsize = 50000
			local tesize = 10000
			-- We load the dataset from disk, it's straightforward
			trainData = {
				data = torch.Tensor(trsize, 3*32*32),
				labels = torch.Tensor(trsize),
				size = function() return trsize end
			}
			for i = 0,4 do
				local subset = torch.load(dataPath .. '/data_batch_' .. (i+1) .. '.t7', 'ascii')
				trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
				trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
			end
			local trsize = nSamples or trainData.data:size(1)
			trainData.labels = trainData.labels + 1
			trainData.data = trainData.data[{ {1,trsize} }]
			trainData.labels = trainData.labels[{ {1,trsize} }]                                                                                    
			trainData.data = trainData.data:reshape(trsize,3,32,32)
		
			local subset = torch.load(dataPath .. '/test_batch.t7', 'ascii')
			testData = {
				data = subset.data:t():double(),
				labels = subset.labels[1]:double(),
				size = function() return tesize end
			}
			local tesize = nSamples or testData.data:size(1)
			testData.labels = testData.labels + 1
			testData.data = testData.data[{ {1,tesize} }]
			testData.labels = testData.labels[{ {1,tesize} }]
			testData.data = testData.data:reshape(tesize,3,32,32)
			preprocessYUV(trainData, testData)
			print('Saving to ' .. fname)
			torch.save(fname,{train=trainData,test=testData})
		end
		return trainData.data, trainData.labels,testData.data, testData.labels
	
	end
end


function preprocessYUV(trainData,testData)
	trainData.data=trainData.data:float()
	testData.data=testData.data:float()
	-- Convert all images to YUV
	print '==> preprocessing data: colorspace RGB -> YUV'
	for i = 1,trainData:size() do
		trainData.data[i] = image.rgb2yuv(trainData.data[i])
	end
	for i = 1,testData:size() do
		testData.data[i] = image.rgb2yuv(testData.data[i])
	end

	-- Name channels for convenience
	local channels = {'y','u','v'}

	-- Normalize each channel, and store mean/std
	-- per channel. These values are important, as they are part of
	-- the trainable parameters. At test time, test data will be normalized
	-- using these values.
	print '==> preprocessing data: normalize each feature (channel) globally'
	local mean = {}
	local std = {}
	for i,channel in ipairs(channels) do
		-- normalize each channel globally:
		mean[i] = trainData.data[{ {},i,{},{} }]:mean()
		std[i] = trainData.data[{ {},i,{},{} }]:std()
		trainData.data[{ {},i,{},{} }]:add(-mean[i])
		trainData.data[{ {},i,{},{} }]:div(std[i])
	end

	-- Normalize test data, using the training means/stds
	for i,channel in ipairs(channels) do
		-- normalize each channel globally:
		testData.data[{ {},i,{},{} }]:add(-mean[i])
		testData.data[{ {},i,{},{} }]:div(std[i])
	end

	-- Local normalization
	print '==> preprocessing data: normalize all three channels locally'

	-- Define the normalization neighborhood:
	local neighborhood = image.gaussian1D(13)

	-- Define our local normalization operator (It is an actual nn module, 
	-- which could be inserted into a trainable model):
	local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

	-- Normalize all channels locally:
	for c in ipairs(channels) do
		for i = 1,trainData:size() do
			trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
		end
		for i = 1,testData:size() do
			testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
		end
	end

	----------------------------------------------------------------------
	print '==> verify statistics'

	-- It's always good practice to verify that data is properly
	-- normalized.

	for i,channel in ipairs(channels) do
		local trainMean = trainData.data[{ {},i }]:mean()
		local trainStd = trainData.data[{ {},i }]:std()

		local testMean = testData.data[{ {},i }]:mean()
		local testStd = testData.data[{ {},i }]:std()

		print('training data, '..channel..'-channel, mean: ' .. trainMean)
		print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

		print('test data, '..channel..'-channel, mean: ' .. testMean)
		print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
	end

end



function showImages(data)
   first256Samples_y = data[{ {1,256},1 }]
   first256Samples_u = data[{ {1,256},2 }]
   first256Samples_v = data[{ {1,256},3 }]
   image.display{image=first256Samples_y, nrow=16, legend='Some training examples: Y channel'}
   image.display{image=first256Samples_u, nrow=16, legend='Some training examples: U channel'}
   image.display{image=first256Samples_v, nrow=16, legend='Some training examples: V channel'}
end













