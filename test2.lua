require 'torch'
require 'nn'
require 'FeatureNormalization'

mytester = torch.Tester()
jac = nn.Jacobian

precision = 1e-5
expprecision = 1e-4

nntest = {}

function nntest.FeatureNormalization()
	nSamples=math.random(1,1)
	nFeatures=math.random(2,3)
	height=math.random(3,3)
	width=math.random(3,3)
	input1 = torch.randn(nSamples,nFeatures,height,width)
	-- generate input with some norms below 0.1
	flag=false
	while not flag do
		input2 = torch.randn(nSamples,nFeatures,height,width):mul(0.1)
		if torch.lt(input2:norm(2,2),0.1):max()==1 then
			flag=true
		end
	end
	module = nn.FeatureNormalization(input1:size())
	err1 = jac.testJacobian(module,input1)
	jac_fprop = nn.Jacobian.forward(module,input2)
	jac_bprop = nn.Jacobian.backward(module,input2)
	err2 = jac_fprop - jac_bprop
	err2 = err2:abs():max()
	print('\nError: ' .. err1)
	print('\nError with small inputs: ' .. err2)
	print('Desired precision: ' .. precision)
	mytester:assertlt(err1,precision, 'error on state')
	mytester:assertlt(err2,precision,'error on small inputs')

   local ferr,berr = jac.testIO(module,input1)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')


end

mytester:add(nntest)
mytester:run()
exit()


