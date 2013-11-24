require 'torch'
require 'nn'
require 'LinearCentered'
require 'stats'

mytester = torch.Tester()
jac = nn.Jacobian

precision = 1e-5
expprecision = 1e-4

nntest = {}



function nntest.LinearCentered()
   local ini = math.random(5,10)
   local inj = math.random(5,10)
   local input = torch.Tensor(ini):zero()
   local module = nn.Linear(ini,inj)
	local lambda = 0
	local centering = 2
	local module = nn.LinearCentered(ini,inj,centering,lambda)

   -- 2D
   local nframe = math.random(50,70)
   local input = torch.Tensor(nframe, ini):zero()

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err,precision, 'error on bias ')

   local err = jac.testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   local err = jac.testJacobianUpdateParameters(module, input, module.bias)
   mytester:assertlt(err,precision, 'error on bias [direct update] ')

   for t,err in pairs(jac.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   for t,err in pairs(jac.testAllUpdate(module, input, 'bias', 'gradBias')) do
      mytester:assertlt(err, precision, string.format(
                         'error on bias [%s]', t))
   end

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')



end









function nntest.LinearCentered0()
	centering = 1
	nSamples = math.random(5,8)
	nInputs = math.random(5,8)
	nOutputs = math.random(3,5)
	lambda = 0.5
	input = torch.randn(nSamples,nInputs)
	module = nn.LinearCentered(nInputs,nOutputs,centering,lambda)
	err1 = jac.testJacobian(module,input)
	jac_fprop = nn.Jacobian.forward(module, input)
	jac_bprop = nn.Jacobian.backward(module, input)
	err2 = jac_fprop - jac_bprop
	err2 = err2:abs():max()
	print('\nError: ' .. err1)
	print('\nError2: ' .. err2)
	print('\nDesired precision: ' .. precision)
	mytester:assertlt(err1,precision)
	mytester:assertlt(err2,precision)

end

mytester:add(nntest)
mytester:run()
exit()


