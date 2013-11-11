-- Simple statistical functions for normal distributions
-- Note that there are probably better approximations to erf. 

function normpdf(x, mu, sigma)
   return  torch.exp(-torch.pow(x-mu,2)/(2*sigma^2)) * (1/(sigma*math.sqrt(2*math.pi)))
end

function erf(x)
   local a=8*(math.pi-3)/(3*math.pi*(4-math.pi))
   local xsq=torch.pow(x,2)
   local d1=torch.cmul(xsq,xsq*a+4/math.pi)
   local d2=torch.pow(xsq*a+1,-1)
   return torch.cmul(torch.sign(x),torch.sqrt(-torch.exp(-torch.cmul(d1,d2))+1))
end
   
function erfinv(x)
   local xsq=torch.pow(x,2)
   local a=8*(math.pi-3)/(3*math.pi*(4-math.pi))
   local d1=torch.log(-xsq+1)
   local d2=d1/2+2/(math.pi*a)
   return torch.cmul(torch.sign(x),torch.sqrt(torch.sqrt(torch.pow(d2,2)-d1/a) - d2))
end

-- CDF for normal distribution
function normcdf(x, mu, sigma)
   if type(mu)=='number' and type(sigma)=='number' then
      return (erf((x-mu)/(sigma*math.sqrt(2)))+1)*0.5
   else
      return (erf(torch.cmul(x-mu,torch.pow(sigma*math.sqrt(2),-1)))+1)*0.5
   end
   --return (erf((x-mu)/(sigma*math.sqrt(2)))+1)*0.5
end
   
-- inverse CDF for normal distribution
function inormcdf(p, mu, sigma)
   if type(mu)=='number' and type(sigma)=='number' then
      return erfinv(p*2-1)*math.sqrt(2)*sigma + mu
   else 
      return torch.cmul(erfinv(p*2-1),sigma*math.sqrt(2)) + mu
   end
   --return erfinv(p*2-1)*math.sqrt(2)*sigma + mu
end



