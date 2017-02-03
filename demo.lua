require('paths')
require('mattorch')
require('cutorch')

dofile('structure_from_category.lua')

function test_real(cls)
	-- Load annotations
	struc_file = paths.concat('data', cls..'_pascal.mat')
	mat_content = mattorch.load(struc_file)
	W_gt = mat_content['W']:t()
	Gamma = mat_content['Gamma']:t()

	-- Preprocess data
	W, mean_w, std_w = normalize_proj(W_gt, Gamma)

	-- Initialize ADMM
	parm = parm_of_cls(cls)
	admm = ADMM(parm)
	-- Run ADMM
	S_est = admm:run()
	W_est = admm.M*admm.B + admm.T

	for i = 1, W_est:size(1)/2 do
		W_est[{{2*i-1, 2*i}, {}}] = W_est[{{2*i-1, 2*i}, {}}] * std_w[i]
	end
	W_est = W_est + torch.expand(mean_w, W_est:size()):cuda()

	-- Save
	list = {W_gt = W_gt, W_est = W_est:double(), S_est = S_est:double(), 
		mask = Gamma, dict = admm.B:double()}
	output_file = paths.concat('data', 'results', cls..'.mat')
	mattorch.save(output_file, list)
end

test_real('chair')
