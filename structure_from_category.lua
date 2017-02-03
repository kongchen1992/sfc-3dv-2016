-- This function normalize structure by removing mean and std
function normalize_struc(structure)
    structure = structure - torch.expand(structure:mean(2), structure:size())
    structure = structure / torch.std(structure, 2, true):mean()
    return structure
end

function normalize_proj(projection, mask)
	local mean_proj = torch.cdiv(projection:sum(2), mask:sum(2))
	projection = projection - torch.cmul(mask, 
		torch.expand(mean_proj, projection:size()))
	local num_frames = projection:size(1)/2
	--local std_proj = torch.std(projection, 2, true)
	--std_proj = std_proj:reshape(num_frames, 2):mean(2):select(2, 1)
	local std_proj = torch.cmul(projection, projection):sum(2)
	std_proj = torch.sqrt(std_proj:reshape(num_frames, 2):sum(2)):select(2, 1)
	for i = 1, num_frames do
		projection[{{2*i-1, 2*i}, {}}] = 
			projection[{{2*i-1, 2*i}, {}}]/std_proj[i]
	end
	return projection, mean_proj, std_proj
end

-- This function compute the determinant of a 3X3 Tensor
function det(A)
	local a_1 = A[{1, 1}]*(A[{2, 2}]*A[{3, 3}] - A[{2, 3}]*A[{3, 2}])
	local a_2 = A[{1, 2}]*(A[{2, 1}]*A[{3, 3}] - A[{2, 3}]*A[{3, 1}])
	local a_3 = A[{1, 3}]*(A[{2, 1}]*A[{3, 2}] - A[{2, 2}]*A[{3, 1}])
	return a_1 - a_2 + a_3
end

-- This function compute the kronecher product
function kron(X, Z, buffer)
	assert(X:dim() == 2 and Z:dim() == 2) -- should generalize this
	local N, M = X:size(1), X:size(2)
	local P, Q = Z:size(1), Z:size(2)
	local K    = buffer or torch.CudaTensor(N*P, M*Q)
	for row = 1,N do
		for col = 1,M do
			K[{{(row - 1)*P + 1, row*P},{(col - 1)*Q + 1, col*Q}}]
				= torch.mul(Z, X[row][col])
		end
	end
	return K
end

function cad_of_cls(cls)
	local cads_available = torch.LongTensor()
	if cls == 'aeroplane' then
		cads_available = torch.LongTensor():range(1, 8)
	elseif cls == 'boat' then
		cads_available = torch.LongTensor{1, 2, 3, 4, 6}
	elseif cls == 'car' then
		cads_available = torch.LongTensor{1, 2, 3, 4, 5, 6, 7, 9, 10}
	elseif cls == 'chair' then
		cads_available = torch.LongTensor{1, 3, 4, 5, 6, 7, 8, 9}
	elseif cls == 'tvmonitor' then
		cads_available = torch.LongTensor{2, 3}
	elseif cls == 'bicycle' then
		cads_available = torch.LongTensor():range(1, 6)
	elseif cls == 'bus' then
		cads_available = torch.LongTensor():range(1, 6)
	elseif cls == 'diningtable' then
		cads_available = torch.LongTensor():range(1, 5)
	elseif cls == 'motorbike' then
		cads_available = torch.LongTensor():range(1, 5)
	elseif cls == 'sofa' then
		cads_available = torch.LongTensor():range(1, 6)
	elseif cls == 'train' then
		cads_available = torch.LongTensor{2, 4}
	elseif cls == 'bottle' then
		cads_available = torch.LongTensor():range(1, 8)
	else
		print('The class '..cls..' has not been found. Ignored.')
	end
	return cads_available
end

function parm_of_cls(cls)
	local cads_available = torch.LongTensor()
	if cls == 'aeroplane' then
		param = {W = W, num_bases = 10, lambda = 0.05, mu_tau = 1.1, 
			rho_tau = 1.1, d_tol = 3e-6, Gamma = Gamma, is_biGamma = true,
			is_gradient_descent = true, mu = 1e-2, rho = 1e-1}
	elseif cls == 'boat' then
		param = {W = W, num_bases = 3, lambda = 0.05, mu_tau = 1.1, 
			rho_tau = 1.1, d_tol = 3e-6, Gamma = Gamma, is_biGamma = true,
			is_gradient_descent = true, mu = 1e-2, rho = 1e-1}
	elseif cls == 'car' then
		param = {W = W, num_bases = 2, lambda = 0.06, mu_tau = 1.1, 
			rho_tau = 1.1, d_tol = 3e-6, Gamma = Gamma, is_biGamma = true,
			is_gradient_descent = true, mu = 1e-2, rho = 1e-1}
	elseif cls == 'chair' then
		param = {W = W, num_bases = 5, lambda = 0.1, mu_tau = 1.1, 
			rho_tau = 1.1, d_tol = 3e-6, Gamma = Gamma, is_biGamma = true,
			is_gradient_descent = true, mu = 1e-2, rho = 1e-1}
	elseif cls == 'tvmonitor' then
		param = {W = W, num_bases = 1, lambda = 0.05, mu_tau = 1.1, 
			rho_tau = 1.1, d_tol = 3e-6, Gamma = Gamma, is_biGamma = true,
			is_gradient_descent = true, mu = 1e-2, rho = 1e-1}
	elseif cls == 'bicycle' then
		param = {W = W, num_bases = 5, lambda = 0.05, mu_tau = 1.1, 
			rho_tau = 1.1, d_tol = 3e-6, Gamma = Gamma, is_biGamma = true,
			is_gradient_descent = true, mu = 1e-2, rho = 1e-1}
	elseif cls == 'bus' then
		param = {W = W, num_bases = 1, lambda = 0.05, mu_tau = 1.1, 
			rho_tau = 1.1, d_tol = 3e-6, Gamma = Gamma, is_biGamma = true,
			is_gradient_descent = true, mu = 1e-2, rho = 1e-1}
	elseif cls == 'diningtable' then
		param = {W = W, num_bases = 1, lambda = 0.01, mu_tau = 1.1, 
			rho_tau = 1.1, d_tol = 3e-6, Gamma = Gamma, is_biGamma = true,
			is_gradient_descent = true, mu = 1e-2, rho = 1e-1}
	elseif cls == 'motorbike' then
		param = {W = W, num_bases = 5, lambda = 0.05, mu_tau = 1.1, 
			rho_tau = 1.1, d_tol = 3e-6, Gamma = Gamma, is_biGamma = true,
			is_gradient_descent = true, mu = 1e-2, rho = 1e-1}
	elseif cls == 'sofa' then
		param = {W = W, num_bases = 5, lambda = 0.05, mu_tau = 1.1, 
			rho_tau = 1.1, d_tol = 3e-6, Gamma = Gamma, is_biGamma = true,
			is_gradient_descent = true, mu = 1e-2, rho = 1e-1}
	elseif cls == 'train' then
		param = {W = W, num_bases = 1, lambda = 0.05, mu_tau = 1.1, 
			rho_tau = 1.1, d_tol = 3e-6, Gamma = Gamma, is_biGamma = true,
			is_gradient_descent = true, mu = 1e-2, rho = 1e-1}
	elseif cls == 'bottle' then
		param = {W = W, num_bases = 2, lambda = 0.01, mu_tau = 1.1, 
			rho_tau = 1.1, d_tol = 3e-6, Gamma = Gamma, is_biGamma = true,
			is_gradient_descent = true, mu = 1e-2, rho = 1e-1}
	else
		print('The class '..cls..' has not been found. Ignored.')
	end
	return param
end

do
	-- This class is the implementation of ADMM to solve structure from category problem
	local ADMM = torch.class('ADMM')
	
	-- The initializer
	function ADMM:__init(options)

		-- Set W
		if type(options.W) == 'nil' then
			error('No projection matrix W')
		else
			self.W = options.W:cuda()
		end

		-- Set Dimension
		self.num_frames = self.W:size(1)/2
		self.num_points = self.W:size(2)

		-- Set B
		if type(options.B) == 'nil' then
			print('No initial B, initializing B by random')
			if type(options.num_bases) == 'nil' then
				error('Not Found either number of bases or inital B')
			else
				-- Initial B by random
				self.num_bases = options.num_bases
				self.B = torch.CudaTensor():randn(self.num_bases*3, self.num_points)
				-- Normalize bases
				self.B = normalize_struc(self.B)
				for i = 1, self.num_bases do
					self.B[{{3*i-2, 3*i}, {}}] = self.B[{{3*i-2, 3*i}, {}}] / 
						torch.norm(self.B[{{3*i-2, 3*i}, {}}])
				end
			end
		else
			self.B = options.B:cuda()
			self.num_bases = self.B:size(1)/3
			-- Normalize bases
			self.B = normalize_struc(self.B)
			for i = 1, self.num_bases do
				self.B[{{3*i-2, 3*i}, {}}] = self.B[{{3*i-2, 3*i}, {}}] / 
					torch.norm(self.B[{{3*i-2, 3*i}, {}}])
			end
		end

		-- Set A
		self.A = self.B:clone()

		-- Set M
		if type(options.M) == 'nil' then
			print('No initial M, initializing M as zeros')
			self.M = torch.CudaTensor(self.num_frames*2, 3*self.num_bases):fill(0)
		else
			self.M = options.M:cuda()
		end

		-- Set Z
		self.Z = self.M:clone()

		-- Set T
		if type(options.T) == 'nil' then
			print('No initial T, initializing T as zeros')
			self.T = torch.CudaTensor(self.W:size()):fill(0)
		else
			self.T = options.T:cuda()
		end

		-- Set mask Gamma
		self.Gamma = options.Gamma
		self.is_biGamma = options.is_biGamma or true
		if type(self.Gamma) ~= 'nil' then
			self.ind_Gamma = torch.nonzero(self.Gamma:t():reshape(
				2*self.num_frames*self.num_points)):select(2, 1)
			self.Gamma_col = {}
			for i = 1, self.num_points do
				self.Gamma_col[i] = 
					torch.nonzero(self.Gamma:select(2, i)):select(2, 1)
			end
			self.Gamma = self.Gamma:cuda()
		end

		-- Set dual variables
		self.Lambda = torch.CudaTensor(self.M:size()):fill(0)
		self.Pi = torch.CudaTensor(self.B:size()):fill(0)
			
		-- Set parameters
		self.lambda = options.lambda or 1

		self.is_auto_penalty = options.is_auto_penalty or false
		if self.is_auto_penalty then
			self.mu = options.mu or 1e2
			self.rho = options.rho or 1e2
			self.prim_res_Z = 0
			self.prim_res_B = 0
			self.dual_res_Z = 0
			self.dual_res_B = 0
			self.Z0 = self.Z:clone()
			self.B0 = self.B:clone()
		else
			self.mu = options.mu or 1e-2
			self.mu_max = options.mu_max or 1e5
			self.mu_tau = options.mu_tau or 1.01
			self.rho = options.rho or 1e-2
			self.rho_max = options.rho_max or 1e5
			self.rho_tau = options.rho_tau or 1.01
		end

		self.max_iters = options.max_iters or 1e6
		self.obj_tol = options.obj_tol or 1e-6
		self.d_tol = options.d_tol or 1e-7
		self.is_detail = options.is_detail or true
		self.how_detail = options.how_detail or 1

		self.is_gradient_descent = options.is_gradient_descent or false
		self.gd_iters_Z = options.gs_iters_Z or 10
		self.gd_iters_B = options.gs_iters_B or 20
		self.alpha = options.alpha or 0.5
		self.beta = options.beta or 0.5

		-- Set output
		self.cost = torch.CudaTensor(self.max_iters):fill(0)
		self.proj_err = 0
		self.mean_sps = 0
		self.dist_M_Z = 0
		self.dist_A_B = 0
		
		-- Set hidden variables
		self.C = torch.CudaTensor(self.num_frames, self.num_bases)
		self.R = torch.CudaTensor(3*self.num_frames, 3*self.num_bases)
	end

	-- The function to run algorithm
	function ADMM:run()
		timer = torch.Timer()
		if self.is_detail then
			self:to_output(-1)
			self:to_output(0)
		end
		for i = 1, self.max_iters do

			--local tim = torch.Timer()
			--local cost = admm:compute_cost()
			--tim:reset()
			--admm:update_M()
			--local cost_new = admm:compute_cost()
			--print(string.format('%-20s   %4.4e   Time:    %2.2e', 'Updating M', 
			--	cost_new - cost, tim:time().real))
			--cost = cost_new
			--tim:reset()
			--admm:update_Z()
			--cost_new = admm:compute_cost()
			--print(string.format('%-20s   %4.4e   Time:    %2.2e', 'Updating Z', 
			--	cost_new - cost, tim:time().real))
			--cost = cost_new
			--tim:reset()
			--admm:update_B()
			--cost_new = admm:compute_cost()
			--print(string.format('%-20s   %4.4e   Time:    %2.2e', 'Updating B', 
			--	cost_new - cost, tim:time().real))
			--cost = cost_new
			--tim:reset()
			--admm:update_A()
			--cost_new = admm:compute_cost()
			--print(string.format('%-20s   %4.4e   Time:    %2.2e', 'Updating A', 
			--	cost_new - cost, tim:time().real))
			--cost = cost_new
			--tim:reset()
			--admm:update_T()
			--cost_new = admm:compute_cost()
			--print(string.format('%-20s   %4.4e   Time:    %2.2e', 'Updating T', 
			--	cost_new - cost, tim:time().real))
			--cost = cost_new
			--tim:reset()
			--admm:update_dual()
			--cost_new = admm:compute_cost()
			--print(string.format('%-20s   %4.4e   Time:    %2.2e', 'Updating dual', 
			--	cost_new - cost, tim:time().real))
			--cost = cost_new
			--tim:reset()
			--admm:update_penalty()
			--cost_new = admm:compute_cost()
			--print(string.format('%-20s   %4.4e   Time:    %2.2e', 'Updating penalty', 
			--	cost_new - cost, tim:time().real))
			--cost = cost_new
			--if self.is_auto_penalty then
			--	print('primal residual Z', self.prim_res_Z)
			--	print('dual residual Z', self.dual_res_Z)
			--	print('primal residual B', self.prim_res_B)
			--	print('dual residual B', self.dual_res_B)
			--end

			self:update_M()
			self:update_Z()
			self:update_B()
			self:update_A()
			self:update_T()
			self:update_dual()
			self:update_penalty()

			self.cost[i] = self:compute_cost()
			if self.is_detail and i%self.how_detail == 0 then
				self:to_output(i)
			end
			if self.is_detail and i%(self.how_detail*10) == 0 then
				self:to_output(0)
			end
			if self:is_stop(i) then
				if self.how_detail ~= 1 then
					self:to_output(i)
				end
				break
			end
		end
		return self:estimate_struc()
	end

	-- This function is to update variable M
	function ADMM:update_M()
		ZLm = self.Z - self.Lambda/self.mu
		lm = self.lambda/self.mu
		for f = 1, self.num_frames do
			for i = 1, self.num_bases do
				self.M[{{2*f-1, 2*f}, {3*i-2, 3*i}}], self.C[{f, i}] = 
					self:prox_2norm(ZLm[{{2*f-1, 2*f}, {3*i-2, 3*i}}], lm)
			end
		end
	end

	function ADMM:prox_2norm(X, lam)
		-- This function utilize proximal operator to solve spectral norm min
		local U, W, V = torch.svd(X:double())
		if W:sum() <= lam then
			W:fill(0)
		elseif W[1] - W[2] <= lam then
			W[1] = (W:sum() - lam)/2
			W[2] = W[1]
		else
			W[1] = W[1] - lam
		end
		local X = U*torch.diag(W)*V:t()
		local normX = W[1]
		return X, normX
	end

	-- This function is to update auxiliary variable Z
	function ADMM:update_Z()
		if self.is_auto_penalty then
			self.Z0 = self.Z:clone()
		end
		if type(self.Gamma) == 'nil' then
			local A = (self.W - self.T)*self.B:t() + self.Lambda + self.mu*self.M
			local B = self.B*self.B:t()
			for i = 1, 3*self.num_bases do
				B[{i, i}] = B[{i, i}] + self.mu
			end
			self.Z = torch.gels(A:t(), B:t()):t()
		elseif self.is_biGamma then
			if not self.is_gradient_descent then
				local eye = torch.CudaTensor(2*self.num_frames, 2*self.num_frames):fill(0)
				for i = 1, 2*self.num_frames do
					eye[{i, i}] = 1
				end
				local BI = kron(self.B, eye)
				-- A = kron(B, I)*diag(vec(Gamma.*Gamma))kron(B', I)
				BI = BI:index(2, self.ind_Gamma)
				local A = BI*BI:t()
				-- A = A + mu*I
				for i = 1, A:size(1) do
					A[{i, i}] = A[{i, i}] + self.mu
				end
				-- B = vec(WB + ...)
				local B = ((self.W - torch.cmul(self.Gamma, self.T))*self.B:t() + 
					self.Lambda + self.mu*self.M):t():reshape(6*self.num_frames*
					self.num_bases, 1)
				-- Compute Z
				self.Z = torch.gels(B, A):reshape(3*self.num_bases, 2*self.num_frames):t()
			else
				for iloop = 1, self.gd_iters_Z do
					local gradient = (torch.cmul(self.Gamma, self.Z*self.B + self.T) -
						self.W)*self.B:t() - self.Lambda + self.mu*(self.Z - self.M)
					local t = 10
					local Z = self.Z - gradient
					local objective = 1/2*torch.norm(torch.cmul(self.Gamma, self.Z*self.B + 
						self.T) - self.W)^2 + torch.cmul(self.Lambda, self.M - self.Z):sum() + 
						self.mu/2*torch.norm(self.M - self.Z)^2
					local obj = 1/2*torch.norm(torch.cmul(self.Gamma, Z*self.B + 
						self.T) - self.W)^2 + torch.cmul(self.Lambda, self.M - Z):sum() + 
						self.mu/2*torch.norm(self.M - Z)^2
					while obj > objective - self.alpha*t*torch.norm(gradient)^2 do
						t = self.beta*t
						Z = self.Z - t*gradient
						obj = 1/2*torch.norm(torch.cmul(self.Gamma, Z*self.B + 
							self.T) - self.W)^2 + torch.cmul(self.Lambda, self.M - Z):sum() + 
							self.mu/2*torch.norm(self.M - Z)^2
					end
					self.Z = Z
				end
			end
		else
			if not self.is_gradient_descent then
				-- Need to do
			else
				for iloop = 1, self.gd_iters_Z do
					local gradient = (torch.cmul(torch.cmul(self.Gamma, self.Gamma), 
						self.Z*self.B + self.T) - self.W)*self.B:t() - self.Lambda +
						self.mu*(self.Z - self.M)
					local t = 10
					local Z = self.Z - gradient
					local objective = 1/2*torch.norm(torch.cmul(self.Gamma, self.Z*self.B + 
						self.T) - self.W)^2 + torch.cmul(self.Lambda, self.M - self.Z):sum() + 
						self.mu/2*torch.norm(self.M - self.Z)^2
					local obj = 1/2*torch.norm(torch.cmul(self.Gamma, Z*self.B + 
						self.T) - self.W)^2 + torch.cmul(self.Lambda, self.M - Z):sum() + 
						self.mu/2*torch.norm(self.M - Z)^2
					while obj > objective - self.alpha*t*torch.norm(gradient)^2 do
						t = self.beta*t
						Z = self.Z - t*gradient
						obj = 1/2*torch.norm(torch.cmul(self.Gamma, Z*self.B + 
							self.T) - self.W)^2 + torch.cmul(self.Lambda, self.M - Z):sum() + 
							self.mu/2*torch.norm(self.M - Z)^2
					end
					self.Z = Z
				end
			end
		end
	end

	-- This function is to update variable B
	function ADMM:update_B()
		if self.is_auto_penalty then
			self.B0 = self.B:clone()
		end
		if type(self.Gamma) == 'nil' then
			local A = self.Z:t()*(self.W - self.T) + self.Pi + self.rho*self.A
			local B = self.Z:t()*self.Z
			for i = 1, 3*self.num_bases do
				B[{i, i}] = B[{i, i}] + self.rho
			end
			self.B = torch.gels(A, B)
		elseif self.is_biGamma then
			if self.rho > 1e2 then
				local right_matrix = self.Z:t()*(self.W - torch.cmul(self.Gamma, 
					self.T)) + self.Pi + self.rho*self.A
				for p = 1, self.num_points do
					-- A = Z'*G*G*Z
					local ZG = self.Z:index(1, self.Gamma_col[p])
					local A = ZG:t()*ZG
					-- A = A + rho*I
					for i = 1, 3*self.num_bases do
						A[{i, i}] = A[{i, i}] + self.rho
					end
					-- Compute B
					self.B[{{}, {p}}] = torch.gels(right_matrix[{{}, {p}}], A)
				end
			else
				for iloop = 1, self.gd_iters_B do
					local gradient = self.Z:t()*(torch.cmul(self.Gamma, 
						self.Z*self.B + self.T) - self.W) - self.Pi + 
						self.rho*(self.B - self.A)
					local t = 1
					local B = self.B - t*gradient
					local objective = 1/2*torch.norm(torch.cmul(self.Gamma, self.Z*self.B + 
						self.T) - self.W)^2 + torch.cmul(self.Pi, self.A - self.B):sum() + 
						self.rho/2*torch.norm(self.A - self.B)^2
					local obj = 1/2*torch.norm(torch.cmul(self.Gamma, self.Z*B + 
						self.T) - self.W)^2 + torch.cmul(self.Pi, self.A - B):sum() + 
						self.rho/2*torch.norm(self.A - B)^2
					while obj > objective - self.alpha*t*torch.norm(gradient)^2 do
						t = self.beta*t
						B = self.B - t*gradient
						obj = 1/2*torch.norm(torch.cmul(self.Gamma, self.Z*B + 
							self.T) - self.W)^2 + torch.cmul(self.Pi, self.A - B):sum() + 
							self.rho/2*torch.norm(self.A - B)^2
					end
					self.B = B
				end
			end
		else
			if self.rho > 1e2 then
				local right_matrix = self.Z:t()*(self.W - torch.cmul(torch.cmul(self.Gamma, 
					self.Gamma), self.T)) + self.Pi + self.rho*self.A
				local Z = torch.cmul(self.Z, self.Gamma:expand(2*self.num_frames, 3*self.num_bases))
				for p = 1, self.num_points do
					-- A = Z'*G*G*Z
					local ZG = self.Z:index(1, self.Gamma_col[p])
					local A = ZG:t()*ZG
					-- A = A + rho*I
					for i = 1, 3*self.num_bases do
						A[{i, i}] = A[{i, i}] + self.rho
					end
					-- Compute B
					self.B[{{}, {p}}] = torch.gels(right_matrix[{{}, {p}}], A)
				end
			else
				for iloop = 1, self.gd_iters_B do
					local gradient = self.Z:t()*(torch.cmul(torch.cmul(self.Gamma, self.Gamma), 
						self.Z*self.B + self.T) - self.W) - self.Pi + 
						self.rho*(self.B - self.A)
					local t = 1
					local B = self.B - t*gradient
					local objective = 1/2*torch.norm(torch.cmul(self.Gamma, self.Z*self.B + 
						self.T) - self.W)^2 + torch.cmul(self.Pi, self.A - self.B):sum() + 
						self.rho/2*torch.norm(self.A - self.B)^2
					local obj = 1/2*torch.norm(torch.cmul(self.Gamma, self.Z*B + 
						self.T) - self.W)^2 + torch.cmul(self.Pi, self.A - B):sum() + 
						self.rho/2*torch.norm(self.A - B)^2
					while obj > objective - self.alpha*t*torch.norm(gradient)^2 do
						t = self.beta*t
						B = self.B - t*gradient
						obj = 1/2*torch.norm(torch.cmul(self.Gamma, self.Z*B + 
							self.T) - self.W)^2 + torch.cmul(self.Pi, self.A - B):sum() + 
							self.rho/2*torch.norm(self.A - B)^2
					end
					self.B = B
				end
			end
		end
	end

	-- This function is to update auxializry variable A
	function ADMM:update_A()
		local BPi = self.B - self.Pi/self.rho
		for i = 1, self.num_bases do
			self.A[{{3*i-2, 3*i}, {}}] = BPi[{{3*i-2, 3*i}, {}}] / 
				torch.norm(BPi[{{3*i-2, 3*i}, {}}])
		end
	end

	-- This function is to update translation T
	function ADMM:update_T()
		if type(self.Gamma) == 'nil' then
			local tau = (self.W:sum(2) - (self.Z*self.B):sum(2))/self.num_points
			for i = 1, self.num_points do
				self.T[{{}, i}] = tau
			end
		elseif self.is_biGamma then
			local tau = self.W:sum(2) - torch.cmul(self.Gamma, self.Z*self.B):sum(2)
			tau = torch.cdiv(tau, self.Gamma:sum(2))
			self.T = torch.expand(tau, self.T:size())
		else
			local tau = self.W:sum(2) - torch.cmul(torch.cmul(self.Gamma, self.Gamma),
				self.Z*self.B):sum(2)
			tau = torch.cdiv(tau, torch.cmul(self.Gamma, self.Gamma):sum(2))
			self.T = torch.expand(tau, self.T:size())
		end
	end

	-- This function is to update dual variables, Lambda, Pi
	function ADMM:update_dual()
		self.Lambda = self.Lambda + self.mu*(self.M - self.Z)
		self.Pi = self.Pi + self.rho*(self.A - self.B)
	end

	-- This function is to update penalty mu, rho
	function ADMM:update_penalty()
		if self.is_auto_penalty then
			if self.prim_res_Z > 10*self.dual_res_Z then
				self.mu = self.mu*2
			elseif self.dual_res_Z > 10*self.prim_res_Z then
				self.mu = self.mu/2
			end
			if self.prim_res_B > 10*self.dual_res_B then
				self.rho = self.rho*2
			elseif self.dual_res_B > 10*self.prim_res_B then
				self.rho = self.rho/2
			end
		else
			if self.mu < self.mu_max then
				self.mu = self.mu*self.mu_tau
			end
			if self.rho < self.rho_max then
				self.rho = self.rho*self.rho_tau
			end
		end
	end

	-- This function is to compute cost
	function ADMM:compute_cost()
		if type(self.Gamma) == 'nil' then
			self.proj_err = torch.norm(self.Z*self.B + self.T - self.W)^2
		else
			self.proj_err = torch.norm(torch.cmul(self.Gamma, self.Z*self.B+self.T)
				- self.W)^2
		end
		self.mean_sps = self.C:ge(1e-4):sum() / self.num_frames
		local spec_norm = self.C:sum()
		local dual_M_inner_mul = torch.cmul(self.Lambda, self.M - self.Z):sum()
		self.dist_M_Z = torch.norm(self.M - self.Z)^2
		local dual_B_inner_mul = torch.cmul(self.Pi, self.A - self.B):sum()
		self.dist_A_B = torch.norm(self.A - self.B)^2

		local cost = 1/2*self.proj_err + self.lambda*spec_norm + dual_M_inner_mul + 
			self.mu/2*self.dist_M_Z + dual_B_inner_mul + self.rho/2*self.dist_A_B

		if self.is_auto_penalty then
			self.prim_res_Z = torch.norm(self.M - self.Z)/torch.norm(self.Z0)
			self.prim_res_B = torch.norm(self.A - self.B)/torch.norm(self.B0)
			self.dual_res_Z = torch.norm(self.Z - self.Z0)/torch.norm(self.Z0)
			self.dual_res_B = torch.norm(self.B - self.B0)/torch.norm(self.B0)
		end

		return cost
	end

	-- This function is used to decide convergence
	function ADMM:is_stop(it)
		if self.is_auto_penalty then
			local is_prim_Z = self.prim_res_Z < self.obj_tol
			local is_prim_B = self.prim_res_B < self.obj_tol
			local is_dual_Z = self.dual_res_Z < self.obj_tol
			local is_dual_B = self.dual_res_B < self.obj_tol
			return is_prim_Z and is_prim_B and is_dual_Z and is_dual_B
		else
			if it == 1 then
				return false
			end
			local is_obj_tol = torch.abs(self.cost[it]) < self.obj_tol
			local is_d_tol = torch.abs(self.cost[it] - self.cost[it-1]) < self.d_tol*self.cost[it]
			local is_descent = self.cost[it] - self.cost[it-1] < 0

			return (is_obj_tol or is_d_tol) and is_descent
		end
	end

	-- The function to print details
	function ADMM:to_output(it)
		if it == -1 then
			print('======================================================================'..
				'================================')
			print('                           Alternating Direction Method of Multipliers')
			print('======================================================================'..
				'================================')
			print(string.format('%-20s%10d %-20s%10d %-20s%10d', 'Num of Points: ', self.num_points,
				'Num of Frames: ', self.num_frames, 'Num of Bases: ', self.num_bases))
			print(string.format('%-20s%6.4e %-20s %6.4e %-20s%6.4e', 'Maximum iterations: ', self.max_iters,
				'Objective tolerance:', self.obj_tol, 'Delta Obj Tolerance:', self.d_tol))
		elseif it == 0 then
			print('---------------------------------------------------------------------'..
				'---------------------------------')
			print(string.format('%6s|%10s|%10s|%10s|%10s|%10s|%10s|%10s|%10s|%10s', 'iters', 
				'rho', 'mu', 'Proj Err', 'mean[c]0', '[M-Z]_F', '[A-B]_F', 'cost', 'd(cost)', 'time(sec)'))
		elseif it == 1 then
			print(string.format('%6d|%6.4e|%6.4e|%6.4e|%6.4e|%6.4e|%6.4e|%6.4e|%+6.4e|%6.4e', 
				it, self.rho, self.mu, self.proj_err, self.mean_sps, self.dist_M_Z, self.dist_A_B, 
				self.cost[it], self.cost[it], timer:time().real))
		else
			print(string.format('%6d|%6.4e|%6.4e|%6.4e|%6.4e|%6.4e|%6.4e|%6.4e|%+6.4e|%6.4e', 
				it, self.rho, self.mu, self.proj_err, self.mean_sps, self.dist_M_Z, self.dist_A_B, 
				self.cost[it], self.cost[it] - self.cost[it-1], timer:time().real))
		end
	end

	-- This function is to estimate structure from estimated variables
	function ADMM:estimate_struc()
		local CR = torch.CudaTensor(3*self.num_frames, 3*self.num_bases)
		local R = torch.CudaTensor(3, 3)
		for f = 1, self.num_frames do
			for i = 1, self.num_bases do
				if self.C[{f, i}] == 0 then
					R:fill(0)
				else
					R[{{1, 2}, {}}] = self.M[{{2*f-1, 2*f}, {3*i-2, 3*i}}]/self.C[{f, i}]
					R[{{3}, {}}] = torch.cross(R[{{1}, {}}], R[{{2}, {}}])
					if det(R) < 0 then
						R[{{3}, {}}] = -R[{{3}, {}}]
					end
				end
				CR[{{3*f-2, 3*f}, {3*i-2, 3*i}}] = self.C[{f, i}]*R
				self.R[{{3*f-2, 3*f}, {3*i-2, 3*i}}] = R
			end
		end
		return CR*self.B
	end
end
