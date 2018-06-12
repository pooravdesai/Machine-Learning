import numpy as np

class mlp:
	
	activation = {}
	derivative = {}

	def logistic(x):
		return 1/(1+np.exp(-x))

	def relu(x):
		return np.maximum(x,0,x)

	def softmax(x):
		exps = np.exp(x)
		return exps/np.sum(exps)

	activation['logistic'] = logistic
	activation['relu'] = relu
	activation['softmax'] = softmax

	def derivative_of_logistic(x):
		return x*(1-x)

	def derivative_of_relu(x):
		dx = np.empty_like(x)
		dx[x < 0] = 0
		dx[x >= 0] = 1
		return dx

	derivative['logistic'] = derivative_of_logistic
	derivative['relu'] = derivative_of_relu
	

	logistic = lambda x: 1.0/(1.0+np.exp(-x))

	def __init__(self, layers, activation_function = 'relu', alpha = 0.001, l = 0.001, update = 'adam',batch_size = 16,epoch = 10):
		self.number_of_features = 0
		self.labels = None
		self.layers = layers
		self.activation_function = activation_function
		self.alpha = alpha
		self.l = l
		self.update = update
		self.batch_size = batch_size
		self.epoch = epoch
		self.theta = []
		self.delta = []
		self.cost = np.inf
		self.a = []
		for i in layers:
			self.a.append(np.zeros((i,1)))	


	def train(self,X,y):
		''' mini-batch gradient descent '''
		if len(X.shape) == 1:
			X.shape = X.shape[0],1
		m,n = X.shape

		if X.dtype == 'object':
			X = np.float64(X)

		self.number_of_features = n

		y = self.onehotencode(y)		
		
		''' initializing the theta(feature) matrices and delta(derivatives w.r.t theta) matrices. '''
		self.theta.append(np.random.randn(self.layers[0],n)*(2/np.sqrt(n))-(1/np.sqrt(n)))
		self.delta.append(np.zeros((self.layers[0],n)))
		for i in range(1,len(self.layers)):
			self.theta.append(np.random.randn(self.layers[i],self.layers[i-1])*(2/np.sqrt(self.layers[i-1]))-(1/np.sqrt(self.layers[i-1])))
			self.delta.append(np.zeros((self.layers[i],self.layers[i-1])))
		self.theta.append(np.random.randn(len(self.labels),self.layers[-1])*(2/np.sqrt(self.layers[-1]))-(1/np.sqrt(self.layers[-1])))
		self.delta.append(np.zeros((len(self.labels),self.layers[-1])))
		self.a.append(np.zeros((len(self.labels),1)))

		''' initialization of variables for the adam update. '''
		if self.update == 'adam':
			beta_1 = 0.9
			beta_2 = 0.999
			epsilon = 1e-8
			m_t = []
			v_t = []
			t = 0
			m_t.append(np.zeros((self.layers[0],n)))
			v_t.append(np.zeros((self.layers[0],n)))
			for i in range(1,len(self.layers)):
				m_t.append(np.zeros((self.layers[i],self.layers[i-1])))
				v_t.append(np.zeros((self.layers[i],self.layers[i-1])))
			m_t.append(np.zeros((len(self.labels),self.layers[-1])))
			v_t.append(np.zeros((len(self.labels),self.layers[-1])))

		counter = 1
		itr = 0
		while counter <= self.epoch:
			prevcost = np.inf
			cost = 0
			
			start = 0
			end = start + self.batch_size
			
			if self.update == 'adam':
				t += 1
			while(end <= m):
				prevcost = cost
				cost = 0
				
				''' forward and backward pass. '''
				regsum = 0 #for regularization
				for i in range(len(self.theta)):
					regsum += np.sum(self.theta[i]*self.theta[i])
				for x,target in zip(X[start:end,:],y[start:end]):
					x.shape = x.shape[0],1
					target.shape = target.shape[0],1
					self.forward(x)
					self.backward(x,target)
					cost += ((-np.log(self.a[-1][target]))+self.l*regsum)/self.batch_size

				''' update the theta matrices. '''
				for i in range(len(self.theta)):
					if self.update == 'gd':
						self.theta[i] -= self.alpha*(1/self.batch_size)*(self.delta[i])
					elif self.update == 'adam':
						self.delta[i] = (1/self.batch_size)*self.delta[i]
						m_t[i] = beta_1*m_t[i] + (1-beta_1)*(self.delta[i])
						v_t[i] = beta_2*v_t[i] + (1-beta_2)*(self.delta[i]*self.delta[i])
						m_bias_correction = 1/(1-beta_1**t)
						v_bias_correction = 1/(1-beta_2**t)
						self.theta[i] -= (self.alpha * m_t[i] * m_bias_correction) / (np.sqrt(v_t[i]*v_bias_correction) + epsilon)

				''' initialize the delta matrices to zero. '''
				for index in range(len(self.delta)):
					r,c = self.delta[index].shape
					for i in range(r):
						for j in range(c):
							self.delta[index][i,j] = 0

				itr += 1

				''' increament to the next batch. '''
				start = end
				end = start + self.batch_size

				print(itr,cost,prevcost,prevcost - cost)
				self.cost = cost
			counter += 1
				

	def forward(self,x):
		self.a[0] = type(self).activation[self.activation_function](self.theta[0].dot(x))
		for i in range(1,len(self.a)-1):
			self.a[i] = type(self).activation[self.activation_function](self.theta[i].dot(self.a[i-1]))
		self.a[-1] = type(self).activation['softmax'](self.theta[-1].dot(self.a[-2]))


	def backward(self,x,y):
		delta1 = self.a[-1] - y #works for softmax output layer too.
		for i in range(len(self.theta)-1,0,-1):
			self.delta[i] += delta1.dot(self.a[i-1].T) + self.l*self.theta[i]
			delta0 = self.theta[i].T.dot(delta1)*type(self).derivative[self.activation_function](self.a[i-1])
			delta1 = delta0
		self.delta[0] += delta1.dot(x.T) + self.l*self.theta[0]


	def backwardtest(self,x,y):
		delta1 = self.a[-1] - y #works for softmax output layer too.
		for i in range(len(self.theta)-1,0,-1):
			self.delta[i] = delta1.dot(self.a[i-1].T) + self.l*self.theta[i]
			delta0 = self.theta[i].T.dot(delta1)*type(self).derivative[self.activation_function](self.a[i-1])
			delta1 = delta0
		self.delta[0] = delta1.dot(x.T) + self.l*self.theta[0]


	def predict(self,X):
		if X.dtype == 'object':
			X = np.float64(X)
		if (self.number_of_features == 1 and len(X) == 1) or (self.number_of_features != 1 and len(X.shape) == 1):
			#single example case
			X.shape = X.shape[0],1
			self.forward(X)
			prediction = np.ravel(self.a[-1] >= 0.5)
			if np.sum(prediction) > 1:
				return None
			else:
				return self.labels[prediction]
		else:
			#multiple example case
			res = []
			for i in X:
				i.shape = i.shape[0],1
				self.forward(i)
				prediction = np.ravel(self.a[-1] >= 0.5)
				if np.sum(prediction) > 1:
					res.append(None)
				else:
					res.append(self.labels[prediction])
			return res


	def gradientcheck(self,X,y):
		if len(X.shape) == 1:
			X.shape = X.shape[0],1
		m,n = X.shape

		if X.dtype == 'object':
			X = np.float64(X)

		self.number_of_features = n

		y = self.onehotencode(y)

		gdelta = []
		self.theta.append(np.random.randn(self.layers[0],n)*(2/np.sqrt(n))-(1/np.sqrt(n)))
		self.delta.append(np.zeros((self.layers[0],n)))
		gdelta.append(np.zeros((self.layers[0],n)))
		for i in range(1,len(self.layers)):
			self.theta.append(np.random.randn(self.layers[i],self.layers[i-1])*(2/np.sqrt(self.layers[i-1]))-(1/np.sqrt(self.layers[i-1])))
			self.delta.append(np.zeros((self.layers[i],self.layers[i-1])))
			gdelta.append(np.zeros((self.layers[i],self.layers[i-1])))
		self.theta.append(np.random.randn(len(self.labels),self.layers[-1])*(2/np.sqrt(self.layers[-1]))-(1/np.sqrt(self.layers[-1])))
		self.delta.append(np.zeros((len(self.labels),self.layers[-1])))
		gdelta.append(np.zeros((len(self.labels),self.layers[-1])))
		self.a.append(np.zeros((len(self.labels),1)))

		epsilon = 1e-4
		for x,target in zip(X,y):
			x.shape = x.shape[0],1
			target.shape = target.shape[0],1
			self.forward(x)
			self.backward(x,target)
			difference = 0
			for i in range(len(self.theta)):
				row,column = self.theta[i].shape
				for r in range(row):
					for c in range(column):
						self.theta[i][r,c] += epsilon
						self.forward(x)
						costplus = (-np.log(self.a[-1][target]))
						self.theta[i][r,c] -= (2*epsilon)
						self.forward(x)
						costminus = (-np.log(self.a[-1][target]))
						self.theta[i][r,c] += epsilon
						gdelta[i][r,c] += (costplus - costminus)/(2*epsilon)
		difference = (1/m)*np.sum(self.delta[i] - gdelta[i])
		print("gradientcheck complete with difference of: "+str(difference))


	def onehotencode(self,y):
		self.labels = np.array(list(set(y)))
		target = []
		for i in y:
			target.append(self.labels == i)
		return target


	def params(self):
		return self.theta


	def loss(self):
		return self.cost
