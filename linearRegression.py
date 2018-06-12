import numpy as np

class linearRegression:
	''' 
		numberOfFeatures: number of features excluding bias.
		alpha: learning rate
		l: regularization parameter
	'''
	def __init__(self,numberOfFeatures,alpha = 0.01,l = 0):
		self.numberOfFeatures = numberOfFeatures
		self.alpha = alpha
		self.l = l
		self.theta = np.random.randn(self.numberOfFeatures+1,1)	#theta is a n+1 * 1 parameter vector where +1 is for the bias.
		#self.theta = np.zeros((self.numberOfFeatures+1,1))

	def train(self,X,y):
		if len(X.shape) == 1:
			X.shape = X.shape[0],1
		y.shape = y.shape[0],1
		m,n = X.shape
		if n != self.numberOfFeatures:
			print("dimension mismatch.")
			return

		''' adding a column of ones to account for the bias. '''
		X = np.hstack((np.ones((m,1)),X))

		''' Applying gradiant descent. '''
		prevCost = np.inf
		while(True):
			hTheta = X.dot(self.theta)
			cost = (0.5/m)*(hTheta-y).T.dot(hTheta-y) + self.l*(self.theta.T.dot(self.theta) - self.theta[0]**2)
			temp = self.theta[0,0]
			self.theta -= (self.alpha/m)*(X.T.dot(hTheta-y) + self.l*self.theta)
			self.theta[0,0] += self.l*temp	#we dont regularize theta0
			if prevCost - cost > 0.000001:
				prevCost = cost
			else:
				self.cost = cost
				return

	def predict(self,X):
		'''
			X: an input vector for size (m,n) or a single input array of size (n,).
			retval: returns the prediction or the vector of predictions.
		'''
		
		if self.numberOfFeatures == 1:
			X.shape = X.shape[0],1
		elif len(X.shape) == 1:
			X.shape = 1,X.shape[0]
		m,n = X.shape 
		X = np.hstack((np.ones((m,1)),X))
		return X.dot(self.theta)

	def params(self):
		return self.theta

	def loss(self):
		return self.cost


