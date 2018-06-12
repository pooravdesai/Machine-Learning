import numpy as np

class logisticRegression:

	logistic = lambda x: 1/(1+np.exp(-x))

	def __init__(self,numberOfFeatures,alpha = 0.01,l = 0):
		self.numberOfFeatures = numberOfFeatures
		self.alpha = alpha
		self.l = l
		self.cost = 0
		#self.theta = np.random.randn(self.numberOfFeatures+1,1)
		self.theta = np.zeros((self.numberOfFeatures+1,1))

	def train(self,X,y):
		if self.numberOfFeatures == 1:
			X.shape = X.shape[0],1
		y.shape = y.shape[0],1

		m,n = X.shape
		X = np.hstack((np.ones((m,1)),X))

		htheta = type(self).logistic(X.dot(self.theta))
		cost = -(1/m)*(y.T.dot(np.log(htheta))+(1-y).T.dot(np.log(1-htheta))) + (0.5*self.l)*(self.theta.T.dot(self.theta) - self.theta[0,0]**2)
		prevCost = np.inf
		while prevCost - cost > 0.000001:
			temp = self.theta[0,0]
			self.theta -= (self.alpha/m)*(X.T.dot(htheta-y) + self.l*self.theta)
			self.theta[0,0] += self.l*temp	#we dont regularize theta0
			htheta = type(self).logistic(X.dot(self.theta))
			prevCost = cost
			cost = -(1/m)*(y.T.dot(np.log(htheta))+(1-y).T.dot(np.log(1-htheta))) + (0.5*self.l)*(self.theta.T.dot(self.theta) - self.theta[0,0]**2)

		self.cost = cost
		return

	def predict(self,X):
		'''
			X: an input vector for size (m,n) or a single input array of size (n,).
			retval: returns the prediction or the vector of predictions. In case of multi class classification, it returns a one-hot vector.
		'''
		if self.numberOfFeatures == 1:
			X.shape = X.shape[0],1
		elif len(X.shape) == 1:
			X.shape = 1,X.shape[0]
		m,n = X.shape
		X = np.hstack((np.ones((m,1)),X))
		prediction = type(self).logistic(X.dot(self.theta))
		prediction[prediction >= 0.5] = 1
		prediction[prediction < 0.5] = 0
		return prediction

	def params(self):
		return self.theta

	def loss(self):
		return self.cost


