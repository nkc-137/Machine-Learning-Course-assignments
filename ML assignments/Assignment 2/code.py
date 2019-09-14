"""                                        FIRST PART                     """

# Helper functions used
def compute_cost(yhat,y):
    m = len(y)
    tot_error = 0
    error = 0
    for i in range(m):
        error1 = y[i]*np.log(sigmoid(yhat[i]))
        error2 = (1-y[i])*np.log(sigmoid(1-yhat[i]))
        tot_error += (error1 + error2)

    return -1/m*tot_error

def sigmoid(z):
    ans = 1/(1+np.exp(-1*z))
    return ans

def initialize_model(X_train,y_train):
    model = {}
    w = np.zeros((X_train.shape[1],1))
    b = np.zeros((1))

    model['w'] = w
    model['b'] = b
#     yhat = np.dot(X_train,w) + b
#     cost = compute_cost(yhat,y_train)
    return model

def gen_data(w1,w2,b,n):
    x = []
    y = []
    for i in range(n):
        point = [20*np.random.ranf() - 10,20*np.random.ranf() - 10]
        x.append(point)
        if w1*point[0] + w2*point[1] + b >= 0:
            y.append(1)
        else:
            y.append(0)
    x = np.array(x)
    y = np.array(y)
    return (x,y)
def gen_ellipse(w1,w2,b,n):
    x = []
    y = []
    for i in range(n):
        point = [20*np.random.ranf() - 10,20*np.random.ranf() - 10]
        x.append(point)
        if (1/w1**2)*point[0]**2 + (1/w2**2)*point[1]**2 + b >= 0:
            y.append(1)
        else:
            y.append(0)
    x = np.array(x)
    y = np.array(y)
    return (x,y)
def accuracy(yhat,y):
    m = len(y)
    correct = 0
    yhat = np.array(list(map(lambda x: x>=0.5,yhat)))
#     print(yhat.shape)
    for i in range(m):
        if (yhat[i] == y[i]):
            correct += 1
    return correct/m


"""                      MOST GENERAL CODE (USED FOR BOTH THE PARTS)        """

def train_model(model,X_train,y_train,num_iterations=100,learning_rate=0.01,grad_type='batch'):
    if (grad_type == 'batch'):
        costs = []
        m = len(y_train)
        y_train = y_train.reshape(y_train.shape[0],1)
        model = initialize_model(X_train,y_train)
        w = model['w']
        b = model['b']
        yhat = sigmoid(np.dot(X_train,w) + b)
        for i in range(num_iterations):
            w = w - 1/m*np.dot(X_train.T,(yhat-y_train))*learning_rate
            b = b - 1/m*np.sum(yhat-y_train)*learning_rate
            yhat = sigmoid(np.dot(X_train,w) + b)

        model['w'] = w
        model['b'] = b
    if (grad_type == 'stochastic'):
        costs = []
        model = initialize_model(X_train,y_train)
        w = model['w']
        b = model['b']
        m = len(y_train)
        yhat = sigmoid(np.dot(X_train,w) + b)
        for i in range(num_iterations):
            for j in range(m):
                yhat = sigmoid(np.dot(X_train,w) + b)
                arr = X_train[j,:].reshape(X_train.shape[1],1)
                grad = np.dot(arr,(yhat[j]-y_train[j]))
                grad = grad.reshape(X_train.shape[1],1)
                vb = (yhat[j]-y_train[j])
                w = w - learning_rate*grad
                b = b - learning_rate*(yhat[j]-y_train[j])
#                 cost = compute_cost(y_train,yhat)
#                 if (count%1==0):
#                     costs.append(cost)
        model['w'] = w
        model['b'] = b
    return (costs,model)
