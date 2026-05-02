import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class nn():
    def __init__(self,learning_rate=0.1):
        np.random.seed(1)

        self.learning_rate=learning_rate
        
        #Layers
        input_nodes= 4
        hidden1_nodes= 6
        hidden2_nodes= 4
        output_nodes= 3

        self.weights= 2*np.random.random((input_nodes,hidden1_nodes))-1 #input->hidden1
        self.weights2= 2*np.random.random((hidden1_nodes,hidden2_nodes))-1 #hidden1->hidden2
        self.weights3= 2*np.random.random((hidden2_nodes,output_nodes))-1 #hidden2->output

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def derived_sigmoid(self,x):
        return x*(1-x)

    def softmax(self,x):
        e=np.exp(x-x.max(axis=1).reshape(-1,1))
        return e/e.sum(axis=1).reshape(-1,1)

    def forward_pass(self,input):
        self.input=input.astype(float)
        self.hidden1= self.sigmoid(np.dot(self.input,self.weights)) #input->hidden1
        self.hidden2=  self.sigmoid(np.dot(self.hidden1,self.weights2)) #hidden1->hidden2
        return self.softmax(np.dot(self.hidden2,self.weights3)) #hidden2->output"softmax" (3 classes)
         

    def training(self,input,actual_output,epochs):
        #calculating the errors
        error_history=[]
        for e in range(epochs):
            prediction= self.forward_pass(input)

            output_error= actual_output-prediction
            error_history.append(np.mean(np.abs(output_error)))  

            #Backpropagation
            output_gradient= -(output_error)/len(input)
            hidden2_error= output_gradient.dot(self.weights3.T)
            hidden2_gradient= hidden2_error*self.derived_sigmoid(self.hidden2)

            hidden1_error= hidden2_gradient.dot(self.weights2.T)
            hidden1_gradient= hidden1_error*self.derived_sigmoid(self.hidden1)
            
            #Weights update
            self.weights3-= self.learning_rate * self.hidden2.T.dot(output_gradient)
            self.weights2-= self.learning_rate * self.hidden1.T.dot(hidden2_gradient)
            self.weights-= self.learning_rate * input.T.dot(hidden1_gradient)

        return error_history


    def predict(self,X):
        probs=self.forward_pass(X)
        return np.argmax(probs,axis=1) #choosing the max col_val in each row

    def accuracy(self,X,y):
        predictions=self.predict(X)
        return np.mean(predictions == np.argmax(y,axis=1))*100


if __name__ == "__main__":

    def load_dataset():
        iris= load_iris()
        
        X=iris.data
        y= pd.get_dummies(iris.target).values

        #splitting
        X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=1)

        #Scaling the inputs
        scaler=StandardScaler() 
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)

        return X_train,X_test,y_train,y_test

    X_train,X_test,y_train,y_test= load_dataset()

    #Model Initialization
    model = nn(learning_rate=0.1)

    errors= model.training(X_train, y_train,5000)

    #Evaluation
    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)

    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy : {test_acc:.2f}%")

    plt.plot(errors)
    plt.title("Training Error")
    plt.show()
