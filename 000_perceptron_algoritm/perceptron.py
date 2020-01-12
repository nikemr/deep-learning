''' Slightly diffent solution is here: https://www.youtube.com/watch?v=jbluHIgBmBo '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
"""this %matplotlib command opens the figure below in a seperate window in Ipython but
doesn't work in terminal. Need to learn how is it possible in terminal""" 
%matplotlib

#exampleFile = open('./000_perceptron_algoritm/data.csv')
#exampleReader = csv.reader(exampleFile)
#exampleData = list(exampleReader) 
exampleData=pd.read_csv('./data.csv')
#all dataframe from csv
df=pd.DataFrame(data=exampleData)
#new dataframe for labels (yy)
yy=df.y
#new dataframe contains only X1 X2 (two data point)
dd=pd.DataFrame(df, columns=['X1','X2'])

values=dd.values

#print(dd.X1.values)
#print(df.head())
#print(yy.head())
#dd.X1[0]

#just for understanding data, it is not necessary
df1=df[df.y==1]#label 1 (above)
df2=df[df.y==0]#label 0 (below)

#only works in jupyter notebook
df1.head()

Weight = np.array(np.random.rand(2,1))
b= (np.random.rand(1)[0])# Bias between 0 and 1

# multiplication
#print(Weight[0]*dd.X1[0]+Weight[1]*dd.X2[0]+b)

'''numPy multiplication w1*x1+w2*x2+bias Weight and bias is always updated according to 
perceptronStep'''
def predictions(values,Weight,b):    
    pred=np.matmul(values,Weight)+b
    return pred
'''checks if multiplication is below or above the seperation line and 
used in the percetronStep at the comparison piece(if statement) '''

def side(predictions):
    si=[]
    for i in range(len(predictions)):
        if predictions[i]>=0:
            si.append(1)
        else:
            si.append(0)
    return si    

def line(Weight,b):#for drawing seperation line according to weight and biases
    ys=[]
    #regular x values
    xs=[.1,.2,.3,.4,.5,.6,.7,.8,.0,1.0]
    #calculated y values for line
    [ys.append((-(Weight[0]*xs[i])-b)/Weight[1]) for i in range(len(xs))]            
    return xs,ys

"""most parts coming from Udacity perceptron trick lesson.It adjusts weights according
to points that on the wrong side of the seperation line"""
def perceptronStep(yy,side,b,dd,learn_rate = 0.001):
    g=0
    for i in range(100):        
        if yy[i]-side[i] == -1:
            Weight[0] -= dd.X1[i]*learn_rate
            Weight[1] -= dd.X2[i]*learn_rate
            b -= learn_rate
        elif yy[i]-side[i] == 1:
            Weight[0] += dd.X1[i]*learn_rate
            Weight[1] += dd.X2[i]*learn_rate
            b += learn_rate
        if yy[i]-side[i]==0:
            g+=1        
            
    return Weight,b,g
#epochs
for t in range(1800):    
    Weight,b,g=perceptronStep(yy,side(predictions(dd.values,Weight,b)),b,dd, learn_rate = 0.0001)    
    x,y=line(Weight,b)    
    if t%10==1:              
        plt.plot(x,y,)
        plt.show(block=False)        
        plt.scatter(data=df1,x='X1',y='X2',color='salmon' )
        plt.scatter(data=df2,x='X1',y='X2',color='violet' )
        plt.pause(.1)       
        print('Number of points at the right side of the line:',g)   