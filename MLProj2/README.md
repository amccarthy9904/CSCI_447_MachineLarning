# MLProj2
This is a project for CSCI 447: Soft Computing with Dr. John Sheppard at Montana State Univeristy.
This program is a runnable .jar file that builds and tests a neural network for approximating the Rosenbrock function.   

# How To Use
For MLP network: `java -jar neuralNet.jar mlp <numInputs>-<numHiddenNodes>-<numOutputs>`    
For RBF network: `java -jar neuralNet.jar rbf <numInputs>-<numFunctions>-<numOutputs>`   
   
# Examples
MLP, 2 inputs, no hidden layers, 1 output   
`java -jar neuralNet.jar mlp 2-1`   

MLP, 2 inputs, 1 hidden layer with 10 nodes, 1 output   
`java -jar neuralNet.jar mlp 2-10-1`   

MLP, 2 inputs, 1 hidden layer with 10 nodes, 1 hidden layer with 5 nodes, 1 output   
`java -jar neuralNet.jar mlp 2-10-5-1`    

RBF, 2 inputs, 10 functions, 1 output   
`java -jar neuralNet.jar rbf 2-10-1`

