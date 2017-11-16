/**
 * 
 */
package neuralNetworkTrainer;

import java.util.ArrayList;

/**
 * The Network class is a container for a neural network.
 * 
 * @author Elias
 *
 */
class  Network implements Comparable<Network> {

	/**
	 * The input layer
	 */
	private Layer inputLayer;
	
	/**
	 * The hidden layers
	 */
	private ArrayList<Layer> hiddenLayers;
	
	/**
	 * The output layer
	 */
	private Layer outputLayer;

	/**
	 * The fitness of this network
	 */
	private double fitness;


	/**
	 * Constructs a network clone based off another Network
	 */
	Network(Network clone){
		this.inputLayer = clone.inputLayer;
		this.outputLayer = clone.outputLayer;
		this.hiddenLayers = clone.hiddenLayers;
		this.fitness = clone.fitness;
	}

	/**
	 * Constructs a network based off the configuration in the Driver
	 */
	Network(boolean setRandomWeights){
		
		this.inputLayer = new Layer();
		this.hiddenLayers = new ArrayList<>();
		this.outputLayer = new Layer();

		ArrayList<ArrayList<Double>> weights = new ArrayList<>();
		ArrayList<ArrayList<Double>> weightChange = new ArrayList<>();
		
		// create output layer, these nodes use linear function for regression problem and sigmoidal for classification; they have no downstream nodes
		for(int nodeIter = 0; nodeIter < Driver.configuration.get(Driver.configuration.size() - 1); nodeIter++){
			if(Driver.isClassificationNetwork){
				this.outputLayer.getNodes().add(nodeIter, new Node(new SigmoidalFunction(), new ArrayList<Node>(), nodeIter));
//				this.outputLayer.getNodes().add(nodeIter, new Node(new LinearFunction(), new ArrayList<Node>(), nodeIter));
			}

			else{
				this.outputLayer.getNodes().add(nodeIter, new Node(new LinearFunction(), new ArrayList<Node>(), nodeIter));
			}
			if(setRandomWeights){
				ArrayList<Double> weightVector = new ArrayList<>();
				ArrayList<Double> weightChangeVector = new ArrayList<>();
				for(int weightIter = 0; weightIter < Driver.configuration.get(Driver.configuration.size() - 2); weightIter++){
					weightVector.add(weightIter, Math.pow(-1, (int)(Math.random() * 2)) * Math.random() * 0.5);
					weightChangeVector.add(weightIter, 0.0);
				}
				weights.add(weightVector);
				weightChange.add(weightChangeVector);
			}
		}
		
		// create hidden layers in reverse, starting at the second to last index of configuration
		ArrayList<Node> downstreamNodes = this.outputLayer.getNodes();
		for(int layerIter = Driver.configuration.size() - 2; layerIter > 0; layerIter--){
			this.hiddenLayers.add(new Layer());
		}
		for(int layerIter = Driver.configuration.size() - 2; layerIter > 0; layerIter--){

			// create hidden nodes for this layer, these nodes use sigmoidal function
			for(int nodeIter = 0; nodeIter < Driver.configuration.get(layerIter); nodeIter++){
				this.hiddenLayers.get(layerIter - 1).getNodes().add(nodeIter, new Node(new SigmoidalFunction(), downstreamNodes, nodeIter));

				// set hidden weights between -0.5 and +0.5
				if(setRandomWeights){
					ArrayList<Double> weightVector = new ArrayList<>();
					ArrayList<Double> weightChangeVector = new ArrayList<>();
					for(int weightIter = 0; weightIter < Driver.configuration.get(layerIter - 1); weightIter++){
						weightVector.add(weightIter, Math.pow(-1, (int)(Math.random() * 2)) * Math.random() * 0.5);
						weightChangeVector.add(weightIter, 0.0);
					}
					weights.add(nodeIter, weightVector);
					weightChange.add(nodeIter, weightChangeVector);
				}
			}
			downstreamNodes = this.hiddenLayers.get(layerIter - 1).getNodes();
		}
		
		// create input layer, these node use sigmoidal function
		for(int nodeIter = 0; nodeIter < Driver.configuration.get(0); nodeIter++){
			this.inputLayer.getNodes().add(nodeIter, new Node(new SigmoidalFunction(), downstreamNodes, nodeIter));

			// set input weights to 1
			if(setRandomWeights){
				ArrayList<Double> weightVector = new ArrayList<>();
				ArrayList<Double> weightChangeVector = new ArrayList<>();
				weightVector.add(0, 1.0);
				weightChangeVector.add(0, 0.0);
				weights.add(nodeIter, weightVector);
				weightChange.add(nodeIter, weightChangeVector);
			}
		}

		this.setWeights(weights, false);
		this.setWeights(weightChange, true);
	}
	
	/**
	 * Gets the input layer of this network
	 * @return the input layer of this network
	 */
	Layer getInputLayer(){
		return this.inputLayer;
	}
	
	/**
	 * Gets the hidden layers of this network
	 * @return the hidden layers of this network
	 */
	ArrayList<Layer> getHiddenLayers(){
		return this.hiddenLayers;
	}
	
	/**
	 * Gets the output layer of this network
	 * @return the output layer of this network
	 */
	Layer getOutputLayer(){
		return this.outputLayer;
	}
	
	/**
	 * Creates a weighted adjacency "matrix" (really a list of lists) representing this network in its current state.
	 * @param network the network to serialize
	 * @return an adjacency "matrix" representing this network and its weights.
	 * 		   The top-level list contains an entry for each node; index 0 is the first input node; the final index is the final output node.
	 * 		   The sub-list contains weights for the top-level index Node (ie list.get(A) is the weights of node A).
	 */
	static ArrayList<ArrayList<Double>> serializeNetwork(Network network, boolean isWeightChange){
		
		ArrayList<ArrayList<Double>> weights = new ArrayList<ArrayList<Double>>();
		for(Node node : network.inputLayer.getNodes()){
			if(isWeightChange){
				weights.add((ArrayList<Double>)node.getPrevWeightChange().clone());
			}
			else{
				weights.add((ArrayList<Double>)node.getWeights().clone());
			}
		}
		for(Layer hiddenLayer : network.getHiddenLayers()){
			for(Node node : hiddenLayer.getNodes()){
				if(isWeightChange){
					weights.add((ArrayList<Double>)node.getPrevWeightChange().clone());
				}
				else{
					weights.add((ArrayList<Double>)node.getWeights().clone());
				}
			}
		}
		for(Node node : network.outputLayer.getNodes()){
			if(isWeightChange){
				weights.add((ArrayList<Double>)node.getPrevWeightChange().clone());
			}
			else{
				weights.add((ArrayList<Double>)node.getWeights().clone());
			}
		}
		return weights;
	}
	
	/**
	 * Creates a Network given a representative weighted adjacency matrix
	 * @param weights matrix the weighted adjacency matrix representing a network
	 * @return the network represented by the weighted adjacency matrix
	 */
	static Network deserializeToNetwork(ArrayList<ArrayList<Double>> weights){
		
		Network network = new Network(true);
		network.setWeights(weights, false);
		return network;
	}
	
	/**
	 * Sets the weights in this network
	 * @param weights the weight "matrix" (list of lists) representing the weights
	 * 		  The top-level list should contain an entry for each node; index 0 is the first input node; the final index is the final output node.
	 * 		  The sub-list should contain the weights for the top-level index Node (ie list.get(A) is the weights of node A).
	 * @param isWeightChange if true, sets the prevWeightChange of the nodes instead
	 */
	void setWeights(ArrayList<ArrayList<Double>> weights, Boolean isWeightChange){

		int weightsIter = 0;
		for(Node node : this.inputLayer.getNodes()){
			if(isWeightChange){
				node.getPrevWeightChange().clear();
			}
			else{
				node.getWeights().clear();
			}
			for(Double weight : weights.get(weightsIter)){
				if(isWeightChange){
					node.getPrevWeightChange().add(weight);
				}
				else{
					node.getWeights().add(weight);
				}
			}
			weightsIter++;
		}
		for(Layer hiddenLayer : this.getHiddenLayers()){
			for(Node node : hiddenLayer.getNodes()){
				if(isWeightChange){
					node.getPrevWeightChange().clear();
				}
				else{
					node.getWeights().clear();
				}
				for(Double weight : weights.get(weightsIter)){
					if(isWeightChange){
						node.getPrevWeightChange().add(weight);
					}
					else{
						node.getWeights().add(weight);
					}
				}
				weightsIter++;
			}
		}
		for(Node node : this.outputLayer.getNodes()){
			if(isWeightChange){
				node.getPrevWeightChange().clear();
			}
			else{
				node.getWeights().clear();
			}
			for(Double weight : weights.get(weightsIter)){
				if(isWeightChange){
					node.getPrevWeightChange().add(weight);
				}
				else{
					node.getWeights().add(weight);
				}
			}
			weightsIter++;
		}
	}

	/**
	 * runs the network on one datapoint
	 * @param inputs datapoint to run the network on
	 * @return true if the network classified the point properly, false otherwise
	 */
	public boolean evaluate(ArrayList<Object> inputs){

		this.setInputs(inputs);
		ArrayList<Object> output = this.executeNodes();
		double correctIndex = (double)inputs.get(inputs.size() - 1);
		int largestValuedIndex = 0;
		for(int outputIter = 1; outputIter < output.size(); outputIter++){
			if((double)output.get(outputIter) > (double)output.get(largestValuedIndex)){
				largestValuedIndex = outputIter;
			}
		}
		if(largestValuedIndex == correctIndex){
			return true;
		}
		return false;
	}


	/**
	 * execute the nodes in the network
	 * @return the computed output of the network
	 */
	public ArrayList<Object> executeNodes(){

		for(Node inputNode : this.getInputLayer().getNodes()){
			inputNode.activateNode();
		}
		for(Layer hiddenLayer : this.getHiddenLayers()){
			for(Node hiddenNode : hiddenLayer.getNodes()){
				hiddenNode.activateNode();
			}
		}
		ArrayList<Object> networkOutput = new ArrayList<>();
		for(Node outputNode : this.getOutputLayer().getNodes()){
			outputNode.activateNode();
			networkOutput.add(outputNode.getComputedOutput());
		}
		return networkOutput;
	}

	/**
	 * sets all inputs in the network
	 * @param inputs values to assign to the input nodes -- might contain class too, need to modify for loop
	 */
	public void setInputs(ArrayList<Object> inputs){

		//for each input node assign the corresponding input to that node
		for (int inputIter = 0; inputIter < (inputs.size() - 1); inputIter++) {
			this.inputLayer.getNodes().get(inputIter).getInputs().clear();
			this.inputLayer.getNodes().get(inputIter).getInputs().add(inputs.get(inputIter));
		}
	}

	/**
	 * clears all inputs in the network
	 */
	public void clearInputs(){
		for(Node node : this.inputLayer.getNodes()){
			node.getInputs().clear();
		}
		for(Layer layer : this.hiddenLayers){
			for(Node node : layer.getNodes()){
				node.getInputs().clear();
			}
		}
		for(Node node : this.outputLayer.getNodes()){
			node.getInputs().clear();
		}
	}

	public double getFitness() {
		return this.fitness;
	}

	public void setFitness(double fitness) {
		this.fitness = fitness;
	}


	@Override
	public int compareTo(Network other) {

		if (this.getFitness() < other.getFitness()) {
			return -1;
		} else if (this.getFitness() > other.getFitness()) {
			return 1;
		} else {
			return 0;
		}
	}
}

