/**
 * 
 */
package neuralNetworkTrainer;

import java.util.ArrayList;

/**
 * @author Elias
 *
 */
class Node {
	
	/**
	 * the inputs to this node
	 */
	private ArrayList<Object> inputs;
	
	/**
	 * the associated weights to the inputs
	 */
	private ArrayList<Double> weights;

	/**
	 * the previous change in weights
	 */
	private ArrayList<Double> prevWeightChange;

	/**
	 * the downstream nodes
	 */
	private final ArrayList<Node> downstream;
	
	/**
	 * the index of this node in its containing layer
	 */
	private final int indexInLayer;
	
	/**
	 * the output computed by this node
	 */
	private Object computedOutput;
	
	/**
	 * the delta value for this node used by the Backprop algorithm
	 */
	private Double backpropDelta;
	
	/**
	 * the NodeFunction that defines how this node computes an output
	 */
	private final INodeFunction nodeFunction;
	
	/**
	 * Constructs a node
	 * 
	 * @param nodeFunction the INodeFunction used to compute the output for this node
	 * @param downstreamNodes the list of downstream nodes
	 */
	Node(INodeFunction nodeFunction, ArrayList<Node> downstreamNodes, int indexInLayer){
		this.inputs = new ArrayList<>();
		this.weights = new ArrayList<>();
		this.prevWeightChange = new ArrayList<>();
		this.downstream = downstreamNodes;
		this.nodeFunction = nodeFunction;
		this.indexInLayer = indexInLayer;
	}
	
	/**
	 * Gets the inputs to this node
	 * @return the inputs to this node
	 */
	ArrayList<Object> getInputs(){
		return this.inputs;
	}
	
	/**
	 * Gets the weights associated to the inputs of this node
	 * @return the weights of this node
	 */
	ArrayList<Double> getWeights(){
		return this.weights;
	}

	/**
	 * Gets the previous changes in weights of this node
	 * @return the previous changes in weights of this node
	 */
	ArrayList<Double> getPrevWeightChange(){ return this.prevWeightChange; }
	
	/**
	 * Gets the downstream nodes of this node
	 * @return the downstream nodes of this node
	 */
	ArrayList<Node> getDownstreamNodes(){
		return this.downstream;
	}
	
	/**
	 * Gets the index of this node in its containing layer
	 * @return the index of this node in its containing layer
	 */
	int getIndexInLayer(){
		return this.indexInLayer;
	}
	
	/**
	 * Gets the computed output of this node
	 * @return the computed output of this node
	 */
	Object getComputedOutput(){
		return this.computedOutput;
	}
	
	/**
	 * Gets the backprop delta value of this node
	 * @return the backprop delta value
	 */
	Double getBackpropDelta(){
		return this.backpropDelta;
	}
	
	/**
	 * Sets the backprop delta value of this node
	 * @param delta the backprop delta value to set for this node
	 */
	void setBackpropDelta(Double delta){
		this.backpropDelta = delta;
	}
	
	/**
	 * Calls the associated node function execute method and sets the input of its downstream nodes accordingly
	 */
	void activateNode(){
		this.computedOutput = this.nodeFunction.execute(this);
		for(Node downstreamNode : this.downstream){
			if(downstreamNode.getInputs().size() > this.indexInLayer){
				downstreamNode.getInputs().remove(this.indexInLayer);
			}
			downstreamNode.getInputs().add(this.indexInLayer, this.computedOutput);
		}
	}
	
	/**
	 * Gets the derivative of the activation function computed on the output
	 * @return the derivative of the activation function computed on the output
	 */
	Object getDerivative(){
		return this.nodeFunction.getDerivative();
	}
}
