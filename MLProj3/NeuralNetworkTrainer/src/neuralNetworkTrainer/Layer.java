/**
 * 
 */
package neuralNetworkTrainer;

import java.util.ArrayList;

/**
 * The Layer class is a container for a set of nodes.
 * 
 * @author Elias
 *
 */
class Layer {

	/**
	 * The set of nodes in this layer
	 */
	private ArrayList<Node> nodes;
	
	/**
	 * Constructs a new Layer
	 */
	Layer(){
		nodes = new ArrayList<Node>();
	}
	
	/**
	 * Gets the nodes in this layer
	 * @return the nodes in this layer
	 */
	ArrayList<Node> getNodes(){
		return this.nodes;
	}
}
