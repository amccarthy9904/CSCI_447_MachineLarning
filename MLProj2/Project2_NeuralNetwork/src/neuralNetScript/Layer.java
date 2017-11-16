/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class Layer {
	// the layer of nodes
	private Node[] nodes;
	
	// return the nodes
	public Node[] getNodes(){
		return nodes;
	}
	
	// set the nodes
	public void setNodes(Node[] nodes){
		this.nodes = nodes;
	}
}
