/**
 * 
 */
package neuralNetworkTrainer;

/**
 * Defines how a Node computes an output
 * 
 * @author Elias
 *
 */
interface INodeFunction {

	/**
	 * Computes the output for the given node
	 * @param node the node to compute the output for
	 * @return the computed output of the given node
	 */
	Object execute(Node node);

	/**
	 * Returns the derivative of the node function on the node's output
	 * @return the derivative of the node function computed on the output
	 */
	Object getDerivative();
}
