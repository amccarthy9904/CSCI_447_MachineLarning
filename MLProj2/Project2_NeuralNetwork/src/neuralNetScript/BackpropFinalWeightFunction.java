/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class BackpropFinalWeightFunction implements IWeightFunction {
	 
	//Ada, the learning rate
	private double learningRate = 0.05;

	// updates the weights using the backpropagation rule for output nodes
	@Override
	public void computeWeights(Node node) {
		double delta = 0; 
		// determine current node delta error term
		delta = node.getComputedOutput() * 
				(1 - node.getComputedOutput()) *
				(Driver.expectedOutput - node.getComputedOutput());
		node.setDeltaValue(delta);
		// update weights
		for(int i = 0; i < node.inputs[0].length; i++){
			node.inputs[1][i] += this.learningRate * delta * node.inputs[0][i];
		}
	}
}
