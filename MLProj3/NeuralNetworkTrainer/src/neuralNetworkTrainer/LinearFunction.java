/**
 * 
 */
package neuralNetworkTrainer;

/**
 * @author Elias
 *
 */
class LinearFunction implements INodeFunction {

	/**
	 * The derivative of the linear function (sum of the weights) computed on the output after the node is activated is constant. So we make it 1.0
	 */
	private final Double derivative = 1.0;

	/* 
	 * (non-Javadoc)
	 * @see neuralNetworkTrainer.INodeFunction#execute(neuralNetworkTrainer.Node)
	 */
	@Override
	public Object execute(Node node) {
		Double weightedSum = 0.0;
		for(int inputIter = 0; inputIter < node.getInputs().size(); inputIter++){
			weightedSum += (Double)node.getInputs().get(inputIter) * node.getWeights().get(inputIter);
		}
		return weightedSum;
	}

	/*
	 * (non-Javadoc)
	 * @see neuralNetworkTrainer.INodeFunction#getDerivative()
	 */
	@Override
	public Object getDerivative() {
		return this.derivative;
	}
}
