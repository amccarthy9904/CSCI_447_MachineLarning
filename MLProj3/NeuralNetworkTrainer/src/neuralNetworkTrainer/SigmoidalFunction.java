/**
 * 
 */
package neuralNetworkTrainer;

/**
 * @author Elias
 *
 */
class SigmoidalFunction implements INodeFunction{

	/**
	 * The derivative of the sigmoid function (x*(1-x)) computed on the output after the node is activated
	 */
	private Double derivative;
	
	/*
	 * (non-Javadoc)
	 * @see neuralNetworkTrainer.INodeFunction#execute(neuralNetworkTrainer.Node)
	 */
	@Override
	public Object execute(Node node){
		
		Double weightedSum = 0.0;
		for(int inputIter = 0; inputIter < node.getInputs().size(); inputIter++){
			if(node.getInputs().get(inputIter).getClass().getTypeName().equals("java.lang.Double")){
				weightedSum += (Double)node.getInputs().get(inputIter) * node.getWeights().get(inputIter);
			}
			else if(node.getInputs().get(inputIter).getClass().getTypeName().equals("java.lang.Integer")){
				weightedSum += (Integer)node.getInputs().get(inputIter) * node.getWeights().get(inputIter);
			}
		}
		Double output = 1 / (1 + Math.exp(-1 * weightedSum));
		this.derivative = output * (1 - output);
		return output;
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
