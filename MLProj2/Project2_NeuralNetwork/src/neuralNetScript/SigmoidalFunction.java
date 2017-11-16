/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class SigmoidalFunction implements INodeFunction {

	// computes the hyperbolic tangent of the weighted sum for hidden nodes in a multilayer perceptron network
	@Override
	public double computeOutput(double[][] inputs) {
		// sum weighted inputs
		double sum = 0;
		for (int i = 0; i < inputs[0].length; i++) {
			sum += (inputs[0][i] * inputs[1][i]);
		}
		// return hyperbolic tangent
		return Math.tanh(sum) + 1.0;
	}
}
