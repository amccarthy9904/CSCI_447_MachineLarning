/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
class NoFunction implements INodeFunction {

	// returns the weighted sum
	@Override
	public double computeOutput(double[][] inputs) {
		// sum weighted inputs
		double sum = 0;
		for (int i = 0; i < inputs[0].length; i++) {
			sum += (inputs[0][i] * inputs[1][i]);
		}
		return sum;
	}

}
