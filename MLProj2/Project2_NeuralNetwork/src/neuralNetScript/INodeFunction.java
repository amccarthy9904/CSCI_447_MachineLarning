/**
 * 
 */
package neuralNetScript;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
interface INodeFunction {
	// computes the output for any node given a weighted sum of its inputs
	double computeOutput(double[][] inputs);
}
