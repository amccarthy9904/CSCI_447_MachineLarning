/**
 * 
 */
package neuralNetworkTrainer;

/**
 * @author Elias
 *
 */
abstract class TrainingAlgorithm {

	/**
	 * trains a network using a concrete TrainingAlgorithm
	 * @return the trained network
	 */
	abstract Network train();
}
