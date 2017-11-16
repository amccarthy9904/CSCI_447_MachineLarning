/**
 * 
 */
package neuralNetworkTrainer;

import java.util.ArrayList;
import java.util.Random;
import java.util.regex.Pattern;

/**
 * @author Elias
 *
 */
class Driver {

	/**
	 * The data set
	 */
	public static Data dataset;

	/**
	 * The training algorithm to use
	 */
	private static TrainingAlgorithm trainingAlgorithm;

	/**
	 * The configuration of the network
	 */
	static ArrayList<Integer> configuration = new ArrayList<>();

	/**
	 * True if the current problem is a classification problem; false if it is a linear regression problem
	 */
	static boolean isClassificationNetwork;

	/**
	 * Backprop parameters
	 */
	static double learningRate;
	static double momentum;

	/**
	 * Evolutionary algorithm parameters
	 */
	static int populationSize;  // "mu"
	static int numberOffspring; // "lambda"
	static double mutationRate; // used by GA and ES
	static double beta;         // used by DE
	
	/**
	 * The entry point of the application
	 * @param args args[0] is the datafile
	 *             args[1] is -bp/-ga/-es/-de specifying the training algorithm
	 *             args[2] is the configuration of the network(ie 1-2-3)
	 */
	public static void main(String[] args) {

		// Check for three required parameters
		if(args.length < 3){
			Driver.displayHelpText();
			System.exit(0);
		}
		else{
			// set required parameters, exit if they are not correct
			if(!Driver.setDataFile(args[0])
				|| !Driver.setTrainingAlgorithm(args[1])
				|| !configureNetwork(args[2]))
			{
				Driver.displayHelpText();
				System.exit(0);
			}

			// set default values for all optional parameters
			Driver.learningRate = 0.01;
			Driver.momentum = 0.5;
			Driver.populationSize = 25;
			Driver.numberOffspring = 100;
			Driver.mutationRate = 0.05;
			Driver.beta = 0.1;

			// check for any additional options (parameters), set accordingly
			Pattern dashPattern = Pattern.compile("\\A-\\w+");
			for(int argIter = 3; argIter < args.length; argIter++){
				if(dashPattern.matcher(args[argIter]).matches()){
					switch(args[argIter]){
						// learning rate
						case "-lr":
							if(argIter + 1 < args.length && Pattern.matches("\\d+\\.\\d+", args[argIter + 1])){
								Driver.learningRate = Float.parseFloat(args[++argIter]);
							}
							else{
								System.out.println("-lr must be followed by a float value for the Backprop learning rate\n");
								System.exit(0);
							}
							break;
						// momentum
						case "-m":
							if(argIter + 1 < args.length && Pattern.matches("\\d+\\.\\d+", args[argIter + 1])){
								Driver.momentum = Float.parseFloat(args[++argIter]);
							}
							else{
								System.out.println("-m must be followed by a float value for the Backprop momentum\n");
								System.exit(0);
							}
							break;
						// size of population
						case "-p":
							if(argIter + 1 < args.length && Pattern.matches("\\d+", args[argIter + 1])){
								Driver.populationSize = Integer.parseInt(args[++argIter]);
							}
							else{
								System.out.println("-p must be followed by a positive integer for the population size\n");
								System.exit(0);
							}
							break;
						// number of offspring generated each iteration
						case "-o":
							if(argIter + 1 < args.length && Pattern.matches("\\d+", args[argIter + 1])){
								Driver.numberOffspring = Integer.parseInt(args[++argIter]);
							}
							else{
								System.out.println("-o must be followed by a positive integer for the number of offspring generated each generation\n");
								System.exit(0);
							}
							break;
						// mutation rate
						case "-mr":
							if(argIter + 1 < args.length && Pattern.matches("\\d+\\.\\d+", args[argIter + 1])){
								Driver.mutationRate = Float.parseFloat(args[++argIter]);
							}
							else{
								System.out.println("-mr must be followed by a float value for the mutation rate\n");
								System.exit(0);
							}
							break;
						// DE beta parameter
						case "-b":
							if(argIter + 1 < args.length && Pattern.matches("\\d+\\.\\d+", args[argIter + 1])){
								Driver.beta = Float.parseFloat(args[++argIter]);
							}
							else{
								System.out.println("-b must be followed by a float value for the Differential Evolution beta value\n");
								System.exit(0);
							}
							break;
						// wrong option
						default:
							System.out.println(args[argIter] + " is not a valid option\n");
							System.exit(0);
					}
				}
			}
			System.out.println("Starting training...\n");
			double start = System.currentTimeMillis();
			Driver.train();
			double time = System.currentTimeMillis() - start;
			System.out.println("Training Completed in " + time + " ms\n");
		}
	}
	
	/**
	 * Trains a network using the current Training Algorithm
	 * @return a trained network
	 */
	private static Network train(){
		
		return Driver.trainingAlgorithm.train();
	}

	/**
	 * Sets the data file to learn from
	 * @param input the file path
	 * @return true if the operation is successful, false otherwise
	 */
	private static boolean setDataFile(String input){
		boolean flag = true;
		switch(input){
			case "tictactoe":
				Driver.dataset = new TTTData();
				Driver.isClassificationNetwork = true;
				break;
			case "glass":
				Driver.dataset = new GlassData();
				Driver.isClassificationNetwork = true;
				break;
			case "ionosphere":
				Driver.dataset = new IonData();
				Driver.isClassificationNetwork = true;
				break;
			case "letter":
				Driver.dataset = new LetterRecogData();
				Driver.isClassificationNetwork = true;
				break;
			case "waveform":
				Driver.dataset = new WaveformData();
				Driver.isClassificationNetwork = true;
				break;
			default:
				System.out.println(input + "is not a valid datafile.\n");
				flag = false;
		}
		return flag;
	}

	/**
	 * Sets the training algorithm based on a string
	 * @param input the training algorithm
	 * @return true if successful, false otherwise
	 */
	private static boolean setTrainingAlgorithm(String input){
		boolean flag = true;
		switch(input){
			case "bp":
				Driver.trainingAlgorithm = new Backprop();
				break;
			case "ga":
				Driver.trainingAlgorithm = new GeneticAlgorithm();
				break;
			case "es":
				Driver.trainingAlgorithm = new EvolutionStrategy();
				break;
			case "de":
				Driver.trainingAlgorithm = new DifferentialEvolution();
				break;
			default:
				System.out.println(input + " is not a valid training algorithm.\n");
				flag = false;
		}
		return flag;
	}

	/**
	 * Splits the input string into an ArrayList of Integers and passes it to the Network constructor
	 * @param input the network hidden layer configuration; inputs and outputs defined by data set
	 * @return true if successful, false otherwise
	 */
	private static boolean configureNetwork(String input){
		if(Pattern.matches("\\d+(-\\d+)*", input)){
			String[] layers = input.split("-");
			Driver.configuration.add(Driver.dataset.numInputs);
			for(String layerSize : layers){
				Driver.configuration.add(Integer.parseInt(layerSize));
			}
			Driver.configuration.add(Driver.dataset.numOutputs);
			return true;
		}
		else{
			System.out.println(input + " is an invalid hidden layer configuration.\n");
			return false;
		}
	}

	/**
	 * Displays the help text for the program
	 */
	private static void displayHelpText(){
		System.out.println("usage:   java -jar NeuralNetworkTrainer.jar <datafile> <training-algorithm> <hidden-layers> [parameters]");
		System.out.println("\n<datafile>:                glass");
		System.out.println("                           ionosphere");
		System.out.println("                           letter");
		System.out.println("                           tictactoe");
		System.out.println("                           waveform");
		System.out.println("\n<training-algorithm>:      bp (backprop)");
		System.out.println("                           ga (genetic algorithm)");
		System.out.println("                           es (evolution strategy)");
		System.out.println("                           de (differential evolution)");
		System.out.println("\n<hidden-layers>:           a[-b]*");
		System.out.println("                           a,b... are positive integers representing the number of nodes in each respective hidden layer");
		System.out.println("                           leftmost value is the first hidden layer, rightmost is the last");
		System.out.println("                           examples:   0 is the network with no hidden layer");
		System.out.println("                                       10 is the network with one layer of 10 hidden nodes");
		System.out.println("                                       20-10 is the network with two hidden layers; first has 20 nodes, second has 10 nodes");
		System.out.println("\nparameters:     [-lr <learning rate>][-m <momentum>][-p <population-size>]");
		System.out.println("                [-o <offspring-size>][-mr <mutation rate>][-b <beta>]");
		System.out.println("\n-lr   defines the LEARNING RATE used by backprop");
		System.out.println("-m    defines the MOMENTUM used by backprop");
		System.out.println("-p    defines the POPULATION SIZE used by evolutionary algorithms");
		System.out.println("-o    defines the OFFSPRING SIZE used by evolutionary algorithms");
		System.out.println("-mr   defines the MUTATION RATE used by genetic algorithm and evolution strategy");
		System.out.println("-b    defines the BETA parameter used by differential evolution\n");
	}
}
