/**
 * 
 */
package neuralNetScript;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

/**
 * @author Elias Athey, Tia Smith, Aaron McCarthy
 *
 */
public class Driver {
	// the type of network being implemented
	private static String networkType;
	
	// the number of inputs
	private static int numInNodes;
	
	// an array list containing the number of nodes in each hidden layer. size() is number of hidden layers
	private static ArrayList<Integer> numHiddenLayers = new ArrayList<Integer>();
	
	// the number of output nodes
	private static int numOutNodes;
	
	// the convergence time
	private static double convergenceTime;
	
	// the previous iteration's weights, used upon convergence
	private static ArrayList<Double> prevWeights = new ArrayList<Double>();
	
	// the training sample
	private static Double[][] sample;
	
	// the k-value used for k-means clustering
	private static int k;
	private static int kMeansConvergenceTracker = 0; 
	
	// the sigma value used for rbf
	private static final double sigma = 40;
	
	// the network itself
	private static ArrayList<Layer> network;
	
	// package accessible sample expected output
	static double expectedOutput;
	
	// the entry point of the program
	//args[0] = networkType
	//args[1] = layers
	public static void main(String args[]){
		Driver.networkType = args[0];
		String[] layers = args[1].split("-");
		
		Driver.numInNodes = Integer.parseInt(layers[0]);
		
		for(int layerMaker = 1; layerMaker < (layers.length - 1); layerMaker++) {
			
			Driver.numHiddenLayers.add(Integer.parseInt(layers[layerMaker]));
		}

		Driver.numOutNodes = Integer.parseInt(layers[(layers.length - 1)]);
		
		try{
			Driver.sample = Driver.getSample((int)Math.pow(1.8, Driver.numInNodes) * 100000);
			
			Driver.buildNetwork();
			Driver.crossValidation(10);
		}
		catch(Exception e){
			System.out.println("Error...");
			System.out.println(e.getMessage());
			for(StackTraceElement stack : e.getStackTrace()){
				System.out.println(stack);
			}
		}
		
		// final print stmt
		System.out.println("\nNetwork has been trained and tested.");
	}
	
	// return a sample dataset of the Rosenbrock function
	// [m][n] contains n data points, each with m-1 inputs and 1 output at last index
	private static Double[][] getSample(int size){
		Double[][] outputs = new Double[Driver.numInNodes + 1][size];
		
		// generate *size number of sample data points
		for(int setIter = 0; setIter < size; setIter++) {
			// generate random inputs from -rangeScale to +rangeScale
			int rangeScale = 10;
			ArrayList<Double> inputs = new ArrayList<Double>();
			for(int inputIter = 0; inputIter < Driver.numInNodes; inputIter++ ) {
				inputs.add(inputIter, Math.random() * Math.pow(-1, (int)(Math.random() * 2)) * rangeScale);
				outputs[inputIter][setIter] = inputs.get(inputIter);
			}
			
			// set the rosenbrock output
			try{
				outputs[Driver.numInNodes][setIter] = Driver.rosenbrock(inputs);
			}
			catch(Exception e){
				System.out.println(e.getMessage());
			}
		}
		return outputs;
	}
	
	// the Rosenbrock function accepting at least 2 inputs
	private static double rosenbrock(ArrayList<Double> input) throws Exception{
		if(input.size() < 2){
			throw new Exception("Rosenbrock function input must have at least two elements.");
		}
		
		double output = 0f;
		for(int i = 0; i < input.size() - 1; i++){
			output += Math.pow(1 - input.get(i), 2) + (100 * Math.pow(input.get(i + 1) - Math.pow(input.get(i), 2), 2));
		}
		return output;
	}
	
	// create Node objects and set downstream attribute for each
	private static void buildNetwork() throws Exception{
		// print status message and model visualization
		System.out.println("Building network...");
		System.out.print(Driver.numInNodes + "(in) -> ");
		for(int i = 0; i < Driver.numHiddenLayers.size(); i++){
			System.out.print(Driver.numHiddenLayers.get(i) + " -> ");
		}
		System.out.print(Driver.numOutNodes + "(out)\n\n");
		
		// initialize Layers and network
		Layer inputLayer = new Layer();
		Layer[] hiddenLayers = new Layer[Driver.numHiddenLayers.size()];
		for(int i = 0; i < hiddenLayers.length; i++){
			hiddenLayers[i] = new Layer();
		}
		Layer outputLayer = new Layer();
		Driver.network = new ArrayList<Layer>();
		Driver.network.add(inputLayer);
		for(Layer layer : hiddenLayers){
			Driver.network.add(layer);
		}
		Driver.network.add(outputLayer);
		
		// create output nodes and store in output layer
		Node[] outputNodes = new Node[Driver.numOutNodes];
		for(int i = 0; i < outputNodes.length; i++){
			// set the node functions for output nodes
			outputNodes[i] = new Node(new NoFunction(), new BackpropFinalWeightFunction(), new Node[0]);
			outputNodes[i].setLayerIndex(i);
		}
		outputLayer.setNodes(outputNodes);
		
		// set k-value, clusters, and sigma for rbf network
		ArrayList<Double[]> clusters = new ArrayList<Double[]>();
		if(Driver.networkType.equals("rbf")){
			Driver.k = Driver.numHiddenLayers.get(0);
			clusters = cluster();
			RadialBasisFunction.setSigma(Driver.sigma);
		}
		
		// create hidden layer nodes and store in hidden layer
		Node[] prevHiddenNodes = outputNodes;
		for(int i = hiddenLayers.length - 1; i >= 0; i--){
			Node[] hiddenNodes = new Node[Driver.numHiddenLayers.get(i)];
			for(int j = 0; j < hiddenNodes.length; j++){
				// set the node functions for hidden nodes
				if(i == hiddenLayers.length - 1){
					switch(Driver.networkType){
						case "rbf": 
							hiddenNodes[j] = new Node(new RadialBasisFunction(), new NoWeightFunction(), outputNodes);
							// set the associated cluster
							hiddenNodes[j].setAssociatedCluster(clusters.get(j));
							break;
						case "mlp":
							hiddenNodes[j] = new Node(new SigmoidalFunction(), new BackpropHiddenWeightFunction(), outputNodes);
							break;
					}
				}
				else{
					if(Driver.networkType.equals("rbf")){
						throw new Exception("An rbf network cannot have more than one hidden layer.");
					}
					hiddenNodes[j] = new Node(new SigmoidalFunction(), new BackpropHiddenWeightFunction(), prevHiddenNodes);
				}
				hiddenNodes[j].setLayerIndex(j);
			}
			
			// initialize input arrays with random weights between -0.5 and 0.5 for downstream nodes
			for(int m = 0; m < prevHiddenNodes.length; m++){
				prevHiddenNodes[m].inputs = new double[2][hiddenNodes.length];
				for(int k = 0; k < prevHiddenNodes[m].inputs[1].length; k++){
					int sign = 1;
					if(Math.random() < 0.5){
						sign = -1;
					}
					prevHiddenNodes[m].inputs[1][k] = sign * (Math.random() * 0.5);
				}
			}
			
			// set hidden nodes to this layer, set reference to these hidden nodes
			hiddenLayers[i].setNodes(hiddenNodes);
			prevHiddenNodes = hiddenNodes;
		}
		
		// create input nodes and store in input layer
		Node[] inputNodes = new Node[Driver.numInNodes];
		for(int i = 0; i < inputNodes.length; i++){
			// set the node functions for input nodes
			switch(Driver.networkType){
				case "rbf": 
					inputNodes[i] = new Node(new NoFunction(), new NoWeightFunction(), prevHiddenNodes);
					break;
				case "mlp":
					inputNodes[i] = new Node(new SigmoidalFunction(), new NoWeightFunction(), prevHiddenNodes);
					break;
			}
			
			// initialize input node weights with 1
			inputNodes[i].inputs = new double[2][1];
			inputNodes[i].inputs[1][0] = 1;
			inputNodes[i].setLayerIndex(i);
		}
		
		// initialize first hidden layer (or output layer in the case with no hidden layers) input arrays with random weights
		for(int j = 0; j < prevHiddenNodes.length; j++){
			prevHiddenNodes[j].inputs = new double[2][inputNodes.length];
			for(int k = 0; k < prevHiddenNodes[j].inputs[1].length; k++){
				switch(Driver.networkType){
					case "rbf": 
						prevHiddenNodes[j].inputs[1][k] = 1.0;
						break;
					case "mlp":
						int sign = 1;
						if(Math.random() < 0.5){
							sign = -1;
						}
						prevHiddenNodes[j].inputs[1][k] = sign * (Math.random() * 0.5);
						break;
				}
			}
		}
		inputLayer.setNodes(inputNodes);
	}

	private static void crossValidation(int k) {

		System.out.println("Training network...\n");
		
		//keeps track of number of times trained, trains k-1 times
		for (int trainIter = 0; trainIter < k; trainIter++) {

			// a list of all weights, will hold the average weights of all iterations in one fold
			ArrayList<Double> averageWeights = new ArrayList<Double>();
			
			//start convergence timer
			double startTime = System.currentTimeMillis();
			
			// counter for training samples
			int samplesInFold = 0;
			
			//loops over all data points
			for (int sampleIter = 0; sampleIter < Driver.sample[0].length; sampleIter++) {
				
				//train on k-1 folds
				if (sampleIter % k != trainIter) { 
					
					//increment counter
					samplesInFold++;
					
					//loops over dimensions of each point
					for (int dimensionIter = 0; dimensionIter < Driver.numInNodes; dimensionIter++) {
						
						//assign sample to input nodes
						Driver.network.get(0).getNodes()[dimensionIter].inputs[0][0] = Driver.sample[dimensionIter][sampleIter];
					}

					//set the expected output for this sample point
					Driver.expectedOutput = Driver.sample[Driver.numInNodes][sampleIter];
					
					// execute the nodes in the network
					for(Layer layer : Driver.network){
						for(Node node : layer.getNodes()){
							node.execute();
						}
					}

					// save previous weights to revert the updated weights for this iteration
					Driver.prevWeights = new ArrayList<Double>();
					for(Layer layer : Driver.network){
						for(Node node : layer.getNodes()){
							for(double weight : node.inputs[1]){
								Driver.prevWeights.add(weight);
							}
						}
					}
					
					// update weights in the network
					for(int updateIter = Driver.network.size() - 1; updateIter >= 0 ; updateIter--){
						for(Node node : Driver.network.get(updateIter).getNodes()){
							node.updateWeights();
						}
					}
					
					// add updated weights to the list of average weights
					int weightIter = 0;
					for(Layer layer : Driver.network){
						for(Node node : layer.getNodes()){
							for(double weight : node.inputs[1]){
								if(samplesInFold == 1){
									averageWeights.add(weightIter, weight);
								}
								else{
									Double summedWeight = averageWeights.get(weightIter);
									summedWeight += weight;
									averageWeights.remove(weightIter);
									averageWeights.add(weightIter, summedWeight);
								}
								weightIter++;
							}
						}
					}
					
					// revert weights in the network
					weightIter = 0;
					for(Layer layer : Driver.network){
						for(Node node : layer.getNodes()){
							for(int i = 0; i < node.inputs[1].length; i++){
								node.inputs[1][i] = Driver.prevWeights.get(weightIter);
								weightIter++;
							}
						}
					}
				}
			}
			
			// average the weights
			for(int weightIter = 0; weightIter < averageWeights.size(); weightIter++){
				Double weight = averageWeights.get(weightIter);
				weight = weight / samplesInFold;
				averageWeights.remove(weightIter);
				averageWeights.add(weightIter, weight);
			}
			
			// assign the averaged weights to the network
			int weightIter = 0;
			for(Layer layer : Driver.network){
				for(Node node : layer.getNodes()){
					for(int i = 0; i < node.inputs[1].length; i++){
						node.inputs[1][i] = averageWeights.get(weightIter);
						weightIter++;
					}
				}
			}
			
			// save convergence time (really just training time)
			Driver.convergenceTime = System.currentTimeMillis() - startTime;
			System.out.println("\nFold " + (trainIter + 1) + " trained in: " + Driver.convergenceTime + " ms.");
			
			//test data after every train and determine average error
			double averageError[] = new double[Driver.numOutNodes];
			for (int sampleIter = 0; sampleIter < Driver.sample[0].length; sampleIter++) {
				
				//test on kth set of data 
				if(sampleIter % k == trainIter){
					
					// set inputs
					Double[] inputs = new Double[Driver.numInNodes];
					for (int dimensionIter = 0; dimensionIter < Driver.numInNodes; dimensionIter++) {
						inputs[dimensionIter] = Driver.sample[dimensionIter][sampleIter];
					}
					
					//set the expected output for this sample point
					Driver.expectedOutput = Driver.sample[Driver.numInNodes][sampleIter];
					
					// test network
					double[] networkOutput = Driver.testNetwork(inputs);
					
					// sum average error
					averageError[0] += (networkOutput[0] - Driver.expectedOutput);
				}
			}

			// compute average error for this fold, if it has increased, revert the weights
			averageError[0] = averageError[0] / (Driver.sample[0].length / k);
			System.out.println("Average error for fold " + (trainIter + 1) + ": " + averageError[0]);
		}
	}
	
	// given an input vector, return the output of the network as the approximation of the Rosenbrock function
	private static double[] testNetwork(Double[] input){
		// set inputs
		for(int i = 0; i < Driver.network.get(0).getNodes().length; i++){
			Driver.network.get(0).getNodes()[i].inputs[0][0] = input[i];
		}
		
		// execute the nodes in the network
		for(Layer layer : Driver.network){
			for(Node node : layer.getNodes()){
				node.execute();
			}
		}
		
		// get computed output from output nodes
		double[] output = new double[Driver.numOutNodes];
		for(int i = 0; i < output.length; i++){
			output[i] = Driver.network.get(Driver.network.size() - 1).getNodes()[i].getComputedOutput();
		}
		return output;
	}

	// returns a set of clusters for rbf
	private static ArrayList<Double[]> cluster() {

		// convert Driver.sample to ArrayList<Double[]>
		ArrayList<Double[]> sample = new ArrayList<Double[]>();
		for(int i = 0; i < Driver.sample[0].length; i++){
			Double[] point = new Double[Driver.sample.length];
			for(int j = 0; j < point.length; j++){
				point[j] = Driver.sample[j][i];
			}
			sample.add(i, point);
		}
		
		// sort the sample data by distance from origin
		Collections.sort(sample, new Comparator<Double[]>() {
		    @Override
		    public int compare(Double[] datapoint1, Double[] datapoint2) {
		    	double dist1 = 0, dist2 = 0;
				for(int coordinate = 0; coordinate < datapoint1.length; coordinate++){
					dist1 += Math.pow(datapoint1[coordinate], 2);
					dist2 += Math.pow(datapoint2[coordinate], 2);
				}
				dist1 = Math.sqrt(dist1);
				dist2 = Math.sqrt(dist2);
		        return Double.compare(dist1, dist2);
		    }
		});
		
		// average datapoints in each cluster and store in clusters list
		ArrayList<Double[]> clusters = new ArrayList<Double[]>();
		int clusterIter = 0;
		int pointsPerCluster = sample.size() / Driver.k;
		Double[] averagePoint = new Double[sample.get(0).length];
		for(int pointIter = 0; pointIter < sample.size(); pointIter++){
			
			// store average point in cluster
			if((pointIter + 1) % pointsPerCluster == 0){
				for(int coordinate = 0; coordinate < sample.get(0).length; coordinate++){
					averagePoint[coordinate] = averagePoint[coordinate] / pointsPerCluster;
				}
				clusters.add(clusterIter, averagePoint);
				clusterIter++;
			}
			// assign first averagePoint for this cluster
			else if(pointIter % pointsPerCluster == 0){
				for(int coordinate = 0; coordinate < sample.get(0).length; coordinate++){
					averagePoint[coordinate] = sample.get(pointIter)[coordinate];
				}
			}
			// sum average point (divided by num points later)
			else{
				for(int coordinate = 0; coordinate < sample.get(0).length; coordinate++){
					averagePoint[coordinate] += sample.get(pointIter)[coordinate];
				}
			}
		}
		return clusters;
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	/*
	 * Below is K-Means functionality. To be used in a later project.
	 */
	
	
//******************************************************************************************************************//
	
//	// given a k and the training set return k centroids that
//	// define the centers of the clusters
//	private static ArrayList<Double[]> kmeans(){
//		
//		ArrayList<Double[]> centroids = new ArrayList<Double[]>(Driver.k);
//		int[] labels = new int[Driver.sample[0].length];
//		int convergenceTracker = 0; 
//		long start = System.currentTimeMillis();
//		
//		// pick initial random data points to be centroids
//		for(int randClusterIter = 0; randClusterIter < Driver.k; randClusterIter++) {
//			Double[] randCentroid = new Double[Driver.numInNodes];
//			int randIndex = (int) (Math.random() * Driver.sample[0].length);
//			
//			for(int randSampler = 0; randSampler < Driver.numInNodes; randSampler++) {
//				randCentroid[randSampler] = sample[randSampler] [randIndex]; 
//			}
//			centroids.add(randCentroid);
//		}
//		
////		int iterations = 0;
//		ArrayList<Double[]> oldCentroids = null;
//		
//		do{
//			
//			// save old for convergence test
//			oldCentroids = centroids;
////			iterations ++;
//			labels = getLabels(centroids);
//			centroids = getNewCentroids(centroids, labels);
//		}while (!stopKmeans(oldCentroids, centroids));
//		System.out.println("kmeans convergence time = " + (System.currentTimeMillis() - start) + " milliseconds");
//		return centroids;
//	}
//	
//	// assigns a label for every datapoint in the sample set
//	// uses distance function to find closest centroid
//	// label is index of centroid in centroids[]
//	private static int[] getLabels(ArrayList<Double[]> centroids) {
//		
//		int[] labels = new int[Driver.sample[0].length];
//		Double[] distances = new Double[Driver.k];
//		Double sum = null;
//		
//		for(int sampleIter = 0; sampleIter < Driver.sample[0].length; sampleIter++) {//loop thru samples
//			
//			for (int centroidIter = 0; centroidIter < Driver.k; centroidIter++) {//loop thru centroids
//				sum = (double) 0;
//				
//				for (int dimensionIter = 0; dimensionIter < Driver.numInNodes; dimensionIter++){//loop thru dimensions of centroids
//					//sum for Euclidean distance function
//					sum += Math.pow(Driver.sample[dimensionIter][sampleIter] - centroids.get(centroidIter)[dimensionIter], 2);
//				}
//				//calc distance to each centroid from each sample
//				distances[centroidIter] = Math.sqrt(sum);
//			}
//			//store index of min distance in labels
//			labels[sampleIter] = findMin(distances);
//		}
//		return labels;
//	}
//	
//	// calculate geometric mean of all sample points with a common label
//	// make this point a new centroid
//	// dimensionSums[l][d] holds the label, [l] with the sum of all 
//	// dimensions of all samples with that label, [d]
//	private static ArrayList<Double[]> getNewCentroids(ArrayList<Double[]> centroids, int[] labels) {
//		
//		
//		ArrayList<Double[]> newCentroids = new ArrayList<Double[]>(Driver.k);
//		int[] labelDivisors = new int[Driver.k];
//		Double[][] dimensionSums = new Double[Driver.k][Driver.numInNodes];
//		// initialize dimensionSums to 0
//		for (int i = 0; i < dimensionSums.length; i++) {
//			for (int x = 0; x < dimensionSums[0].length; x++) {
//				dimensionSums[i][x] = (double) 0;
//			}
//		}
//		
//		for(int sampleIter = 0; sampleIter < Driver.sample[0].length; sampleIter++) {//loop thru samples
//			for(int dimIter = 0; dimIter < Driver.numInNodes; dimIter++) {//loop thru dimensions of each sample
//				
//				dimensionSums[labels[sampleIter]][dimIter] += Driver.sample[dimIter][sampleIter];
//				labelDivisors[labels[sampleIter]] ++;
//			}
//		}
//		
//		for(int labelIter = 0; labelIter < Driver.k; labelIter++) {
//			Double[] newCent = new Double[Driver.numInNodes];
//			for(int dimIter = 0; dimIter < Driver.numInNodes; dimIter++) {//loop thru dimensions of each sample
//
//				newCent[dimIter] = dimensionSums[labelIter][dimIter] / labelDivisors[labelIter];
//			}
//			newCentroids.add(labelIter, newCent);
//		}
//		return newCentroids;
//	}
//
//	// determines the minimum value in the distance array
//	private static int findMin(Double[] distanceArray) {
//		int minIndex = 0;
//		for(int index = 1; index < distanceArray.length; index++) {
//			if (distanceArray[minIndex] > distanceArray[index]) {
//				minIndex = index;
//			}
//		}
//		return minIndex;
//	}
//	
//	// a convergence test for the k-means algorithm that returns true if the cluster vectors have converged
//	private static boolean stopKmeans(ArrayList<Double[]> centroids, ArrayList<Double[]> oldCentroids) {
//		
//		int index = 0;
//		int flags = 0;
//		while(centroids.size() > index) {
//			
//			for(int arrayIter = 0; arrayIter < centroids.get(index).length ; arrayIter++) {
//				
//				if (centroids.get(index)[arrayIter] - (oldCentroids.get(index)[arrayIter]) < 0.0001) {
//					flags++;
//				}
//			}
//			index++;
//		}
//
//		//System.out.println(Driver.kMeansConvergenceTracker + " num flags: " + flags);
//
//		if (flags >= (Driver.k * Driver.numInNodes)) {
//			Driver.kMeansConvergenceTracker++;
//			if(Driver.kMeansConvergenceTracker >= 3) {
//				return true;
//			}
//		}
//		else {
//			Driver.kMeansConvergenceTracker = 0;
//		}
//		return false;
//	}
}
