package neuralNetworkTrainer;

import java.util.ArrayList;

/**
 * @author Elias
 *
 */
class Backprop extends TrainingAlgorithm {
	
	/**
	 * Initializes a network with weights
	 * @return the initialized network
	 */
	private Network initializeNetwork(){
		
		// construct network with random weights
		return new Network(true);
	}

	/**
	 * trains a network using backprop
	 */
	@Override
	Network train() {
		
		// initialize network
		Network network = this.initializeNetwork();
		
		// do until convergence
		while(true){

			ArrayList<Object> computedOutput;
			
			// holds the last expected output
			ArrayList<Object> expectedOutput = new ArrayList<>();
			
			// all the serialized networks (weights) of a single run
			ArrayList<ArrayList<ArrayList<Double>>> allWeights = new ArrayList<>();

			// all the squared errors, used to find average
			ArrayList<ArrayList<Double>> errors = new ArrayList<>();
			
			// get sample data set
			int numInputs = Driver.dataset.getNumInputs();
			ArrayList<ArrayList<Object>> dataset = Driver.dataset.getDataPoints();
			
			// iterate over each sample data point
			int samplePointIter = 0;
			for(ArrayList<Object> samplePoint : dataset){

				// set inputs and expected outputs
				network.setInputs(samplePoint);
				expectedOutput.add(samplePoint.get(samplePoint.size() - 1));
				
				// execute the nodes in the network and save computed output
				computedOutput = network.executeNodes();

				// add squared error to list
				ArrayList<Double> currentError = this.getError(expectedOutput, computedOutput);
				errors.add(samplePointIter, currentError);
				
				// Save original weights
				ArrayList<ArrayList<Double>> originalWeights = (ArrayList<ArrayList<Double>>)Network.serializeNetwork(network, false).clone();
				
				// set delta values then update weights
				this.setOutputDeltas(network, expectedOutput, currentError);
				this.setHiddenDeltas(network);
				this.updateFinalNodeWeights(network);
				this.updateHiddenNodeWeights(network);
				
				// Save updated weights
				allWeights.add((ArrayList<ArrayList<Double>>)Network.serializeNetwork(network, false).clone());
				
				// Reset to original weights
				network.setWeights(originalWeights, false);
				
				samplePointIter++;
			}
			
			// find average of all weights
			ArrayList<ArrayList<Double>> averagedWeights = new ArrayList<>();
			int numWeights = 0;
			for(ArrayList<ArrayList<Double>> weights : allWeights){
				numWeights++;
				int nodeIter = 0;
				for(ArrayList<Double> weightVector : weights){
					if(numWeights == 1){
						averagedWeights.add(nodeIter, weightVector);
					}
					else{
						for(int weightIter = 0; weightIter < weightVector.size(); weightIter++){
							Double current = averagedWeights.get(nodeIter).get(weightIter);
							current += weightVector.get(weightIter);
						}
					}
					nodeIter++;
				}
			}
			for(ArrayList<Double> weightVectors : averagedWeights){
				for(Double weight : weightVectors){
					weight = weight / numWeights;
				}
			}
			ArrayList<ArrayList<Double>> originalWeights = Network.serializeNetwork(network, false);

			// get and print error
			Double percentError = this.getPercentError(errors);
			System.out.println("Percent error: " + percentError + "\n");
			errors.clear();

			// check convergence
			if(percentError != 0 && !this.hasConverged(originalWeights, averagedWeights)){
				network.setWeights(this.getChangeInWeights(averagedWeights, originalWeights), true);
				network.setWeights(averagedWeights, false);

//				// print previous weights
//				System.out.println("Previous Weights");
//				for(ArrayList<Double> node : originalWeights){
//					System.out.print("Node: ");
//					for(Double weight : node){
//						System.out.print(weight + " ");
//					}
//					System.out.print("\n");
//				}
//
//				// print new weights
//				System.out.println("\nNew Weights");
//				for(ArrayList<Double> node : averagedWeights){
//					System.out.print("Node: ");
//					for(Double weight : node){
//						System.out.print(weight + " ");
//					}
//					System.out.print("\n");
//				}
			}
			else{
				// break out of while loop if we have converged
				break;
			}
		}
		
		return network;
	}

	/**
	 * Gets the list of lists of differences in weights
	 * @param newWeights the new weight matrix
	 * @param oldWeights the previous weight matrix
	 * @return the matrix of differences
	 */
	private ArrayList<ArrayList<Double>> getChangeInWeights(ArrayList<ArrayList<Double>> newWeights, ArrayList<ArrayList<Double>> oldWeights){

		ArrayList<ArrayList<Double>> change = new ArrayList<>();
		for(int nodeIter = 0; nodeIter < newWeights.size(); nodeIter++){
			change.add(nodeIter, new ArrayList<>());
			for(int weightIter = 0; weightIter < newWeights.get(nodeIter).size(); weightIter++){
				Double difference = newWeights.get(nodeIter).get(weightIter) - oldWeights.get(nodeIter).get(weightIter);
				change.get(nodeIter).add(difference);
			}
		}
		return change;
	}
	
	/**
	 * Determines if the given network's weights have converged
	 * @param originalWeights the last iteration's weights
	 * @param currentWeights the current iteration's weights
	 * @return true if the network has converged
	 */
	private boolean hasConverged(ArrayList<ArrayList<Double>> originalWeights, ArrayList<ArrayList<Double>> currentWeights){

		for(int i = 0; i < originalWeights.size(); i++){
			for(int j = 0; j < originalWeights.get(i).size(); j++){
				double difference = originalWeights.get(i).get(j) - currentWeights.get(i).get(j);
				int estimate = (int)(difference * 10000);
				if(estimate != 0){
					return false;
				}
			}
		}
		return true;
	}
	
	/**
	 * Computes the squared error between the network's computed output and the sample expected output
	 * @param expectedOutput the output defined by the sample
	 * @return the squared error between the network's computed output and the sample expected output
	 */
	private ArrayList<Double> getError(ArrayList<Object> expectedOutput, ArrayList<Object> computedOutput){

		ArrayList<Double> error = new ArrayList<>();
		int largestValuedOutputIndex = 0;
		for(int outputIter = 0; outputIter < computedOutput.size(); outputIter++){
			if(Driver.isClassificationNetwork){
				if((double)computedOutput.get(outputIter) > (double)computedOutput.get(largestValuedOutputIndex)){
					largestValuedOutputIndex = outputIter;
				}
				if(outputIter == computedOutput.size() - 1){
					if(largestValuedOutputIndex == (double)expectedOutput.get(0)){
						error.add(0, 0.0);
					}
					else{
						error.add(0, 1.0);
					}
				}
			}
			else{
				double expected = 0;
				if(expectedOutput.get(outputIter).getClass().getTypeName().equals("java.lang.Double")){
					expected = (double)expectedOutput.get(outputIter);
				}
				else if(expectedOutput.get(outputIter).getClass().getTypeName().equals("java.lang.Integer")){
					expected = (int)expectedOutput.get(outputIter);
				}
				double computed = (double)computedOutput.get(outputIter);
				error.add(outputIter, expected - computed);
			}
		}
		return error;
	}

	/**
	 * Computes the average square error given a list of errors
	 * @param allErrors the list of all squared errors
	 * @return the average squared error
	 */
	private Double getPercentError(ArrayList<ArrayList<Double>> allErrors){

		double[] sums = new double[allErrors.get(0).size()];
		for(int errorIter = 0; errorIter < allErrors.size(); errorIter++){
			for(int i = 0; i < allErrors.get(errorIter).size(); i++){
				sums[i] += allErrors.get(errorIter).get(i); // not squared
			}
		}
		Double averageError = sums[0] / allErrors.size();
		return averageError;
	}
	
	/**
	 * Sets the delta values for all output nodes.
	 * @param network the Network to reference
	 * @param expectedOutput the expected output of the network
	 * @param errors the computed error for this iteration
	 */
	private void setOutputDeltas(Network network, ArrayList<Object> expectedOutput, ArrayList<Double> errors){

		for(Node outputNode : network.getOutputLayer().getNodes()){
			if(Driver.isClassificationNetwork){
//				if(errors.get(0) == 1.0){
//					outputNode.setBackpropDelta((Double)outputNode.getDerivative());
//				}
//				else if(outputNode.getIndexInLayer() == (double)expectedOutput.get(0)){
//					outputNode.setBackpropDelta((1.0 - (double)outputNode.getComputedOutput()) * (Double)outputNode.getDerivative());
//				}
//				else{
//					outputNode.setBackpropDelta((0.0 - (double)outputNode.getComputedOutput()) * (Double)outputNode.getDerivative());
//				}
				if(outputNode.getIndexInLayer() == (double)expectedOutput.get(0)){
					outputNode.setBackpropDelta((1.0 - (double)outputNode.getComputedOutput()) * (Double)outputNode.getDerivative());
				}
				else{
					outputNode.setBackpropDelta((0.0 - (double)outputNode.getComputedOutput()) * (Double)outputNode.getDerivative());
				}
			}
			else{
				double expected = 0;
				if(expectedOutput.get(outputNode.getIndexInLayer()).getClass().getTypeName().equals("java.lang.Double")){
					expected = (double)expectedOutput.get(outputNode.getIndexInLayer());
				}
				else if(expectedOutput.get(outputNode.getIndexInLayer()).getClass().getTypeName().equals("java.lang.Integer")){
					expected = (int)expectedOutput.get(outputNode.getIndexInLayer());
				}
				Double computed = (Double)outputNode.getComputedOutput();
				Double delta = (expected - computed) * (Double)outputNode.getDerivative();
				outputNode.setBackpropDelta(delta);
			}
		}
	}
	
	/**
	 * Sets the delta values for all hidden nodes. The derivative used is(output * (1 - output)) because the hidden nodes function is sigmoidal.
	 * @param network the Network to reference
	 */
	private void setHiddenDeltas(Network network){
		
		for(int layerIter = network.getHiddenLayers().size() - 1; layerIter >= 0; layerIter--){
			for(Node hiddenNode : network.getHiddenLayers().get(layerIter).getNodes()){
				Double downstreamSum = 0.0;
				for(Node downstreamNode : hiddenNode.getDownstreamNodes()){
					downstreamSum += (downstreamNode.getBackpropDelta() * downstreamNode.getWeights().get(hiddenNode.getIndexInLayer()));
				}
				hiddenNode.setBackpropDelta(downstreamSum * (Double)hiddenNode.getDerivative());
			}
		}
	}
	
	/**
	 * update the weights for every hidden node
	 * @param network the Network to update weights on
	 */
	private void updateHiddenNodeWeights(Network network){
		
		for(int layerIter = 0; layerIter < network.getHiddenLayers().size(); layerIter++){
			for(Node hiddenNode : network.getHiddenLayers().get(layerIter).getNodes()){
				for(int inputIter = 0; inputIter < hiddenNode.getInputs().size(); inputIter++){
					Double weightChange = 0.0;
					weightChange += ((1 - Driver.momentum) * Driver.learningRate * hiddenNode.getBackpropDelta() * (Double)hiddenNode.getInputs().get(inputIter));
					weightChange += (Driver.momentum * hiddenNode.getPrevWeightChange().get(inputIter));
					Double originalWeight = hiddenNode.getWeights().get(inputIter);
					hiddenNode.getWeights().remove(inputIter);
					hiddenNode.getWeights().add(inputIter, originalWeight + weightChange);
					hiddenNode.getPrevWeightChange().remove(inputIter);
					hiddenNode.getPrevWeightChange().add(inputIter, weightChange);
				}
			}
		}
	}
	
	/**
	 * update the weights to the final nodes
	 * @param network the Network to update weights on
	 */
	private void updateFinalNodeWeights(Network network){
		
		for(Node outputNode : network.getOutputLayer().getNodes()){
			for(int inputIter = 0; inputIter < outputNode.getInputs().size(); inputIter++){
				Double weightChange = 0.0;
				weightChange += ((1 - Driver.momentum) * Driver.learningRate * outputNode.getBackpropDelta() * (Double)outputNode.getInputs().get(inputIter));
				weightChange += (Driver.momentum * outputNode.getPrevWeightChange().get(inputIter));
				Double originalWeight = outputNode.getWeights().get(inputIter);
				outputNode.getWeights().remove(inputIter);
				outputNode.getWeights().add(inputIter, originalWeight + weightChange);
				outputNode.getPrevWeightChange().remove(inputIter);
				outputNode.getPrevWeightChange().add(inputIter, weightChange);
			}
		}
	}
}
