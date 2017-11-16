package neuralNetworkTrainer;

import com.sun.org.apache.bcel.internal.generic.NEW;
import sun.nio.ch.Net;

import java.util.ArrayList;
import java.util.Collections;

public class DifferentialEvolution extends TrainingAlgorithm {

	/**
	 * The previous iteration's population
	 */
	private ArrayList<Network> previousPopulation;

	/**
	 * The current iteration's population
	 */
	private ArrayList<Network> currentPopulation;

	/**
	 * The current iteration's error rate of the best individual
	 */
	private double currentPercentError = 1.0;
	private int gencounter = 0;

	/**
	 * Keep creating new generations until convergence
	 * @return the best network after convergence
	 */
	@Override
	Network train() {

		this.currentPopulation = this.generatePopulation();

		do{
			this.previousPopulation = this.currentPopulation;
			this.currentPopulation = this.generateNewPopulation(this.serializePopulation(this.previousPopulation));
			this.currentPercentError = 1 - this.currentPopulation.get(0).getFitness();
			System.out.println("Current best percent error: " + this.currentPercentError);

		}while( this.currentPercentError != 0.0 && !this.hasConverged());

		this.evalFitness(this.currentPopulation);
		return this.currentPopulation.get(this.currentPopulation.size() - 1);
	}

	/**
	 * Given the previous generation, generates a new population
	 * @param currentGeneration the current generation
	 * @return the new generation
	 */
	private ArrayList<Network> generateNewPopulation(ArrayList<ArrayList<ArrayList<Double>>> currentGeneration){

		ArrayList<ArrayList<ArrayList<Double>>> newGeneration = new ArrayList<>();
		for(int parentIter = 0; parentIter < currentGeneration.size(); parentIter++){

			// randomly choose 3 "parent" individuals
			int[] otherParentIndices = new int[3];
			for(int otherParentIter = 0; otherParentIter < 3; otherParentIter++){
				int index = parentIter;
				while(
						index == parentIter
								|| index == otherParentIndices[0]
								|| index == otherParentIndices[1]
								|| index == otherParentIndices[2]){
					index = (int)(Math.random() * currentGeneration.size());
				}
				otherParentIndices[otherParentIter] = index;
			}

			// generate a single offspring
			ArrayList<ArrayList<ArrayList<Double>>> chosenIndividuals = new ArrayList<>();
			chosenIndividuals.add((ArrayList<ArrayList<Double>>) currentGeneration.get(parentIter).clone());
			chosenIndividuals.add((ArrayList<ArrayList<Double>>) currentGeneration.get(otherParentIndices[0]).clone());
			chosenIndividuals.add((ArrayList<ArrayList<Double>>) currentGeneration.get(otherParentIndices[1]).clone());
			chosenIndividuals.add((ArrayList<ArrayList<Double>>) currentGeneration.get(otherParentIndices[2]).clone());
			newGeneration.add(this.generateOffspring(chosenIndividuals));
		}

		return this.replacePopulation(this.deserializePopulation(currentGeneration), this.deserializePopulation(newGeneration));
	}

	/**
	 * Given 2 populations, returns a population of the same size with the best from each
	 * @param prevGen population 1
	 * @param currGen population 2
	 * @return best population where the first element is the best, last is the worst
	 */
	private ArrayList<Network> replacePopulation(ArrayList<Network> prevGen, ArrayList<Network> currGen){
		ArrayList<Network> sortedPrevGen = this.evalFitness(prevGen);
		ArrayList<Network> sortedCurrGen = this.evalFitness(currGen);
		ArrayList<Network> bestOfBoth = new ArrayList<>();

		int prevGenIter = sortedPrevGen.size() - 1;
		int currGenIter = sortedCurrGen.size() - 1;
		int iter = 0;
		while(iter < prevGen.size()){
			if(sortedPrevGen.get(prevGenIter).getFitness() > sortedCurrGen.get(currGenIter).getFitness()){
				bestOfBoth.add(iter, sortedPrevGen.get(prevGenIter));
				prevGenIter--;
			}
			else{
				bestOfBoth.add(iter, sortedCurrGen.get(currGenIter));
				currGenIter--;
			}
			iter++;
		}
		return bestOfBoth;
	}

	/**
	 * Generates a single offspring using the conventional DE algorithm
	 * @param parents parents[0] contains the original parent
	 *                parents[1] contains the target individual
	 *                parents[2] and parents[3] contain the difference individuals
	 * @return the "mutated" and crossed offspring
	 */
	private ArrayList<ArrayList<Double>> generateOffspring(ArrayList<ArrayList<ArrayList<Double>>> parents){
		ArrayList<ArrayList<Double>> trialIndividual = new ArrayList<>();

		// iterate through the weight vectors
		for(int nodeIter = 0; nodeIter < parents.get(0).size(); nodeIter++){
			ArrayList<Double> weightVector = new ArrayList<>();

			// iterate through each weight and assign trial individual weight
			for(int weightIter = 0; weightIter < parents.get(0).get(nodeIter).size(); weightIter++){
				double x1 = parents.get(1).get(nodeIter).get(weightIter);
				double x2 = parents.get(2).get(nodeIter).get(weightIter);
				double x3 = parents.get(3).get(nodeIter).get(weightIter);
				weightVector.add(x1 + (Driver.beta * (x2 - x3)));
			}
			trialIndividual.add(weightVector);
		}
		return this.crossover(parents.get(0), trialIndividual);
	}

	/**
	 * Performs uniform crossover between two parents
	 * @param p1 parent 1
	 * @param p2 parent 2
	 * @return the crossed child
	 */
	private ArrayList<ArrayList<Double>> crossover(ArrayList<ArrayList<Double>> p1, ArrayList<ArrayList<Double>> p2){
		// iterate through the weight vectors
		for(int nodeIter = 0; nodeIter < p1.size(); nodeIter++){

			// iterate through each weight, randomly (uniformly) swap p1 weight with p2 weight
			for(int weightIter = 0; weightIter < p1.get(nodeIter).size(); weightIter++){
				if(Math.random() < 0.5){
					p1.get(nodeIter).remove(weightIter);
					p1.get(nodeIter).add(weightIter, p2.get(nodeIter).get(weightIter));
				}
			}
		}
		return (ArrayList<ArrayList<Double>>) p1.clone();
	}

	/**
	 * Generates a population of networks
	 * @return a population of networks
	 */
	public ArrayList<Network> generatePopulation() {

		ArrayList<Network> population = new ArrayList<>();

		// create populationSize number of individuals and add them to population
		for (int popIter = 0; popIter < Driver.populationSize; popIter++) {

			// adding a global config in driver might be worth doing
			// or passing config through train() to all these methods
			Network individual = new Network(true);
			population.add(individual);
		}
		return population;
	}

	/**
	 * Deserializes a population of ArrayList<ArrayList<ArrayList<Double>>>
	 * @param population a population of ArrayList<ArrayList<ArrayList<Double>>>
	 * @return a population of networks
	 */
	//converts matrixes into networks for fitness evaluation
	public ArrayList<Network>  deserializePopulation (ArrayList<ArrayList<ArrayList<Double>>> population) {
		ArrayList<Network> deserializedPopulation = new ArrayList<>();

		for (ArrayList<ArrayList<Double>> individual : population) {
			deserializedPopulation.add(Network.deserializeToNetwork(individual));
		}

		return deserializedPopulation;
	}

	/**
	 * Serializes a population of networks
	 * @param population a population of networks
	 * @return a population of ArrayList<ArrayList<ArrayList<Double>>>
	 */
	//converts networks into matrixes for reproduction
	public ArrayList<ArrayList<ArrayList<Double>>> serializePopulation( ArrayList<Network> population){
		ArrayList<ArrayList<ArrayList<Double>>> serializedPopulation = new ArrayList<>();

		for (Network individual : population) {
			serializedPopulation.add(Network.serializeNetwork(individual, false));
		}

		return serializedPopulation;
	}

	/**
	 * Checks convergence
	 * @return true if the populations have converged, false otherwise
	 */
	public Boolean hasConverged() {
		this.gencounter++;
		if (this.gencounter < 200){
			return false;
		}
		return true;
	}

	/**
	 * calculates the fitness of the population using a random subset of data and sorts according to that fitness
	 * @param population individuals to calculate fitness for
	 * @return population sorted according to their fitness
	 */
	private ArrayList<Network> evalFitness(ArrayList<Network> population){

		ArrayList<ArrayList<Object>> evalSet = Driver.dataset.getEvalDataSet(0);

		for(Network individual: population) {
			double fitness = 0;

			for(ArrayList<Object> datapoint: evalSet) {
				if(individual.evaluate(datapoint)) { //returns true or false for classification
					fitness++;
				}
			}

			fitness = fitness/evalSet.size();
			individual.setFitness(fitness);
		}

		//sorts population based on fitness
		Collections.sort(population);
		return population;
	}

}