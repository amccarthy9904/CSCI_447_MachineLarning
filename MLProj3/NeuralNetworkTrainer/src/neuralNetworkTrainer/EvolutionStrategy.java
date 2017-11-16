package neuralNetworkTrainer;

import sun.nio.ch.Net;

import java.util.ArrayList;
import java.util.Collections;
import java.util.ListIterator;
import java.util.Random;

public class EvolutionStrategy extends TrainingAlgorithm {

    private double oneFith = .82;
    private double mutationRedux = .25;
    private double mutationIncreace = 1.5;
    private RouletteWheel rouletteWheel; //used to randomly select parents weighted by their rank
    Random randNum = new Random();
    int gencounter = 0;


    public ArrayList<IndividualES> generatePopulation() {

        ArrayList<IndividualES> population = new ArrayList<>();

        //create populationSize number of individuals and add them to population
        for (int popIter = 0; popIter < Driver.populationSize; popIter++) {

            IndividualES individual = new IndividualES(true);
            population.add(individual);
        }
        this.rouletteWheel = new RouletteWheel();
        //Collections.sort(population);
        return population;
    }

    // converts matrixes into networks for fitness evaluation
    public void deserializePopulation(ArrayList<IndividualES> population) {

        for (IndividualES individual : population) {
            individual.setNetwork(Network.deserializeToNetwork(individual.getGenome()));
        }
    }

    // creates a new generation using rank based selection of parents, crossover and
    // mutation, then returns the new generation
    // This is specific to having 2 offspring from 2 parents but is generalizable to
    // any number of parents and offspring with a little refactoring
    public ArrayList<IndividualES> newGeneration(ArrayList<IndividualES> population) {

        ArrayList<IndividualES> offspringPool = new ArrayList<>();
        ArrayList<IndividualES> parentPair;
        ArrayList<IndividualES> offspringPair;

        while (offspringPool.size() < Driver.numberOffspring) {                                                    //continue until the new generation is the required offspring size
            parentPair = selectParents(population);                                            //selects 2 parents with rank based selection
            offspringPair = crossoverOffspring(parentPair.get(0), parentPair.get(1));        //creates 2 new offspring via crossover
            offspringPool.add(offspringPair.get(0));            //adds new offspring to the new generation
            offspringPool.add(offspringPair.get(1));            //adds new offspring to the new generation
        }

        offspringPool = mutateOffspringFeatures(offspringPool);
        //gives the individuals Networks
        deserializePopulation(offspringPool);
        offspringPool = evalFitness(offspringPool);
        offspringPool = oneFithRule(offspringPool);

        return offspringPool;
    }

    public ArrayList<IndividualES> selectParents(ArrayList<IndividualES> population) {

        ArrayList<IndividualES> parentPair = new ArrayList<>();

        //select 2 parents randomly based on rank using the roulette wheel
        parentPair.add(population.get(this.rouletteWheel.get(this.randNum.nextInt(this.rouletteWheel.size))));
        parentPair.add(population.get(this.rouletteWheel.get(this.randNum.nextInt(this.rouletteWheel.size))));

        return parentPair;
    }

    // creates a List of boolean Lists that mirrors the dimensions and structure of
    // an individual and uses the list to randomly assign genes from parents to offspring
    public ArrayList<IndividualES> crossoverOffspring(IndividualES parent1, IndividualES parent2) {

        //List of boolean Lists to decide if offspring get their genes from parent1 or parent2
        ArrayList<ArrayList<Boolean>> randomizer = new ArrayList<ArrayList<Boolean>>();

        for (ArrayList<Double> chromosome : parent1.getGenome()) {

            //a chromosome of random booleans that is the same length of the chromosomes of the parents and offspring
            ArrayList<Boolean> randChrom = new ArrayList<Boolean>();

            for (Double gene : chromosome) {
                //fill every randChrom with random booleans
                randChrom.add(this.randNum.nextBoolean());
            }
            randomizer.add(randChrom);
        }

        //make offspring
        IndividualES offspring1 = new IndividualES(false);
        IndividualES offspring2 = new IndividualES(false);


        ArrayList<ArrayList<Double>> genomeOffspring1 = new ArrayList<>();
        ArrayList<ArrayList<Double>> genomeOffspring2 = new ArrayList<>();
        ArrayList<ArrayList<Double>> stratParamsOffspring1 = new ArrayList<>();
        ArrayList<ArrayList<Double>> stratParamsOffspring2 = new ArrayList<>();
        ArrayList<Double> chromosomeO1 = new ArrayList<>();
        ArrayList<Double> chromosomeO2 = new ArrayList<>();
        ArrayList<Double> stratChromosomeO1 = new ArrayList<>();
        ArrayList<Double> stratChromosomeO2 = new ArrayList<>();

        //randomizer selects which offspring gets which gene
        for (int chromIter = 0; chromIter < parent1.getGenome().size(); chromIter++) {

            chromosomeO1.clear();
            chromosomeO2.clear();
            stratChromosomeO1.clear();
            stratChromosomeO2.clear();

            for (int geneIter = 0; geneIter < parent1.getGenome().get(chromIter).size(); geneIter++) {

                if (randomizer.get(chromIter).get(geneIter)) {//give the double from parent1 to offspring1 and p2 to offspring2
                    chromosomeO1.add(parent1.getGenome().get(chromIter).get(geneIter));
                    chromosomeO2.add(parent2.getGenome().get(chromIter).get(geneIter));

                    stratChromosomeO1.add((parent1.getStrategyParams().get(chromIter).get(geneIter)));
                    stratChromosomeO2.add((parent2.getStrategyParams().get(chromIter).get(geneIter)));
                } else {//flip which offspring get which gene
                    chromosomeO1.add(parent2.getGenome().get(chromIter).get(geneIter));
                    chromosomeO2.add(parent1.getGenome().get(chromIter).get(geneIter));

                    stratChromosomeO1.add((parent2.getStrategyParams().get(chromIter).get(geneIter)));
                    stratChromosomeO2.add((parent1.getStrategyParams().get(chromIter).get(geneIter)));
                }
            }

            genomeOffspring1.add(new ArrayList<>(chromosomeO1));
            genomeOffspring2.add(new ArrayList<>(chromosomeO2));
            stratParamsOffspring1.add(new ArrayList<>(stratChromosomeO1));
            stratParamsOffspring2.add(new ArrayList<>(stratChromosomeO2));

        }
        //add the genomes and the stratParams to the offspring
        offspring1.setGenome(genomeOffspring1);
        offspring2.setGenome(genomeOffspring2);
        offspring1.setStrategyParams(stratParamsOffspring1);
        offspring2.setStrategyParams(stratParamsOffspring2);

        //create the offspring pair that will be returned
        ArrayList<IndividualES> offspringPair = new ArrayList<>();
        offspringPair.add(offspring2);
        offspringPair.add(offspring1);
        return offspringPair;
    }

    // for any number of offspring:
    // look at every Double and with a Driver.mutationRate chance
    // change that value by getting a random gaussian distributed number centered at
    // 0 with a standard deviation of 1
    public ArrayList<IndividualES> mutateOffspringFeatures(ArrayList<IndividualES> offspring) {

        for (final ListIterator<IndividualES> individualIter = offspring.listIterator(); individualIter.hasNext(); ) {
            final IndividualES individual = individualIter.next();


            for (final ListIterator<ArrayList<Double>> chromIter = individual.getGenome().listIterator(); chromIter.hasNext(); ) {
                final ArrayList<Double> chromosome = chromIter.next();

                for (final ListIterator<Double> geneIter = chromosome.listIterator(); geneIter.hasNext(); ) {
                    final Double gene = geneIter.next();

                    if (randNum.nextDouble() <= Driver.mutationRate) {
                        Double newVal = gene + individual.getStrategyParams()            //gets strategy parameter associated with the current gene
                                .get(individual.getGenome().indexOf(chromosome))        //and uses it as a standard deviation for a random gaussian number
                                .get(chromosome.indexOf(gene)) * randNum.nextGaussian();//that is used to mutate the gene

                        geneIter.set(newVal);
                    }

                }
            }
        }

        return offspring;
    }

    public ArrayList<IndividualES> oneFithRule(ArrayList<IndividualES> offspring) {

        for (final ListIterator<IndividualES> individualIter = offspring.listIterator(); individualIter.hasNext(); ) {
            final IndividualES individual = individualIter.next();

            for (final ListIterator<ArrayList<Double>> chromIter = individual.getStrategyParams().listIterator(); chromIter.hasNext(); ) {
                final ArrayList<Double> chromosome = chromIter.next();

                for (final ListIterator<Double> geneIter = chromosome.listIterator(); geneIter.hasNext(); ) {
                    final Double gene = geneIter.next();

                    //reduce the top fitness individuals mutation stdev, increace the bottom fitness individuals stdev
                    if (offspring.indexOf(individual) >= (this.oneFith * offspring.size())) {
                        Double newVal = gene + this.mutationRedux;
                        geneIter.set(newVal);
                    } else {
                        Double newVal = gene + this.mutationIncreace;
                        geneIter.set(newVal);
                    }

                }
            }
        }
        return offspring;
    }


    // evaluated the fitness of the population
    public ArrayList<IndividualES> evalFitness(ArrayList<IndividualES> population) {

        ArrayList<ArrayList<Object>> evalSet = Driver.dataset.getEvalDataSet2(50);

        for (IndividualES individual : population) {
            double fitness = 0;

            for (ArrayList<Object> datapoint : evalSet) {
                if (individual.getNetwork().evaluate(datapoint)) { //returns true or false for classification
                    fitness++;
                }
            }

            fitness = (fitness / evalSet.size()) ;
            individual.getNetwork().setFitness(fitness);
        }
        //System.out.println("===========================");

        Collections.sort(population);
        //System.out.println("Generation " + gencounter);
        //Double best = population.get(population.size() - 1).getNetwork().getFitness();
        //Double worst = population.get(0).getNetwork().getFitness();
        //System.out.println("best = " +best+ "\tworst = " + worst );
        return population;
    }


    public Boolean hasConverged(ArrayList<IndividualES> currPop, ArrayList<IndividualES> prevPop) {
        if (this.gencounter < 200){
            return false;
        }
        return true;
    }

    @Override
    Network train() {

        ArrayList<IndividualES> population = generatePopulation();
        population = evalFitness(population);
        ArrayList<IndividualES> prevPopulation = null;
        ArrayList<IndividualES> offspring = null;

        do {
            gencounter++;
            prevPopulation = population;
            offspring = newGeneration(population);
            population = new ArrayList<>(replacePop(offspring, population));

        } while (!hasConverged(population, prevPopulation));
        //returns highest fit individual after convergence
        return population.get(population.size() - 1).getNetwork();
    }





    private ArrayList<IndividualES> replacePop(ArrayList<IndividualES> offspring, ArrayList<IndividualES> prevGeneration) {


        ArrayList<IndividualES> nextGeneration = new ArrayList<>();
        IndividualES mostFit = null;
        while (nextGeneration.size() < Driver.populationSize) {

            int comparator = offspring.get(offspring.size() - 1).compareTo(prevGeneration.get(prevGeneration.size() - 1));

            if (comparator >= 0) {
                mostFit = new IndividualES(offspring.get(offspring.size() - 1));

            } else if (comparator == -1) {
                mostFit = new IndividualES(prevGeneration.get(prevGeneration.size() - 1));
            }

            nextGeneration.add(mostFit);
        }

        Collections.sort(nextGeneration);
        System.out.println("Current best percent error: " +  (1 - nextGeneration.get(nextGeneration.size()-1).getNetwork().getFitness()));
        return nextGeneration;

    }

}
