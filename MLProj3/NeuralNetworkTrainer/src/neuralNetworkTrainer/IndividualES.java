package neuralNetworkTrainer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Random;

public class IndividualES implements Comparable<IndividualES> {
    private ArrayList<ArrayList<Double>> genome;
    private ArrayList<ArrayList<Double>> strategyParams;
    private Network network;



    public IndividualES(boolean  initialPopulation){
        Random randNum = new Random();
        strategyParams = new ArrayList<>();
        genome = new ArrayList<>();
        if(initialPopulation){
            this.network = new Network(true);
            genome = Network.serializeNetwork(this.network, false);

            for (ArrayList<Double> chromosome: genome){
                ArrayList<Double> stratChrom = new ArrayList<>();
                for (Double gene: chromosome ) {
                    stratChrom.add(randNum.nextGaussian() * 15);
                }
                this.strategyParams.add(stratChrom);
            }
        }
    }

    public IndividualES(IndividualES copy){
        strategyParams = new ArrayList<>(copy.strategyParams);
        genome = new ArrayList<>(copy.genome);
        network = copy.network;
    }

    public ArrayList<ArrayList<Double>> getGenome() {
        return genome;
    }

    public void setGenome(ArrayList<ArrayList<Double>> genome) {
        this.genome = genome;
    }

    public void setStrategyParams(ArrayList<ArrayList<Double>> strategyParams) {
        this.strategyParams = strategyParams;
    }

    public ArrayList<ArrayList<Double>> getStrategyParams() {
        return strategyParams;
    }

    public Network getNetwork() {
        return network;
    }

    public void setNetwork(Network network) {
        this.network = network;
    }

    @Override
    public int compareTo(IndividualES individual) {

        if (this.network.getFitness() < individual.getNetwork().getFitness()) {
            return -1;
        } else if (this.network.getFitness() > individual.getNetwork().getFitness()) {
            return 1;
        } else {
            return 0;
        }
    }
}
