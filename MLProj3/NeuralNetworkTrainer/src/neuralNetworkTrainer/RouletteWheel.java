package neuralNetworkTrainer;

import java.util.ArrayList;

public class RouletteWheel {

    private ArrayList<Integer> wheel;
    public int size;

    public RouletteWheel (){

        this.wheel = new ArrayList<>();

        for (int rank = 0; rank < Driver.populationSize; rank++) {
            for (int rankIter = 0; rankIter <= (rank/2); rankIter++) {
                wheel.add(rank);
            }
        }
        this.size = wheel.size();
    }

    public int get(int index){
        return this.wheel.get(index);
    }
}
