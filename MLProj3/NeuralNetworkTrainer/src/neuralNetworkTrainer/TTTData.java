package neuralNetworkTrainer;

import java.util.ArrayList;
import java.util.NoSuchElementException;
import java.util.Scanner;

public class TTTData extends Data{

    /**
     * Constructs TTTData
     */
    TTTData(){
        numInputs = 9;
        numOutputs = 2;
        dataPoints = new ArrayList<>();
        this.setDataPoints();
    }

    /**
     * Sets the data points by scanning the file
     */
    private void setDataPoints(){
        Scanner fileScanner = new Scanner(Thread.currentThread().getContextClassLoader().getResourceAsStream("tic-tac-toe.data"));
        do{
            try{
                String line = fileScanner.nextLine();
                String[] entries = line.split(",");
                ArrayList<Object> values = new ArrayList<>();
                for(int entryIter = 0; entryIter < entries.length; entryIter++){
                    switch(entries[entryIter]){
                        case "x": values.add(-1.0); break;
                        case "b": values.add(0.0); break;
                        case "o": values.add(1.0); break;
                        case "positive": values.add(0.0); break;
                        case "negative": values.add(1.0); break;
                        default: System.out.println("Error in reading tic-tac-toe.data"); System.exit(0);
                    }
                }
                dataPoints.add(values);
            }
            catch(NoSuchElementException e){
                break;
            }
        }
        while(true);
    }
}
