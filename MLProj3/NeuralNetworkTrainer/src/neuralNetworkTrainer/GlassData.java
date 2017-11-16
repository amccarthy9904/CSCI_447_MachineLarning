package neuralNetworkTrainer;

import java.util.ArrayList;
import java.util.NoSuchElementException;
import java.util.Scanner;

public class GlassData extends Data{

    /**
     * Constructs GlassData
     */
    GlassData(){
        numInputs = 9;
        numOutputs = 7;
        dataPoints = new ArrayList<>();
        this.setDataPoints();
    }

    /**
     * Sets the data points by scanning the file
     */
    private void setDataPoints(){
        Scanner fileScanner = new Scanner(Thread.currentThread().getContextClassLoader().getResourceAsStream("glass.data"));
        do{
            try{
                String line = fileScanner.nextLine();
                String[] entries = line.split(",");
                ArrayList<Object> values = new ArrayList<>();
                for(int entryIter = 0; entryIter < entries.length - 1; entryIter++){
                    values.add(Double.parseDouble(entries[entryIter]));
                }
                values.add(Double.parseDouble(entries[entries.length - 1]) - 1);
                dataPoints.add(values);
            }
            catch(NoSuchElementException e){
                break;
            }
        }
        while(true);
    }
}
