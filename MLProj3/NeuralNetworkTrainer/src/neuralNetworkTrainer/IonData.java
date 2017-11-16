package neuralNetworkTrainer;

import java.util.ArrayList;
import java.util.NoSuchElementException;
import java.util.Scanner;

public class IonData extends Data {

    /**
     * Constructs IonosphereData
     */
    IonData(){
        numInputs = 34;
        numOutputs = 2;
        dataPoints = new ArrayList<>();
        this.setDataPoints();
    }

    /**
     * Sets the data points by scanning the file
     */
    private void setDataPoints(){
        Scanner fileScanner = new Scanner(Thread.currentThread().getContextClassLoader().getResourceAsStream("ionosphere.data"));
        do{
            try{
                String line = fileScanner.nextLine();
                String[] entries = line.split(",");
                ArrayList<Object> values = new ArrayList<>();
                for(int entryIter = 0; entryIter < entries.length - 1; entryIter++){
                    values.add(Double.parseDouble(entries[entryIter]));
                }
                if(entries[entries.length - 1].equals("g")){
                    values.add(1.0);
                }
                else if(entries[entries.length - 1].equals("b")){
                    values.add(0.0);
                }
                else{
                    System.out.println("Error parsing ionosphere.data");
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
