package neuralNetworkTrainer;

import java.util.ArrayList;
import java.util.NoSuchElementException;
import java.util.Scanner;

public class LetterRecogData extends Data{

    /**
     * Constructs LetterRecogData
     */
    LetterRecogData(){
        numInputs = 16;
        numOutputs = 26;
        dataPoints = new ArrayList<>();
        this.setDataPoints();
    }

    /**
     * Sets the data points by scanning the file
     */
    private void setDataPoints(){
        Scanner fileScanner = new Scanner(Thread.currentThread().getContextClassLoader().getResourceAsStream("letter-recognition.data"));
        do{
            try{
                String line = fileScanner.nextLine();
                String[] entries = line.split(",");
                ArrayList<Object> values = new ArrayList<>();
                double classValue = (int)entries[0].charAt(0) - 65; // this maps A to 0, B to 1, etc
                for(int entryIter = 1; entryIter < entries.length; entryIter++){
                    values.add(Double.parseDouble(entries[entryIter]));
                }
                values.add(classValue);
                dataPoints.add(values);
            }
            catch(NoSuchElementException e){
                break;
            }
        }
        while(true);
    }
}
