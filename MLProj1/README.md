# MLProj1

This is a CLI ARFF Converter built for CSCI447 - Soft Computing.
Built by Tiana Smith, Elias Athey, and Aaron McCarthy

This converter takes in a data file (.dat, .data, .csv, or .txt) and converts it into an ARFF file that [WEKA](http://www.cs.waikato.ac.nz/ml/weka/) can use. Python 3.x is required to run this script.

We are building this so we can test some of WEKA's Machine Learning algorithms on datasets from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php).

## How to Use
- Open a terminal to the location of the script.
- Run the command: python ARFF_Converter.py \<path-to-dataset\>
- Enter the classes according to the specified format.
- The script will show a sample datapoint (probably a bunch of numbers)
- Enter the names and datatypes of each attribute in the sample datapoint.
- Format it as specified by the script.
- If succesful, the script will end and a new ARFF file will be created in the current directory.


Here is sample run of the script using the Iris Dataset:

![A sample run using the Iris dataset](/SampleRun.png?raw=true "A sample run using the Iris dataset")

## The 5 Datasets
- [Iris Dataset](http://archive.ics.uci.edu/ml/datasets/Iris)
- [Hayes-Roth](https://archive.ics.uci.edu/ml/datasets/Hayes-Roth)
- [Vertebral Column Dataset](http://archive.ics.uci.edu/ml/datasets/Vertebral+Column)
- [Abalone Dataset](http://archive.ics.uci.edu/ml/datasets/Abalone)
- [Haberman's Survival](http://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival)
