# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:13:53 2017

@author: Aaron McCarthy
@author: Elias Athey
@author: Tia Smith
"""
import sys
import re

# Prompts the user for the attributes and associated data types
# Returns a list of 2-tuples, each corresonding to an attribute name and datatype pair
def getAttributes(classes):
	global dataFile, classColumnIndex
	with open(dataFile, 'r') as file:
		# Get the first data point as a list of strings
		firstLine = file.readline().strip()
		sampleDataPoint = []
		
		# split data on commas or spaces
		if re.search(',', firstLine):
			sampleDataPoint = firstLine.split(',')
		else:
			sampleDataPoint = re.split('\s+', firstLine)
		sampleDataPoint = [attr.strip() for attr in sampleDataPoint]
		
		# Finds the column in which the class is defined
		# Order of checking: last index, index 0, then index 1
		# This does not handle every case - open to other suggestions
		classColumnIndex = -1
		if (sampleDataPoint[len(sampleDataPoint) - 1] in classes):
			classColumnIndex = len(sampleDataPoint) - 1
		elif(sampleDataPoint[0] in classes):
			classColumnIndex = 0
		else:
			classColumnIndex = 1
		sampleClass = sampleDataPoint[classColumnIndex]
		sampleDataPoint.pop(classColumnIndex)
		sampleAttributes = re.sub('\[|\]|\'', '', str(sampleDataPoint))
		
		# Message to user and sample data point
		print("\nThank you.\nBelow is a sample data point.")
		print("Please enter the attribute names and associated data types in the order they appear in the sample.")
		print("---------------------------------------------------------------------------------------------------")
		print("Follow this format:\n\n\t<attributeName1>:<dataType1>, <attributeName2>:<dataType2>, ...etc\n\n<dataType> can be one of NUMERIC, STRING, DATE, or a nominal type.\nFormat nominal types as {nominalValue1 & nominalValue2 &...etc}\n")
		print("Here is the sample data point of the class", sampleClass + ".")
		print("Enter the " + str(len(sampleDataPoint)) + " attributes in the order they appear in the sample.")
		dashCount = len(sampleAttributes) + 4
		for dash in range(dashCount):
			print("-", end='')
		print("\n| " + sampleAttributes + " |")
		for dash in range(dashCount):
			print("-", end='')
		
		# check user input and return the attributes as a list of 2-tuples (attribute name, data type)
		oneAttrRegex = '([\w -]+:(NUMERIC|STRING|DATE|(\{(([\w-]+)|(([\w -]+&)+[\w -]+))\})))'
		formatRegex = '(' + oneAttrRegex + '(, )?){' + str(len(sampleDataPoint)) + '}'
		userInput = ""
		inputIsValid = False
		while not inputIsValid:
			userInput = input('\n> ')
			userInput = userInput.strip()
			if len(userInput) is 0:
				print("Error: You must enter the corresonding attributes.\n")
			elif not re.match(formatRegex, userInput):
				print("Error: You must enter the " + str(len(sampleDataPoint)) + " attributes in the specified format.\n")
			else:
				inputIsValid = True
		
		return [attrTuple.strip().split(':') for attrTuple in userInput.split(',')]

# Prompts the user for the data classes. Verifies the uses inputs a comma separated list
# Returns a list of the class names
def getClasses():
	classRegex = '((^(([\w -])+,)+([\w -])+$)|(^(\w-)+$))|(^(\w-)+$),( )?NUMERIC|(^(\w-)+$),( )?STRING|(^(\w-)+$),( )?DATE'
	classes = ""
	inputIsValid = False
	while not inputIsValid:
		print("\nPlease enter the exact class names from the data (case sensitive) separated by commas.")
		print("Use the format specified below depending upon the class type:\n")
		print("Class is Nominal: <class1>, <class2>,...etc")
		print("Continuous Number: <classLabel>, NUMERIC")
		print("A Calendar Date: <classLabel>, DATE")
		print("A Unique String: <classLabel>, STRING\n")
		classes = input('> ')
		classes = classes.strip()
		if len(classes) is 0:
			print("Error: You must enter at least 1 class name.\n")
		elif not re.match(classRegex, classes):
			print("Error: Class names must follow the specified format.\n")
		else:
			inputIsValid = True
		
	classes = classes.split(',')
	classes = [c.strip() for c in classes]
	return classes

# Given the classes and attributes (with datatypes)
# Create and format an .arff file in the current directory
def createARFF(classes, attributes):
	global dataFile
	fileName = re.search('(\w)+\..', dataFile).group(0).split('.')[0]
	outputString = ""	
	# write @RELATION, @ATTRIBUTE, and @DATA tags specified by ARFF format
	with open(dataFile, "r") as file:
		outputString += "@RELATION " + fileName + "\n"
		for  attr in attributes:
			outputString += "\n@ATTRIBUTE " + attr[0].replace(' ', '') + ' ' + attr[1].replace(' ','').replace('&',',')
		
		# if class is a nominal value
		if(not re.match('NUMERIC|STRING|DATE', classes[1])):
			outputString += "\n@ATTRIBUTE class {"
			for c in classes:
				outputString += c
				if classes.index(c) is not len(classes)-1:
					outputString += ','
			
			outputString += '}\n'
		else:
			outputString += "\n@ATTRIBUTE " + classes[0] + " " + classes[1] + "\n"
			
		outputString += formatData(len(attributes))

			
	#create the ARFF file and write ouputString into the file
	arffFile = fileName + ".arff"
	with open(arffFile, "w") as newFile:
		newFile.write(outputString)

# Given the classes and attributes
# Format the @DATA section and return it as a string
def formatData(numAttributes):
	global dataFile
	with open(dataFile, 'r') as file:

		#check for correct file extension
		# continue with warning if file extension wrong
		if not re.search("\.txt|\.data|\.csv|\.dat", dataFile):
			print("File format not supported\nFile must be .txt, .data, .dat, or .csv")
			print("Data transfer continuing...\nResults may be unreliable")

		#make header
		data = "\n@DATA\n"
		lines = file.readlines()
		lines[0] = lines[0].lstrip()
		lines[len(lines) - 1] = lines[len(lines) - 1].rstrip()

		# check if the file is space delimited with range to account for class names with spaces
		# replace spaces with commas
		if len(re.split(" ", lines[0])) in range(numAttributes, numAttributes + 4):
			for i in range(0, len(lines)):
				# replace numAttributes number of spaces with commas on every line
				lines[i] = re.sub(" ", ",", lines[i], numAttributes)

		for line in lines:
			data = data + line
		data = data + "\n%\n%\n%\n"
		return data

# Entry point of the program
# Calls methods to collect classes and attributes, then sends that information to createARFF
if __name__ == "__main__":
	global dataFile
	if len(sys.argv) > 1:
		dataFile = sys.argv[1]
		classes = getClasses()
		attributes = getAttributes(classes)
		createARFF(classes, attributes)
	else:
		print ("Error: must enter a single valid datafile to convert.")
	
