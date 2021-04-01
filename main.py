#Ahmed Mohamed Flair Tutorial

# import flair for sentiment model and importing pandas for managing the dataFrame
import flair 
import pandas as pd
# import csv to write results
import csv
from itertools import zip_longest

# make sire the the dataframe displays with max column width
pd.set_option('display.max_colwidth', None)
# import model
sentimentModel = flair.models.TextClassifier.load('en-sentiment')


# read in files and put them into a dataframe using pandas
dataFrame = pd.read_csv('test.tsv', sep='\t')

# drop the duplicates in the data by Sentence
dataFrame.drop_duplicates(subset='SentenceId', keep='first', inplace=True)


print("These are the Sentences that will be analyzed and given a sentiment")

print(dataFrame.head(3)) #should show that you can only print head in a specific part

#create lists to store the sentiments and confidence of each sentence in the list
sentiment =[]
confidence =[]

# loop through each "sentence" in the data frame in the Phrase column 
for sentence in dataFrame['Phrase']:
    if sentence.strip() == "": # strip spaces and check for empty cells
        sentiment.append("") 
        confidence.append("")

    else:
        sample = flair.data.Sentence(sentence) 
        sentimentModel.predict(sample, mini_batch_size=64)

        sentiment.append(sample.labels[0].value) # add the label of the first index to a list (contains the sentiment value)
        confidence.append(sample.labels[0].score) # add the label of the first index to a list (contains the confidence value)


dataFrame['sentiment'] = sentiment # add the list of the the new sentiments to the data frame as a column 

dataFrame['confidence'] = confidence # add the list of the the new sentiments to the data frame as a column 

print(dataFrame.head(3)) # Print the final dataFrame head to display sentiments

#create a sentences list to hold the sentences for the csv 
sentences = []
# loop through each "sentence" in the data frame in the Phrase column 
for sentence in dataFrame['Phrase']:
    if sentence.strip() == "": # strip spaces and check for empty cells
        sentiment.append("") 
        confidence.append("")
    else:
        sentences.append(sentence)  #add sentences to list

# set up columns and rows 
rows = [['Sentence','Sentiment','Confidence']]
columns = [sentences,sentiment,confidence]
file = open('results.csv','w')
columnData = zip_longest(*columns, fillvalue = '')

with file: # write it into csv file
    writer = csv.writer(file)
    writer.writerows(rows)
    writer.writerows(columnData)


file.close() # close the file