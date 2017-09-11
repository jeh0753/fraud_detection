# Fraud Detection Case Study

Our team built a Gradient Boosted Model that triages fraudulent events, and recommends whether or not to flag a given event for further investigation. Our model produces an F1 score (weighted precision and recall) of 0.66. 

We identified these factors as key indicators of fraud, and used our available data to find appropriate proxies:

* Completeness of the Event Listing
* Country where Event Occurs
* Veracity of Event Listing
* Transaction History of Event Organizer
* Payment Method Requested
* 'Value' of the Event (Expected Payout Size)
* Conference vs. Non-Conference

# File Structure

Data:
The data is *confidential* and cannot be shared outside of Galvanize is in that directory, under `files/data.zip`.

Model:
Our model is in the home directory, under model.py. 
Predictions are generated via predict.py

A few notebooks can also be found there outlining our exploratory data analysis process, our feature engineering process, and our model development process. You can uncompress the data file with this command: `unzip data.zip`.


