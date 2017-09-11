# Fraud Detection Case Study

Our team built a Gradient Boosted Model that triages fraudulent events. Our model produces an F1 score (weighted precision and recall) of 0.66. 

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
The data used in this case study is confidential, and therefore not included in this repository.

Model:
Our model is in the home directory, under model.py. 
Predictions are generated via predict.py

A few notebooks can also be found outlining our feature engineering process, and our model development process. 


