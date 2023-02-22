# Product Price Prediction
##### [Project Explanation](https://medium.com/@priyankkhanna2001/product-price-prediction-using-flipkart-data-934e227da5f8)
### Runing web interface
1) If you already have a python env you can open the env in terminal and run the following commands
- pip install -r requirements.txt
- python models\model2.py

2) You can also create a docker image to run it using -
- docker build -t exp:v1 -f docker/Dockerfile .
- docker run -it --rm --add-host=host.docker.internal:host-gateway exp:v1

### flipkart_categories.xlsx
Contains all the different categories along with their url. These url were used to scrape different products information present in them.
### data_url
First page of flipkart_categories.xlsx. Used by flipkart_scrapper.ipynb to navigate to different pages and generate dataset.csv.
### dataset.csv
Complete flipkart dataset generated using 1st page of each categories. Size can further be increased if different footnote pages are also included.
### url.csv
Used by flipkart_scrapper.ipynb to navigate to different pages and generate poc.csv dataset.
### Flipkart_scrapper.ipynb
Code used to generate the poc.csv and complete dataset for flipkart product description. \
Link for complete dataset - [Raw csv file](https://raw.githubusercontent.com/priyankkhanna/product_price_prediction/main/models/dataset.csv)
### preprocess_model.ipynb
Code regarding preprocessing the data, making it model ready, training and fine tuning the model.
### poc.csv
Proof of Concept dataset.
### int1.csv
Intermidiate data used by flipkart_scrapper.ipynb.
### int2.csv
Intermidiate data used by flipkart_scrapper.ipynb.

###models contains results, actual dataset and code for running the file
