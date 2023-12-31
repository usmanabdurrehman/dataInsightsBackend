from flask import Flask, jsonify, request
from flask_cors import CORS,cross_origin
import pandas as pd
import math
import pickle
import json

from constants import COLUMN_NAN_THRESHOLD, SAVED_MODEL_NAME
from utils import find, get_decomposed_plot_data, get_problem_type, get_feature_importances, get_best_fit_model

app = Flask(__name__)
CORS(app,support_credentials=True) 

best_model = ''

@app.route('/')
@cross_origin(origin='*',supports_credentials=True)
def index():
	return 'Nice'

@app.route('/getData', methods = ['POST'])
@cross_origin(origin='*',supports_credentials=True)
def getData():
	showTicks = json.loads(request.form['showTicks'])
	df = ''
	target = ''
	file = ''

	if(showTicks['custom']):
		file = request.files['file']
		df = pd.read_csv(file)
		filename = file.filename.split('.')[0] + ' Dataset'
		target = request.form['target']
	elif(showTicks['heart']):
		filename = 'heart dataset'
		# df = pd.read_csv(r'/home/usmanabdurrehman/mysite/Data-Insights.io/heart.csv')
		df = pd.read_csv('example_datasets/heart.csv')
		target = 'target'
	elif(showTicks['prostate']):
		filename = 'prostate dataset'
		df = pd.read_csv('example_datasets/prostate.csv')
		target = 'lpsa'
	else:
		filename = 'Stock Prices dataset'
		df = pd.read_csv('example_datasets/stock_prices.csv') 
		target = 'Stock price'

	X,y = df.drop(target,axis=1),df[target]

	data_display = {
		'title':filename,
		'rows':pd.concat([X,y],axis=1).values.tolist(),
		'columns':pd.concat([X,y],axis=1).columns.to_list()
	}

	### 		Dropping Cols with Nans greater than 27%      ###
	null_vals = X.isnull().sum()/len(X) * 100
	columns_to_be_dropped = []
	j = 0
	while(j<len(null_vals)):
		if(null_vals[j]>COLUMN_NAN_THRESHOLD):
			columns_to_be_dropped.append(null_vals.index[j])
		j+=1	

	X = X.drop(columns_to_be_dropped,axis=1)

	###  		Check if categorical and then filling with mode or mean       ###
	columns_iscategorical = []
	for i in X.columns:
		unique_vals = X[i].unique()
		if(len(unique_vals)<12):
			columns_iscategorical.append(True)
			X[i].fillna(X[i].mode()[0],inplace=True)
		else:
			X[i].fillna(X[i].mean(),inplace=True)
			columns_iscategorical.append(False)

	### 		Converting string data into numerical form       ###
	columns_to_be_dummied = []
	columns_to_be_dropped = []
	j = 0
	for i in X.columns:
		if((X[i].dtype.name=='object' and columns_iscategorical[j]==True) or columns_iscategorical[j]==True):
			columns_to_be_dummied.append(i)
		elif(X[i].dtype.name=='object' and columns_iscategorical[j]==False):
			columns_to_be_dropped.append(i)
		j+=1	

	X = pd.get_dummies(X,columns=columns_to_be_dummied)
	X = X.drop(columns_to_be_dropped,axis=1)
		
	stats = {
		'problemType':get_problem_type(y),
		'numSamples':len(X),
		'numFeatures':len(X.columns),
		'numTarget':1,
		'featureStats':[]
	}

	####       Feature Importances        ####
	feature_importances = get_feature_importances(X,y,number_of_features_to_select=math.ceil(len(X.columns)/2))
	feature_importances_client = []	
	idx = 0

	for feature in X.columns:
		tempdict = {
			'featureName':feature,
			'tree':find(feature_importances['tree'],feature)  + 1,
			'statistical':find(feature_importances['statistical'],feature) + 1
		}
		feature_importances_client.append(tempdict)
		idx+=1

	### 	 Plotting compressed features against the target       ###
	plot_data = get_decomposed_plot_data(X,y)
	plot_data['x'] = [ round(elem,2) for elem in plot_data['x']]
	plot_data['y'] = [ round(elem,2) for elem in plot_data['y']]
	p = 0
	xy = []		
	for i in plot_data['x']:
		xy.append({
			'x':round(plot_data['x'][p],2),
			'y':round(plot_data['y'][p],2)
		})
		p+=1
	plot_data['xy'] = xy	

	###			Fitting a model 		###
	model__ = get_best_fit_model(X,y)
	model_info = model__['model_info']
	best_model = model__['best_model']
	filename = SAVED_MODEL_NAME

	pickle.dump(best_model, open(filename, 'wb'))

	finalDict = {
		'dataDisplay':data_display,
		'stats':stats,
		'featureImportances':feature_importances_client,
		'plotData':plot_data,
		'modelInfo':model_info
	}

	return jsonify(finalDict)

@app.route('/predict', methods = ['POST'])
@cross_origin(origin='*',supports_credentials=True)
def predict():
	features = request.get_json()['features']
	filename = SAVED_MODEL_NAME
	model = pickle.load(open(filename, 'rb'))
	prediction = model.predict([features])
	prediction[0] = str(round(prediction[0],2))

	return jsonify({
		'prediction':str(prediction[0])
	})


if __name__ == "__main__":
	app.run(debug=True)	