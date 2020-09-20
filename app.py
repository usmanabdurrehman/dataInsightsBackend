from flask import Flask, jsonify, request
from flask_cors import CORS,cross_origin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_iris,load_diabetes
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVR,SVC
from sklearn.feature_selection import RFE,SelectKBest,f_regression,f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,GridSearchCV
import math
import pickle

app = Flask(__name__, static_folder='./build', static_url_path='/')
CORS(app,support_credentials=True) 

best_model = ''

@app.route('/')
def index():
	return app.send_static_file('index.html')

@app.route('/getData', methods = ['GET','POST'])
@cross_origin(origin='*',supports_credentials=True)
def getData():
	file = request.files['file']
	target = request.form['target']

	print(file)

	path = './prostate.csv'
	df = pd.read_csv(file)
	# df = pd.DataFrame({
	# 	'f1':[np.nan,np.nan,np.nan,45],
	# 	'f2':[1,63,np.nan,45],
	# 	'target':[23,45,1,23]
	# 	})

	# target = 'lpsa'
	X,y = df.drop(target,axis=1),df[target]



	### 		Dropping Cols with Nans greater than 27%      ###
	null_vals = X.isnull().sum()/len(X) * 100
	columns_to_be_dropped = []
	j = 0
	while(j<len(null_vals)):
		if(null_vals[j]>27):
			columns_to_be_dropped.append(null_vals.index[j])
		j+=1	

	X = X.drop(columns_to_be_dropped,axis=1)






	###  		Check if categorical and then filling with mode or mean       ###
	columns_iscategorical = []
	for i in X.columns:
		unique_vals = X[i].unique()
		# print(unique_vals)
		if(len(unique_vals)<12):
			columns_iscategorical.append(True)
			X[i].fillna(stats.mode(X[i])[0][0],inplace=True)
		else:
			X[i].fillna(X[i].mean(),inplace=True)
			columns_iscategorical.append(False)






	### 		Converting string data into numerical form       ###
	dtypes = X.dtypes
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



	dataDisplay = {
		'title':file.filename.split('.')[0] + ' dataset',
		'rows':pd.concat([X,y],axis=1).values.tolist(),
		'columns':pd.concat([X,y],axis=1).columns.to_list()
	}

	# print(dataDisplay)

	statss = {
		'problemType':'',
		'numSamples':len(X),
		'numFeatures':len(X.columns),
		'numTarget':1,
		'featureStats':[]
	}

	# print(statss)

	def problem_type(target):
		unique_vals = target.unique()
		threshold_unique_categories = 12
		if(len(unique_vals)>threshold_unique_categories):
			return 'regression'
		else:
			return 'classification'

	statss['problemType'] = problem_type(y)



	####       Feature Importances        ####
	def feature_importance(X,y,number_of_features_to_select):
		if(problem_type(y)=='classification'):
			feature_importances_tree = RandomForestClassifier(random_state=0).fit(X,y).feature_importances_
			features_sorted_tree = [columns for _,columns in sorted(zip(feature_importances_tree,X.columns),reverse=True)]
			features_selected_RFE = list(X.columns[RFE(RandomForestClassifier(random_state=0),n_features_to_select=number_of_features_to_select).fit(X,y).support_])

			feature_importances_statistical = SelectKBest(score_func=f_classif,k=number_of_features_to_select).fit(X,y).scores_
			features_sorted_statistical = [columns for _,columns in sorted(zip(feature_importances_statistical,X.columns),reverse=True)]
			features_sorted_linear = False

		else:
			feature_importances_tree = RandomForestRegressor(random_state=0).fit(X,y).feature_importances_
			features_sorted_tree = [columns for _,columns in sorted(zip(feature_importances_tree,X.columns),reverse=True)]
			feature_importances_linear = LinearRegression().fit(X,y).coef_

			features_sorted_linear = [columns for _,columns in sorted(zip(feature_importances_linear,X.columns),reverse=True)]
			features_selected_RFE = list(X.columns[RFE(RandomForestRegressor(random_state=0),n_features_to_select=number_of_features_to_select).fit(X,y).support_])
			feature_importances_statistical = SelectKBest(score_func=f_regression,k=number_of_features_to_select).fit(X,y).scores_
			features_sorted_statistical = [columns for _,columns in sorted(zip(feature_importances_statistical,X.columns),reverse=True)]
		return {
			'linear':features_sorted_linear,
			'tree':features_sorted_tree,
			'statistical':features_sorted_statistical
		}
	feature_importances = feature_importance(X,y,number_of_features_to_select=math.ceil(len(X.columns)/2))
	feature_importances_client = []
	cols = X.columns	
	idx = 0

	def find(stack,needle):
		if needle in stack:
			return stack.index(needle)
		else:
			return -1	

	for feature in X.columns:
		tempdict = {
			'featureName':feature,
			'tree':find(feature_importances['tree'],feature)  + 1,
			'statistical':find(feature_importances['statistical'],feature) + 1
		}
		feature_importances_client.append(tempdict)
		idx+=1
	# print(feature_importances_client)





	### 			Plotting compressed features against the target       ###
	def the_plot(X,y):
		plt.figure()
		colors = ['red','blue','green','cyan','purple','magenta',
			'brown','black','gray','yellow']
		classes = ['class {}'.format(i) for i in range(20)]
		if(problem_type(y)=='classification'):
			reduced_X = PCA(n_components=2).fit_transform(X)
			for i in range(len(np.unique(y))):
				px = reduced_X[:,0][y==i]
				py = reduced_X[:,1][y==i]
				plt.scatter(px,py,c=colors[i],label=classes[i])
			plt.xlabel('First Principal Component')
			plt.ylabel('Second Principal Component')
			plt.legend(loc=0)
			return({
				'problem_type':'classification',
				'x':reduced_X[:,0].tolist(),
				'y':reduced_X[:,1].tolist()
			})	
			# plt.show()
		else:
			reduced_X = PCA(n_components=1).fit_transform(X)
			plt.scatter(reduced_X,y)
			plt.xlabel('Features projected to a 1D space')
			plt.ylabel(target)
			# plt.show()
			return({
				'problem_type':'regression',
				'x':reduced_X.ravel().tolist(),
				'y':y.tolist()
			})			

	plot_data = the_plot(X,y)
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

	###               Stats of features              ###
	# Should be according to feature type(nominal,ordinal,interval)
	# statistics = []
	# for i in range(len(X.columns)):
	# 	statistic = {
	# 		'feature':X.columns[i],
	# 		'mean':X[X.columns[i]].mean(),
	# 		'median':X[X.columns[i]].median(),
	# 		'mode':stats.mode(X[X.columns[i]])[0][0]
	# 	}
	# 	statistics.append(statistic)	

	###			Fitting a model 			###
	def fitting_model(X,y):
		clfs = [GradientBoostingClassifier(random_state=0),RandomForestClassifier(random_state=0),
		SVC(),KNeighborsClassifier()]
		regs = [GradientBoostingRegressor(random_state=0),RandomForestRegressor(random_state=0),
		SVR(),KNeighborsRegressor()]

		optimized_clfs = []
		optimized_regs = []

		paramgrids = [
		{
			'learning_rate':[0.01,0.1],
			'n_estimators':[100,125,150],
			'min_samples_split':[2,4]
		},
		{
			'n_estimators':[10,20,30],
			'min_samples_split':[2,4]
		},
		{
			'kernel':['linear','poly','rbf'],
			'C':[0.1,0.3,1]
		},
		{
			'n_neighbors':[4,5,6]
		}
		]

		best_model = ''
		scores = []
		X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=33)
		idx=0
		if(problem_type(y)=='classification'):
			for clf in clfs:
				# clf = GridSearchCV(estimator=i,param_grid=paramgrids[idx]) 
				clf.fit(X_train,y_train)
				optimized_clfs.append(clf)
				scores.append(optimized_clfs[idx].score(X_test,y_test))
				idx+=1
			best_model = optimized_clfs[np.argmax(scores)]
		else:
			for reg in regs:
				# reg = GridSearchCV(estimator=i,param_grid=paramgrids[idx])
				reg.fit(X_train,y_train)
				optimized_regs.append(reg)
				scores.append(optimized_regs[idx].score(X_test,y_test))
				idx+=1
			best_model = optimized_regs[np.argmax(scores)]
		return {
			'best_model':best_model,
			'model_info':{
				'best_model_name':type(best_model).__name__,
				'accuracy':round((np.max(scores)*100),2)
			}
		}		
	'''
	Best model 
	GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
	             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
	             max_leaf_nodes=None, min_impurity_decrease=0.0,
	             min_impurity_split=None, min_samples_leaf=1,
	             min_samples_split=4, min_weight_fraction_leaf=0.0,
	             n_estimators=100, presort='auto', random_state=0,
	             subsample=1.0, verbose=0, warm_start=False) 
	r2_score 
	0.9832532381346817
	'''
	model__ = fitting_model(X,y)
	model_info = model__['model_info']
	best_model = model__['best_model']
	filename = 'model.sav'

	pickle.dump(best_model, open(filename, 'wb'))

	finalDict = {
		'dataDisplay':dataDisplay,
		'stats':statss,
		'featureImportances':feature_importances_client,
		'plotData':plot_data,
		'model_info':model_info
	}

	return jsonify(finalDict)

@app.route('/predict', methods = ['GET','POST'])
@cross_origin(origin='*',supports_credentials=True)
def predict():
	features = request.get_json()['features']
	filename = 'model.sav'
	model = pickle.load(open(filename, 'rb'))
	prediction = model.predict([features])
	prediction[0] = str(round(prediction[0],2))
	print(prediction)
	return jsonify({
		'prediction':str(prediction[0])
	})

if __name__ == "__main__":
	app.run(debug=True)	