from flask import Flask, render_template, request, url_for
import pickle
app = Flask(__name__)

clf = pickle.load(open("classifier.pkl", "rb"))

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/iris_classification', methods = ['GET', 'POST'])
def iris_classification():
	plt = [[url_for('static', filename="sb_fig/rel_sl_sw_plt.svg"), url_for('static', filename="sb_fig/rel_pl_pw_plt.svg")], 
	[url_for('static', filename="sb_fig/cat_sl_plot.svg"), url_for('static', filename="sb_fig/cat_sw_plot.svg")],
	[url_for('static', filename="sb_fig/cat_pl_plot.svg"), url_for('static', filename="sb_fig/cat_pw_plot.svg")],
	[url_for('static', filename="sb_fig/box_sl_plt.svg"), url_for('static', filename="sb_fig/box_sw_plt.svg")],
	[url_for('static', filename="sb_fig/box_pl_plt.svg"), url_for('static', filename="sb_fig/box_pw_plt.svg")]]

	train_clf = url_for('static', filename="sb_fig/train.png")
	test_clf = url_for('static', filename="sb_fig/test.png")

	plt_head = [['sepal_length vs sepal_width', 'petal_length vs petal_width'],
				['sepal_length', 'sepal_width'], ['petal_length', 'petal_width'],
				['sepal_length', 'sepal_width'], ['petal_length', 'petal_width']]

	plt_ctnt = [['Setosa has small sepal_length and high sepal_width. Versicolor has medium sepal_length and medium sepal_width, Virginica has high sepal_length and medium sepal_width',
	'Setosa has less petal_length and less petal_width, Versicolor has medium petal_length and medium petal_width, Virginica has more petal_length and more petal_width'],
	['We can classify between setosa and virginica by means of sepal_length', 'We can classify setosa and versicolor by means of sepal_width'],
	['Petal_length column will be working as the major classifier which clearly classifies all the three types setosa, versicolor and virginica', 
	'Petal_width column is also a major classifier, which helps us to classify setosa, versicolor and virginica'],
	['Virginica has the outliers with respect to sepal_length', 'Setosa and Virginica has two outliers with respect to sepal_width'],
	['Setosa and Versicolor has one minor outliers with respect to petal_length', 'Setosa has minor outlier with respect to petal_width']]

	if request.method == 'POST':

		sep_len = request.form['sep_len']
		sep_wid = request.form['sep_wid']
		pet_len = request.form['pet_len']
		pet_wid = request.form['pet_wid']

		species = clf.predict([[sep_len, sep_wid, pet_len, pet_wid]])

		return render_template('iris classification.html', species = species[0], sub = False, plt = plt, plt_head = plt_head, train_clf = train_clf, test_clf = test_clf, plt_ctnt = plt_ctnt)
	return render_template('iris classification.html', sub = True, plt = plt, plt_head = plt_head, train_clf = train_clf, test_clf= test_clf, plt_ctnt = plt_ctnt)

if __name__ == '__main__':
	app.run(debug = True)