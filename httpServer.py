from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
@app.route("/sentiment_result",methods= ["POST"])
def sentiment_result():
	data = request.get_json(force=True)
	sent_result = model.sentiment_result([[np.array(data["exp"])]])
	output = sent_result[0]
	return jsonify(output)
if __name__ == "__main__":
	try:
		app.run(host="0.0.0.0", port = 5000, debug = True)
	except:
		print('could not creating hosting server')

