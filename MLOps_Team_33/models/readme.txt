
#Make a prediction

#Send X_test[0]

curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"input": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0,0.0, 1.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,0.0,0.0, 1.0, 0.0, 0.0, 1.0, 0.0]}'  