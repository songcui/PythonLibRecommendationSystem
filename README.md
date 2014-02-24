PythonLibRecommendationSystem
=============================

This libraries contains various standard algorithms for recommendation system in python. 
It assumes the training dataset is stored in a txt file with the following format:

userID <tab> ItemID <tab> Rating <br />
5             19            4    <br /> 
4             6             3    <br />
...... <br />

1. method "qr" in qrlatentfactor.py builds up and optimize the qr matrix factorization model 
where we assume rating[user i, item j] = transpose(p[user i, :])*q[item j,:]. p[user i,:] and q[item j,:] are vectors with dimension 1 * k. k represents the number of hidden dimensions which is set by the user. In the optimization stage, we are trying to minimize the following cost function with L2 regularization on the parameters p and q: <br />
cost function = sum_{x} (rating[user i, item j] - transpose(p[user i, :]) * q[item j,:])^{2} + lamda1 transpose(p[user i, :]) * p[user i, :] +  lamda2 transpose(q[item j, :]) * p[item j, :]. The stochastic gradient decent is used to update the parameters. This method will output RMSE and the matrix p and q. 
