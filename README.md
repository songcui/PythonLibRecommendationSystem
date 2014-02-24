PythonLibRecommendationSystem
=============================

This libraries contains various standard algorithms for recommendation system in python. 
It assumes the training dataset is stored in a txt file with the following format:

userID <tab> ItemID <tab> Rating <br />
5             19            4    <br /> 
4             6             3    <br />
...... <br />

1. The method "qr" in qrlatentfactor.py builds up and optimize the qr matrix factorization model 
where we assume rating[user i, item j] = transpose(p[user i, :])*q[item j,:]. p[user i,:] and q[item j,:] are vectors with dimension 1 * k. k represents the number of hidden dimensions which is set by the user. In the optimization stage, we are trying to minimize the following cost function with L2 regularization on the parameters p and q: <br />
cost function = sum_{x} (rating[user i, item j] - transpose(p[user i, :]) * q[item j,:])^{2} + lamdauser transpose(p[user i, :]) * p[user i, :] +  lamdaitem transpose(q[item j, :]) * p[item j, :]. The stochastic gradient decent is used to update the parameters. This method will output RMSE at each iteraction and the matrix p and q. 

2. The method "qrPlusBaseline" in qrlatentfactor.py builds up and optimize the qr matrix factorization model plus the baseline method where we assume rating[user i, item j] = global_average + userbias[user i] + itembias[item j]+  transpose(p[user i, :])*q[item j,:]. p[user i,:] and q[item j,:] are vectors with dimension 1 * k. k represents the number of hidden dimensions which is set by the user. In the optimization stage, we are trying to minimize the following cost function with L2 regularization on the parameters userbias, itembias, p and q: <br />
cost function = sum_{x} (rating[user i, item j] - userbias[user i] - itembias[item j] - transpose(p[user i, :]) * q[item j,:])^{2} + lamda_qr_user transpose(p[user i, :]) * p[user i, :] +  lamda_qr_item transpose(q[item j, :]) * p[item j, :] + lamda_user * (userbias[user i])^2 + lamda_item * (itembias[item j])^2
