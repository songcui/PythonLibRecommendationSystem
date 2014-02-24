PythonLibRecommendationSystem
=============================

This libraries contains various standard algorithms for recommendation system in python. 
It assumes the training dataset is stored in a txt file with the following format:

userID <tab> ItemID <tab> Rating <br />
5             19            4    <br /> 
4             6             3    <br />
...... <br />

1. method "qr" in qrlatentfactor.py builds up and optimize the qr factorization model 
where rating[user i, item j] = transpose(p[user i, :])*q[item j,:]. p[user i,:] is a vector of 1*k
