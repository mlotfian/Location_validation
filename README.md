# Location validation API

This API aims at performing a real-time validation of location of bird species in Switzerland. It takes the species name, and the location of observation, and creates environmental variables in an area of 1km2 around the observation point. Then it uses trained species distribution models to predict the probability of occurance of the species. 

## how to run

python api_RF.py

