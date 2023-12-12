Please install all needed dependencies using the following pip statement: 
    pip install flask scikit-learn numpy joblib nltk deeplake

Ensure all of the following files are located in the same directory:
app.py, dataLoader.py, lieDetector.py, trained-ModelBin.joblib, trained-ModelSix.joblib, vectorizerBin.joblib, vectorizerSix.joblib

Also ensure that the 'templates' and 'static' folders are present in this same directory as well. 

If any or all of the joblib files are missing, the program will create them from scratch. This process may take up to 5 minutes and requires an internet connection.
It will be announced in the console if this process is occuring. 
As long as the files are present when the program starts it will use them and this is much much faster of an outcome. 

This is a flask project, in order to run it:
1. cd to the project directory from terminal or cmd
2. Run the following command using the enviroment you loaded the dependencies into: "flask --app app --debug  run"
3. It will either go through the model building process if the model and vectorizer files are missing, or it will start the flask server.
4. To access the user interface, open a web browser and enter "127.0.0.1:5000" into your URL field. 
5. You should now be able to enter in text fields and preform a prediction. 

An exmaple of input parameters is as follows:
Statement: There are already more American jobs in the solar industry than in coal mining.
subject(Whats it about? (few words)): climate-change,energy,environment,jobs
speaker(Name): sheldon-whitehouse
job title: U.S. Senator
state: Rhode Island
party: democrat
context(Where did it come from?): a newspaper commentary

This should check out as a true statement. 

https://app.activeloop.ai/activeloop/liar-test