# Jue-Mo Thermal-Treatment-Programm

## 1. Data cleaning
Run `$python3 clean_data.py` command after changing the experiment name to choose the correct data file.

## 2. Model training 
Run `python3 train.py` to train a model. The code will output a `.pt` file like `Oak_model.pt` and will used for inference.

## 3. Inference and Plotting
Run `python3 inference.py` to generate R plots. Change plot colour, box colour, legend, etc. from here.

## 4.To open the writting pool which is TERMINAL
control + `

## 5.How to write infe
Run `python3 inference_time.py`
And the fig is in oak_E_orig

## 6.Train the model
Run `python3 train.py` & `network.py` modify the model:
>"self.linear1 = torch.nn.Linear(5, 12)" 

**`5`** here means the input *parameter (T,t,l,a,b before)*; **`12`** here means how many *hidden layers* we have in the model

>"self.linear2 = torch.nn.Linear(12, 1)"

**`12`** here: still *hidden layer number*
**`1`** mean *the output number*

# GitHub
Sync local repository with web: `git pull`   
To check the changes: `git status`   
Add the commits： `git add .` (. means all)   
Commit tag: `git commit -m "<put the tag here>"`   
push to Gihub: `git push`
