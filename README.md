# Installation

Model training was based on superb simple Q-learning algorithm


https://github.com/baybaraandrey/tic-tac-toe/blob/main/tic-tac-toe3.gif

## Windows Users
```powershell
python -m venv env
.\env\Scripts\activate
pip instal -r requirements.txt

python .\main.py --load model.pkl # using already trained model
python .\main.py --save your-model.pkl # or you can train and save your model on your own
```

## Linux Users
```sh
python -m venv env
source env/bin/activate
pip instal -r requirements.txt

python main.py --load model.pkl # using already trained model
python main.py --save your-model.pkl # or you can train and save your model on your own
```

### Try to win AI)
