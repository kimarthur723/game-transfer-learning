CALL "G:\Desktop\game\venv\Scripts\activate"
cd "G:\Desktop\game"
set FLASK_APP=server.py
start chrome http://localhost:5000/game
flask run