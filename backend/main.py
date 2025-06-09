import string
import os
import urllib
import uuid
import pickle
import datetime
import time
import shutil

import cv2
from fastapi import FastAPI
from fastapi import File, UploadFile, Form, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import starlette

ATTENDANCE_LOG_DIR = './logs'
DB_PATH = './db'

# Créer les répertoires si non existants
for dir_ in [ATTENDANCE_LOG_DIR, DB_PATH]:
    if not os.path.exists(dir_):
        os.mkdir(dir_)

app = FastAPI()

origins = ["*"]

# Ajouter la gestion CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route d'accueil pour éviter l'erreur 404
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Face Attendance App!"}

# Route de login
@app.post("/login")
async def login(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.png"
    contents = await file.read()

    # Sauvegarder le fichier
    with open(file.filename, "wb") as f:
        f.write(contents)

    user_name, match_status = recognize(cv2.imread(file.filename))

    if match_status:
        epoch_time = time.time()
        date = time.strftime('%Y%m%d', time.localtime(epoch_time))
        with open(os.path.join(ATTENDANCE_LOG_DIR, f'{date}.csv'), 'a') as f:
            f.write(f'{user_name},{datetime.datetime.now()},IN\n')

    return {'user': user_name, 'match_status': match_status}

# Route de logout
@app.post("/logout")
async def logout(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.png"
    contents = await file.read()

    # Sauvegarder le fichier
    with open(file.filename, "wb") as f:
        f.write(contents)

    user_name, match_status = recognize(cv2.imread(file.filename))

    if match_status:
        epoch_time = time.time()
        date = time.strftime('%Y%m%d', time.localtime(epoch_time))
        with open(os.path.join(ATTENDANCE_LOG_DIR, f'{date}.csv'), 'a') as f:
            f.write(f'{user_name},{datetime.datetime.now()},OUT\n')

    return {'user': user_name, 'match_status': match_status}

# Route pour l'enregistrement d'un nouvel utilisateur
@app.post("/register_new_user")
async def register_new_user(file: UploadFile = File(...), text: str = Form(...)):
    file.filename = f"{uuid.uuid4()}.png"
    contents = await file.read()

    # Sauvegarder le fichier
    with open(file.filename, "wb") as f:
        f.write(contents)

    # Copier dans la base de données
    shutil.copy(file.filename, os.path.join(DB_PATH, f'{text}.png'))

    # Extraire les embeddings
    embeddings = face_recognition.face_encodings(cv2.imread(file.filename))

    # Sauvegarder les embeddings dans un fichier pickle
    with open(os.path.join(DB_PATH, f'{text}.pickle'), 'wb') as file_:
        pickle.dump(embeddings, file_)

    os.remove(file.filename)

    return {'registration_status': 200}

# Route pour récupérer les logs d'assiduité
@app.get("/get_attendance_logs")
async def get_attendance_logs():
    filename = 'out.zip'
    shutil.make_archive(filename[:-4], 'zip', ATTENDANCE_LOG_DIR)
    return starlette.responses.FileResponse(filename, media_type='application/zip', filename=filename)

# Fonction de reconnaissance faciale
def recognize(img):
    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found', False
    else:
        embeddings_unknown = embeddings_unknown[0]

    match = False
    db_dir = sorted([j for j in os.listdir(DB_PATH) if j.endswith('.pickle')])

    j = 0
    while not match and j < len(db_dir):
        path_ = os.path.join(DB_PATH, db_dir[j])

        with open(path_, 'rb') as file:
            embeddings = pickle.load(file)[0]

        match = face_recognition.compare_faces([embeddings], embeddings_unknown)[0]
        j += 1

    if match:
        return db_dir[j - 1][:-7], True
    else:
        return 'unknown_person', False
