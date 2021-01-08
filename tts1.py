# - coding: utf-8 --
from aip import AipSpeech
import csv

APP_ID = '###'
API_KEY = '###'
SECRET_KEY = '###'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

def createAudioFile(text, name):
    result  = client.synthesis(text, 'zh', 1, {
        'vol': 5,
    })
    if not isinstance(result, dict):
        with open(name, 'wb') as f:
            f.write(result)



csvFileName = "EnterCSVFileNameHere"
newRowsList = []
with open ((csvFileName+".csv"), 'rb') as csvfile:
    charReader = csv.reader(csvfile, delimiter=',')
    for inter, row in enumerate(charReader):
        if not row[0]:
            break
        audioName = str(inter) + "-" + csvFileName + ".mp3"
        ankiSoundField = "[sound:" + audioName + "]"
        newRow = [row[0], row[1], row[2], row[3], row[4], ankiSoundField]
        createAudioFile(row[0], audioName)
        print ', '.join(newRow)
        newRowsList.append(newRow)
            #synthesize word to audio file
            
    
with open ((csvFileName+"Write"+".csv"), 'wb') as csvfile:
    charWriter = csv.writer(csvfile, delimiter=',')
    charWriter.writerows(newRowsList)

    
        
        
