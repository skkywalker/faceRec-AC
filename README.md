# face_recognition
Learning how to use face_recognition and dlib python libraries.

**Software menu:**

```
1. Listar Pessoas
2. Registrar
3. Start
4. Deletar
5. Salvar e Sair
0. Dev Tools

DEV TOOLS
1. Toggle Debugging
2. Set Confianca
3. Resetar log
4. Resetar todos os arquivos
5. Voltar para menu
```

This script is capable of recognizing, at real time, people's faces and who they belong to. It is also possible to register new people pressing `SPACE` - while on program 2. - however many times you wish to train a picture (the more pictures, the more accurate the system will be, but also slower), then pressing `ESC` and naming the person. You can also delete previously registered people and change the threshold to associate a face with someone (default is .4)

Whenever unknown faces are present in the frame, a counter will increase and, upon reaching an user defined value (default is ~6 seconds), will take a snapshot of the frame every ~6 seconds and save it to safe/{datetime}.png

The system also mantains a log file, which registers new and deleted users, as well as when snapshots were taken and when some known face entered or exited the frame

### Requirements
[face_recognition](https://github.com/ageitgey/face_recognition) (requires dlib)

[openCV](https://opencv.org/)

Don't know if it works on windows! (not officially supported by face_recognition)

### Instructions

Just download the .py file and run it in an environment with the required libraries and a folder named "safe" at the same location as the script, to store snapshots.

Three files will be created, if non-existant, named nomes.txt - which stores every registered name in binary form - log.txt - which can be opened with a text editor to be read - and encodings.npy - also stored in binary, this is the array with every previously trained face.

**Remember to always exit the program pressing 5. and ENTER, otherwise it will not save any file changes**

### Tested on

|   |   |
|---|---|
| `Ubuntu 18.04`          ||
| `face_recognition 1.2.2`||
| `dlib 19.15.0`          |`works!`|
| `openCV 3.4.2`          ||
| `Python 3.6.5`          ||
