import face_recognition
import cv2
import numpy as np
import datetime
import pickle
from pathlib import Path


START_LOG_STR = str(datetime.datetime.now()) + ' - ' + 'INICIO DO LOG\n'
LOG_MSG = ''
DEBUG = False
CONF = 0.4


# Abre os arquivos que contem os nomes e as definicoes do rosto

# ENCODINGS
if Path('encodings.npy').is_file():
    _encodings = np.load('encodings.npy')
else:
    _encodings = np.zeros([1,128])
    np.save('encodings', _encodings)

# NOMES
if Path('nomes.txt').is_file():
    with open('nomes.txt', 'rb') as fp:
        _nomes = pickle.load(fp)
else:
    _nomes = []
    with open('nomes.txt', 'wb') as fp:
        pickle.dump(_nomes, fp)

# LOG
if Path('log.txt').is_file():
    _log = open('log.txt', 'a')
else:
    _log = open('log.txt', 'a')
    _log.write(START_LOG_STR)


# Loop Principal
while(True):
    print(chr(27) + '[2J')
    print('RECONHECIMENTO DE FACES - NEPAS (EESC-USP)')
    print('1. Listar Pessoas')
    print('2. Registrar')
    print('3. Start')
    print('4. Deletar')
    print('5. Salvar e Sair')
    print('0. Dev Tools')

    escolha_menu = int(input('Escolha uma opcao do menu: '))

    # Le o set da lista que contem os nomes e retorna para o usuario
    if (escolha_menu == 1):
        if len(_nomes) == 0:
            print('Nao ha pessoas cadastradas')

        else:
            print(set(_nomes))

            # Imprime todo o vetor _nomes, ou seja, as pessoas que tem mais de uma foto aparecem varias vezes
            if DEBUG:
                print('Vetor com todos os nomes: ', _nomes)

        input('Pressione ENTER para continuar...')

    # Abre a camera(0) e permite a captura de imagens, treina a rede e coloca o vetor na array _encodings
    elif escolha_menu == 2:

        camera = cv2.VideoCapture(0)

        valido = 0  # Checa se ha algum rosto na foto
        img_counter = 0 # Conta quantas fotos foram tiradas na sessao
        sample_face_encoding = np.zeros([1,128])

        while (camera.isOpened()):
            ret, frame = camera.read()
            cv2.imshow('frame', frame)
            if not ret:
                break
            
            k = cv2.waitKey(1)

            # Teclou ESC para sair da captura de imagem
            if k%256 == 27:
                print('Saindo...')
                break

            # Teclou SPACE para adicionar o rosto ao vetor _encodings
            elif k%256 == 32:
                img_name = 'img.png'
                cv2.imwrite(img_name, frame) # Salva a foto
                sample_image = face_recognition.load_image_file(img_name)

                # Se nao reconhece nenhum rosto na foto, nao faz nada
                if not face_recognition.face_encodings(sample_image):
                    print('Nenhum rosto encontrado!')

                # Se existe algum rosto, ele passa a imagem pela rede, e salva na variavel _encodings o array resposta
                else:
                    valido = 1
                    img_counter += 1
                    print('Capturado')
                    encoding_imagem = face_recognition.face_encodings(sample_image)[0]
                    sample_face_encoding = np.vstack((sample_face_encoding, encoding_imagem))

        camera.release()
        cv2.destroyAllWindows()

        if (_encodings == np.zeros([1,128])).all():
            primeiro = 1
        else:
            primeiro = 0

        if valido:
            # Salva, na mesma posicao do array _encodings, o nome da pessoa no array _nomes
            sample_name = input('Qual e o nome da pessoa na(s) foto(s)? ')

            # Acrescenta no vetor _nomes o numero de fotos (encodings)
            for i in range(img_counter):
                _nomes.append(sample_name)

            # Se for o primeiro encoding, substitui a linha, senao acresenta
            if primeiro:
                sample_face_encoding = np.delete(sample_face_encoding, (0), axis=0)
                _encodings = np.vstack((_encodings, sample_face_encoding))
                _encodings = np.delete(_encodings, (0), axis=0)
            else:
                sample_face_encoding = np.delete(sample_face_encoding, (0), axis=0)
                _encodings = np.vstack((_encodings, sample_face_encoding))

            # Salva no log o novo usuario
            LOG_MSG = '\n\nNOVO USUARIO\n' \
                      + str(datetime.datetime.now()) + ' '\
                      + sample_name \
                      + ' adicionado com ' \
                      + str(img_counter) \
                      + ' fotos'
            _log.write(LOG_MSG)

            print('SUCESSO!')
        else:
            print('ERROR!')
            
        input('Pressione ENTER para continuar...')
        

    # Roda o programa principal, praticamente igual ao que tem na documentacao
    elif escolha_menu == 3:
        LOG_MSG = '\n\n########## Iniciado Monitoramento ' \
                      + str(datetime.datetime.now())
        _log.write(LOG_MSG)

        video_capture = cv2.VideoCapture(0)

        face_locations = []                     # Contem a localizacao de todas as faces do frame
        face_encodings = []                     # Contem o encoding de todas as faces do frame
        face_names = []                         # Contem os nomes ...
        process_this_frame = True               # Para alternar e processar frame sim, frame nao - para otimizacao

        contador_desconhecido = 0                                   # Contador para marcar o tempo que um desconhecido esta na imagem para uma SNAPSHOT
        list_set__nomes = list(set(_nomes)) + ['Desconhecido']      # Lista com todos os nomes aparecendo uma unica vez
        status = np.zeros([len(list_set__nomes)])                   # Variavel que varia de 0~100 mostrando o indice de confianca que a pessoa esta realmente ali
        status_TF = status.copy()                                   # Variavel que contem True/False para cara nome, analisando o frame e frames anteriores
        last_status_TF = status_TF.copy()                           # Equivale ao status_TF da ultima iteracao
        range_len_status = range(len(status))                       # Usado em alguns loops, aqui calcula-se uma unica vez para otimizacao

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=2)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                qtd_d = 0      # Quantidade de desconhecidos no frame

                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                        # codigo antigo: matches = face_recognition.compare_faces(_encodings, face_encoding, tolerance = 0.5)
                    distance = face_recognition.face_distance(_encodings, face_encoding)

                    '''
                    for i in range(len(_encodings)):
                        print(_nomes[i], ':', distance[i])
                    '''

                    if DEBUG:
                        print(distance)

                    menor = CONF # Confianca
                    name = 'Desconhecido'

                    menor_i = 0
                    for i in range(len(distance)):
                        if distance[i] <= menor:
                            menor = distance[i]
                            name = _nomes[i]
                            menor_i = i

                    if DEBUG:
                        print('Menor distancia:', menor)
                        print('Indice:', menor_i)

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Diminui a confianca de todos os nomes em 3 pts
            for i in range_len_status:
                if status[i] > 2:
                    status[i] -= 2

            # Das faces que estao presentes, aumenta 6 pts de confianca (resultando em +3 pts por iteracao nas presentes)
            for i in face_names:
                match_index = list_set__nomes.index(i)
                if status[match_index] < 96:
                    status[match_index] += 5

            # Analiza se status_TF para cada pessoa True/False
            for i in range_len_status:
                if (status[i] > 40 and status[i] < 60): # De 40~60 pts, mantem a ultima decisao
                    status_TF[i] = last_status_TF[i]
                elif status[i] >= 60:
                    status_TF[i] = True
                else:
                    status_TF[i] = False
            
            # Para cada pessoa registrada (ou desconhecido), registra no log se entrou ou saiu
            for i in range_len_status:
                if status_TF[i] != last_status_TF[i]:
                    if status_TF[i]:
                        LOG_MSG = '\n' + str(datetime.datetime.now()) + ' entrou ' + list_set__nomes[i]
                        _log.write(LOG_MSG)
                    else:
                        LOG_MSG = '\n' + str(datetime.datetime.now()) + ' saiu ' + list_set__nomes[i]
                        _log.write(LOG_MSG)

            # Para cada frame, analisa se existe um desconhecido. Se sim, aumenta o contador, se nao zera
            # Depois salva uma imagem com a hora, caso o contador chegue a 100
            #index_desconhecido = list_set__nomes.index('Desconhecido')

            if 'Desconhecido' in face_names:
                contador_desconhecido += 1
            elif contador_desconhecido > 0:
                contador_desconhecido -= 1

            if contador_desconhecido == 100:
                img_name = 'safe/{}.png'.format(str(datetime.datetime.now()))
                cv2.imwrite(img_name, frame) # Salva a foto
                contador_desconhecido = 0

                LOG_MSG = '\n' + str(datetime.datetime.now()) + ' SNAPSHOT com {} DESCONHECIDO(S)!'.format(face_names.count('Desconhecido'))
                _log.write(LOG_MSG)

            # Atualiza o ultimo status para comparacao
            last_status_TF = status_TF.copy()

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

        LOG_MSG = '\n########## Monitoramento Encerrado ' \
                      + str(datetime.datetime.now())
        _log.write(LOG_MSG)

        input('Pressione ENTER para continuar...')

    # Permite tirar algum nome do vetor de nomes
    elif escolha_menu == 4:
        perde_acesso = input('Nome a ser deletado: ')

        # Monta um array do tipo [n, n-x, ...] com todos os index que contem o nome a ser excluido
        indices = [i for i, x in enumerate(_nomes) if x == perde_acesso]
        indices = list(reversed(indices)) #inverte o array

        if not indices:
            print('Nome nao consta nos registros')
        else:
            print('Nome encontrado!')
            print('Deletando...')
            for j in indices:
                del _nomes[j]
                _encodings = np.delete(_encodings, (j), axis = 0)

            # Escreve nos registros
            LOG_MSG = '\n\nUSUARIO DELETADO\n' \
                  + str(datetime.datetime.now()) + ' '\
                  + perde_acesso \
                  + ' excluido dos registros'
            _log.write(LOG_MSG)

        input('Pressione ENTER para continuar...')

    # Entra no DEV TOOLS
    elif escolha_menu == 0:
        print(chr(27) + '[2J')
        print('DEV TOOLS')
        print('1. Toggle Debugging')
        print('2. Set Confianca')
        print('3. Resetar log')
        print('4. Resetar todos os arquivos')
        print('5. Voltar para menu')

        escolha_menu2 = int(input('Entre com a opcao: '))

        # Ativa ou desativa o modo de debug
        if escolha_menu2 == 1:
            DEBUG = not DEBUG
            print('Modo Debug:' , DEBUG)
            input('Pressione ENTER para continuar...')

        # Set um novo indice de confianca
        elif escolha_menu2 == 2:
            print('Confianca atual:', conf)
            CONF = float(input('Nova confianca: '))
            print('SUCESSO!')
            input('Pressione ENTER para continuar...')

        # Reseta o arquivo LOG
        elif escolha_menu2 == 3:
            print('Resetando o log')
            _log.close()
            _log = open('log.txt', 'w')
            _log.write(START_LOG_STR)
            _log.write('\n\nNomes registrados: ')
            _log.write(str(list(set(_nomes))))

            print('SUCESSO!')
            input('Aperte ENTER para voltar para o programa...')

        # Reseta os arquivos encodings.npy, nomes.txt e log.txt
        elif escolha_menu2 == 4:
            print('Apagando todos os arquivos...')

            _encodings = np.zeros([1,128])
            np.save('encodings', _encodings)

            _nomes = []
            with open('nomes.txt', 'wb') as fp:
                pickle.dump(_nomes, fp)

            _log.close()
            _log = open('log.txt', 'w')
            _log.write(START_LOG_STR)        

            print('SUCESSO!')
            input('Aperte ENTER para voltar para o programa...')

        else:
            print('Voltando para o programa principal...')

    # Salva os vetores em seus respectivos arquivos e sai do programa
    else:
        np.save('encodings', _encodings)
        with open('nomes.txt', 'wb') as fp:
            pickle.dump(_nomes, fp)
        _log.close()
        break