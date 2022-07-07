def naivemodel(fichier):
    '''
    Simulation des résultats données par l'outil SAM "Synthesis Analysis Motor" de MyTeam qui reçoit en entrée un document et renvoie sa catégorie.  
    '''

    innovationp = "Très probablement innovation "
    innovation =  "Innovation mais pourrait passer en R&D "
    rd = "Probablement R&D "
    rdp = "Très probablement R&D "

    with open(fichier,'r') as f:
        content = f.readline()
    
    if content[0] == 'a':
        return rdp
    elif content[0] == 'b':
        return rd  
    elif content[0] == 'c':
        return innovation
    else:
        return innovationp

