#Database con 4 tabelle collegate:

#Studenti: dati degli studenti.
#Corsi: elenco dei corsi.
#Esami: risultati degli esami (studente, corso, voto).
#Professori: dati dei professori associati ai corsi.

#parametri studenti = id_studente, nome, cognome, età
#corsi = id_corso , nome_corso, id_professore
#professori = id_professore,nome, cognome
#esami = id_esame, voto,  id_studente , id_corso

#Interfaccia Grafica:

#Visualizzare i dati delle singole tabelle.
#Visualizzare nome e cognome degli studenti, voto di ogni esame,
#corso e cognome del professore che lo insegna

import tkinter as tk
from tkinter import messagebox, simpledialog
import sqlite3
from datetime import date, time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import random
from tkinter import ttk
import pandas as pd

# Funzione per connettersi al database e creare la tabella "rubrica" se non esiste
def init_db():
    conn = sqlite3.connect('ateneo.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS studenti ( 
            id_studente INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            cognome TEXT NOT NULL,
            età INTEGER NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS professori (
            id_professore INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            cognome TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS corsi (
            id_corso INTEGER PRIMARY KEY AUTOINCREMENT,
            nome_corso TEXT NOT NULL,
            id_professore INTEGER NOT NULL,
            CONSTRAINT cor2pro FOREIGN KEY (id_professore) REFERENCES professori(id_professore)                       
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS esami (
            id_studente INTEGER NOT NULL,
            id_corso INTEGER NOT NULL,
            voto INTEGER NOT NULL,
            PRIMARY KEY(id_studente,id_corso),

            CONSTRAINT esa2stu FOREIGN KEY (id_studente) REFERENCES studenti(id_studente),
            CONSTRAINT esa2cor FOREIGN KEY (id_corso) REFERENCES corsi(id_corso)
        )
    ''')
    
init_db()
'''
def populate_db():
    conn = sqlite3.connect('ateneo.db')
    cursor = conn.cursor()

    # Inserimento studenti
    studenti = [
        ('Mario', 'Rossi', 21),
        ('Luca', 'Bianchi', 22),
        ('Giulia', 'Verdi', 20),
        ('Elena', 'Ferrari', 23),
        ('Alessandro', 'Gallo', 24),
        ('Francesca', 'Romano', 22),
        ('Marco', 'Esposito', 21),
        ('Sofia', 'Ricci', 23),
        ('Federico', 'Moretti', 22),
        ('Valentina', 'Conti', 20),
        ('Andrea', 'De Luca', 21),
        ('Martina', 'Greco', 22),
        ('Simone', 'Lombardi', 23),
        ('Chiara', 'Marini', 24),
        ('Giorgio', 'Fabbri', 22)
    ]
    cursor.executemany("INSERT INTO studenti (nome, cognome, età) VALUES (?, ?, ?)", studenti)

    # Inserimento professori
    professori = [
        ('Paolo', 'Rinaldi'),
        ('Laura', 'Vitali'),
        ('Giovanni', 'D\'Angelo'),
        ('Federica', 'Sartori'),
        ('Matteo', 'Fiorini'),
        ('Serena', 'Russo'),
        ('Stefano', 'De Rosa'),
        ('Claudia', 'Leone'),
        ('Alessandro', 'Bruno'),
        ('Livia', 'Colombo')
    ]
    cursor.executemany("INSERT INTO professori (nome, cognome) VALUES (?, ?)", professori)

    # Inserimento corsi
    corsi = [
        ('Matematica', 1),
        ('Fisica', 2),
        ('Chimica', 3),
        ('Informatica', 4),
        ('Biologia', 5),
        ('Letteratura', 6),
        ('Storia', 7),
        ('Filosofia', 8),
        ('Economia', 9),
        ('Giurisprudenza', 10)
    ]
    cursor.executemany("INSERT INTO corsi (nome_corso, id_professore) VALUES (?, ?)", corsi)

    # Inserimento esami
    esami = [
        (1, 1, 28), (1, 2, 30), (1, 3, 25), (1, 4, 27), (1, 5, 24),
        (2, 1, 30), (2, 2, 29), (2, 3, 26), (2, 4, 27), (2, 5, 28),
        (3, 1, 18), (3, 2, 20), (3, 3, 22), (3, 4, 21), (3, 5, 19),
        (4, 1, 30), (4, 2, 29), (4, 3, 28), (4, 4, 27), (4, 5, 30),
        (5, 1, 22), (5, 2, 21), (5, 3, 20), (5, 4, 23), (5, 5, 19),
        (6, 1, 25), (6, 2, 26), (6, 3, 27), (6, 4, 28), (6, 5, 24),
        (7, 1, 19), (7, 2, 20), (7, 3, 21), (7, 4, 22), (7, 5, 23),
        (8, 1, 30), (8, 2, 29), (8, 3, 30), (8, 4, 30), (8, 5, 30),
        (9, 1, 18), (9, 2, 20), (9, 3, 22), (9, 4, 24), (9, 5, 23),
        (10, 1, 28), (10, 2, 27), (10, 3, 26), (10, 4, 25), (10, 5, 30)
    ]
    cursor.executemany("INSERT INTO esami (id_studente, id_corso, voto) VALUES (?, ?, ?)", esami)

    conn.commit()
    conn.close()

populate_db()
'''
def visualizza_dati(tabella):
    conn = sqlite3.connect('ateneo.db')
    cursor = conn.cursor()
    
    query = f"SELECT * FROM {tabella}"
    cursor.execute(query)
    risultati = cursor.fetchall()
    colonne = [desc[0] for desc in cursor.description]  # Ottieni i nomi delle colonne
    conn.close()

    # Creazione della finestra per visualizzare i dati
    finestra = tk.Toplevel()
    finestra.title(f"Dati in {tabella}")
    finestra.geometry("800x400")  # Dimensione iniziale regolabile

    if risultati:
        # Creazione del frame principale
        frame = tk.Frame(finestra)
        frame.pack(expand=True, fill='both')

        # Creazione della Treeview
        tree = ttk.Treeview(frame, columns=colonne, show='headings')

        # Impostazione delle intestazioni e delle colonne
        for col in colonne:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER, width=100)  # Regola la larghezza delle colonne

        # Inserimento dei dati nella Treeview
        for riga in risultati:
            tree.insert("", tk.END, values=riga)

        # Creazione della scrollbar verticale
        scrollbar_vert = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar_vert.set)

        # Posizionamento di Treeview e scrollbar
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_vert.grid(row=0, column=1, sticky="ns")

        # Configurazione del layout
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
    else:
        # Messaggio se non ci sono dati
        label = tk.Label(finestra, text="Nessun dato presente", font=("Arial", 14))
        label.pack(pady=20)


def ricerca_mirata_totale():
    conn = sqlite3.connect('ateneo.db')
    cursor = conn.cursor()
    query = '''
        SELECT stu.nome, stu.cognome , esa.voto, cor.nome_corso, pro.cognome as docente
        FROM studenti stu
        JOIN (esami esa 
        JOIN (corsi cor
        JOIN professori pro
        ON   cor.id_professore = pro.id_professore)
        ON   esa.id_corso = cor.id_corso)
        ON   stu.id_studente = esa.id_studente
            '''
    cursor.execute(query)
    risultati= cursor.fetchall()
    colonne = [desc[0] for desc in cursor.description]  # Ottieni i nomi delle colonne

    '''
    if risultati:
        risultato = "\n".join( f"Studente: {riga[0]} {riga[1]} - Voto: {riga[2]} - Corso:  {riga[3]} - Prof.: {riga[4]} "  for riga in risultati)
        messagebox.showinfo(f"Ricerca Totale", risultato)
    else:
        messagebox.showinfo("Ricerca Totale ", "nessun dato presente")
 '''
    conn.close()
        # Creazione della finestra per visualizzare i dati
    finestra = tk.Toplevel()
    finestra.title(f"Esami registrati per ogni studente")
    finestra.geometry("800x400")  # Dimensione iniziale regolabile

    if risultati:
        # Creazione del frame principale
        frame = tk.Frame(finestra)
        frame.pack(expand=True, fill='both')

        # Creazione della Treeview
        tree = ttk.Treeview(frame, columns=colonne, show='headings')

        # Impostazione delle intestazioni e delle colonne
        for col in colonne:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER, width=100)  # Regola la larghezza delle colonne

        # Inserimento dei dati nella Treeview
        for riga in risultati:
            tree.insert("", tk.END, values=riga)

        # Creazione della scrollbar verticale
        scrollbar_vert = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar_vert.set)

        # Posizionamento di Treeview e scrollbar
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_vert.grid(row=0, column=1, sticky="ns")

        # Configurazione del layout
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
    else:
        # Messaggio se non ci sono dati
        label = tk.Label(finestra, text="Nessun dato presente", font=("Arial", 14))
        label.pack(pady=20)

# Connessione al database
conn = sqlite3.connect('ateneo.db')
query = '''
    SELECT 
        stu.id_studente, 
        stu.età, 
        AVG(esa.voto) AS media_voti, 
        COUNT(esa.id_corso) AS numero_esami
    FROM 
        studenti stu
    LEFT JOIN 
        esami esa ON stu.id_studente = esa.id_studente
    GROUP BY 
        stu.id_studente, stu.età;
'''
# Caricare i dati in un DataFrame
data = pd.read_sql_query(query, conn)
conn.close()

# Gestione dei NaN (nel caso in cui uno studente non abbia sostenuto esami)
data.fillna(0, inplace=True)

from sklearn.preprocessing import StandardScaler

# Selezionare le features
features = data[['età', 'media_voti', 'numero_esami']]

# Normalizzare i dati
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


from sklearn.cluster import KMeans

# Definire il numero di cluster
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_features)

# Visualizzare i gruppi
print(data)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Ridurre le dimensioni per la visualizzazione
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

# Grafico 2D
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=data['cluster'], cmap='viridis')
plt.title("Clustering degli studenti")
plt.xlabel("Componente Principale 1")
plt.ylabel("Componente Principale 2")
plt.colorbar(label='Cluster')
plt.show()    

import sqlite3
import pandas as pd

# Connessione al database
conn = sqlite3.connect('ateneo.db')

# Query per estrarre dati
query = '''
    SELECT 
        stu.id_studente,
        stu.età,
        COUNT(esa.id_corso) AS num_corsi,
        GROUP_CONCAT(DISTINCT c.id_professore) AS professori,
        GROUP_CONCAT(DISTINCT c.id_corso) AS corsi,
        AVG(esa.voto) AS rendimento_medio
    FROM 
        studenti stu
    LEFT JOIN 
        esami esa ON stu.id_studente = esa.id_studente
    LEFT JOIN 
        corsi c ON esa.id_corso = c.id_corso
    GROUP BY 
        stu.id_studente, stu.età;
'''

# Caricare i dati in un DataFrame
data = pd.read_sql_query(query, conn)
conn.close()

# Encoding delle variabili categoriali (professori e corsi)
data = data.explode('professori')  # Separare i professori in righe
data = pd.get_dummies(data, columns=['professori'], prefix='prof')  # One-Hot Encoding professori

data = data.explode('corsi')  # Separare i corsi in righe
data = pd.get_dummies(data, columns=['corsi'], prefix='corso')  # One-Hot Encoding corsi

# Rimuovere eventuali NaN
data.fillna(0, inplace=True)

# Separare feature e target
features = data.drop(['id_studente', 'rendimento_medio'], axis=1)
target = data['rendimento_medio']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import numpy as np

# Suddivisione dei dati in training e test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Creazione e addestramento del modello
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predizioni sul set di test
y_pred = model.predict(X_test)

# Valutazione del modello
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

importances = model.feature_importances_
feature_names = features.columns
sorted_indices = importances.argsort()

plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
plt.title('Importanza delle Feature')
plt.show()

# Visualizzazione della media predetta per ciascuno studente nel set di test
predicted_data = pd.DataFrame({
    'ID Studente': X_test.index,  # Presupponendo che gli ID siano gli indici originali
    'Media Predetta': y_pred,
    'Media Reale': y_test
})
print(predicted_data.head(15))  # Mostra i primi risultati



root = tk.Tk()
root.title("ATENEO Potentissimo")
root.geometry("800x600")
menubarra = tk.Menu(root)
root.config(menu=menubarra)

def mostra_stu():
    visualizza_dati("studenti")
    '''
def inserisci_studente():
    inserisci_dati("studenti",["nome", "cognome", "età"])
    '''
def mostra_pro():
    visualizza_dati("professori")
def visualizza_esami():
    visualizza_dati("esami")
def visualizza_corsi():
    visualizza_dati("corsi")
def mostra_lib():
    ricerca_mirata_totale()
def inserisci_corsi():
    inserisci_dati("corsi",["nome_corso", "id_professore"])
def inserisci_esami():
    inserisci_dati("esami",["id_studente", "id_corso", "voto"])



#commentonissimo: da una bellissima e potentissima query con join estraggo i dati per addestrare 
#una rete neurale Feed Forward Neural Network FNN a predire il voto di un esame data l'età

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
'''
def estrai_training():
    conn = sqlite3.connect('ateneo.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT esami.id_corso,studenti.età,esami.voto
        FROM esami JOIN studenti
        ON esami.id_studente = studenti.id_studente    
    """)
    dati = cursor.fetchall()
    conn.close()
    x = np.array([[riga[0],riga[1]] for riga in dati])
    y = np.array([riga[2]for riga in dati])
    return x,y

def modello_train(x,y):
    xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size= 0.2, random_state= 3)
    modello = Sequential([
        Dense(10,activation = 'relu', input_shape = (x.shape[1], )),
        Dense(10,activation = 'relu'),
        Dense(1)    
    ])
    modello.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
    modello.fit(xtrain,ytrain, epochs = 100, batch_size = 10, verbose = 1)
    loss,mae = modello.evaluate(xtest,ytest, verbose = 0)
    print(f"MSE: {loss:.2f}")
    print(f"MAE: {mae:.2f}")    
    return modello

if __name__ == "__main__":
    x,y = estrai_training()
    modello = modello_train(x,y)
    modello.save("voto_predetto.h5")
'''
# Funzione per addestrare il modello di predizione
def train_model():
    conn = sqlite3.connect('ateneo.db')
    cursor = conn.cursor()
    
    query = '''
        SELECT stu.età, AVG(esa.voto) as voto_medio
        FROM studenti stu
        JOIN esami esa ON stu.id_studente = esa.id_studente
        GROUP BY stu.età
    '''
    cursor.execute(query)
    dati = cursor.fetchall()
    conn.close()
    
    if len(dati) > 1:  # Serve almeno una relazione tra età e voto medio
        X = np.array([d[0] for d in dati]).reshape(-1, 1)  # Età
        y = np.array([d[1] for d in dati])  # Voto medio
        
        modello = LinearRegression()
        modello.fit(X, y)
        return modello
    else:
        return None

# Funzione per predire il voto basato sull'età
def predici_voto(eta):
    modello = train_model()
    if modello:
        voto_previsto = modello.predict(np.array([[eta]]))[0]
        return round(voto_previsto, 2)
    else:
        return "Non abbastanza dati per la predizione"

# Funzione per inserire un nuovo studente e mostrare la predizione
def inserisci_studente():
    nome = simpledialog.askstring("Inserisci Studente", "Nome:")
    cognome = simpledialog.askstring("Inserisci Studente", "Cognome:")
    eta = simpledialog.askinteger("Inserisci Studente", "Età:")
    
    if nome and cognome and eta:
        conn = sqlite3.connect('ateneo.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO studenti (nome, cognome, età) VALUES (?, ?, ?)", (nome, cognome, eta))
        conn.commit()
        conn.close()
        
        voto_previsto = predici_voto(eta)
        messagebox.showinfo("Predizione", f"Studente aggiunto!\nMedia voti prevista per età {eta}: {voto_previsto}")
    else:
        messagebox.showerror("Errore", "Dati non validi!")

def visualizza_esami():
    conn = sqlite3.connect('ateneo.db')
    cursor = conn.cursor()
    query = '''
        SELECT stu.nome, stu.cognome ,cor.nome_corso as esame ,esa.voto
        FROM studenti stu
        JOIN (esami esa
        JOIN corsi cor
        ON   esa.id_corso = cor.id_corso)
        ON   stu.id_studente = esa.id_studente
            '''
    cursor.execute(query)
    risultati= cursor.fetchall()
    colonne = [desc[0] for desc in cursor.description]  # Ottieni i nomi delle colonne

    conn.close()
        # Creazione della finestra per visualizzare i dati
    finestra = tk.Toplevel()
    finestra.title(f"Voti ed esami per ogni studente")
    finestra.geometry("800x400")  # Dimensione iniziale regolabile

    if risultati:
        # Creazione del frame principale
        frame = tk.Frame(finestra)
        frame.pack(expand=True, fill='both')

        # Creazione della Treeview
        tree = ttk.Treeview(frame, columns=colonne, show='headings')

        # Impostazione delle intestazioni e delle colonne
        for col in colonne:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER, width=100)  # Regola la larghezza delle colonne

        # Inserimento dei dati nella Treeview
        for riga in risultati:
            tree.insert("", tk.END, values=riga)

        # Creazione della scrollbar verticale
        scrollbar_vert = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar_vert.set)

        # Posizionamento di Treeview e scrollbar
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_vert.grid(row=0, column=1, sticky="ns")

        # Configurazione del layout
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
    else:
        # Messaggio se non ci sono dati
        label = tk.Label(finestra, text="Nessun dato presente", font=("Arial", 14))
        label.pack(pady=20)


def visualizza_corsi():
    conn = sqlite3.connect('ateneo.db')
    cursor = conn.cursor()
    query = '''
        SELECT cor.id_corso ,cor.nome_corso, pro.cognome as docente
        FROM  corsi cor
        JOIN professori pro
        ON   cor.id_professore = pro.id_professore
'''
    cursor.execute(query)
    risultati= cursor.fetchall()
    colonne = [desc[0] for desc in cursor.description]  # Ottieni i nomi delle colonne

    conn.close()
        # Creazione della finestra per visualizzare i dati
    finestra = tk.Toplevel()
    finestra.title(f"Elenco corsi")
    finestra.geometry("800x400")  # Dimensione iniziale regolabile

    if risultati:
        # Creazione del frame principale
        frame = tk.Frame(finestra)
        frame.pack(expand=True, fill='both')

        # Creazione della Treeview
        tree = ttk.Treeview(frame, columns=colonne, show='headings')

        # Impostazione delle intestazioni e delle colonne
        for col in colonne:
            tree.heading(col, text=col)
            tree.column(col, anchor=tk.CENTER, width=100)  # Regola la larghezza delle colonne

        # Inserimento dei dati nella Treeview
        for riga in risultati:
            tree.insert("", tk.END, values=riga)

        # Creazione della scrollbar verticale
        scrollbar_vert = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar_vert.set)

        # Posizionamento di Treeview e scrollbar
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_vert.grid(row=0, column=1, sticky="ns")

        # Configurazione del layout
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
    else:
        # Messaggio se non ci sono dati
        label = tk.Label(finestra, text="Nessun dato presente", font=("Arial", 14))
        label.pack(pady=20)

def inserisci_dati(tabella,attributi):
    valori = []
    for attributo in attributi:
        valore = simpledialog.askstring("INPUT",f"INSERISCI {attributo}:")
        if valore is None:
            return 
        valori.append(valore)
 
    conn = sqlite3.connect('ateneo.db')
    cursor = conn.cursor()
    placeholders = ', '.join(['?'] *len(attributi))#vado a creare una stringa che mi dava a sostiruire il valore del punto interrogativo
    query = f"INSERT INTO {tabella} ({', '.join(attributi)}) VALUES ({placeholders})"
    try:
        cursor.execute(query, valori)
        conn.commit()
        messagebox.showinfo("successo", f"Studente inserito correttamente nella {tabella}")
    except sqlite3.Error as e:
        messagebox.showerror("Errore" , f"Errore durante l' inserimento {e}")
    conn.close()
# Aggiunta del pulsante per inserire studenti
btn_inserisci_studente = tk.Button(root, text="Inserisci Studente", command=inserisci_studente)
btn_inserisci_studente.pack(pady=10)

# Aggiunta del pulsante per visualizzare gli esami
btn_visualizza_esami = tk.Button(root, text="Visualizza Esami", command=ricerca_mirata_totale)
btn_visualizza_esami.pack(pady=10)

menu_ins = tk.Menu(menubarra, tearoff=0)
#menu_ins.add_command(label="studente", command = inserisci_studente)
#menubarra.add_cascade(label= "Inserisci", menu = menu_ins)


menu_vis = tk.Menu(menubarra, tearoff=0)
menu_vis.add_command(label="Elenco studenti", command = mostra_stu)
menu_vis.add_command(label="Elenco professori", command = mostra_pro)
menu_vis.add_command(label="Elenco corsi", command = visualizza_corsi)
menu_vis.add_command(label="Elenco esami", command = visualizza_esami)
menu_vis.add_command(label="Libretti per studente", command = mostra_lib)
menubarra.add_cascade(label= "Visualizza", menu = menu_vis)

menu_ins = tk.Menu(menubarra, tearoff=0)
menu_ins.add_command(label="esame", command = inserisci_esami)
menu_ins.add_command(label="corso", command = inserisci_corsi)
menubarra.add_cascade(label= "Inserisci", menu = menu_ins)

root.mainloop()


