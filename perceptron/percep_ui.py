import os
import csv
import numpy as np
import re
import tkinter as tk
from tkinter import ttk, messagebox

# --- CONFIGURACIÓN DE PALABRAS CLAVE ---
URGENT_WORDS = ["urgente", "ya", "hoy", "ahora", "expira", "limitada", "gratis", "gana", "ganaste", "garantizado", "secreto", "único", "clic", "click", "reclama", "actúa", "perderás", "aproveche"]
MONEY_WORDS = ["dinero", "pesos", "bitcoin", "btc", "crypto", "inversión", "premio", "millones", "descuento", "oferta", "gratis", "gana", "ganaste", "duplica", "profits"]
LEGIT_WORDS = ["reunión", "proyecto", "informe", "equipo", "estado", "extracto", "repositorio", "pull", "request", "resumen", "playlist", "trimestre", "asistencia", "planeación"]
SUSPICIOUS_DOMAINS = [".xyz", ".biz", ".io", "free-", "crypto", "slim", "pills", "invest", "gana", "promo"]
KNOWN_DOMAINS = ["gmail.com", "empresa.co", "universidad.edu", "github.com", "spotify.com", "banconacional.com.co"]

# --- LÓGICA DEL PERCEPTRÓN ---
class Perceptron:
    def __init__(self, n_features, learning_rate=0.1, epochs=50):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _forward(self, x):
        return self._sigmoid(np.dot(x, self.weights) + self.bias)

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = self._forward(xi)
                error = yi - pred
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

    def predict_proba(self, x):
        return self._forward(x)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

# --- EXTRACCIÓN DE CARACTERÍSTICAS ---
def extract_features(email):
    text_lower = (email["subject"] + " " + email["body"] + " " + email["sender"]).lower()
    text_raw = email["subject"] + " " + email["body"]
    sender_email = email["sender"] # Ajustado para la estructura de la app

    caps_ratio = min(len(re.findall(r"[A-ZÁÉÍÓÚÑÜ]{2,}", text_raw)) / 3, 1.0)
    excl_count = min(text_raw.count("!") / 4, 1.0)
    urgent_score = min(sum(w in text_lower for w in URGENT_WORDS) / 4, 1.0)
    money_score = min(sum(w in text_lower for w in MONEY_WORDS) / 4, 1.0)
    susp_domain = min(sum(s in sender_email for s in SUSPICIOUS_DOMAINS) / 2, 1.0)
    legit_score = min(sum(w in text_lower for w in LEGIT_WORDS) / 3, 1.0)
    long_subject = float(len(email["subject"]) > 50)
    unknown_dom = float(not any(d in sender_email for d in KNOWN_DOMAINS))

    return np.array([caps_ratio, excl_count, urgent_score, money_score, susp_domain, legit_score, long_subject, unknown_dom])

# --- INTERFAZ GRÁFICA ---
class SpamApp:
    def __init__(self, root, modelo):
        self.root = root
        self.modelo = modelo
        self.root.title("Clasificador de Spam - Universidad")
        self.root.geometry("800x600")
        
        self.notebook = ttk.Notebook(root)
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Ingreso Manual")
        self.notebook.add(self.tab2, text="Verificación CSV")
        self.notebook.pack(expand=True, fill="both")
        
        self.setup_manual_tab()
        self.setup_csv_tab()

    def setup_manual_tab(self):
        tk.Label(self.tab1, text="Análisis de Correo Individual", font=("Arial", 14, "bold")).pack(pady=10)
        
        tk.Label(self.tab1, text="Remitente (Email):").pack()
        self.ent_sender = tk.Entry(self.tab1, width=50)
        self.ent_sender.pack(pady=2)
        
        tk.Label(self.tab1, text="Asunto:").pack()
        self.ent_subject = tk.Entry(self.tab1, width=50)
        self.ent_subject.pack(pady=2)
        
        tk.Label(self.tab1, text="Cuerpo del mensaje:").pack()
        self.txt_body = tk.Text(self.tab1, height=5, width=50)
        self.txt_body.pack(pady=5)
        
        tk.Button(self.tab1, text="Clasificar ahora", bg="#005088", fg="white", command=self.analizar_manual).pack(pady=10)
        
        self.lbl_res = tk.Label(self.tab1, text="", font=("Arial", 12, "bold"))
        self.lbl_res.pack(pady=10)

    def analizar_manual(self):
        email = {
            "sender": self.ent_sender.get(),
            "subject": self.ent_subject.get(),
            "body": self.txt_body.get("1.0", tk.END)
        }
        feat = extract_features(email)
        prob = self.modelo.predict_proba(feat)
        clase = "SPAM" if self.modelo.predict(feat) == 1 else "LIMPIO"
        color = "red" if clase == "SPAM" else "green"
        
        self.lbl_res.config(text=f"Resultado: {clase} ({prob*100:.2f}%)", fg=color)

    def setup_csv_tab(self):
        tk.Label(self.tab2, text="Resultados del Dataset (correos.csv)", font=("Arial", 14, "bold")).pack(pady=10)
        self.tree = ttk.Treeview(self.tab2, columns=("Sender", "Probabilidad", "Resultado"), show="headings")
        self.tree.heading("Sender", text="Remitente")
        self.tree.heading("Probabilidad", text="Prob. Spam")
        self.tree.heading("Resultado", text="Predicción")
        self.tree.pack(expand=True, fill="both", padx=10, pady=10)
        
        tk.Button(self.tab2, text="Cargar y Procesar CSV", command=self.cargar_csv).pack(pady=10)

    def cargar_csv(self):
        path_csv = os.path.join(os.path.dirname(__file__), "correos.csv")
        if not os.path.exists(path_csv):
            messagebox.showerror("Error", "No se encontró el archivo correos.csv")
            return
        
        for i in self.tree.get_children(): self.tree.delete(i)
        
        with open(path_csv, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                email_data = {"sender": row["sender"], "subject": row["subject"], "body": row["body"]}
                feat = extract_features(email_data)
                prob = self.modelo.predict_proba(feat)
                pred = "SPAM" if prob >= 0.5 else "OK"
                self.tree.insert("", tk.END, values=(row["sender"], f"{prob*100:.2f}%", pred))

# --- INICIO DE LA APP ---
if __name__ == "__main__":
    path_csv = os.path.join(os.path.dirname(__file__), "correos.csv")
    X_train, y_train = [], []
    
    if os.path.exists(path_csv):
        with open(path_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                X_train.append(extract_features({"sender": r["sender"], "subject": r["subject"], "body": r["body"]}))
                y_train.append(int(r["label"]))
        
        modelo = Perceptron(n_features=8)
        modelo.train(np.array(X_train), np.array(y_train))
        
        root = tk.Tk()
        app = SpamApp(root, modelo)
        root.mainloop()
    else:
        print("Error: Asegúrate de tener el archivo 'correos.csv' en la misma carpeta.")