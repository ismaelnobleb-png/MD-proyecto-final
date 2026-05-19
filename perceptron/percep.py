import os
import csv
import numpy as np
import re
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import math

# ============================================================
#  LISTAS DE PALABRAS CLAVE
# ============================================================
URGENT_WORDS = [
    "urgente", "ya", "hoy", "ahora", "expira", "limitada", "gratis",
    "gana", "ganaste", "garantizado", "secreto", "único", "clic",
    "click", "reclama", "actúa", "perderás", "aproveche",
]

MONEY_WORDS = [
    "dinero", "pesos", "bitcoin", "btc", "crypto", "inversión", "premio",
    "millones", "descuento", "oferta", "gratis", "gana", "ganaste",
    "duplica", "profits",
]

LEGIT_WORDS = [
    "reunión", "proyecto", "informe", "equipo", "estado", "extracto",
    "repositorio", "pull", "request", "resumen", "playlist", "trimestre",
    "asistencia", "planeación",
]

SUSPICIOUS_DOMAINS = [
    ".xyz", ".biz", ".io", "free-", "crypto", "slim",
    "pills", "invest", "gana", "promo",
]

KNOWN_DOMAINS = [
    "gmail.com", "empresa.co", "universidad.edu",
    "github.com", "spotify.com", "banconacional.com.co",
]

FEAT_NAMES = [
    "Mayúsculas", "Exclamación", "Urgencia", "Dinero",
    "Dom. Sospechoso", "Palabras Legítimas", "Asunto Largo", "Dom. Desconocido"
]

# ============================================================
#  EXTRACCIÓN DE CARACTERÍSTICAS
# ============================================================
def extract_features(email):
    text_lower = (email["subject"] + " " + email["body"] + " " + email["sender"]).lower()
    text_raw   = email["subject"] + " " + email["body"]
    sender_email = email["from"]

    caps_ratio  = min(len(re.findall(r"[A-ZÁÉÍÓÚÑÜ]{2,}", text_raw)) / 3, 1.0)
    excl_count  = min(text_raw.count("!") / 4, 1.0)
    urgent_score = min(sum(w in text_lower for w in URGENT_WORDS) / 4, 1.0)
    money_score  = min(sum(w in text_lower for w in MONEY_WORDS) / 4, 1.0)
    susp_domain  = min(sum(s in sender_email for s in SUSPICIOUS_DOMAINS) / 2, 1.0)
    legit_score  = min(sum(w in text_lower for w in LEGIT_WORDS) / 3, 1.0)
    long_subject = float(len(email["subject"]) > 50)
    unknown_dom  = float(not any(d in sender_email for d in KNOWN_DOMAINS))

    return np.array([caps_ratio, excl_count, urgent_score, money_score,
                     susp_domain, legit_score, long_subject, unknown_dom])

# ============================================================
#  PERCEPTRÓN (Regresión Logística con Gradiente Descendente)
# ============================================================
class Perceptron:
    def __init__(self, n_features, learning_rate=0.1, epochs=50):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _forward(self, x):
        return self._sigmoid(np.dot(x, self.weights) + self.bias)

    def train(self, X, y, callback=None):
        self.loss_history = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xi, yi in zip(X, y):
                pred  = self._forward(xi)
                error = yi - pred
                self.weights += self.lr * error * xi
                self.bias    += self.lr * error
                epoch_loss   += -(yi * np.log(pred + 1e-9) + (1 - yi) * np.log(1 - pred + 1e-9))
            avg_loss = epoch_loss / len(y)
            self.loss_history.append(avg_loss)
            if callback:
                callback(epoch + 1, avg_loss)

    def predict_proba(self, x):
        return self._forward(x)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def accuracy(self, X, y, threshold=0.5):
        preds = np.array([self.predict(xi, threshold) for xi in X])
        return np.mean(preds == y)

# ============================================================
#  CARGA DE DATOS
# ============================================================
def load_data(file_path):
    emails_list = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row["label"] = int(row["label"])
            emails_list.append(row)
    return emails_list

# ============================================================
#  PALETA DE COLORES (estilo terminal oscuro)
# ============================================================
BG       = "#0d0d0d"
BG2      = "#111111"
BG3      = "#1a1a1a"
FG       = "#00ff88"       # verde principal
FG2      = "#888888"       # gris texto secundario
FG3      = "#cccccc"       # blanco suave
ACCENT   = "#00cc6a"
RED      = "#ff4444"
ORANGE   = "#ff8844"
YELLOW   = "#ffcc00"
BORDER   = "#2a2a2a"
FONT_MONO = ("Courier New", 10)
FONT_MONO_BIG = ("Courier New", 13, "bold")
FONT_TITLE = ("Courier New", 16, "bold")
FONT_LABEL = ("Courier New", 9)

# ============================================================
#  APLICACIÓN PRINCIPAL
# ============================================================
class SpamApp(tk.Tk):

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.title("Spam Detector — Perceptrón  |  4to Semestre Sistemas")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.geometry("860x680")

        # Modelo y datos
        self.emails = []
        self.modelo = None
        self.X = None
        self.y = None
        self._load_dataset()

        self._build_ui()

    # ----------------------------------------------------------
    def _load_dataset(self):
        try:
            self.emails = load_data(self.csv_path)
            self.X = np.array([extract_features(e) for e in self.emails])
            self.y = np.array([e["label"] for e in self.emails])
            self.modelo = Perceptron(n_features=self.X.shape[1], learning_rate=0.1, epochs=50)
        except FileNotFoundError:
            messagebox.showerror("Error", f"No se encontró el archivo:\n{self.csv_path}")
            self.destroy()

    # ----------------------------------------------------------
    def _build_ui(self):
        # ── Encabezado ──────────────────────────────────────────
        header = tk.Frame(self, bg=BG2, pady=8)
        header.pack(fill="x")

        tk.Label(header, text="●", fg=FG, bg=BG2, font=("Courier New", 14)).pack(side="left", padx=(14, 6))
        tk.Label(header, text="PERCEPTRÓN CLASIFICADOR DE SPAM",
                 fg=FG, bg=BG2, font=FONT_TITLE).pack(side="left")
        tk.Label(header, text=f"  //  {os.path.basename(self.csv_path)}",
                 fg=FG2, bg=BG2, font=FONT_MONO).pack(side="left")
        tk.Label(header, text="Prog. Redes Neuronales — 4to Sem.",
                 fg="#333333", bg=BG2, font=FONT_LABEL).pack(side="right", padx=14)

        sep = tk.Frame(self, bg=FG, height=2)
        sep.pack(fill="x")

        # ── Notebook (pestañas) ──────────────────────────────────
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("Dark.TNotebook",          background=BG,  borderwidth=0)
        style.configure("Dark.TNotebook.Tab",      background=BG2, foreground=FG2,
                        font=FONT_MONO, padding=[14, 6], borderwidth=0)
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BG3)],
                  foreground=[("selected", FG)])

        self.nb = ttk.Notebook(self, style="Dark.TNotebook")
        self.nb.pack(fill="both", expand=True, padx=0, pady=0)

        # Pestañas
        self._build_tab_clasificar()
        self._build_tab_entrenar()
        self._build_tab_dataset()
        self._build_tab_consola()

    # ══════════════════════════════════════════════════════════════
    #  PESTAÑA 1 — CLASIFICAR
    # ══════════════════════════════════════════════════════════════
    def _build_tab_clasificar(self):
        frame = tk.Frame(self.nb, bg=BG)
        self.nb.add(frame, text="[ Clasificar ]")

        pad = {"padx": 18, "pady": 6}

        tk.Label(frame, text="// ingrese los datos del correo a analizar",
                 fg=FG2, bg=BG, font=FONT_LABEL).pack(anchor="w", padx=18, pady=(14, 2))

        # Campos de entrada
        fields_frame = tk.Frame(frame, bg=BG)
        fields_frame.pack(fill="x", **pad)

        def make_field(parent, label, row, col, width=38, span=1):
            tk.Label(parent, text=label, fg=FG2, bg=BG, font=FONT_LABEL,
                     anchor="w").grid(row=row*2, column=col, columnspan=span, sticky="w", padx=6, pady=(6,0))
            e = tk.Entry(parent, width=width, bg=BG2, fg=FG, insertbackground=FG,
                         relief="flat", font=FONT_MONO, bd=1,
                         highlightthickness=1, highlightcolor=FG, highlightbackground=BORDER)
            e.grid(row=row*2+1, column=col, columnspan=span, sticky="ew", padx=6, pady=(0,4))
            return e

        fields_frame.columnconfigure(0, weight=1)
        fields_frame.columnconfigure(1, weight=1)

        self.inp_sender  = make_field(fields_frame, "REMITENTE (nombre)", 0, 0)
        self.inp_from    = make_field(fields_frame, "FROM (email)",       0, 1)
        self.inp_subject = make_field(fields_frame, "ASUNTO",             1, 0, span=2, width=80)

        tk.Label(fields_frame, text="CUERPO DEL MENSAJE", fg=FG2, bg=BG,
                 font=FONT_LABEL, anchor="w").grid(row=4, column=0, columnspan=2, sticky="w", padx=6, pady=(6,0))
        self.inp_body = tk.Text(fields_frame, height=5, bg=BG2, fg=FG3, insertbackground=FG,
                                relief="flat", font=FONT_MONO, bd=1,
                                highlightthickness=1, highlightcolor=FG, highlightbackground=BORDER)
        self.inp_body.grid(row=5, column=0, columnspan=2, sticky="ew", padx=6, pady=(0,4))

        # Valores de ejemplo
        self.inp_sender.insert(0, "CryptoProfits Corp")
        self.inp_from.insert(0, "gana@free-crypto.xyz")
        self.inp_subject.insert(0, "¡GANASTE! Reclama YA tus 5 MILLONES de pesos GRATIS — URGENTE!!!")
        self.inp_body.insert("1.0", "Haz CLIC ahora y duplica tus Bitcoin en 24 horas. Oferta LIMITADA, expira hoy. Actúa ahora o perderás esta oportunidad única y garantizada de ganar millones.")

        # Botón analizar
        btn_frame = tk.Frame(frame, bg=BG)
        btn_frame.pack(fill="x", padx=18, pady=4)

        self.btn_analizar = tk.Button(
            btn_frame, text="▶  ANALIZAR CORREO",
            bg=BG, fg=FG, activebackground=FG, activeforeground=BG,
            relief="flat", font=("Courier New", 11, "bold"),
            bd=0, highlightthickness=2, highlightbackground=FG,
            cursor="hand2", command=self._analizar, pady=8
        )
        self.btn_analizar.pack(fill="x")

        # ── Panel de resultado ───────────────────────────────────
        res_frame = tk.Frame(frame, bg=BG2, bd=0,
                             highlightthickness=1, highlightbackground=BORDER)
        res_frame.pack(fill="both", expand=True, padx=18, pady=10)

        # Veredicto
        top_res = tk.Frame(res_frame, bg=BG2)
        top_res.pack(fill="x", padx=14, pady=(12, 4))

        self.lbl_verdict = tk.Label(top_res, text="— esperando análisis —",
                                    fg=FG2, bg=BG2, font=("Courier New", 18, "bold"), anchor="w")
        self.lbl_verdict.pack(side="left")

        self.lbl_prob = tk.Label(top_res, text="", fg=FG, bg=BG2,
                                 font=("Courier New", 18, "bold"), anchor="e")
        self.lbl_prob.pack(side="right")

        # Barra de probabilidad
        bar_frame = tk.Frame(res_frame, bg=BG2)
        bar_frame.pack(fill="x", padx=14, pady=(0, 8))

        tk.Label(bar_frame, text="Probabilidad de SPAM:", fg=FG2, bg=BG2,
                 font=FONT_LABEL).pack(anchor="w")
        self.canvas_bar = tk.Canvas(bar_frame, height=12, bg=BG3, bd=0,
                                    highlightthickness=0)
        self.canvas_bar.pack(fill="x", pady=3)

        # Features
        tk.Label(res_frame, text="// vector de características (features):",
                 fg=FG2, bg=BG2, font=FONT_LABEL).pack(anchor="w", padx=14)

        feat_grid = tk.Frame(res_frame, bg=BG2)
        feat_grid.pack(fill="x", padx=14, pady=(4, 12))

        self.feat_labels = []
        self.feat_bars   = []
        cols = 4
        for i, name in enumerate(FEAT_NAMES):
            r, c = divmod(i, cols)
            cell = tk.Frame(feat_grid, bg=BG3, bd=0,
                            highlightthickness=1, highlightbackground=BORDER)
            cell.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            feat_grid.columnconfigure(c, weight=1)

            tk.Label(cell, text=name, fg=FG2, bg=BG3, font=FONT_LABEL).pack(pady=(5, 0))
            lv = tk.Label(cell, text="—", fg=FG, bg=BG3, font=("Courier New", 13, "bold"))
            lv.pack(pady=(0, 5))
            self.feat_labels.append(lv)

    # ══════════════════════════════════════════════════════════════
    #  PESTAÑA 2 — ENTRENAMIENTO
    # ══════════════════════════════════════════════════════════════
    def _build_tab_entrenar(self):
        frame = tk.Frame(self.nb, bg=BG)
        self.nb.add(frame, text="[ Entrenamiento ]")

        # ── Parámetros ───────────────────────────────────────────
        tk.Label(frame, text="// configuración del modelo",
                 fg=FG2, bg=BG, font=FONT_LABEL).pack(anchor="w", padx=18, pady=(14, 4))

        params_frame = tk.Frame(frame, bg=BG)
        params_frame.pack(fill="x", padx=18)

        def make_slider(parent, label, from_, to, init, col, fmt=lambda v: str(int(v))):
            box = tk.Frame(parent, bg=BG2, bd=0,
                           highlightthickness=1, highlightbackground=BORDER)
            box.grid(row=0, column=col, padx=6, pady=4, sticky="nsew")
            parent.columnconfigure(col, weight=1)

            tk.Label(box, text=label, fg=FG2, bg=BG2, font=FONT_LABEL).pack(pady=(8, 2))
            val_lbl = tk.Label(box, text=fmt(init), fg=FG, bg=BG2,
                               font=("Courier New", 16, "bold"))
            val_lbl.pack()

            sl = tk.Scale(box, from_=from_, to=to, orient="horizontal",
                          bg=BG2, fg=FG, troughcolor=BG3, activebackground=FG,
                          highlightthickness=0, sliderrelief="flat",
                          showvalue=False, bd=0, length=160,
                          command=lambda v, lbl=val_lbl, f=fmt: lbl.config(text=f(float(v))))
            sl.set(init)
            sl.pack(pady=(0, 8))
            return sl

        self.sl_epochs = make_slider(params_frame, "ÉPOCAS",        10, 200, 50,  0)
        self.sl_lr     = make_slider(params_frame, "LEARNING RATE",  1, 20,  10,  1,
                                     fmt=lambda v: f"{v/100:.2f}")
        self.sl_thr    = make_slider(params_frame, "THRESHOLD",      10, 90, 50,  2,
                                     fmt=lambda v: f"{v/100:.2f}")

        # Botón entrenar
        self.btn_train = tk.Button(
            frame, text="[ ENTRENAR MODELO ]",
            bg=BG, fg=FG, activebackground=FG, activeforeground=BG,
            relief="flat", font=("Courier New", 10, "bold"),
            bd=0, highlightthickness=1, highlightbackground=FG,
            cursor="hand2", command=self._entrenar, pady=7
        )
        self.btn_train.pack(fill="x", padx=18, pady=(10, 6))

        # Log y gráfica lado a lado
        mid = tk.Frame(frame, bg=BG)
        mid.pack(fill="both", expand=True, padx=18, pady=4)
        mid.columnconfigure(0, weight=2)
        mid.columnconfigure(1, weight=3)

        # Log de épocas
        tk.Label(mid, text="// log de entrenamiento", fg=FG2, bg=BG,
                 font=FONT_LABEL).grid(row=0, column=0, sticky="w", padx=(0,8))
        self.train_log = scrolledtext.ScrolledText(
            mid, height=9, bg=BG2, fg=FG2, insertbackground=FG,
            relief="flat", font=("Courier New", 9), bd=0,
            highlightthickness=1, highlightbackground=BORDER
        )
        self.train_log.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.train_log.configure(state="disabled")

        # Gráfica de loss
        tk.Label(mid, text="// curva de pérdida (loss)", fg=FG2, bg=BG,
                 font=FONT_LABEL).grid(row=0, column=1, sticky="w")
        self.loss_canvas = tk.Canvas(mid, bg=BG2, bd=0, highlightthickness=1,
                                     highlightbackground=BORDER)
        self.loss_canvas.grid(row=1, column=1, sticky="nsew")
        mid.rowconfigure(1, weight=1)

        # Métricas
        metrics = tk.Frame(frame, bg=BG)
        metrics.pack(fill="x", padx=18, pady=(8, 6))

        self.metric_vars = {}
        for i, (key, lbl) in enumerate([("acc","ACCURACY"), ("ep","ÉPOCAS"), ("loss","LOSS FINAL"), ("bias","BIAS")]):
            box = tk.Frame(metrics, bg=BG2, bd=0,
                           highlightthickness=1, highlightbackground=BORDER)
            box.pack(side="left", expand=True, fill="x", padx=4)
            tk.Label(box, text=lbl, fg=FG2, bg=BG2, font=FONT_LABEL).pack(pady=(6,0))
            v = tk.StringVar(value="—")
            tk.Label(box, textvariable=v, fg=FG, bg=BG2,
                     font=("Courier New", 15, "bold")).pack(pady=(0,6))
            self.metric_vars[key] = v

        # Pesos
        tk.Label(frame, text="// pesos aprendidos por la neurona",
                 fg=FG2, bg=BG, font=FONT_LABEL).pack(anchor="w", padx=18)
        self.weights_canvas = tk.Canvas(frame, height=120, bg=BG2, bd=0,
                                        highlightthickness=1, highlightbackground=BORDER)
        self.weights_canvas.pack(fill="x", padx=18, pady=(4, 10))

    # ══════════════════════════════════════════════════════════════
    #  PESTAÑA 3 — DATASET
    # ══════════════════════════════════════════════════════════════
    def _build_tab_dataset(self):
        frame = tk.Frame(self.nb, bg=BG)
        self.nb.add(frame, text="[ Dataset ]")

        tk.Label(frame, text=f"// correos cargados desde: {self.csv_path}",
                 fg=FG2, bg=BG, font=FONT_LABEL).pack(anchor="w", padx=18, pady=(14, 6))

        cols = ("REMITENTE", "ASUNTO", "LABEL", "PREDICCIÓN", "OK?", "PROB.")
        style = ttk.Style()
        style.configure("Dark.Treeview",
                        background=BG2, fieldbackground=BG2,
                        foreground=FG3, font=FONT_MONO, rowheight=24,
                        borderwidth=0)
        style.configure("Dark.Treeview.Heading",
                        background=BG3, foreground=FG2, font=FONT_LABEL,
                        relief="flat")
        style.map("Dark.Treeview", background=[("selected", "#1e3a2a")])

        tree_frame = tk.Frame(frame, bg=BG)
        tree_frame.pack(fill="both", expand=True, padx=18, pady=(0, 10))

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        vsb.pack(side="right", fill="y")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        hsb.pack(side="bottom", fill="x")

        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                 yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                                 style="Dark.Treeview")
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        widths = [130, 280, 70, 90, 40, 70]
        for col, w in zip(cols, widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="w")

        self.tree.pack(fill="both", expand=True)
        self.tree.tag_configure("spam",   foreground="#ff6666")
        self.tree.tag_configure("legit",  foreground="#00ff88")
        self.tree.tag_configure("wrong",  foreground="#ff8844")

        self._refresh_dataset_tab()

    # ══════════════════════════════════════════════════════════════
    #  PESTAÑA 4 — CONSOLA
    # ══════════════════════════════════════════════════════════════
    def _build_tab_consola(self):
        frame = tk.Frame(self.nb, bg=BG)
        self.nb.add(frame, text="[ Consola ]")

        tk.Label(frame, text="// salida estándar del programa (equivalente a la terminal)",
                 fg=FG2, bg=BG, font=FONT_LABEL).pack(anchor="w", padx=18, pady=(14, 4))

        self.console = scrolledtext.ScrolledText(
            frame, bg=BG2, fg=FG3, insertbackground=FG,
            relief="flat", font=("Courier New", 10), bd=0,
            highlightthickness=1, highlightbackground=BORDER
        )
        self.console.pack(fill="both", expand=True, padx=18, pady=(0, 10))

        btn_run = tk.Button(
            frame, text="[ EJECUTAR REPORTE COMPLETO ]",
            bg=BG, fg=FG, activebackground=FG, activeforeground=BG,
            relief="flat", font=("Courier New", 10, "bold"),
            bd=0, highlightthickness=1, highlightbackground=FG,
            cursor="hand2", command=self._run_console, pady=7
        )
        btn_run.pack(fill="x", padx=18, pady=(0, 10))

    # ══════════════════════════════════════════════════════════════
    #  LÓGICA — ANALIZAR
    # ══════════════════════════════════════════════════════════════
    def _analizar(self):
        email = {
            "sender":  self.inp_sender.get().strip() or "Desconocido",
            "from":    self.inp_from.get().strip()   or "unknown@unknown.com",
            "subject": self.inp_subject.get().strip(),
            "body":    self.inp_body.get("1.0", "end").strip(),
        }

        feat = extract_features(email)
        thr  = self.sl_thr.get() / 100

        if not np.any(self.modelo.weights):
            # Si no se ha entrenado, usar pesos por defecto ilustrativos
            self.modelo.weights = np.array([0.8, 0.6, 1.2, 1.0, 0.9, -0.7, 0.3, 0.5])
            self.modelo.bias    = -1.2

        prob = self.modelo.predict_proba(feat)
        pred = 1 if prob >= thr else 0

        # Veredicto
        if pred == 1:
            self.lbl_verdict.config(text="⚠  SPAM DETECTADO", fg=RED)
        else:
            self.lbl_verdict.config(text="✓  CORREO LEGÍTIMO", fg=FG)

        self.lbl_prob.config(text=f"{prob*100:.2f}%",
                             fg=RED if pred == 1 else FG)

        # Barra de probabilidad
        self.canvas_bar.update_idletasks()
        w = self.canvas_bar.winfo_width()
        self.canvas_bar.delete("all")
        color = RED if prob > 0.5 else FG
        self.canvas_bar.create_rectangle(0, 0, int(w * prob), 12, fill=color, outline="")

        # Features
        feat_colors = [RED if v > 0.5 else (ORANGE if v > 0.25 else FG)
                       for v in feat]
        for i, (v, lbl) in enumerate(zip(feat, self.feat_labels)):
            lbl.config(text=f"{v*100:.0f}%", fg=feat_colors[i])

    # ══════════════════════════════════════════════════════════════
    #  LÓGICA — ENTRENAR
    # ══════════════════════════════════════════════════════════════
    def _entrenar(self):
        epochs = int(self.sl_epochs.get())
        lr     = self.sl_lr.get() / 100

        self.modelo = Perceptron(n_features=self.X.shape[1],
                                 learning_rate=lr, epochs=epochs)

        self.train_log.configure(state="normal")
        self.train_log.delete("1.0", "end")
        self.train_log.configure(state="disabled")

        self.btn_train.config(state="disabled", text="[ ENTRENANDO... ]")

        def callback(epoch, loss):
            def update():
                self.train_log.configure(state="normal")
                self.train_log.insert("end", f"Epoch {epoch:>3}/{epochs}  |  loss: {loss:.4f}\n")
                self.train_log.see("end")
                self.train_log.configure(state="disabled")
                self._draw_loss(self.modelo.loss_history)
            self.after(0, update)

        def run():
            self.modelo.train(self.X, self.y, callback=callback)
            self.after(0, self._post_train)

        threading.Thread(target=run, daemon=True).start()

    def _post_train(self):
        thr = self.sl_thr.get() / 100
        acc = self.modelo.accuracy(self.X, self.y, thr)
        loss_final = self.modelo.loss_history[-1] if self.modelo.loss_history else 0

        self.metric_vars["acc"].set(f"{acc*100:.1f}%")
        self.metric_vars["ep"].set(str(int(self.sl_epochs.get())))
        self.metric_vars["loss"].set(f"{loss_final:.4f}")
        self.metric_vars["bias"].set(f"{self.modelo.bias:.3f}")

        self.train_log.configure(state="normal")
        self.train_log.insert("end",
            f"\n{'='*38}\n"
            f" ENTRENAMIENTO COMPLETO\n"
            f" Accuracy: {acc*100:.2f}%   Loss: {loss_final:.4f}\n"
            f"{'='*38}\n"
        )
        self.train_log.see("end")
        self.train_log.configure(state="disabled")

        self.btn_train.config(state="normal", text="[ ENTRENAR MODELO ]")
        self._draw_weights()
        self._refresh_dataset_tab()

    # ── Gráfica de pérdida ───────────────────────────────────────
    def _draw_loss(self, history):
        c = self.loss_canvas
        c.update_idletasks()
        W, H = c.winfo_width(), c.winfo_height()
        if W < 10 or len(history) < 2:
            return
        c.delete("all")
        pad = 24
        max_l = max(history)
        min_l = min(history)
        rng   = max_l - min_l if max_l != min_l else 1

        def px(i, v):
            x = pad + i / (len(history) - 1) * (W - pad * 2)
            y = pad + (1 - (v - min_l) / rng) * (H - pad * 2)
            return x, y

        # Ejes
        c.create_line(pad, pad, pad, H - pad, fill=BORDER)
        c.create_line(pad, H - pad, W - pad, H - pad, fill=BORDER)

        # Etiquetas
        c.create_text(pad + 2, pad, text=f"{max_l:.3f}", fill=FG2,
                      font=("Courier New", 8), anchor="nw")
        c.create_text(pad + 2, H - pad, text=f"{min_l:.3f}", fill=FG2,
                      font=("Courier New", 8), anchor="sw")

        # Curva
        pts = [px(i, v) for i, v in enumerate(history)]
        for i in range(len(pts) - 1):
            c.create_line(*pts[i], *pts[i+1], fill=FG, width=1.5)

        # Área bajo la curva
        poly = []
        for p in pts:
            poly.extend(p)
        poly.extend([W - pad, H - pad, pad, H - pad])
        c.create_polygon(poly, fill="#003322", outline="")

    # ── Barras de pesos ──────────────────────────────────────────
    def _draw_weights(self):
        c = self.weights_canvas
        c.update_idletasks()
        W, H = c.winfo_width(), c.winfo_height()
        if W < 10:
            return
        c.delete("all")
        weights = self.modelo.weights
        max_w = max(abs(w) for w in weights) if any(weights) else 1
        n = len(weights)
        bar_h = 16
        spacing = H / n

        for i, (w, name) in enumerate(zip(weights, FEAT_NAMES)):
            y = i * spacing + spacing / 2
            bar_len = abs(w) / max_w * (W / 2 - 110)
            color = FG if w >= 0 else RED
            mid_x = W / 2
            c.create_text(mid_x - 10, y, text=name, fill=FG2,
                          font=("Courier New", 8), anchor="e")
            c.create_rectangle(mid_x, y - bar_h / 2,
                                mid_x + bar_len, y + bar_h / 2,
                                fill=color, outline="")
            c.create_text(mid_x + bar_len + 6, y,
                          text=f"{w:.3f}", fill=color,
                          font=("Courier New", 8), anchor="w")

    # ══════════════════════════════════════════════════════════════
    #  DATASET TAB — REFRESH
    # ══════════════════════════════════════════════════════════════
    def _refresh_dataset_tab(self):
        thr = self.sl_thr.get() / 100 if hasattr(self, "sl_thr") else 0.5
        for row in self.tree.get_children():
            self.tree.delete(row)

        for email in self.emails:
            feat  = extract_features(email)
            prob  = self.modelo.predict_proba(feat)
            pred  = 1 if prob >= thr else 0
            label = email["label"]

            txt_label = "SPAM"    if label == 1 else "OK"
            txt_pred  = "SPAM"    if pred  == 1 else "OK"
            ok_mark   = "✓"       if pred  == label else "✗"

            tag = "spam" if label == 1 else "legit"
            if pred != label:
                tag = "wrong"

            self.tree.insert("", "end",
                values=(
                    email["sender"][:22],
                    email["subject"][:45] + ("..." if len(email["subject"]) > 45 else ""),
                    txt_label,
                    txt_pred,
                    ok_mark,
                    f"{prob*100:.1f}%"
                ),
                tags=(tag,)
            )

    # ══════════════════════════════════════════════════════════════
    #  CONSOLA — REPORTE
    # ══════════════════════════════════════════════════════════════
    def _run_console(self):
        thr = self.sl_thr.get() / 100
        self.console.delete("1.0", "end")
        out = []
        out.append("=" * 60)
        out.append(f" PERCEPTRÓN CLASIFICADOR — DATASET: {self.csv_path}")
        out.append("=" * 60)
        out.append(f" Parámetros -> Épocas: {self.modelo.epochs} | LR: {self.modelo.lr} | Threshold: {thr:.2f}")
        acc = self.modelo.accuracy(self.X, self.y, thr)
        out.append(f" Accuracy sobre el dataset: {acc*100:.2f}%")
        out.append(f" Bias: {self.modelo.bias:.4f}")
        out.append("=" * 60)
        out.append("")
        out.append(f"{'REMITENTE':<22} {'PRED':<10} {'PROB. SPAM':<12} {'REAL'}")
        out.append("-" * 60)
        for email in self.emails:
            feat  = extract_features(email)
            prob  = self.modelo.predict_proba(feat)
            pred  = 1 if prob >= thr else 0
            txt_pred = "SPAM" if pred == 1 else "OK"
            txt_real = "spam" if email["label"] == 1 else "legitimo"
            check    = "✓" if pred == email["label"] else "✗"
            out.append(f"{email['sender'][:20]:<22} {txt_pred:<10} {prob*100:>8.2f}%    {txt_real} {check}")
        out.append("")
        out.append("// pesos finales de la neurona:")
        for name, w in zip(FEAT_NAMES, self.modelo.weights):
            out.append(f"  {name:<22}: {w:+.4f}")
        out.append(f"  {'Bias':<22}: {self.modelo.bias:+.4f}")
        out.append("")
        out.append("// Fin del reporte.")

        self.console.insert("end", "\n".join(out))

# ============================================================
#  PUNTO DE ENTRADA
# ============================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_csv   = os.path.join(script_dir, "correos.csv")
    app = SpamApp(path_csv)
    app.mainloop()

if __name__ == "__main__":
    main()