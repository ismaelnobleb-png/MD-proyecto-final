import os
import csv
import numpy as np
import re

# Listas de palabras clave para la extracción de características (Features)
URGENT_WORDS = [
    "urgente",
    "ya",
    "hoy",
    "ahora",
    "expira",
    "limitada",
    "gratis",
    "gana",
    "ganaste",
    "garantizado",
    "secreto",
    "único",
    "clic",
    "click",
    "reclama",
    "actúa",
    "perderás",
    "aproveche",
]

MONEY_WORDS = [
    "dinero",
    "pesos",
    "bitcoin",
    "btc",
    "crypto",
    "inversión",
    "premio",
    "millones",
    "descuento",
    "oferta",
    "gratis",
    "gana",
    "ganaste",
    "duplica",
    "profits",
]

LEGIT_WORDS = [
    "reunión",
    "proyecto",
    "informe",
    "equipo",
    "estado",
    "extracto",
    "repositorio",
    "pull",
    "request",
    "resumen",
    "playlist",
    "trimestre",
    "asistencia",
    "planeación",
]

SUSPICIOUS_DOMAINS = [
    ".xyz",
    ".biz",
    ".io",
    "free-",
    "crypto",
    "slim",
    "pills",
    "invest",
    "gana",
    "promo",
]

KNOWN_DOMAINS = [
    "gmail.com",
    "empresa.co",
    "universidad.edu",
    "github.com",
    "spotify.com",
    "banconacional.com.co",
]


# Función para normalizar y extraer el vector de características de cada correo
def extract_features(email):
    # Pasamos a minúsculas para evitar problemas de case-sensitivity
    text_lower = (
        email["subject"] + " " + email["body"] + " " + email["sender"]
    ).lower()
    text_raw = email["subject"] + " " + email["body"]
    sender_email = email["from"]

    # 1. Ratio de palabras en mayúsculas (indicador de gritos/spam)
    caps_ratio = min(
        len(re.findall(r"[A-ZÁÉÍÓÚÑÜ]{2,}", text_raw)) / 3, 1.0
    )

    # 2. Conteo de signos de exclamación
    excl_count = min(text_raw.count("!") / 4, 1.0)

    # 3. Frecuencia de palabras urgentes
    urgent_score = min(sum(w in text_lower for w in URGENT_WORDS) / 4, 1.0)

    # 4. Frecuencia de palabras de dinero
    money_score = min(sum(w in text_lower for w in MONEY_WORDS) / 4, 1.0)

    # 5. Si el dominio del remitente está en la lista negra
    susp_domain = min(
        sum(s in sender_email for s in SUSPICIOUS_DOMAINS) / 2, 1.0
    )

    # 6. Frecuencia de palabras legítimas laborales/estudiantiles
    legit_score = min(sum(w in text_lower for w in LEGIT_WORDS) / 3, 1.0)

    # 7. Bandera binaria: asunto excesivamente largo
    long_subject = float(len(email["subject"]) > 50)

    # 8. Bandera binaria: si el dominio no pertenece a los conocidos
    unknown_dom = float(not any(d in sender_email for d in KNOWN_DOMAINS))

    # Retornamos el vector representativo mapeado
    return np.array(
        [
            caps_ratio,
            excl_count,
            urgent_score,
            money_score,
            susp_domain,
            legit_score,
            long_subject,
            unknown_dom,
        ]
    )


# Implementación de la neurona artificial (Perceptrón con Sigmoide / Regresión Logística)
class Perceptron:

    def __init__(self, n_features, learning_rate=0.1, epochs=50):
        self.lr = learning_rate
        self.epochs = epochs
        # Inicializamos pesos en cero y el sesgo (bias)
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

    def _sigmoid(self, z):
        # Función de activación para mapear a probabilidades [0, 1]
        return 1.0 / (1.0 + np.exp(-z))

    def _forward(self, x):
        # Combinación lineal combinada con la activación
        return self._sigmoid(np.dot(x, self.weights) + self.bias)

    def train(self, X, y):
        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for xi, yi in zip(X, y):
                pred = self._forward(xi)
                error = yi - pred

                # Gradiente descendente estocástico para actualizar parámetros
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

                # Entropía cruzada binaria (Binary Cross-Entropy Loss)
                # Sumamos 1e-9 para evitar indeterminaciones matemáticas por log(0)
                epoch_loss += -(
                    yi * np.log(pred + 1e-9)
                    + (1 - yi) * np.log(1 - pred + 1e-9)
                )

            self.loss_history.append(epoch_loss / len(y))

    def predict_proba(self, x):
        return self._forward(x)

    def predict(self, x, threshold=0.5):
        # Clasificación binaria dura basada en el umbral
        if self.predict_proba(x) >= threshold:
            return 1
        else:
            return 0

    def accuracy(self, X, y):
        # Evaluación del desempeño del modelo
        preds = np.array([self.predict(xi) for xi in X])
        return np.mean(preds == y)


# Función para cargar el dataset desde el archivo CSV
def load_data(file_path):
    emails_list = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Casteamos el label a entero ya que el CSV lee strings
            row["label"] = int(row["label"])
            emails_list.append(row)
    return emails_list


# Hilo principal de ejecución
def main():
    # Detectamos la carpeta exacta donde está guardado este script (percep.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Unimos esa ruta con el nombre del archivo para generar la ruta absoluta perfecta
    path_csv = os.path.join(script_dir, "correos.csv")

    emails_dataset = load_data(path_csv)

    # El resto de tu función main se queda exactamente igual...
    X = np.array([extract_features(e) for e in emails_dataset])
    y = np.array([e["label"] for e in emails_dataset])

    num_features = X.shape[1]
    modelo = Perceptron(
        n_features=num_features, learning_rate=0.1, epochs=50
    )

    modelo.train(X, y)
    exactitud = modelo.accuracy(X, y)

    print("============================================================")
    print(f" PERCEPTRÓN CLASIFICADOR - DATASET: {path_csv}")
    print("============================================================")
    print(
        f" Parámetros -> Épocas: {modelo.epochs} | LR: {modelo.lr} | Accuracy: {exactitud*100:.2f}%"
    )
    print("============================================================\n")

    print(f"{'REMITENTE':<22} {'PREDICCIÓN':<12} {'PROB. SPAM':<12} {'REAL'}")
    print("-" * 60)

    for email in emails_dataset:
        feat = extract_features(email)
        prob = modelo.predict_proba(feat)
        pred = modelo.predict(feat)

        txt_pred = "SPAM" if pred == 1 else "OK (Limpio)"
        txt_real = "spam" if email["label"] == 1 else "legitimo"
        check = "✓" if pred == email["label"] else "✗"

        print(
            f"{email['sender'][:20]:<22} {txt_pred:<12} {prob*100:>8.2f}%    {txt_real} {check}"
        )


if __name__ == "__main__":
    main()