import numpy as np
import re

# ─────────────────────────────────────────────
#  DATASET DE CORREOS
# ─────────────────────────────────────────────

EMAILS = [
    {"sender": "Banco Nacional",    "from": "noreply@banconacional.com.co", "subject": "Tu estado de cuenta de abril está listo",           "body": "Hola, tu extracto mensual ya está disponible. Ingresa a tu banca en línea para consultarlo.", "label": 0},
    {"sender": "OFERTA URGENTE!!!", "from": "promo@gana-millones.xyz",      "subject": "¡¡¡GANASTE $50,000,000 GRATIS!!! RECLAMA YA",       "body": "Felicitaciones! Has sido seleccionado. Haz clic AHORA para reclamar tu premio. Oferta expira HOY.", "label": 1},
    {"sender": "Carlos Martínez",   "from": "c.martinez@empresa.co",        "subject": "Reunión de equipo mañana a las 9am",                 "body": "Hola equipo, confirmen asistencia para la reunión de planeación del trimestre. Gracias.", "label": 0},
    {"sender": "FARMACIA DESC.",    "from": "ventas@pharmapills.net",       "subject": "Compra Viagra, Cialis SIN RECETA — 90% descuento",   "body": "Medicamentos de marca sin fórmula médica. Envío discreto. Paga con crypto.", "label": 1},
    {"sender": "GitHub",            "from": "noreply@github.com",           "subject": "[PR #42] Feature: dark mode implementado",          "body": "rodrigo_dev abrió un pull request en tu repositorio. Revisa los cambios propuestos.", "label": 0},
    {"sender": "BITCOIN GRATIS $$$","from": "cryptoking@free-crypto.io",    "subject": "Gana 0.5 BTC hoy GRATIS sin inversión!",            "body": "Sistema automático de multiplicación de criptomonedas. Solo necesitas tu wallet. URGENTE actúa hoy.", "label": 1},
    {"sender": "Ana Sofía López",   "from": "a.lopez@universidad.edu",      "subject": "Re: Entrega del proyecto de machine learning",      "body": "Hola, adjunto la versión final del informe. Quedé muy contenta con los resultados del perceptrón.", "label": 0},
    {"sender": "HERBAL SLIM PRO",   "from": "slim@dietpills-best.biz",      "subject": "Pierde 20 KILOS en 1 semana garantizado",           "body": "Pastillas naturales milagrosas. Oferta limitada. Compra 2 lleva 5 GRATIS.", "label": 1},
    {"sender": "Spotify",           "from": "no-reply@spotify.com",         "subject": "Tu resumen musical de la semana está aquí",         "body": "Descubre tus artistas más escuchados esta semana y crea una playlist personalizada.", "label": 0},
    {"sender": "INVERSIONES",       "from": "profits@invest-now.xyz",       "subject": "Duplica tu dinero en 48 horas — Método SECRETO",    "body": "Inversión garantizada. Miles de clientes satisfechos. Mínimo $100.000 COP. No pierdas esta oportunidad ÚNICA.", "label": 1},
]

URGENT_WORDS = ["urgente","ya","hoy","ahora","expira","limitada","gratis","gana","ganaste",
                "garantizado","secreto","único","clic","click","reclama","actúa","perderás","aproveche"]

MONEY_WORDS  = ["dinero","pesos","bitcoin","btc","crypto","inversión","premio","millones",
                "descuento","oferta","gratis","gana","ganaste","duplica","profits"]

LEGIT_WORDS  = ["reunión","proyecto","informe","equipo","estado","extracto","repositorio",
                "pull","request","resumen","playlist","trimestre","asistencia","planeación"]

SUSPICIOUS_DOMAINS = [".xyz",".biz",".io","free-","crypto","slim","pills","invest","gana","promo"]

KNOWN_DOMAINS = ["gmail.com","empresa.co","universidad.edu","github.com",
                 "spotify.com","banconacional.com.co"]

FEATURE_NAMES = [
    "Palabras en MAYÚSCULAS",
    "Signos de exclamación",
    "Palabras urgentes",
    "Palabras de dinero/premio",
    "Links/dominio sospechoso",
    "Palabras legítimas",
    "Asunto muy largo (>50 chars)",
    "Dominio desconocido",
]

def extract_features(email: dict) -> np.ndarray:
    text_lower = (email["subject"] + " " + email["body"] + " " + email["sender"]).lower()
    text_raw   =  email["subject"] + " " + email["body"]
    sender_email = email.get("from", "")

    caps_ratio   = min(len(re.findall(r'[A-ZÁÉÍÓÚÑÜ]{2,}', text_raw)) / 3, 1.0)
    excl_count   = min(text_raw.count("!") / 4, 1.0)
    urgent_score = min(sum(w in text_lower for w in URGENT_WORDS) / 4, 1.0)
    money_score  = min(sum(w in text_lower for w in MONEY_WORDS)  / 4, 1.0)
    susp_domain  = min(sum(s in sender_email for s in SUSPICIOUS_DOMAINS) / 2, 1.0)
    legit_score  = min(sum(w in text_lower for w in LEGIT_WORDS)  / 3, 1.0)
    long_subject = float(len(email["subject"]) > 50)
    unknown_dom  = float(not any(d in sender_email for d in KNOWN_DOMAINS))

    return np.array([caps_ratio, excl_count, urgent_score, money_score,
                     susp_domain, legit_score, long_subject, unknown_dom])

# ─────────────────────────────────────────────
#  PERCEPTRÓN
# ─────────────────────────────────────────────

class Perceptron:
    """
    Perceptrón de una sola capa con función de activación sigmoide.

    Regla de aprendizaje (descenso de gradiente estocástico):
        error  = y_real - ŷ
        wᵢ    += η · error · xᵢ
        bias  += η · error
    """

    def __init__(self, n_features: int, learning_rate: float = 0.1, epochs: int = 50):
        self.lr      = learning_rate
        self.epochs  = epochs
        self.weights = np.zeros(n_features)
        self.bias    = 0.0
        self.loss_history: list[float] = []

    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    def _forward(self, x: np.ndarray) -> float:
        return self._sigmoid(np.dot(x, self.weights) + self.bias)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entrena el perceptrón con los datos X (n_samples × n_features) e y (n_samples,)."""
        self.loss_history.clear()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            for xi, yi in zip(X, y):
                pred  = self._forward(xi)
                error = yi - pred

                # actualización de pesos
                self.weights += self.lr * error * xi
                self.bias    += self.lr * error

                # log-loss (entropía cruzada binaria)
                epoch_loss += -(yi * np.log(pred + 1e-9) +
                                (1 - yi) * np.log(1 - pred + 1e-9))

            self.loss_history.append(epoch_loss / len(y))

    def predict_proba(self, x: np.ndarray) -> float:
        """Devuelve la probabilidad de que el correo sea spam."""
        return self._forward(x)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> int:
        """Devuelve 1 (spam) o 0 (legítimo)."""
        return int(self.predict_proba(x) >= threshold)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = [self.predict(x) for x in X]
        return np.mean(np.array(preds) == y)

def main():
    # 1. Construir dataset
    X = np.array([extract_features(e) for e in EMAILS])
    y = np.array([e["label"] for e in EMAILS])

    # 2. Entrenar
    model = Perceptron(n_features=X.shape[1], learning_rate=0.1, epochs=50)
    model.train(X, y)

    # 3. Evaluar
    acc = model.accuracy(X, y)
    print(f"\n{'='*60}")
    print(f"  PERCEPTRÓN ANTI-SPAM — RESULTADOS")
    print(f"{'='*60}")
    print(f"  Épocas: {model.epochs}  |  η: {model.lr}  |  Precisión: {acc*100:.0f}%")
    print(f"{'='*60}\n")

    # 4. Clasificar cada correo
    print(f"  {'REMITENTE':<22}  {'RESULTADO':<12}  {'PROB.':<8}  {'REAL'}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*8}  {'-'*10}")

    for email in EMAILS:
        features = extract_features(email)
        prob     = model.predict_proba(features)
        pred     = model.predict(features)
        verdict  = "SPAM" if pred == 1 else "Legítimo"
        real     = "spam"   if email["label"] == 1 else "legítimo"
        correct  = "✓" if pred == email["label"] else "✗"
        print(f"  {email['sender'][:22]:<22}  {verdict:<12}  {prob*100:>5.1f}%   {real} {correct}")

    # 5. Pesos aprendidos
    print(f"\n  PESOS APRENDIDOS:")
    for name, w in zip(FEATURE_NAMES, model.weights):
        bar = "█" * int(abs(w) * 20)
        sign = "+" if w >= 0 else "-"
        print(f"    {sign}{bar:<20}  {w:+.4f}  {name}")
    print(f"    bias: {model.bias:+.4f}\n")

    # 6. Clasificar un correo nuevo
    nuevo = {
        "sender": "Lotería Nacional",
        "from":   "loteria@gana-facil.xyz",
        "subject":"GANASTE el sorteo especial — RESPONDE AHORA",
        "body":   "Has sido elegido ganador. Envía tus datos bancarios para recibir $5.000.000 gratis. Oferta expira hoy.",
    }
    feats_nuevo = extract_features(nuevo)
    prob_nuevo  = model.predict_proba(feats_nuevo)
    pred_nuevo  = model.predict(feats_nuevo)
    print(f"{'='*60}")
    print(f"  CORREO NUEVO:")
    print(f"    Asunto : {nuevo['subject']}")
    print(f"    Veredicto: {'SPAM' if pred_nuevo else 'Legítimo'} (probabilidad: {prob_nuevo*100:.1f}%)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()