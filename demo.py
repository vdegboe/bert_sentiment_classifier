import torch
import gradio as gr
from transformers import BertTokenizer
from model import MyBertModel
from utils import device, set_seed
import os


set_seed(61)

model = MyBertModel(num_classes=5).to(device)
checkpoint_dir = 'checkpoint'
best_model_name = 'best_model_epoch3_f10.8729.pt'
best_model_path = os.path.join(checkpoint_dir, best_model_name)

try:
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✓ Modèle entraîné chargé depuis {best_model_path}")
    print("  Epoch 3 | Accuracy: 87.28% | F1-Score: 0.8729")
    model_loaded = True
except FileNotFoundError:
    print(f"⚠ Attention: {best_model_path} non trouvé.")
    print("  Utilisation du modèle BERT pré-entraîné (non fine-tuné).")
    model_loaded = False

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentiment_map = {0: 'Extremely Negative', 1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Extremely Positive'}


def classify_sentiment(text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    sentiment_label = sentiment_map[pred]

    return sentiment_label, {
        'Extremely Negative': float(probs[0, 0]),
        'Negative': float(probs[0, 1]),
        'Neutral': float(probs[0, 2]),
        'Positive': float(probs[0, 3]),
        'Extremely Positive': float(probs[0, 4])
    }


model_status = "✓ Modèle fine-tuné (Epoch 3, F1=0.8729)" if model_loaded else "⚠ Modèle pré-entraîné (non fine-tuné)"

demo = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(
        label="Texte à analyser",
        placeholder="Entrez un texte en anglais à analyser...",
        lines=3
    ),
    outputs=[
        gr.Label(label="Sentiment Prédit"),
        gr.Label(label="Probabilités par classe")
    ],
    title="Classification de Sentiments avec BERT",
    description=f"""Analysez le sentiment d'un texte avec un modèle BERT fine-tuné sur 41k tweets.

⚠️ **Important:** Veuillez entrer le texte **en anglais** pour des résultats optimaux.

**Statut du modèle:** {model_status}

**Performance (Ensemble de Validation):**
- Précision: 87.28%
- F1-Score: 0.8729

**Classes de sentiments:**
- 😠 Extremely Negative (Très négatif)
- 😞 Negative (Négatif)
- 😐 Neutral (Neutre)
- 😊 Positive (Positif)
- 😍 Extremely Positive (Très positif)""",
    examples=[
        ["I love this movie! It's amazing!"],
        ["This is the worst experience ever"],
        ["The weather is okay today"],
        ["Amazing product, highly recommended!"],
        ["Terrible service, never coming back"]
    ],
    theme=gr.themes.Soft(),
    submit_btn="Analyser",
    clear_btn="Effacer"
)

if __name__ == "__main__":
    demo.launch(share=False)
