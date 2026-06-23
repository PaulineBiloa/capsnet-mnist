import io
import torch
import streamlit as st
from PIL import Image, ImageOps
from torchvision import transforms

from model import CapsNet


@st.cache_resource
def load_model(weights_path="capsnet_best.pth"):
    device = torch.device("cpu")
    model = CapsNet()
    state = torch.load(weights_path, map_location=device)
    # If state is a dict with 'state_dict' key (training script style), handle it
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # Support both plain state_dict and checkpoint formats
    try:
        model.load_state_dict(state)
    except Exception:
        # maybe state dict keys have module. prefix
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k.replace('module.', '')
            new_state[name] = v
        model.load_state_dict(new_state)

    model.to(device)
    model.eval()
    return model


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(image).unsqueeze(0)


def main():
    st.set_page_config(page_title="CapsNet MNIST", layout="centered")
    st.title("CapsNet - Démo MNIST")

    st.markdown("Charge le modèle et faites glisser/déposez une image de chiffre (ou utilisez l'exemple).")

    model = load_model()

    col1, col2 = st.columns(2)

    with col1:
        uploaded = st.file_uploader("Téléverser une image (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
        if uploaded is None:
            st.info("Ou utilisez l'image d'exemple ci-dessous.")
            sample = st.button("Charger l'exemple MNIST")
            if sample:
                # load a small white digit on black background from PIL generated sample (7)
                import numpy as np
                arr = np.zeros((28, 28), dtype='uint8')
                # draw a simple digit-like blob (approx)
                arr[6:22,10:18] = 255
                img = Image.fromarray(arr)
                uploaded = io.BytesIO()
                img.save(uploaded, format='PNG')
                uploaded.seek(0)

    if uploaded is not None:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, caption='Image téléchargée', use_column_width=True)

        # Preprocess and predict
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            v, classes, reconstructions = model(input_tensor, None)

        # Convert classes (norms) to probabilities via softmax
        probs = torch.softmax(classes, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())

        with col2:
            st.subheader('Prédiction')
            st.write(f"Chiffre prédit: **{pred}**")
            st.write('Probabilités:')
            for i, p in enumerate(probs):
                st.write(f"{i}: {p:.3f}")

            st.subheader('Reconstruction')
            recon = reconstructions.cpu().numpy()[0, 0]
            # recon est dans [0,1]
            st.image(recon, width=200, clamp=True)


if __name__ == '__main__':
    main()
