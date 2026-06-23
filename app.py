"""
app.py — CapsNet Dashboard
Capsule Network for MNIST Classification and Reconstruction
Based on: Sabour, Frosst & Hinton (2017) — "Dynamic Routing Between Capsules"
"""

import io
import sys
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Project imports — model.py / train.py / utils.py
# ---------------------------------------------------------------------------
from model import CapsNet, CapsLoss
from train import train, test
from utils import set_seed, get_dataloaders

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CapsNet Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    section[data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1e2130;
    }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

    .main .block-container {
        background: #0d1117;
        padding: 2rem 3rem;
        max-width: 1300px;
    }

    .kpi-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #58a6ff;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.4rem;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #e6edf3;
        border-bottom: 2px solid #21262d;
        padding-bottom: 0.6rem;
        margin-bottom: 1.4rem;
    }
    .section-sub {
        font-size: 0.95rem;
        color: #8b949e;
        margin-bottom: 1.6rem;
        line-height: 1.6;
    }

    .badge {
        display: inline-block;
        background: #1f2937;
        color: #60a5fa;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        margin: 0.15rem;
    }

    .arch-block {
        background: #161b22;
        border: 1px solid #21262d;
        border-left: 3px solid #58a6ff;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .arch-block h4 {
        color: #58a6ff;
        margin: 0 0 0.4rem 0;
        font-size: 0.95rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .arch-block p { color: #8b949e; margin: 0; font-size: 0.88rem; line-height: 1.5; }

    .info-box {
        background: #0d2137;
        border: 1px solid #1d4ed8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        color: #93c5fd;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .stButton > button {
        background: #1d4ed8;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 1.4rem;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #2563eb; }

    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #8b949e !important;
        font-size: 0.88rem;
    }
    div[data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-family: 'JetBrains Mono', monospace;
    }
    h1, h2, h3 { color: #e6edf3 !important; }
    p, li { color: #c9d1d9; }
    hr { border-color: #21262d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "training_history": None,
    "model": None,
    "eval_data": None,       # dict: y_true, y_pred, capsule_vectors, images
    "train_loader": None,
    "test_loader": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLOTLY_BASE = dict(
    plot_bgcolor="#0d1117",
    paper_bgcolor="#161b22",
    font=dict(color="#8b949e", family="Inter"),
)
AXIS_STYLE = dict(gridcolor="#21262d")


@st.cache_resource(show_spinner="Chargement de MNIST…")
def load_mnist(batch_size: int = 128):
    """Charge MNIST via get_dataloaders (utils.py) et met en cache."""
    return get_dataloaders(batch_size)


@st.cache_data(show_spinner="Chargement des images brutes MNIST…")
def load_mnist_raw():
    """Charge le dataset MNIST sans normalisation, pour la galerie visuelle."""
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tf)
    return train_ds, test_ds


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor_to_numpy_img(t: torch.Tensor) -> np.ndarray:
    """(1, 28, 28) float tensor → (28, 28) numpy [0, 1]."""
    return t.squeeze().cpu().numpy()


@torch.no_grad()
def run_full_eval(model: torch.nn.Module, test_loader, device, n_batches: int = 20):
    """
    Parcourt n_batches du test loader.
    Retourne y_true, y_pred, capsule_vectors (N, 10, 16), images (N, 28, 28).
    """
    model.eval()
    all_true, all_pred, all_caps, all_imgs = [], [], [], []

    for i, (data, target) in enumerate(test_loader):
        if i >= n_batches:
            break
        data = data.to(device)
        v, classes, _ = model(data)          # v: (B, 10, 16)
        preds = classes.max(dim=1)[1]

        all_true.append(target.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
        all_caps.append(v.cpu().numpy())
        all_imgs.append(data.cpu().numpy())

    return {
        "y_true": np.concatenate(all_true),
        "y_pred": np.concatenate(all_pred),
        "caps":   np.concatenate(all_caps),    # (N, 10, 16)
        "images": np.concatenate(all_imgs),    # (N, 1, 28, 28)
    }


def plotly_heatmap_img(img: np.ndarray, title: str = "") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=img, colorscale="gray", showscale=False, zmin=0, zmax=1))
    fig.update_layout(
        **PLOTLY_BASE,
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        height=200,
        title=dict(text=title, font=dict(color="#8b949e", size=11)),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def styled_kpi(value: str, label: str, extra_style: str = "") -> str:
    return (
        f'<div class="kpi-card" style="{extra_style}">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:1rem 0 1.5rem 0;border-bottom:1px solid #21262d;margin-bottom:1.2rem;">
            <div style="font-size:1.15rem;font-weight:700;color:#e6edf3;letter-spacing:0.03em;">
                CapsNet Dashboard
            </div>
            <div style="font-size:0.75rem;color:#58a6ff;margin-top:0.2rem;
                        font-family:'JetBrains Mono',monospace;">
                MNIST &nbsp;•&nbsp; Dynamic Routing
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        options=[
            "Accueil",
            "Dataset MNIST",
            "Architecture CapsNet",
            "Entrainement",
            "Evaluation",
            "Predictions",
            "Reconstruction",
            "Visualisations avancees",
            "A propos",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    model_status = "charge" if st.session_state.model is not None else "non charge"
    model_color  = "#3fb950" if st.session_state.model is not None else "#f85149"
    st.markdown(
        f"""
        <div style="font-size:0.78rem;color:#8b949e;">
            <b style="color:#c9d1d9;">Environnement</b><br/>
            Device : <span style="color:#58a6ff;font-family:'JetBrains Mono',monospace;">
                     {DEVICE.type.upper()}</span><br/>
            PyTorch : <span style="color:#58a6ff;font-family:'JetBrains Mono',monospace;">
                      {torch.__version__}</span><br/>
            Modele : <span style="color:{model_color};font-family:'JetBrains Mono',monospace;">
                     {model_status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE — ACCUEIL
# ===========================================================================
if page == "Accueil":
    st.markdown('<div class="section-title">Capsule Networks pour MNIST</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "Implementation du modele CapsNet de Sabour, Frosst et Hinton (NeurIPS 2017) pour la "
        "classification et la reconstruction des chiffres manuscrits du dataset MNIST."
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    for col, (val, label) in zip(
        [col1, col2, col3, col4],
        [("99.75%", "Accuracy cible"), ("10", "Classes MNIST"),
         ("16", "Dim. capsules digit"), ("3", "Iterations routing")],
    ):
        with col:
            st.markdown(styled_kpi(val, label), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### Le probleme traite")
        st.markdown(
            """
Les reseaux de neurones convolutifs (CNN) classiques souffrent d'une limite fondamentale :
ils perdent l'information spatiale et positionnelle lors du pooling. Deux images contenant
les memes caracteristiques mais dans des positions differentes peuvent produire la meme
sortie, ce qui rend le modele fragile aux transformations geometriques.

**CapsNet** propose une alternative : remplacer les neurones scalaires par des **capsules**,
des groupes de neurones dont le **vecteur d'activation** encode simultanement la presence
et les proprietes spatiales d'une entite (position, orientation, echelle, deformation).
"""
        )
        st.markdown("### Dynamic Routing")
        st.markdown(
            """
Le mecanisme de **routage dynamique entre capsules** (Sabour et al., 2017) permet aux
capsules de niveau inferieur de "voter" pour les capsules de niveau superieur. L'accord
entre votes est mesure iterativement, produisant une representation hierarchique coherente
sans supervision explicite.
"""
        )
        st.markdown("### Avantages vis-a-vis des CNN")
        for adv in [
            "Equivariance aux transformations geometriques",
            "Preservation de la structure spatiale",
            "Reconstruction interpretable par le decodeur",
            "Meilleure generalisation sous faible volume de donnees",
            "Representations hierarchiques explicites",
        ]:
            st.markdown(f"- {adv}")

    with col_right:
        st.markdown("### Pipeline CapsNet")
        steps = [
            "Image MNIST\n(28 x 28)",
            "Convolution\n256 filtres, 9x9",
            "Primary Capsules\n32 x 6 x 6, dim=8",
            "Digit Capsules\n10 capsules, dim=16",
            "Classification\n(norme du vecteur)",
            "Reconstruction\n(decodeur FC)",
        ]
        colors_flow = ["#1d4ed8", "#1e40af", "#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa"]
        y_pos = list(range(len(steps), 0, -1))

        fig_flow = go.Figure()
        for i, (step, color, y) in enumerate(zip(steps, colors_flow, y_pos)):
            fig_flow.add_shape(
                type="rect", x0=0.1, x1=0.9, y0=y - 0.35, y1=y + 0.35,
                line=dict(color="#58a6ff", width=1.5), fillcolor=color, opacity=0.9,
            )
            fig_flow.add_annotation(
                x=0.5, y=y, text=step.replace("\n", "<br>"), showarrow=False,
                font=dict(color="#e6edf3", size=11, family="JetBrains Mono"), align="center",
            )
            if i < len(steps) - 1:
                fig_flow.add_annotation(
                    x=0.5, y=y - 0.35, ax=0.5, ay=y_pos[i + 1] + 0.35,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowcolor="#58a6ff", arrowwidth=2,
                )
        fig_flow.update_layout(
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0.4, len(steps) + 0.5]),
            **PLOTLY_BASE, margin=dict(l=10, r=10, t=10, b=10), height=480,
        )
        st.plotly_chart(fig_flow,  width='stretch')

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box"><b>Reference :</b> Sara Sabour, Nicholas Frosst, Geoffrey E. Hinton — '
        "<i>Dynamic Routing Between Capsules</i>, NeurIPS 2017. arXiv:1710.09829</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE — DATASET MNIST
# ===========================================================================
elif page == "Dataset MNIST":
    st.markdown('<div class="section-title">Dataset MNIST</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "Le Modified National Institute of Standards and Technology database contient 70 000 images "
        "de chiffres manuscrits en niveaux de gris (28x28 pixels), partitionnees en 60 000 exemples "
        "d'entrainement et 10 000 exemples de test."
        "</div>",
        unsafe_allow_html=True,
    )

    # Chargement reel
    train_ds, test_ds = load_mnist_raw()
    targets_train = np.array(train_ds.targets)

    col1, col2, col3, col4 = st.columns(4)
    for col, (val, label) in zip(
        [col1, col2, col3, col4],
        [(f"{len(train_ds) + len(test_ds):,}", "Images totales"),
         (f"{len(train_ds):,}", "Entrainement"),
         (f"{len(test_ds):,}", "Test"),
         ("10", "Classes")],
    ):
        with col:
            st.markdown(styled_kpi(val, label), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Distribution des classes (entrainement)")
        class_counts = [(targets_train == i).sum() for i in range(10)]
        fig_dist = go.Figure(go.Bar(
            x=[str(i) for i in range(10)],
            y=class_counts,
            marker_color="#1d4ed8",
            marker_line_color="#58a6ff",
            marker_line_width=1.2,
            text=class_counts,
            textposition="outside",
            textfont=dict(color="#8b949e", size=10),
        ))
        fig_dist.update_layout(
            xaxis_title="Classe (chiffre)", yaxis_title="Nombre d'exemples",
            **PLOTLY_BASE,
            xaxis=AXIS_STYLE, yaxis={**AXIS_STYLE, "range": [0, max(class_counts) * 1.12]},
            margin=dict(l=10, r=10, t=10, b=10), height=320,
        )
        st.plotly_chart(fig_dist,  width='stretch')

    with col_right:
        st.markdown("### Proprietes du dataset")
        props = {
            "Resolution":       "28 x 28 pixels",
            "Canaux":           "1 (niveaux de gris)",
            "Normalisation":    "mean=0.1307 / std=0.3081",
            "Classe minimale":  f"Chiffre {int(np.argmin(class_counts))} ({min(class_counts):,} ex.)",
            "Classe maximale":  f"Chiffre {int(np.argmax(class_counts))} ({max(class_counts):,} ex.)",
            "Desequilibre":     f"{(max(class_counts)-min(class_counts))/max(class_counts)*100:.1f} %",
        }
        for k, v in props.items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:0.5rem 0;'
                f'border-bottom:1px solid #21262d;">'
                f'<span style="color:#8b949e;font-size:0.88rem;">{k}</span>'
                f'<span style="color:#e6edf3;font-family:JetBrains Mono,monospace;font-size:0.85rem;">{v}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # Galerie reelle
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Galerie d'exemples (images reelles MNIST)")

    n_per_class = st.slider("Exemples par classe", 1, 5, 2)

    fig_gallery = make_subplots(
        rows=n_per_class, cols=10,
        horizontal_spacing=0.008, vertical_spacing=0.04,
    )
    for digit in range(10):
        idxs = np.where(targets_train == digit)[0][:n_per_class]
        for row_idx, sample_idx in enumerate(idxs):
            img_np = train_ds[int(sample_idx)][0].squeeze().numpy()
            fig_gallery.add_trace(
                go.Heatmap(z=img_np, colorscale="gray", showscale=False, zmin=0, zmax=1),
                row=row_idx + 1, col=digit + 1,
            )
            if row_idx == 0:
                fig_gallery.update_xaxes(
                    title_text=str(digit),
                    title_font=dict(color="#58a6ff", size=10),
                    showticklabels=False, row=1, col=digit + 1,
                )
            else:
                fig_gallery.update_xaxes(showticklabels=False, row=row_idx + 1, col=digit + 1)
            fig_gallery.update_yaxes(showticklabels=False, row=row_idx + 1, col=digit + 1)

    fig_gallery.update_layout(
        **PLOTLY_BASE,
        margin=dict(l=5, r=5, t=10, b=5),
        height=120 * n_per_class,
    )
    st.plotly_chart(fig_gallery,  width='stretch')


# ===========================================================================
# PAGE — ARCHITECTURE CAPSNET
# ===========================================================================
elif page == "Architecture CapsNet":
    st.markdown('<div class="section-title">Architecture CapsNet</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "Description complete des couches du reseau a capsules tel qu'implemente dans model.py, "
        "conformement a l'architecture originale de Sabour et al. (2017)."
        "</div>",
        unsafe_allow_html=True,
    )

    layers = [
        {"name": "Entree",
         "code": "Input(1, 28, 28)",
         "desc": "Image MNIST en niveaux de gris, 28x28 pixels. Normalisation : mean=0.1307, std=0.3081."},
        {"name": "Conv2d",
         "code": "Conv2d(1→256, kernel=9, stride=1) → ReLU → (256, 20, 20)",
         "desc": "256 filtres 9x9 extraient les caracteristiques locales de bas niveau."},
        {"name": "PrimaryCaps",
         "code": "Conv2d(256→256, kernel=9, stride=2) → reshape → squash → (1152, 8)",
         "desc": "32 groupes de capsules, chacun produisant un vecteur de dimension 8. "
                 "Sortie aplatie : 1152 capsules primaires. Activation : squash."},
        {"name": "DigitCaps",
         "code": "W(1152, 10, 16, 8) · u → routing(3 iter.) → (10, 16)",
         "desc": "10 capsules de dimension 16, une par classe. Routage dynamique sur 3 iterations. "
                 "La norme du vecteur encode la probabilite d'existence de la classe."},
        {"name": "Classification",
         "code": "||v_j|| → argmax → classe predite",
         "desc": "La capsule ayant la plus grande norme determine la classe predite."},
        {"name": "Decoder",
         "code": "FC(160→512) → ReLU → FC(512→1024) → ReLU → FC(1024→784) → Sigmoid → (1,28,28)",
         "desc": "Reconstruction de l'image a partir du vecteur de la capsule active "
                 "(vraie classe a l'entrainement, classe predite en inference). Perte MSE ponderee par 0.0005."},
    ]

    for layer in layers:
        st.markdown(
            f'<div class="arch-block"><h4>{layer["name"]}</h4>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#3fb950;'
            f'margin-bottom:0.5rem;">{layer["code"]}</div>'
            f'<p>{layer["desc"]}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Fonction Squash")
        st.markdown(
            r"""
$$\mathbf{v}_j = \frac{\|\mathbf{s}_j\|^2}{1+\|\mathbf{s}_j\|^2} \cdot \frac{\mathbf{s}_j}{\|\mathbf{s}_j\|}$$

- $\|\mathbf{s}_j\|$ grand → norme proche de 1 (capsule active)
- $\|\mathbf{s}_j\|$ petit → norme proche de 0 (capsule inactive)
"""
        )

    with col_right:
        st.markdown("### Margin Loss")
        st.markdown(
            r"""
$$L_k = T_k\max(0,\,m^+-\|\mathbf{v}_k\|)^2 + \lambda(1-T_k)\max(0,\|\mathbf{v}_k\|-m^-)^2$$

$m^+=0.9$, $m^-=0.1$, $\lambda=0.5$.
Perte totale = margin loss + 0.0005 × MSE reconstruction.
"""
        )

    # Statistiques reelles du modele
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Statistiques du modele")
    _m = CapsNet()
    n_params = count_parameters(_m)
    mem_mb = n_params * 4 / 1e6  # FP32

    col1, col2, col3 = st.columns(3)
    for col, (val, label) in zip(
        [col1, col2, col3],
        [(f"{n_params:,}", "Parametres entrainables"),
         (f"{mem_mb:.1f} MB", "Memoire FP32"),
         ("3", "Iterations routing")],
    ):
        with col:
            st.markdown(styled_kpi(val, label), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    _buf = io.StringIO()
    _old = sys.stdout; sys.stdout = _buf
    print(_m)
    sys.stdout = _old
    st.code(_buf.getvalue(), language="text")


# ===========================================================================
# PAGE — ENTRAINEMENT
# ===========================================================================
elif page == "Entrainement":
    st.markdown('<div class="section-title">Entrainement</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "Configurez les hyperparametres et lancez l'entrainement. Les metriques sont mises "
        "a jour apres chaque epoque."
        "</div>",
        unsafe_allow_html=True,
    )

    col_params, col_run = st.columns([1, 1.4])

    with col_params:
        st.markdown("### Hyperparametres")
        batch_size    = st.selectbox("Batch size", [32, 64, 128, 256], index=2)
        lr            = st.number_input("Learning rate", value=0.001, step=0.0001, format="%.4f")
        epochs        = st.slider("Nombre d'epoques", 1, 50, 10)
        routing_iter  = st.slider("Iterations routing dynamique", 1, 5, 3)
        optimizer_name = st.selectbox("Optimiseur", ["Adam", "SGD"])
        seed          = st.number_input("Seed", value=42, step=1)

    with col_run:
        st.markdown("### Lancement")
        start = st.button("Lancer l'entrainement",  width='stretch')

        if start:
            set_seed(int(seed))
            train_loader, test_loader = get_dataloaders(batch_size)
            st.session_state.train_loader = train_loader
            st.session_state.test_loader  = test_loader

            model = CapsNet(num_routing=routing_iter).to(DEVICE)
            criterion = CapsLoss()

            if optimizer_name == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            progress_bar    = st.progress(0)
            status_text     = st.empty()
            chart_ph        = st.empty()

            train_losses, val_accs = [], []

            for epoch in range(1, epochs + 1):
                t0 = time.time()

                # --- Entrainement sur une epoque ---
                model.train()
                epoch_loss = 0.0
                n_batches  = 0
                for data, target in train_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    target_ohe = torch.eye(10, device=DEVICE)[target]
                    optimizer.zero_grad()
                    _, classes, reconstructions = model(data, target_ohe)
                    loss = criterion(data, target_ohe, classes, reconstructions)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches  += 1
                avg_loss = epoch_loss / n_batches

                # --- Evaluation rapide ---
                acc = test(model, test_loader, criterion, DEVICE)

                train_losses.append(avg_loss)
                val_accs.append(acc)

                elapsed = time.time() - t0
                eta     = elapsed * (epochs - epoch)

                progress_bar.progress(epoch / epochs)
                status_text.markdown(
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:0.85rem;color:#8b949e;">'
                    f"Epoque {epoch}/{epochs} &nbsp;|&nbsp; "
                    f"Loss : <span style='color:#58a6ff'>{avg_loss:.4f}</span> &nbsp;|&nbsp; "
                    f"Accuracy : <span style='color:#3fb950'>{acc:.2f}%</span> &nbsp;|&nbsp; "
                    f"ETA : <span style='color:#d29922'>{eta:.0f}s</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                epochs_range = list(range(1, epoch + 1))
                fig_live = make_subplots(rows=1, cols=2, subplot_titles=["Loss (train)", "Accuracy % (test)"])
                fig_live.add_trace(
                    go.Scatter(x=epochs_range, y=train_losses, name="Loss",
                               line=dict(color="#58a6ff")), row=1, col=1)
                fig_live.add_trace(
                    go.Scatter(x=epochs_range, y=val_accs, name="Accuracy",
                               line=dict(color="#3fb950"), showlegend=False), row=1, col=2)
                fig_live.update_layout(
                    **PLOTLY_BASE,
                    margin=dict(l=10, r=10, t=40, b=10), height=300,
                    legend=dict(bgcolor="#161b22", bordercolor="#21262d"),
                )
                for r, c in [(1, 1), (1, 2)]:
                    fig_live.update_xaxes(gridcolor="#21262d", row=r, col=c)
                    fig_live.update_yaxes(gridcolor="#21262d", row=r, col=c)
                chart_ph.plotly_chart(fig_live,  width='stretch')

            # Sauvegarde
            st.session_state.model = model
            st.session_state.training_history = {
                "train_losses": train_losses,
                "val_accs":     val_accs,
            }
            st.session_state.eval_data = None   # invalider le cache d'eval

            torch.save(model.state_dict(), "capsnet_best.pth")
            st.success(
                f"Entrainement termine — meilleure accuracy : {max(val_accs):.2f}% "
                f"— modele sauvegarde dans capsnet_best.pth"
            )

    # Charger un checkpoint existant
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Charger un checkpoint")
    ckpt_path = st.text_input("Chemin vers un fichier .pth", value="capsnet_best.pth")
    ckpt_routing = st.number_input("Iterations routing du checkpoint", value=3, step=1)
    if st.button("Charger le modele"):
        try:
            m = CapsNet(num_routing=int(ckpt_routing)).to(DEVICE)
            m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            st.session_state.model = m
            st.session_state.eval_data = None
            st.success(f"Modele charge depuis {ckpt_path}")
        except Exception as e:
            st.error(f"Echec du chargement : {e}")


# ===========================================================================
# PAGE — EVALUATION
# ===========================================================================
elif page == "Evaluation":
    st.markdown('<div class="section-title">Evaluation du modele</div>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("Aucun modele en memoire. Lancez un entrainement ou chargez un checkpoint.")
        st.stop()

    model = st.session_state.model

    # Chargement du test loader si absent
    if st.session_state.test_loader is None:
        _, test_loader = get_dataloaders(128)
        st.session_state.test_loader = test_loader
    test_loader = st.session_state.test_loader

    # Inference (avec cache dans session_state)
    if st.session_state.eval_data is None:
        with st.spinner("Calcul des predictions sur le test set…"):
            st.session_state.eval_data = run_full_eval(model, test_loader, DEVICE, n_batches=80)

    ed = st.session_state.eval_data
    y_true, y_pred = ed["y_true"], ed["y_pred"]

    accuracy  = (y_true == y_pred).mean() * 100
    report    = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    precision = report["weighted avg"]["precision"] * 100
    recall    = report["weighted avg"]["recall"] * 100
    f1        = report["weighted avg"]["f1-score"] * 100

    col1, col2, col3, col4 = st.columns(4)
    for col, (name, val) in zip(
        [col1, col2, col3, col4],
        [("Accuracy", accuracy), ("Precision", precision),
         ("Recall", recall), ("F1-Score", f1)],
    ):
        with col:
            st.markdown(styled_kpi(f"{val:.2f}%", name), unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Matrice de confusion")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(
            cm, color_continuous_scale="Blues", text_auto=True,
            labels=dict(x="Classe predite", y="Classe reelle"),
            x=[str(i) for i in range(10)], y=[str(i) for i in range(10)],
        )
        fig_cm.update_layout(
            **PLOTLY_BASE, coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10), height=400,
        )
        st.plotly_chart(fig_cm,  width='stretch')

    with col_right:
        st.markdown("### Rapport de classification")
        header = ["Classe", "Precision", "Recall", "F1", "Support"]
        rows_html = ""
        for digit in range(10):
            r = report[str(digit)]
            rows_html += "<tr>" + "".join(
                f'<td style="padding:0.4rem 0.6rem;color:#c9d1d9;border-bottom:1px solid #21262d;'
                f'font-family:JetBrains Mono,monospace;">{v}</td>'
                for v in [
                    str(digit),
                    f"{r['precision']*100:.1f}%",
                    f"{r['recall']*100:.1f}%",
                    f"{r['f1-score']*100:.1f}%",
                    int(r["support"]),
                ]
            ) + "</tr>"

        table_html = (
            '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
            "<thead><tr>"
            + "".join(
                f'<th style="padding:0.4rem 0.6rem;color:#58a6ff;border-bottom:1px solid '
                f'#21262d;text-align:left;">{h}</th>'
                for h in header
            )
            + f"</tr></thead><tbody>{rows_html}</tbody></table>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

    # Courbes d'apprentissage si disponibles
    history = st.session_state.training_history
    if history:
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("### Courbes d'apprentissage")
        ep_range = list(range(1, len(history["train_losses"]) + 1))
        fig_curves = make_subplots(rows=1, cols=2, subplot_titles=["Loss (train)", "Accuracy % (test)"])
        fig_curves.add_trace(
            go.Scatter(x=ep_range, y=history["train_losses"], name="Loss",
                       line=dict(color="#58a6ff")), row=1, col=1)
        fig_curves.add_trace(
            go.Scatter(x=ep_range, y=history["val_accs"], name="Accuracy",
                       line=dict(color="#3fb950"), showlegend=False), row=1, col=2)
        fig_curves.update_layout(
            **PLOTLY_BASE, margin=dict(l=10, r=10, t=40, b=10), height=300,
        )
        for r, c in [(1, 1), (1, 2)]:
            fig_curves.update_xaxes(gridcolor="#21262d", row=r, col=c)
            fig_curves.update_yaxes(gridcolor="#21262d", row=r, col=c)
        st.plotly_chart(fig_curves,  width='stretch')


# ===========================================================================
# PAGE — PREDICTIONS
# ===========================================================================
elif page == "Predictions":
    st.markdown('<div class="section-title">Predictions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "Deposez une image de chiffre manuscrit. Le modele applique le meme preprocessing "
        "que lors de l'entrainement et retourne la classe predite avec les probabilites."
        "</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.model is None:
        st.warning("Aucun modele en memoire. Lancez un entrainement ou chargez un checkpoint.")
        st.stop()

    model = st.session_state.model.eval()

    col_upload, col_result = st.columns([1, 1.4])

    with col_upload:
        uploaded_file = st.file_uploader("Charger une image (PNG ou JPG)", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            img_pil = Image.open(uploaded_file).convert("L")
            st.image(img_pil, caption="Image originale", use_column_width=True)

            # Preprocessing identique a l'entrainement
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

            # Affichage de l'image preprocessee (denormalisee pour la visu)
            img_vis = img_tensor.squeeze().cpu().numpy()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
            st.markdown("**Image preprocessee 28x28**")
            fig_pre = plotly_heatmap_img(img_vis)
            st.plotly_chart(fig_pre,  width='stretch')

    with col_result:
        if uploaded_file is not None:
            with torch.no_grad():
                v, classes, _ = model(img_tensor)
                probs = classes.squeeze().cpu().numpy()
                pred_class  = int(probs.argmax())
                confidence  = float(probs[pred_class])

            st.markdown(styled_kpi(str(pred_class), "Classe predite", "margin-bottom:1rem;"),
                        unsafe_allow_html=True)
            st.markdown(styled_kpi(f"{confidence*100:.1f}%", "Confiance (norme capsule)",
                                   "margin-bottom:1.5rem;"),
                        unsafe_allow_html=True)

            st.markdown("### Normes des capsules digit")
            fig_probs = go.Figure(go.Bar(
                x=[str(i) for i in range(10)],
                y=probs.tolist(),
                marker_color=["#1d4ed8" if i != pred_class else "#3fb950" for i in range(10)],
                marker_line_color="#21262d", marker_line_width=1,
                text=[f"{p:.3f}" for p in probs],
                textposition="outside",
                textfont=dict(color="#8b949e", size=10),
            ))
            fig_probs.update_layout(
                xaxis_title="Classe", yaxis_title="Norme du vecteur capsule",
                **PLOTLY_BASE,
                xaxis=AXIS_STYLE,
                yaxis={**AXIS_STYLE, "range": [0, max(probs) * 1.18]},
                margin=dict(l=10, r=10, t=10, b=10), height=280,
            )
            st.plotly_chart(fig_probs,  width='stretch')

            # Reconstruction de l'image predite
            st.markdown("### Image reconstruite par le decodeur")
            with torch.no_grad():
                v_full, _, reconstructions = model(img_tensor)
            recon_np = reconstructions.squeeze().cpu().numpy()
            recon_np = (recon_np - recon_np.min()) / (recon_np.max() - recon_np.min() + 1e-8)
            fig_rec = plotly_heatmap_img(recon_np)
            st.plotly_chart(fig_rec,  width='stretch')

        else:
            st.markdown(
                '<div class="info-box">Deposez une image PNG ou JPG d\'un chiffre manuscrit '
                "pour obtenir une prediction.</div>",
                unsafe_allow_html=True,
            )


# ===========================================================================
# PAGE — RECONSTRUCTION
# ===========================================================================
elif page == "Reconstruction":
    st.markdown('<div class="section-title">Reconstruction par le decodeur</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        "Exploration de la capacite de reconstruction de CapsNet. Modifiez les valeurs "
        "du vecteur capsule pour observer l'effet sur l'image reconstruite."
        "</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.model is None:
        st.warning("Aucun modele en memoire. Lancez un entrainement ou chargez un checkpoint.")
        st.stop()

    model = st.session_state.model.eval()

    # Chargement d'un batch de test reel
    if st.session_state.test_loader is None:
        _, test_loader = get_dataloaders(128)
        st.session_state.test_loader = test_loader

    # Recuperer un exemple reel pour chaque chiffre
    @st.cache_data(show_spinner="Preparation des exemples de test…")
    def get_one_sample_per_class(_model_id):
        """Renvoie un dict {digit: (image_tensor, label)} depuis le test set."""
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        ds = datasets.MNIST(root="./data", train=False, download=True, transform=tf)
        samples = {}
        for img, lbl in ds:
            if lbl not in samples:
                samples[lbl] = img.unsqueeze(0)
            if len(samples) == 10:
                break
        return samples

    samples = get_one_sample_per_class(id(model))

    col_ctrl, col_viz = st.columns([1, 1.4])

    with col_ctrl:
        st.markdown("### Parametres")
        selected_digit  = st.selectbox("Chiffre de reference (test reel)", list(range(10)), index=3)
        dim_to_perturb  = st.slider("Dimension du vecteur capsule a modifier", 0, 15, 0)
        perturbation    = st.slider("Amplitude de la perturbation", -0.5, 0.5, 0.0, step=0.05)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">'
            "Chaque dimension du vecteur capsule de 16 elements encode une propriete "
            "geometrique apprise (inclinaison, epaisseur, largeur…). Deplacer une dimension "
            "revele ce qu'elle represente."
            "</div>",
            unsafe_allow_html=True,
        )

    with col_viz:
        st.markdown("### Original vs Reconstruit")

        img_tensor = samples[selected_digit].to(DEVICE)

        with torch.no_grad():
            v, classes, recon_orig = model(img_tensor)
            # Vecteur capsule de la classe selectionnee
            cap_vec = v[0, selected_digit].clone()  # (16,)

            # Perturbation d'une dimension
            cap_vec_perturbed = cap_vec.clone()
            cap_vec_perturbed[dim_to_perturb] += perturbation

            # Reconstruire avec le vecteur perturbe
            mask = torch.zeros(1, 10, 16, device=DEVICE)
            mask[0, selected_digit] = cap_vec_perturbed
            recon_perturbed = model.decoder(mask.view(1, -1))

        def denorm(t):
            a = t.squeeze().cpu().numpy()
            return np.clip((a - a.min()) / (a.max() - a.min() + 1e-8), 0, 1)

        img_orig_np   = denorm(img_tensor)
        img_recon_np  = denorm(recon_orig)
        img_pert_np   = denorm(recon_perturbed)
        error_np      = np.abs(img_recon_np - img_pert_np)
        mse_val       = float(np.mean((img_recon_np - img_pert_np) ** 2))

        fig_recon = make_subplots(
            rows=1, cols=4,
            subplot_titles=["Original", "Reconstruction", "Reconstruction perturbee", "Erreur"],
        )
        for col_idx, (img_np, cscale) in enumerate(
            [(img_orig_np, "gray"), (img_recon_np, "gray"),
             (img_pert_np, "gray"), (error_np, "Reds")], start=1
        ):
            fig_recon.add_trace(
                go.Heatmap(z=img_np, colorscale=cscale, showscale=False, zmin=0, zmax=1),
                row=1, col=col_idx,
            )
            fig_recon.update_xaxes(visible=False, row=1, col=col_idx)
            fig_recon.update_yaxes(visible=False, row=1, col=col_idx)

        fig_recon.update_layout(
            **PLOTLY_BASE, margin=dict(l=5, r=5, t=40, b=5), height=240,
        )
        st.plotly_chart(fig_recon,  width='stretch')
        st.markdown(styled_kpi(f"{mse_val:.5f}", "MSE (recon vs perturbe)",
                               "display:inline-block;min-width:220px;"),
                    unsafe_allow_html=True)

    # Traversee dimensionnelle
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Traversee dimensionnelle")
    st.markdown(
        f"Perturbations de −0.5 a +0.5 sur la dimension **{dim_to_perturb}** "
        f"du vecteur capsule du chiffre **{selected_digit}**."
    )

    perturb_vals = np.linspace(-0.5, 0.5, 9)
    fig_trav = make_subplots(rows=1, cols=9, horizontal_spacing=0.006, vertical_spacing=0.0)

    with torch.no_grad():
        base_vec = v[0, selected_digit].clone()
        for col_idx, pv in enumerate(perturb_vals, start=1):
            cv = base_vec.clone()
            cv[dim_to_perturb] += float(pv)
            mask_t = torch.zeros(1, 10, 16, device=DEVICE)
            mask_t[0, selected_digit] = cv
            recon_t = model.decoder(mask_t.view(1, -1))
            img_t   = denorm(recon_t)
            fig_trav.add_trace(
                go.Heatmap(z=img_t, colorscale="gray", showscale=False, zmin=0, zmax=1),
                row=1, col=col_idx,
            )
            fig_trav.update_xaxes(
                title_text=f"{pv:+.2f}", showticklabels=False,
                title_font=dict(size=9, color="#8b949e"), row=1, col=col_idx,
            )
            fig_trav.update_yaxes(visible=False, row=1, col=col_idx)

    fig_trav.update_layout(
        **PLOTLY_BASE, margin=dict(l=5, r=5, t=10, b=20), height=140,
    )
    st.plotly_chart(fig_trav,  width='stretch')


# ===========================================================================
# PAGE — VISUALISATIONS AVANCEES
# ===========================================================================
elif page == "Visualisations avancees":
    st.markdown('<div class="section-title">Visualisations avancees</div>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("Aucun modele en memoire. Lancez un entrainement ou chargez un checkpoint.")
        st.stop()

    model = st.session_state.model

    if st.session_state.test_loader is None:
        _, test_loader = get_dataloaders(128)
        st.session_state.test_loader = test_loader

    # Inference (cache session)
    if st.session_state.eval_data is None:
        with st.spinner("Calcul des representations sur le test set…"):
            st.session_state.eval_data = run_full_eval(
                model, st.session_state.test_loader, DEVICE, n_batches=40
            )

    ed     = st.session_state.eval_data
    y_true = ed["y_true"]
    y_pred = ed["y_pred"]
    caps   = ed["caps"]    # (N, 10, 16)
    images = ed["images"]  # (N, 1, 28, 28)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Activations des capsules",
        "Heatmap des poids",
        "Vecteurs capsules (PCA)",
        "Analyse des erreurs",
    ])

    # ------------------------------------------------------------------
    with tab1:
        st.markdown("#### Distribution des activations par capsule de classe")
        st.markdown(
            "Normes des 10 vecteurs capsules digit pour un exemple du test set. "
            "Un modele converge produit une capsule dominante nettement au-dessus des autres."
        )
        digit_sel = st.selectbox("Chiffre a visualiser", list(range(10)), key="tab1_digit")

        # Chercher le premier exemple correct de ce chiffre
        mask_correct = (y_true == digit_sel) & (y_pred == digit_sel)
        if mask_correct.any():
            idx = int(np.where(mask_correct)[0][0])
            cap_norms = np.linalg.norm(caps[idx], axis=-1)  # (10,)
        else:
            idx = int(np.where(y_true == digit_sel)[0][0])
            cap_norms = np.linalg.norm(caps[idx], axis=-1)

        fig_act = go.Figure(go.Bar(
            x=[str(i) for i in range(10)],
            y=cap_norms.tolist(),
            marker_color=["#1d4ed8" if i != digit_sel else "#3fb950" for i in range(10)],
            marker_line_color="#21262d", marker_line_width=1,
            text=[f"{a:.3f}" for a in cap_norms],
            textposition="outside",
            textfont=dict(color="#8b949e", size=10),
        ))
        fig_act.update_layout(
            xaxis_title="Capsule digit", yaxis_title="Norme du vecteur",
            **PLOTLY_BASE,
            xaxis=AXIS_STYLE, yaxis={**AXIS_STYLE, "range": [0, cap_norms.max() * 1.2]},
            margin=dict(l=10, r=10, t=10, b=10), height=320,
        )
        st.plotly_chart(fig_act,  width='stretch')

        # Afficher l'image correspondante
        col_img, _ = st.columns([1, 3])
        with col_img:
            img_np = images[idx].squeeze()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            fig_img = plotly_heatmap_img(img_np, title=f"Chiffre {digit_sel} (test)")
            st.plotly_chart(fig_img,  width='stretch')

    # ------------------------------------------------------------------
    with tab2:
        st.markdown("#### Heatmap des poids W — DigitCaps")
        st.markdown(
            "Norme des vecteurs de prediction u_hat pour chaque paire "
            "(capsule primaire, capsule digit), aggregee sur les 32 premiers exemples."
        )
        model.eval()
        # Extraire W directement depuis le module DigitCaps
        W = model.digit_caps.W.detach().cpu().numpy()  # (10, 1152, 16, 8)
        # Norme sur la dimension out_dim : (10, 1152)
        W_norm = np.linalg.norm(W, axis=2)  # (10, 1152)
        # Sous-echantillonner : regrouper par blocs de 32 primaires
        W_agg = W_norm.reshape(10, -1, 32).mean(axis=2).T  # (36, 10)

        fig_hw = px.imshow(
            W_agg, color_continuous_scale="Blues",
            labels=dict(x="Capsule digit", y="Bloc de capsules primaires"),
            x=[str(i) for i in range(10)],
        )
        fig_hw.update_layout(
            **PLOTLY_BASE, margin=dict(l=10, r=10, t=10, b=10), height=420,
        )
        st.plotly_chart(fig_hw,  width='stretch')

    # ------------------------------------------------------------------
    with tab3:
        st.markdown("#### Projection 2D des vecteurs capsules — PCA")
        st.markdown(
            "Chaque point est la projection du vecteur capsule de la vraie classe "
            "(dim=16) en 2D par ACP. Une separation nette des clusters indique que "
            "les capsules ont appris des representations discriminantes."
        )
        # Extraire les vecteurs de la vraie classe pour chaque exemple
        true_caps = caps[np.arange(len(y_true)), y_true]  # (N, 16)

        pca = PCA(n_components=2)
        proj = pca.fit_transform(true_caps)  # (N, 2)
        var  = pca.explained_variance_ratio_ * 100

        fig_pca = go.Figure()
        colors_pca = px.colors.qualitative.Plotly
        for cls in range(10):
            mask = y_true == cls
            fig_pca.add_trace(go.Scatter(
                x=proj[mask, 0], y=proj[mask, 1],
                mode="markers", name=str(cls),
                marker=dict(size=4, color=colors_pca[cls], opacity=0.7),
            ))
        fig_pca.update_layout(
            xaxis_title=f"PC1 ({var[0]:.1f}%)", yaxis_title=f"PC2 ({var[1]:.1f}%)",
            **PLOTLY_BASE,
            xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
            legend=dict(bgcolor="#161b22", bordercolor="#21262d", title="Classe"),
            margin=dict(l=10, r=10, t=10, b=10), height=420,
        )
        st.plotly_chart(fig_pca,  width='stretch')

    # ------------------------------------------------------------------
    with tab4:
        st.markdown("#### Analyse des erreurs de classification")

        correct_mask = y_true == y_pred
        error_mask   = ~correct_mask

        col_a, col_b = st.columns(2)

        def show_grid(title, indices, n=10):
            n = min(n, len(indices))
            if n == 0:
                st.markdown(f"*Aucun exemple disponible pour : {title}*")
                return
            cols_g = min(5, n)
            rows_g = (n + cols_g - 1) // cols_g
            fig_g = make_subplots(rows=rows_g, cols=cols_g,
                                  horizontal_spacing=0.01, vertical_spacing=0.05)
            for k, idx in enumerate(indices[:n]):
                r_g, c_g = k // cols_g + 1, k % cols_g + 1
                img_g = images[idx].squeeze()
                img_g = (img_g - img_g.min()) / (img_g.max() - img_g.min() + 1e-8)
                fig_g.add_trace(
                    go.Heatmap(z=img_g, colorscale="gray", showscale=False, zmin=0, zmax=1),
                    row=r_g, col=c_g,
                )
                lbl = str(y_true[idx]) if error_mask.sum() == 0 else f"{y_true[idx]}→{y_pred[idx]}"
                fig_g.update_xaxes(
                    title_text=lbl, showticklabels=False,
                    title_font=dict(size=8, color="#8b949e"), row=r_g, col=c_g,
                )
                fig_g.update_yaxes(visible=False, row=r_g, col=c_g)
            fig_g.update_layout(
                **PLOTLY_BASE,
                margin=dict(l=0, r=0, t=0, b=20),
                height=130 * rows_g,
            )
            st.markdown(f"**{title}**")
            st.plotly_chart(fig_g,  width='stretch')

        with col_a:
            correct_idxs = np.where(correct_mask)[0][:10]
            show_grid("Exemples correctement classes", correct_idxs)

        with col_b:
            error_idxs = np.where(error_mask)[0][:10]
            show_grid("Exemples incorrectement classes (vrai → predit)", error_idxs)

        st.markdown("**Matrice des confusions hors-diagonale**")
        cm_err = confusion_matrix(y_true, y_pred)
        np.fill_diagonal(cm_err, 0)
        fig_err = px.imshow(
            cm_err, color_continuous_scale="Reds", text_auto=True,
            x=[str(i) for i in range(10)], y=[str(i) for i in range(10)],
            labels=dict(x="Classe predite", y="Classe reelle"),
        )
        fig_err.update_layout(
            **PLOTLY_BASE, coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10), height=360,
        )
        st.plotly_chart(fig_err,  width='stretch')


# ===========================================================================
# PAGE — A PROPOS
# ===========================================================================
elif page == "A propos":
    st.markdown('<div class="section-title">A propos</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown("### Projet")
        st.markdown(
            """
**Capsule Network for MNIST Classification and Reconstruction**

Ce projet implemente le modele CapsNet decrit dans l'article fondateur de Sabour, Frosst et
Hinton (NeurIPS 2017). Il couvre l'architecture complete — convolution, capsules primaires,
capsules digits et decodeur — ainsi que le mecanisme de routage dynamique. Le dashboard
Streamlit permet de former, evaluer et explorer le comportement du modele de maniere
interactive, en s'appuyant directement sur les modules model.py, train.py et utils.py du
projet.
"""
        )

        st.markdown("### Reference scientifique")
        st.markdown(
            '<div class="arch-block">'
            "<h4>Dynamic Routing Between Capsules</h4>"
            "<p>Sara Sabour, Nicholas Frosst, Geoffrey E. Hinton<br/>"
            "31st Conference on Neural Information Processing Systems (NeurIPS 2017)<br/>"
            "Long Beach, CA, USA — arXiv : 1710.09829</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Technologies")
        techs = [
            ("Python", "3.9+"), ("PyTorch", "2.x"), ("Streamlit", "1.x"),
            ("Plotly", "5.x"), ("NumPy", "1.24+"), ("scikit-learn", "1.x"),
            ("Pillow", "10.x"),
        ]
        st.markdown(
            "".join(f'<span class="badge">{n} {v}</span>' for n, v in techs),
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("### Auteur")
        st.markdown(
            '<div class="arch-block" style="border-left-color:#3fb950;">'
            "<h4>Informations auteur</h4>"
            '<p style="color:#c9d1d9;">'
            "Nom &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b>BILOA Pauline & NGBAYAFOU Lynette</b><br/>"
            "Institution : <b>Ecole Normale Supérieure de Yaoundé </b><br/>"
            "Email &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b>line@gmail.com</b><br/>"
            "GitHub &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b>github.com/PaulineBiloa</b>"
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Licence")
        st.markdown(
            '<div class="arch-block" style="border-left-color:#d29922;">'
            "<h4>MIT License</h4>"
            "<p>Copyright (c) 2024 — Votre Nom.<br/>"
            "Permission is hereby granted, free of charge, to any person obtaining a copy "
            "of this software and associated documentation files (the \"Software\"), to deal "
            "in the Software without restriction, including without limitation the rights to "
            "use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies "
            "of the Software, and to permit persons to whom the Software is furnished to do "
            "so, subject to the following conditions: The above copyright notice and this "
            "permission notice shall be included in all copies or substantial portions of "
            "the Software. THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, "
            "EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF "
            "MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Structure du projet")
        st.code(
            "CapsNet_Project/\n"
            "├── app.py          # Dashboard Streamlit\n"
            "├── model.py        # CapsNet, PrimaryCaps, DigitCaps, Decoder, CapsLoss\n"
            "├── train.py        # train(), test()\n"
            "├── utils.py        # set_seed(), get_dataloaders()\n"
            "├── requirements.txt\n"
            "└── LICENSE",
            language="text",
        )

    st.markdown("<br/><hr/>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;color:#8b949e;font-size:0.8rem;'
        'font-family:JetBrains Mono,monospace;">'
        "CapsNet Dashboard — Sabour, Frosst &amp; Hinton (2017) — Streamlit + PyTorch"
        "</div>",
        unsafe_allow_html=True,
    )