"""Local, stateless Streamlit interface for the saved SMS classifier."""
import json
from pathlib import Path
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Message Check · Spam Detector", page_icon="✉️", layout="centered")
st.markdown("""<style>
.stApp { background: var(--background-color); }
.block-container { max-width: 780px; padding-top: 3.5rem; }
h1 { letter-spacing: -1.8px; }
div.stButton > button[kind="primary"] { border-radius: 10px; }
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "model/spam_pipeline.joblib")


st.caption("MESSAGE CHECK  /  A TEXT CLASSIFICATION LAB")
st.title("A little clarity for your inbox.")
st.write("Paste an English text message to see whether a machine learning model thinks it looks like spam.")
st.caption("Runs locally · Your submitted messages are not saved")

examples = {
    "Prize offer": "Congratulations! You have won a FREE cash prize. Call now to claim your reward!",
    "Urgent promotion": "URGENT! You have been selected for a guaranteed cash award. Call 09061701461 to claim now.",
    "Dinner plans": "Hey, are we still meeting for dinner at 7? I'll book a table.",
    "Everyday update": "I'm running a few minutes late. See you at the station soon.",
}


def use_example(text):
    st.session_state.message = text


st.subheader("Try a message")
cols = st.columns(2)
for i, (name, message) in enumerate(examples.items()):
    cols[i % 2].button(name, on_click=use_example, args=(message,), use_container_width=True)

message = st.text_area("Your message", key="message", height=160,
                       placeholder="Type or paste a text message here…")
st.caption(f"{len(message):,} characters · English messages work best")

if st.button("Check Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please type or paste a message, or choose an example above.")
    elif not (ROOT / "model/spam_pipeline.joblib").exists():
        st.error("The trained model is missing. Run `python train.py` in the app folder, then try again.")
    else:
        try:
            prediction = load_model().predict([message.strip()])[0]
            if prediction == "spam":
                st.error("### Spam\nThis message resembles spam in the training data.")
            else:
                st.success("### Not Spam\nThis message resembles everyday messages in the training data.")
            st.caption("Edit the message or select another example to try again.")
        except Exception:
            st.error("The model could not check this message. Reinstall the pinned dependencies and run `python train.py` to rebuild it.")

st.divider()
st.info("Predictions can be incorrect. “Not Spam” does not guarantee a message is safe. This learning app does not block messages or scan accounts.")
with st.expander("How does it work?"):
    st.write("TF-IDF converts words and short word pairs into numbers. A Multinomial Naive Bayes classifier learns which patterns appear in spam and everyday messages from the UCI SMS Spam Collection. It then predicts a category for your message.")
    st.write("Duplicate messages are removed before a stratified 80/20 split. The text processor learns only from the training portion; evaluation uses the held-out portion.")
    metrics_path = ROOT / "evaluation/metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        spam = metrics["classification_report"]["spam"]
        columns = st.columns(3)
        for col, name, key in zip(columns, ["Spam precision", "Spam recall", "Spam F1"], ["precision", "recall", "f1-score"]):
            col.metric(name, f"{spam[key]:.1%}")
        st.caption(f"Held-out messages: {metrics['test_messages']:,}. These scores describe this dataset, not all real-world messages.")
        st.image(str(ROOT / "evaluation/confusion_matrix.png"), caption="Rows: actual category. Columns: predicted category.")
