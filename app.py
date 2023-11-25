import streamlit as st
from transformers import pipeline, AutoTokenizer

# Function to load available models
def load_models():
    models_dict = {
        "Text Classification (DistilBERT)": "distilbert-base-uncased-finetuned-sst-2-english",
        "Sentiment Analysis (BERT)": "nlptown/bert-base-multilingual-uncased-sentiment"
        # Add more models as needed with their corresponding Hugging Face model names
    }
    return models_dict.copy()  # Return a copy of the dictionary to prevent mutation

# Streamlit UI
def main():
    st.title("Advanced Text Classification App")

    # Cache the model loading function
    models_dict = load_models()

    model_option = st.sidebar.selectbox("Select a model", list(models_dict.keys()))

    selected_model = models_dict[model_option]

    @st.cache(allow_output_mutation=True)
    def get_classifier(model_name):
        return pipeline("text-classification", model=model_name)

    classifier = get_classifier(selected_model)
    tokenizer = AutoTokenizer.from_pretrained(selected_model)

    user_input = st.text_area("Enter text for classification:", "")

    if st.button("Classify"):
        if user_input:
            encoded_input = tokenizer.encode(user_input, return_tensors="pt")
            result = classifier(user_input)

            # Display prediction label and score
            predicted_label = result[0]["label"]
            predicted_score = result[0]["score"]
            st.success(f"Predicted Label: {predicted_label}, Score: {predicted_score}")

            # Display additional information
            st.subheader("Additional Information:")
            st.json(result[0])

        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()