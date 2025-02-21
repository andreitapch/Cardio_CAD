import os
import logging
import torch
from dash import html, dcc
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


class MedicalLLMInterpreter:
    def __init__(self, model_dir=None, max_new_tokens=512, temperature=0.5, top_p=0.9):
        self.model_name = "microsoft/BioGPT-Large"
        self.model_dir = model_dir or os.getcwd()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.logger.warning("⚠️ GPU is not available.")

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.interpreter = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.model_dir).to(self.device)
            self.interpreter = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            self.logger.info(f"✅ Successfully loaded {self.model_name}.")
        except Exception as e:
            self.logger.error(f"❌ Error loading model {self.model_name}: {e}")
            self.interpreter = None

    def interpret_results(self, selected_features):
        if not self.interpreter:
            return self._handle_failure("Model failed to load.")

        if not selected_features or not isinstance(selected_features, dict):
            return self._handle_failure("Invalid input: 'selected_features' must be a dictionary.")

        feature_string = self._format_features(selected_features)
        prompt = self._generate_prompt(feature_string)
        return self._generate_local_analysis(prompt)

    def _generate_prompt(self, feature_string):
        return (
            "You are an expert in cardiology and machine learning.\n\n"
            "**Task:** Evaluate the provided features for their suitability in predicting left ventricular (LV) remodeling.\n\n"
            "**Selected Features:**\n"
            f"{feature_string}\n\n"
            "**Expected Output:**\n"
            "1️ **Feature Name:** The name of the feature.\n"
            "2️ **Description:** Clinical meaning and how it is measured.\n"
            "3️ **Relevance to LV Remodeling:** Why this feature matters.\n"
            "4️ **Potential for Classification:** How this feature helps classify LV remodeling.\n"
            "Provide the response in a structured format using bullet points."
        )

    def _format_features(self, selected_features):
        formatted_features = []
        for method, features in selected_features.items():
            feature_names = [str(f).strip() for f in features]
            feature_names = list(set(feature_names))  # Remove duplicates
            formatted_features.append(f"**{method}**:\n- " + "\n- ".join(feature_names))
        return "\n\n".join(formatted_features)

    def _generate_local_analysis(self, prompt):
        if not self.interpreter:
            return self._handle_failure("Model or tokenizer failed to load.")

        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True
            )
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
            return generated_text
        except Exception as e:
            return self._handle_failure(f"Failed to generate analysis. Error: {e}")

    def _handle_failure(self, error_message):
        self.logger.error(error_message)
        return f"### Analysis Generation Failed ###\n**Reason:** {error_message}"

    def _format_analysis(self, analysis_text):
        return html.Div([
            html.H5("AI-Powered Medical Feature Analysis"),
            dcc.Markdown(analysis_text, style={'white-space': 'pre-wrap'})
        ])
