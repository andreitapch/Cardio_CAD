import os
import logging
from dash import html, dcc

# Optional imports (done safely)
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    torch = None

try:
    import openai
except ImportError:
    openai = None


class MedicalLLMInterpreter:
    def __init__(self,
                 mode="openai",  # or "local"
                 openai_api_key=None,
                 model_name="gpt-3.5-turbo",
                 local_model_path="medalpaca/medalpaca-7b",
                 max_tokens=512,
                 temperature=0.5,
                 top_p=0.9):

        self.mode = mode
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Logging setup
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Initialize model depending on mode
        if self.mode == "openai":
            if not openai:
                raise ImportError("OpenAI module is not available. Install with `pip install openai`.")
            self.model_name = model_name
            self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is required.")
            openai.api_key = self.api_key
            self.logger.info(f"✅ OpenAI GPT-3.5 model loaded: {self.model_name}")

        elif self.mode == "local":
            if not torch:
                raise ImportError("Transformers/torch not installed. Run `pip install transformers torch`.")
            self.model_name = local_model_path
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.float16 if self.device == "cuda" else None)
                self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == "cuda" else -1)
                self.logger.info(f"✅ Local model loaded: {self.model_name} on {self.device}")
            except Exception as e:
                self.logger.error(f"❌ Failed to load local model: {e}")
                self.generator = None

    def interpret_results(self, selected_features):
        if not selected_features or not isinstance(selected_features, dict):
            return self._handle_failure("Invalid input: 'selected_features' must be a dictionary.")

        feature_string = self._format_features(selected_features)
        prompt = self._generate_prompt(feature_string)

        if self.mode == "openai":
            return self._generate_analysis_openai(prompt)
        elif self.mode == "local":
            return self._generate_analysis_local(prompt)
        else:
            return self._handle_failure("Unsupported mode selected.")

    def _generate_prompt(self, feature_string):
        return (
            "You are an expert cardiovascular specialist contributing to a clinical study on predicting adverse left ventricular (LV) remodeling.\n\n"
            "Below is a list of clinical and imaging features measured in patients:\n"
            f"{feature_string}\n\n"
            "For each feature, assess whether it is likely to be associated with adverse LV remodeling. Briefly explain the reasoning based on cardiac physiology or clinical evidence. If there is no clear link, say so.\n\n"
            "Example:\n"
            "- EF_percent_combined: A reduced ejection fraction reflects poor systolic function and is commonly associated with adverse remodeling.\n\n"
            "Now continue with the remaining features:"
        )

    def _format_features(self, selected_features):
        features_flat = []
        for features in selected_features.values():
            features_flat.extend([str(f).strip() for f in features])
        features_flat = sorted(set(features_flat))
        return "\n".join(f"- {f}" for f in features_flat)

    def _generate_analysis_openai(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a medical assistant specialized in cardiac function."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            result = response['choices'][0]['message']['content'].strip()
            return self._format_analysis(result)
        except Exception as e:
            return self._handle_failure(f"Failed to get OpenAI response. Error: {e}")

    def _generate_analysis_local(self, prompt):
        if not self.generator:
            return self._handle_failure("Local model not available.")

        try:
            output = self.generator(prompt, max_new_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, do_sample=True)
            result = output[0]["generated_text"].replace(prompt, "").strip()
            return self._format_analysis(result)
        except Exception as e:
            return self._handle_failure(f"Failed to generate with local model. Error: {e}")

    def _format_analysis(self, analysis_text):
        return html.Div([
            html.H5(f"{'GPT-3.5' if self.mode == 'openai' else 'Local'} Medical Feature Analysis"),
            dcc.Markdown(analysis_text, style={'white-space': 'pre-wrap'})
        ])

    def _handle_failure(self, message):
        self.logger.error(message)
        return html.Div([
            html.H5("Analysis Generation Failed"),
            dcc.Markdown(f"**Reason:** {message}", style={'color': 'red'})
        ])
