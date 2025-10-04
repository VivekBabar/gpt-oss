import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, ViTModel, ViTImageProcessor
import pandas as pd
import os

class GPTOSSModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.text_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.vision_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vision_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        
        # Load CSV data
        self.csv_data = self.load_csv_data()
        
    def load_csv_data(self):
        """Load all CSV files from data directory"""
        csv_data = {}
        data_dir = "data"
        
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    try:
                        file_path = os.path.join(data_dir, file)
                        df = pd.read_csv(file_path)
                        csv_data[file] = df
                        print(f"✅ Loaded CSV: {file} ({len(df)} rows, {len(df.columns)} columns)")
                    except Exception as e:
                        print(f"❌ Error loading {file}: {e}")
        else:
            print("📁 Creating data directory...")
            os.makedirs(data_dir, exist_ok=True)
        
        return csv_data
    
    def search_csv_data(self, query):
        """Search through CSV data for relevant information"""
        results = []
        
        for filename, df in self.csv_data.items():
            try:
                # Search in all string columns
                for col in df.columns:
                    if df[col].dtype == 'object':  # String columns
                        matches = df[df[col].str.contains(query, case=False, na=False)]
                        for _, row in matches.iterrows():
                            results.append({
                                'file': filename,
                                'column': col,
                                'data': row.to_dict()
                            })
            except Exception as e:
                print(f"Error searching {filename}: {e}")
        
        return results
    
    def generate_response(self, text_input, image_path=None, use_csv=False):
        """Generate response using GPT-OSS model"""
        
        # Search CSV data if requested
        csv_results = []
        if use_csv and self.csv_data:
            csv_results = self.search_csv_data(text_input)
        
        # Process image if provided
        image_analysis = ""
        if image_path and os.path.exists(image_path):
            image_analysis = self.analyze_image(image_path)
        
        # Generate response based on available data
        if csv_results:
            response = self._generate_response_with_csv(text_input, csv_results, image_analysis)
        elif image_analysis:
            response = self._generate_response_with_image(text_input, image_analysis)
        else:
            response = self._generate_text_response(text_input)
        
        return response
    
    def analyze_image(self, image_path):
        """Analyze image and return description"""
        try:
            if image_path.endswith('.txt'):
                with open(image_path, 'r') as f:
                    content = f.read()
                # Extract key information from placeholder
                if "Description:" in content:
                    desc_start = content.find("Description:") + len("Description:")
                    desc_end = content.find("\n", desc_start)
                    return content[desc_start:desc_end].strip()
                return "Image placeholder file"
            else:
                # For real images, you would process them here
                return f"Analyzed image: {os.path.basename(image_path)}"
        except Exception as e:
            return f"Error analyzing image: {e}"
    
    def _generate_text_response(self, text_input):
        """Generate text-only response"""
        # Simple response generation - you can enhance this
        responses = {
            "hello": "Hello! I'm your GPT-OSS assistant. How can I help you today?",
            "hi": "Hi there! I can help you analyze data, images, and answer questions.",
            "help": "I can:\n• Analyze CSV data\n• Process images\n• Answer questions\n• Provide Earth Observation insights",
            "what can you do": "I'm a multimodal AI assistant that can:\n📊 Analyze CSV data\n🖼️ Process and describe images\n🌍 Provide Earth Observation analysis\n💬 Answer questions using available data"
        }
        
        # Check for simple responses first
        lower_input = text_input.lower()
        for key, response in responses.items():
            if key in lower_input:
                return response
        
        # Default response
        return f"I understand you're asking about: {text_input}. I can help you analyze this using available data and models."

    def _generate_response_with_csv(self, text_input, csv_results, image_analysis):
        """Generate response with CSV data"""
        csv_context = "Relevant data found:\n"
        for i, result in enumerate(csv_results[:3]):  # Limit to 3 results
            csv_context += f"{i+1}. From {result['file']}: {str(result['data'])}\n"
        
        prompt = f"""Based on the following data and context:

{csv_context}

Image Analysis: {image_analysis}

User Question: {text_input}

Please provide a helpful response using the available data."""
        
        return self._generate_text_response(prompt)
    
    def _generate_response_with_image(self, text_input, image_analysis):
        """Generate response with image analysis"""
        prompt = f"""Image Analysis: {image_analysis}

User Question: {text_input}

Please provide a response based on the image analysis."""
        
        return self._generate_text_response(prompt)

# Global model instance
gptoss_model = GPTOSSModel()