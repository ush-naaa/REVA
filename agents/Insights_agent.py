import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

class InsightsAgent:
    def __init__(self, groq_api_key, model_name="llama-3.3-70b-versatile"): 
        self.client = Groq(api_key=groq_api_key)
        self.model = model_name
        
        # Setup Vector Database (RAG)
        self.chroma_client = chromadb.PersistentClient(path="./market_db")
        
        # Embedding model
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="pakistan_real_estate", 
            embedding_function=self.emb_fn
        )

        # Load RAG file if empty
        if self.collection.count() == 0:
            self._load_knowledge("market_knowledge.txt")

    def _load_knowledge(self, file_path):
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found. Please run your generator script first.")
            return
            
        with open(file_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if len(l.strip()) > 15]
            ids = [f"id_{i}" for i in range(len(lines))]
            self.collection.add(documents=lines, ids=ids)
            print(f"✅ RAG: Loaded {len(lines)} market insights into database.")

    def explain_prediction(self, user_input, predicted_class, confidence):
        # 1. Retrieve the local data
        query = f"{user_input['property_type']} in {user_input['location']}, {user_input['city']}"
        results = self.collection.query(query_texts=[query], n_results=1)
        
        # Check if we found context
        context_data = results['documents'][0][0] if results['documents'] and results['documents'][0] else "No specific historical pricing found for this exact sub-sector."

        # ---------------------------------------------------------
        # 2. DEFINING THE PERSONA (SYSTEM PROMPT)
        # This tells the AI HOW to think and format the answer.
        # ---------------------------------------------------------
        system_instruction = """
        You are **Reva AI**, a friendly and expert Real Estate Consultant for the Pakistan market. 
        
        Your Goal: Explain property valuations to a non-technical buyer in simple, everyday language.
        
        GUIDELINES:
        1. **Tone:** Warm, professional, and clear. Avoid jargon like "coefficients" or "embeddings."
        2. **Evidence-Based:** You must explain WHY a property is classified as a certain tier based ONLY on the provided Context Data.
        3. **Formatting:** Use Markdown to bold key prices or locations (e.g., **1.2 Crore**).
        4. **Structure:**
           - Start with a direct, one-sentence summary.
           - Provide 3 bullet points explaining the 'Why'.
           - End with a 'Reva's Pro Tip' specific to the property size.
        
        CONSTRAINT: If the Context Data contains specific prices (e.g., "15,000,000 PKR"), you MUST quote them to justify your explanation.
        """

        # ---------------------------------------------------------
        # 3. DEFINING THE TASK (USER PROMPT)
        # This provides the dynamic data to analyze.
        # ---------------------------------------------------------
        user_payload = f"""
        Here is the market data and the property details. Please generate the explanation report.

        ### 1. MARKET CONTEXT (RAG Data):
        "{context_data}"

        ### 2. PROPERTY DETAILS:
        - **City:** {user_input['city']}
        - **Location:** {user_input['location']}
        - **Type:** {user_input['property_type']}
        - **Size:** {user_input['Area_in_Marla']} Marla
        - **AI Prediction:** This property is rated as **'{predicted_class}'** tier.

        ### QUESTION:
        Why does this specific property fall into the '{predicted_class}' category based on the market context provided?
        """

        # ---------------------------------------------------------
        # 4. SEND TO API
        # ---------------------------------------------------------
        completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload}
            ],
            model=self.model,
            temperature=0.6, # Slight creativity for the "friendly" tone, but strictly constrained by system prompt
        )
        
        return completion.choices[0].message.content