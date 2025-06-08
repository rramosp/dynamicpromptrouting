
from google import genai
from google.genai.types import EmbedContentConfig
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from time import sleep
import json

class GenAI:

    def __init__(self, project, location):
        
        self.client = genai.Client(vertexai=True, project=project, location=location)        
    
    def get_embeddings(self, list_of_texts, n_jobs=-1, sleep_time=None):
    
        def f(text):

            response = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=EmbedContentConfig(
                    task_type="CLUSTERING",  # Optional
                    output_dimensionality=3072,  # Optional
                ),
            )
            r = np.r_[response.embeddings[0].values]
            
            if sleep_time is not None:
                sleep(sleep_time)
            return r
        
        embeddings = Parallel(n_jobs=n_jobs, verbose=5)(delayed(f)(text) for text in list_of_texts)
        return np.r_[embeddings]
        
    def generate_contents(self, list_of_texts, model, n_jobs=-1, sleep_time=None):
    
        def f(text):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=text,
                )

                r = response.text
                code = 'ok'
                
            except Exception as e:
                r = str(e)
                code = e.__class__.__name__
                
            if sleep_time is not None:
                sleep(sleep_time)
            return {'response': r, 'code': code}
            
        responses = Parallel(n_jobs=n_jobs, verbose=5)(delayed(f)(text) for text in list_of_texts)
        return pd.DataFrame(responses)

def extract_json_rating(text): 

    r = None
    i0 = text.find ("```json")
    if i0>=0:
        i1 = text[i0+7:].find("```")
        if i1>=0:
            r = text[i0+7:i0+i1+7]
    
    if r is not None:
        try:
            r = json.loads(r)
            return r['rating']
        except Exception as e:
            return None
    else:
        return None