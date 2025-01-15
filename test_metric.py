#/usr/bin/env python3
import evaluate 
from pprint import pprint
perplexity = evaluate.load("perplexity", module_type="metric")
input_texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]
results = perplexity.compute(model_id='gpt2',
                             add_start_token=False,
                             predictions=input_texts)
print(list(results.keys()))

pprint(results)
