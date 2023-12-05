# chatlm

Use LLMs to generate chat responses lazily and asynchronously and control different threads of conversation with ease.

## Installation

```bash
pip install chatlm
```

## Usage

Make lazy evaluations of chat responses:

```python
from chatlm.models.openai import GPT
# at the moment only OpenAI's GPT using the API is supported

gpt = GPT(temperature=0.9, max_tokens=100, chat=True, lazy=True, async_=True)

with gpt.lazy():
    gpt("Hello, I will give you a series of questions and you must answer them with honesty and sincerity. Understood?", role="user")
    gpt("What is the capital of Spain?", role="user")

print(gpt.threads[0]) # by default the first thread is the main thread
# User:
# > Hello, I will give you a series of questions and you must answer them with honesty and sincerity. Understood?
# Assistant:
# > [LAZY EVALUATION - Not yet evaluated]
# User:
# > What is the capital of Spain?
# Assistant:
# > [LAZY EVALUATION - Not yet evaluated]

await gpt.evaluate_async() 
# this will evaluate all lazy evaluations asynchronously

print(gpt.threads[0])
# User:
# > Hello, I will give you a series of questions and you must answer them with honesty and sincerity. Understood?
# Assistant:
# > Understood. Please proceed with any questions you may have.
# User:
# > What is the capital of Spain?
# Assistant:
# > The capital of Spain is Madrid.
```

Simultaneous evaluations of multiple threads will be handled automatically:

```python
countries = ["France", "Italy", "Germany"]
for i, country in enumerate(countries):
    with gpt.create_thread(thread_id=f"thread_{i+1}", copy_from=0):
        # this will create a new conversation in a different thread
        # forked from the main thread (thread_id=0)
        gpt(f"What is the capital of {country}?", role="user")
    
await gpt.evaluate_async()
# all 5 threads will be evaluated and sent to the API simultaneously (asynchronously)

for thread in gpt.threads:
    if thread==0:
        continue
    print(f"Thread {thread.thread_id}:")
    print(thread)
# Thread thread_1:
# User:
# Hello, I will give you a series of questions and you must answer them with honesty and sincerity. Understood?
# ...
# User:
# > What is the capital of France?
# Assistant:
# > The capital of France is Paris.
# Thread thread_2:
# Hello, I will give you a series of questions and you must answer them with honesty and sincerity. Understood?
# ...
# User:
# > What is the capital of Italy?
# Assistant:
# > The capital of Italy is Rome.
# Thread thread_3:
# Hello, I will give you a series of questions and you must answer them with honesty and sincerity. Understood?
# ...
# User:
# > What is the capital of Germany?
# Assistant:
# > The capital of Germany is Berlin.
```
