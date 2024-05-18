# lazyGPT

Use LLMs to generate chat responses lazily and asynchronously and control different threads of conversation with ease.

## Installation

```bash
pip install git+https://github.com/simonsanvil/lazygpt.git                 
```

## Usage

Make lazy evaluations of chat responses:

```python
from lazygpt import GPT

gpt = GPT(model="gpt-4o", chat=True, async_=True, temperature=0.9)

with gpt.lazy():
    # The response to all of these won't be evaluated until a call to gpt.evaluate() is made
    gpt("Hello, I will give you a series of questions and you must answer them with honesty and sincerity. Understood?", role="user")
    gpt("What is the capital of Spain?", role="user")

print(gpt.threads[0]) # by default the first thread is the main conversation thread
# User:
# > Hello, I will give you a series of questions and you must answer them with honesty and sincerity. Understood?
# Assistant:
# > [LAZY EVALUATION - Not yet evaluated]
# User:
# > What is the capital of Spain?
# Assistant:
# > [LAZY EVALUATION - Not yet evaluated]

await gpt.evaluate_async()  # this will evaluate all lazy evaluations.

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

The evaluations are done sequentially for each individual conversation thread and in parallel for different threads. 
Here's another example illustrating that:

```python
countries = ["France", "Italy", "Germany"]
for i, country in enumerate(countries):
    with gpt.create_thread(thread_id=f"thread_{i+1}", copy_from=0):
        # this will create a new conversation in a different thread
        # forked from the main thread (thread_id=0)
        gpt(f"What is the capital of {country}?", role="user")
        gpt("Thank you.", role="user", model=None)
        # setting model=None will not trigger a response from the model
    
await gpt.evaluate_async()
# Since the messages to evaluate are all in different threads, 
# they will all be sent to the API simultaneously

for thread in gpt.threads:
    if thread==0: # don't print the main thread
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
# User:
# > Thank you.

# Thread thread_2:
# Hello, I will give you a series of questions and you must answer them with honesty and sincerity. Understood?
# ...
# User:
# > What is the capital of Italy?
# Assistant:
# > The capital of Italy is Rome.
# User:
# > Thank you.

# Thread thread_3:
# Hello, I will give you a series of questions and you must answer them with honesty and sincerity. Understood?
# ...
# User:
# > What is the capital of Germany?
# Assistant:
# > The capital of Germany is Berlin.
# User:
# > Thank you.
```
