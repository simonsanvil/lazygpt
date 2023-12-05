# chatlm

Use OpenAI's API to generate chat responses lazily and asynchronously and control different threads of conversation with ease.

## Installation

```bash
pip install chatlm
```

## Usage

```python
from chatlm.models.openai import GPT

gpt = GPT(temperature=0.9, max_tokens=100, chat=True, lazy=True, async_=True)

with gpt.lazy():
    gpt("Hello, how are you?", role="user")
    gpt("Thank you", role="user")

print(gpt.threads[0])
# User:
# > Hello, how are you?
# Assistant:
# > [LAZY EVALUATION - Not yet evaluated]
# User:
# > Thank you

await gpt.evaluate_async()

print(gpt.threads[0])
# User:
# > Hello, how are you?
# Assistant:
# > I'm good, how are you?
# User:
# > Thank you
```

