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

await gpt.evaluate_async()

print(gpt.threads[0])
```

