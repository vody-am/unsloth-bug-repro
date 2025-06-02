# unsloth-bug-repro
reproducing a bug in unsloth-zoo


## Running

`uv run main.py`

```
...
n_items = ().get("num_items_in_batch", None) or ().get("n_items", None)
AttributeError: 'tuple' object has no attribute 'get'
```
