# Jax Llama

![](pics/llama_3.png)
*(Many thanks to Kandinsky 2.2 for this beautiful image)*

This repository is just a very simple implementation of the [Llama model](https://arxiv.org/abs/2302.13971). It is implented in Jax & Flax. It works fine on `cpu` and all the config is located and `jax_llama/config.py`. Many thanks to the Meta team and to the blog post https://blog.briankitano.com/llama-from-scratch/ (highly recommend to read it). It was implemented and tested using `python==3.10.12`. In order to start run the following:

```bash
pip install -r requirements.txt
python jax_llama/main.py
```
