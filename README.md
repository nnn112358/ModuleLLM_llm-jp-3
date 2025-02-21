# ModuleLLM_llm-jp-3

## Result


## Model Convert

```
huggingface-cli download --resume-download  llm-jp/llm-jp-3-440m-instruct3 --local-dir llm-jp-3-440m-instruct3
huggingface-cli download --resume-download  llm-jp/llm-jp-3-150m-instruct3 --local-dir llm-jp-3-150m-instruct3
huggingface-cli download --resume-download  llm-jp/llm-jp-3-980m-instruct3 --local-dir llm-jp-3-980m-instruct3
huggingface-cli download --resume-download  llm-jp/llm-jp-3-1.8b-instruct3 --local-dir llm-jp-3-1.8b-instruct3
```

```
sudo docker run -it --net host -v $PWD:/data pulsar2:3.3
pulsar2 llm_build --input_path llm-jp-3-440m-instruct3 --output_path llm-jp-3-440m-instruct3-AX620E --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX620E
pulsar2 llm_build --input_path llm-jp-3-150m-instruct3 --output_path llm-jp-3-150m-instruct3-AX620E --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX620E
pulsar2 llm_build --input_path llm-jp-3-980m-instruct3 --output_path llm-jp-3-980m-instruct3-AX620E --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX620E
pulsar2 llm_build --input_path llm-jp/llm-jp-3-1.8b-instruct3 --output_path llm-jp/llm-jp-3-1.8b-instruct3-AX620E --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX620E
```


