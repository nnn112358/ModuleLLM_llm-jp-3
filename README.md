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
 ./tools/embed_process.sh llm-jp-3-440m-instruct3/ llm-jp-3-440m-instruct3-AX620E/

Config(
    model_name='/data/experiments/0087_llmjp3-440m/checkpoints_hf/iter_0988240',
    model_type='llama',
    num_hidden_layers=16,
    num_attention_heads=8,
    num_key_value_heads=8,
    hidden_size=1024,
    head_dim=128,
    intermediate_size=3584,
    vocab_size=99584,
    rope_theta=10000,
    max_position_embeddings=4096,
    rope_partial_factor=1.0,
    rms_norm_eps=1e-05,
    norm_type='rms_norm',
    hidden_act='silu',
    hidden_act_param=0.03,
    scale_depth=1.4,
    scale_emb=1,
    dim_model_base=256,
    origin_model_type=''
)

pulsar2 llm_build --input_path llm-jp-3-150m-instruct3 --output_path llm-jp-3-150m-instruct3-AX620E --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX620E
 ./tools/embed_process.sh llm-jp-3-150m-instruct3/ llm-jp-3-150m-instruct3-AX620E/

Config(
    model_name='/data/experiments/0016_v3-152m/checkpoints_hf/iter_0988240',
    model_type='llama',
    num_hidden_layers=12,
    num_attention_heads=8,
    num_key_value_heads=8,
    hidden_size=512,
    head_dim=64,
    intermediate_size=2048,
    vocab_size=99584,
    rope_theta=10000,
    max_position_embeddings=4096,
    rope_partial_factor=1.0,
    rms_norm_eps=1e-05,
    norm_type='rms_norm',
    hidden_act='silu',
    hidden_act_param=0.03,
    scale_depth=1.4,
    scale_emb=1,
    dim_model_base=256,
    origin_model_type=''
)

pulsar2 llm_build --input_path llm-jp-3-980m-instruct3 --output_path llm-jp-3-980m-instruct3-AX620E --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX620E
 ./tools/embed_process.sh llm-jp-3-980m-instruct3/ llm-jp-3-980m-instruct3-AX620E/

Config(
    model_name='/data/experiments/0088_llmjp3-980m/checkpoints_hf/iter_0988240',
    model_type='llama',
    num_hidden_layers=20,
    num_attention_heads=8,
    num_key_value_heads=8,
    hidden_size=1536,
    head_dim=192,
    intermediate_size=5376,
    vocab_size=99584,
    rope_theta=10000,
    max_position_embeddings=4096,
    rope_partial_factor=1.0,
    rms_norm_eps=1e-05,
    norm_type='rms_norm',
    hidden_act='silu',
    hidden_act_param=0.03,
    scale_depth=1.4,
    scale_emb=1,
    dim_model_base=256,
    origin_model_type=''
)


pulsar2 llm_build --input_path llm-jp-3-1.8b-instruct3 --output_path llm-jp-3-1.8b-instruct3-AX620E --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX620E

  model_name='llm-jp/llm-jp-3-1.8b',
    model_type='llama',
    num_hidden_layers=24,
    num_attention_heads=16,
    num_key_value_heads=16,
    hidden_size=2048,
    head_dim=128,
    intermediate_size=7168,
    vocab_size=99584,
    rope_theta=10000,
    max_position_embeddings=4096,
    rope_partial_factor=1.0,
    rms_norm_eps=1e-05,
    norm_type='rms_norm',
    hidden_act='silu',
    hidden_act_param=0.03,
    scale_depth=1.4,
    scale_emb=1,
    dim_model_base=256,
    origin_model_type=''

```


