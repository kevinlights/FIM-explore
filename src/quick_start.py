#%%
from model import FIMModel, FIMConfig, FIM_TEMPLATES

# # Ollama backend / Ollama 后端
# model = FIMModel(
#     FIMConfig(
#         base_url="http://127.0.0.1:11434/v1",
#         model="qwen2.5-coder:7b",
#         template=FIM_TEMPLATES["qwen2.5-coder"],
#     )
# )

# LM Studio backend / LM Studio 后端
model = FIMModel(
    FIMConfig(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        model="qwen2.5-coder-7b-instruct",
        template=FIM_TEMPLATES["qwen2.5-coder"],
    )
)

result = model.complete(
    prefix="# Implement quick sort algorithm",
    suffix="# this is the suffix",
)
FIMModel.print_stream(result)



# %%
