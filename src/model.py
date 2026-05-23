"""
FIM (Fill-in-the-Middle) Model Support Module.
Supports Ollama and OpenAI-compatible backends (LM Studio, etc.)

FIM (Fill-in-the-Middle) 模型支持模块
支持 Ollama 和兼容 OpenAI 的后端（如 LM Studio 等）
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Generator, Optional

from openai import OpenAI


# ---------------------------------------------------------------------------
# FIM Templates for different models
# 不同模型的 FIM 模板
# ---------------------------------------------------------------------------
@dataclass
class FIMTemplate:
    """FIM template definition for a model family.

    FIM 模板定义，用于不同模型系列
    """

    name: str
    prefix_token: str
    suffix_token: str
    middle_token: str
    end_token: Optional[str] = None

    def build_prompt(self, prefix: str, suffix: str) -> str:
        """Build the FIM prompt by inserting prefix and suffix into the template.

        通过将前缀和后缀插入模板来构建 FIM 提示
        """
        return f"{self.prefix_token}{prefix}{self.suffix_token}{suffix}{self.middle_token}"


# Pre-defined FIM templates for popular local code models
# 为流行的本地代码模型预定义的 FIM 模板
FIM_TEMPLATES: dict[str, FIMTemplate] = {
    "qwen2.5-coder": FIMTemplate(
        name="Qwen2.5-Coder",
        prefix_token="<|fim_prefix|>",
        suffix_token="<|fim_suffix|>",
        middle_token="<|fim_middle|>",
    ),
    "starcoder2": FIMTemplate(
        name="StarCoder2 / Stable Code",
        prefix_token="<fim_prefix>",
        suffix_token="<fim_suffix>",
        middle_token="<fim_middle>",
    ),
    "codelama": FIMTemplate(
        name="CodeLlama",
        prefix_token="<PRE> ",
        suffix_token=" <SUF>",
        middle_token=" <MID>",
    ),
    "deepseek-coder": FIMTemplate(
        name="DeepSeek Coder",
        prefix_token="rule\n",
        suffix_token="??\n",
        middle_token="??\n",
    ),
}


# ---------------------------------------------------------------------------
# FIM Model Client
# FIM 模型客户端
# ---------------------------------------------------------------------------
@dataclass
class FIMConfig:
    """Configuration for the FIM model client.

    FIM 模型客户端配置
    """

    base_url: str = "http://127.0.0.1:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen2.5-coder:7b"
    template: FIMTemplate = field(default_factory=lambda: FIM_TEMPLATES["qwen2.5-coder"])
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = True


class FIMModel:
    """A model client for Fill-in-the-Middle (FIM) completion.

    用于 Fill-in-the-Middle (FIM) 补全的模型客户端

    Supports both Ollama (via /v1/completions) and OpenAI-compatible backends
    like LM Studio, vLLM, etc.
    同时支持 Ollama（通过 /v1/completions）和兼容 OpenAI 的后端，如 LM Studio、vLLM 等
    """

    def __init__(self, config: Optional[FIMConfig] = None):
        """Initialize the FIM model client.

        初始化 FIM 模型客户端
        """
        self.config = config if config is not None else FIMConfig()
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def complete(
        self,
        prefix: str,
        suffix: str = "",
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: Optional[bool] = None,
    ) -> str | Generator[str, None, None]:
        """Perform FIM completion.

        执行 FIM 补全

        Args:
            prefix: Code/text before the cursor (光标前的代码/文本)
            suffix: Code/text after the cursor (光标后的代码/文本)
            max_tokens: Maximum tokens to generate (最大生成 token 数)
            temperature: Sampling temperature (采样温度)
            stream: Whether to stream the output (是否流式输出)

        Returns:
            If stream=True: a generator yielding text chunks (流式返回文本块生成器)
            If stream=False: the full generated text (非流式返回完整文本)
        """
        prompt = self.config.template.build_prompt(prefix, suffix)
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        stream = stream if stream is not None else self.config.stream

        response = self._client.completions.create(
            model=self.config.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self.config.template.end_token,
            stream=stream,
        )

        if stream:
            return self._stream_response(response)
        else:
            return response.choices[0].text

    @staticmethod
    def _stream_response(response) -> Generator[str, None, None]:
        """Stream response chunks from the API.

        从 API 流式获取响应块
        """
        for chunk in response:
            if chunk.choices and chunk.choices[0].text:
                yield chunk.choices[0].text

    @staticmethod
    def print_stream(stream: Generator[str, None, None]) -> str:
        """Print streaming output to stdout and return the full text.

        将流式输出打印到标准输出并返回完整文本
        """
        full_text: list[str] = []
        for text_chunk in stream:
            print(text_chunk, end="", flush=True)
            full_text.append(text_chunk)
        print()  # final newline / 最终换行
        return "".join(full_text)


# ---------------------------------------------------------------------------
# CLI Entry Point
# 命令行入口
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="FIM (Fill-in-the-Middle) completion client",
        epilog="Example: python model.py --prefix 'def fib(n):' --suffix '# test' --model qwen2.5-coder:7b",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="# Implement quick sort algorithm",
        help="Code/text before the cursor (default: quick sort comment)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="# this is the suffix",
        help="Code/text after the cursor (default: suffix marker)",
    )
    parser.add_argument(
        "--model", type=str, default="qwen2.5-coder:7b", help="Model name (default: qwen2.5-coder:7b)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:11434/v1",
        help="API base URL (default: http://127.0.0.1:11434/v1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="ollama",
        help="API key (default: ollama)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="qwen2.5-coder",
        choices=list(FIM_TEMPLATES.keys()),
        help=f"FIM template to use (default: qwen2.5-coder). Choices: {', '.join(FIM_TEMPLATES.keys())}",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List all available FIM templates and exit",
    )
    return parser.parse_args(argv)


def list_templates() -> None:
    """Print all available FIM templates.

    打印所有可用的 FIM 模板
    """
    print("Available FIM Templates / 可用的 FIM 模板:\n")
    print(f"{'Key':<20} {'Name':<25} {'Prefix Token':<30}")
    print("-" * 75)
    for key, tmpl in FIM_TEMPLATES.items():
        print(f"{key:<20} {tmpl.name:<25} {tmpl.prefix_token!r:<30}")


def main() -> None:
    """Main CLI entry point.

    主命令行入口
    """
    args = parse_args()

    if args.list_templates:
        list_templates()
        return

    # Build config
    # 构建配置
    config = FIMConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        template=FIM_TEMPLATES[args.template],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=not args.no_stream,
    )

    # Create model and run completion
    # 创建模型并执行补全
    model = FIMModel(config)

    print(f"Model / 模型: {config.model}")
    print(f"Template / 模板: {config.template.name}")
    print(f"Backend / 后端: {config.base_url}")
    print(f"Prefix / 前缀: {args.prefix!r}")
    print(f"Suffix / 后缀: {args.suffix!r}")
    print(f"{'─' * 40}")
    print("Generated output / 生成输出:")
    print()

    result = model.complete(prefix=args.prefix, suffix=args.suffix)

    if isinstance(result, Generator):
        FIMModel.print_stream(result)
    else:
        print(result)


# ---------------------------------------------------------------------------
# Jupyter / Interactive cell support
# Jupyter / 交互式单元格支持
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# %% Example usage in VS Code Interactive / VS Code 交互式示例
# ```python
# from model import FIMModel, FIMConfig, FIM_TEMPLATES
#
# # Ollama backend / Ollama 后端
# model = FIMModel(FIMConfig(
#     base_url="http://127.0.0.1:11434/v1",
#     model="qwen2.5-coder:7b",
#     template=FIM_TEMPLATES["qwen2.5-coder"],
# ))
#
# result = model.complete(
#     prefix="# Implement quick sort algorithm",
#     suffix="# this is the suffix",
# )
# FIMModel.print_stream(result)
#
# # LM Studio backend / LM Studio 后端
# model_lm = FIMModel(FIMConfig(
#     base_url="http://127.0.0.1:1234/v1",
#     api_key="lm-studio",
#     model="qwen2.5-coder-7b-instruct",
#     template=FIM_TEMPLATES["qwen2.5-coder"],
# ))
# ```
