"""集中管理 ABook 的运行配置。

项目统一使用 OpenAI 兼容协议，因此这里只保留模型 ID、API Key 和 API 根地址。
模型对象在 CLI 启动时创建，模块导入阶段不会连接外部服务。
"""

from dataclasses import dataclass
import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


@dataclass(frozen=True)
class Settings:
    """一次 CLI 会话使用的不可变配置。

    Attributes:
        model: OpenAI 兼容服务提供的原始模型 ID。
        api_key: 兼容服务的访问密钥。
        base_url: OpenAI 兼容 API 的根地址，通常以 ``/v1`` 结尾。
    """

    model: str
    api_key: str
    base_url: str

    @classmethod
    def from_env(cls) -> "Settings":
        """从三个必需的环境变量构建配置。

        ``load_dotenv`` 由 CLI 边界调用，本方法不主动读取文件，因此也可被
        测试或其他入口复用。空字符串只包含空白时也视为缺失。

        Raises:
            ValueError: 缺少一个或多个必需的环境变量。
        """
        values = {
            "ABOOK_MODEL": os.getenv("ABOOK_MODEL", "").strip(),
            "ABOOK_API_KEY": os.getenv("ABOOK_API_KEY", "").strip(),
            "ABOOK_BASE_URL": os.getenv("ABOOK_BASE_URL", "").strip(),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")

        return cls(
            model=values["ABOOK_MODEL"],
            api_key=values["ABOOK_API_KEY"],
            base_url=values["ABOOK_BASE_URL"],
        )


def create_model(settings: Settings) -> OpenAIChatModel:
    """为统一的 OpenAI 兼容接口创建 PydanticAI 模型。

    构造过程只初始化客户端，不会发起网络请求。模型 ID 会原样传给兼容
    服务，不接受或解析 ``provider:model`` 形式的供应商前缀。
    """
    provider = OpenAIProvider(
        base_url=settings.base_url,
        api_key=settings.api_key,
    )
    return OpenAIChatModel(settings.model, provider=provider)
