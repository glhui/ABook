# Pydantic `BaseModel` 与 `model_config` 使用指南

> 注意：本文的 `EvidenceRecord` 示例说明早期的扁平证据模型。当前 Runtime 使用
> `ToolCallRecord` 保存结构化工具调用结果，并以 `tool_call_id` 供 `FactClaim`
> 引用；请以 `context.py` 中的当前模型为准。

本文介绍 ABook 中使用的 Pydantic 2 模型写法，重点说明 `BaseModel`、
`ConfigDict`、字段类型、验证和序列化之间的关系。文中的示例使用 Pydantic
2.x 语法，与项目当前依赖一致。

## 1. 为什么使用 `BaseModel`

`BaseModel` 是 Pydantic 提供的结构化数据模型基类。继承它之后，类上的类型
注解不再只是给编辑器看的说明，而会参与运行时的数据处理：

```python
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


user = User(name="Alice", age=30)
print(user.name)
print(user.model_dump())
```

上面的 `User` 获得了几项通用能力：

- 根据字段注解检查输入数据；
- 检查必填字段和字段约束；
- 将嵌套字典转换为嵌套模型；
- 把模型转换为字典或 JSON；
- 用统一的 `ValidationError` 报告输入错误。

因此，`BaseModel` 对应的是“数据结构加运行时验证”，而不是简单的字典别名。
模型可以通过 `model_dump()` 转换成字典，但在转换之前，模型对象仍然可以提供
类型检查、嵌套验证和配置控制。

## 2. 最基本的模型定义

```python
from pydantic import BaseModel


class Evidence(BaseModel):
    evidence_id: str
    source: str
    content: str
```

创建模型时，字段名必须与定义一致：

```python
evidence = Evidence(
    evidence_id="evidence-1",
    source="config.py",
    content="ABOOK_MODEL=demo",
)
```

读取字段时使用属性访问：

```python
print(evidence.evidence_id)
print(evidence.content)
```

如果缺少必填字段，或者字段值无法通过类型验证，Pydantic 会抛出
`pydantic.ValidationError`，调用方不应该把它当作一个普通的、不完整的对象继续使用。

## 3. `BaseModel`、`ConfigDict` 和 `model_config` 的关系

项目中常见的写法是：

```python
from pydantic import BaseModel, ConfigDict


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    evidence_id: str
    content: str
```

三者的关系可以这样理解：

```text
BaseModel
    提供模型的通用能力：初始化、校验、序列化、错误处理

字段类型注解
    定义模型有哪些数据，以及这些数据的基本类型和约束

ConfigDict
    保存模型行为配置

model_config
    把 ConfigDict 挂到当前模型上，让 BaseModel 使用这些配置
```

`ConfigDict` 本质上是 Pydantic 识别的一组配置项集合；它不是用来保存业务数据的。
`model_config` 也不是一个需要由调用方传入的业务字段，而是模型类的配置入口。

## 4. 项目最重要的两个配置

### 4.1 `extra="forbid"`

`extra` 控制输入中出现模型未声明字段时的行为。

```python
class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    content: str
```

下面的输入会失败，因为 `agent_id` 没有在模型中声明：

```python
EvidenceRecord(
    evidence_id="evidence-1",
    content="...",
    agent_id="worker-1",
)
```

常见的 `extra` 选项有：

- `"ignore"`：忽略未声明字段；
- `"forbid"`：遇到未声明字段时抛出验证错误；
- `"allow"`：保留未声明字段。

ABook 对 Runtime 状态、工具结果和模型输出普遍使用 `"forbid"`，因为这些数据
有明确的协议。拼写错误或模型多生成的字段应该尽早暴露，而不是静默丢弃。

### 4.2 `frozen=True`

`frozen=True` 用于阻止模型字段在创建后被重新赋值：

```python
class EvidenceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    content: str


record = EvidenceRecord(
    evidence_id="evidence-1",
    content="原始工具结果",
)

record.content = "被修改的结果"  # ValidationError
```

ABook 的 `EvidenceRecord` 使用它是为了保证：证据一旦登记，就不能被后续逻辑
悄悄改写。`TaskFact` 和证据引用也使用相同原则，避免已经验证过的结果失去可追溯性。

`frozen=True` 主要保护模型字段的重新赋值。它不等于对所有嵌套对象做深度不可变
保护；如果模型中保存了可变的嵌套列表，仍应根据业务需要选择元组、嵌套冻结模型
或在边界复制数据。

## 5. 字段类型和常见约束

### 5.1 基本类型

```python
class Settings(BaseModel):
    model: str
    timeout_seconds: int
    enabled: bool
```

Pydantic 会读取这些注解并验证输入。默认模式下，Pydantic 可能对部分输入做合理
转换，例如把可以表示整数的字符串转换为整数。需要禁止隐式转换时，可以使用
`StrictInt`、`StrictStr` 等严格类型，或配置 `strict=True`。

### 5.2 `Literal`

`Literal` 用于限制字段只能取固定值：

```python
from typing import Literal


class EvidenceRecord(BaseModel):
    kind: Literal[
        "file_listing",
        "file_read",
        "text_search",
        "file_change",
        "command",
    ]
```

这正是 ABook 用来限制 `EvidenceRecord.kind` 的方式。它把可接受的证据类型
变成显式协议，避免出现任意字符串。

### 5.3 `Field`

`Field` 可以为字段增加长度、范围、默认值和说明：

```python
from pydantic import BaseModel, Field


class FactClaim(BaseModel):
    statement: str = Field(min_length=1, max_length=2_000)
    evidence_ids: list[str] = Field(min_length=1, max_length=10)
```

ABook 中的 `CompactionCheckpoint`、`FactClaim` 和其他模型使用 `Field` 限制模型
输出的大小，防止异常长的内容消耗过多上下文或持久化空间。

### 5.4 可选值和默认值

```python
class TaskState(BaseModel):
    title: str
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
```

`summary` 可以没有值；`tags` 默认是一个新列表。对于列表、字典等可变默认值，
应使用 `default_factory`，不要直接写成 `tags: list[str] = []`。

## 6. 嵌套模型

Pydantic 支持用模型组合模型：

```python
class Citation(BaseModel):
    evidence_id: str
    quote: str


class Fact(BaseModel):
    statement: str
    citations: list[Citation]
```

调用方可以传入字典，Pydantic 会把它们转换成 `Citation` 对象：

```python
fact = Fact(
    statement="配置中声明了模型名称",
    citations=[
        {
            "evidence_id": "evidence-8",
            "quote": "ABOOK_MODEL",
        }
    ],
)

print(type(fact.citations[0]))
# <class '...Citation'>
```

ABook 的数据链条就是类似的嵌套关系：

```text
TaskFact
    -> EvidenceCitation
        -> EvidenceRecord
```

这使得保存下来的事实不仅有陈述文本，也保留了已经验证过的证据记录和引用原文。

## 7. 创建模型时的几种入口

### 7.1 使用关键字参数

```python
record = EvidenceRecord(
    evidence_id="evidence-1",
    content="实际工具结果",
)
```

这是业务代码中最直观的方式。

### 7.2 从字典验证：`model_validate`

```python
payload = {
    "evidence_id": "evidence-1",
    "content": "实际工具结果",
}

record = EvidenceRecord.model_validate(payload)
```

适用于读取 JSON、HTTP 请求或其他外部数据。外部数据进入业务逻辑前，应先经过
这种模型验证，而不是直接当作可信字典使用。

### 7.3 从 JSON 验证：`model_validate_json`

```python
record = EvidenceRecord.model_validate_json(
    '{"evidence_id":"evidence-1","content":"工具结果"}'
)
```

它会先解析 JSON，再执行模型验证。

## 8. 模型序列化

### 8.1 `model_dump`

```python
data = record.model_dump()
```

返回普通 Python 字典，适合交给 JSON 存储层继续处理。

### 8.2 `model_dump_json`

```python
json_text = record.model_dump_json()
```

直接返回 JSON 字符串。

### 8.3 嵌套模型的序列化

`model_dump()` 会递归处理嵌套的 Pydantic 模型：

```python
fact.model_dump()
# {
#     "statement": "...",
#     "citations": [
#         {"evidence_id": "evidence-8", "quote": "ABOOK_MODEL"}
#     ]
# }
```

持久化层通常应该保存 `model_dump()` 的结果，而不是依赖模型对象本身可被直接
写入 JSON。

## 9. ABook 中 `EvidenceRecord` 的完整链路

### 9.1 工作区工具登记证据

工作区工具执行成功后，会调用 Runtime 的 `register_evidence`：

```python
evidence = ctx.deps.runtime.register_evidence(
    "file_read",
    "config.py",
    "lines 1-80",
    result,
    ctx.deps.agent_context.agent_id,
)
```

Runtime 创建：

```python
EvidenceRecord(
    evidence_id="evidence-8",
    agent_id="worker-1",
    kind="file_read",
    source="config.py",
    detail="lines 1-80",
    content=result,
)
```

并把它放进 `runtime.evidence_records`。

### 9.2 工具结果暴露证据 ID

返回给模型的工具文本第一行包含：

```text
[evidence_id=evidence-8]
ABOOK_MODEL=demo
```

这样模型知道以后应该引用哪个 ID。证据 ID 不是事实本身，而是找到原始工具结果
的索引。

### 9.3 模型提交事实声明

模型提交的结构化内容类似：

```python
FactClaim(
    statement="配置文件中声明了模型名称",
    citations=[
        EvidenceQuoteClaim(
            evidence_id="evidence-8",
            quote="ABOOK_MODEL=demo",
        )
    ],
)
```

### 9.4 Runtime 验证引用

Runtime 会检查：

```python
record = runtime.evidence_records.get(citation.evidence_id)
if record is None:
    raise ValueError("未知证据 ID")

if citation.quote.strip() not in record.content:
    raise ValueError("证据中不存在引用原文")
```

验证成功后，`FactClaim` 才会转换成 `TaskFact`，再保存到
`TaskState.important_facts`。因此模型不能只凭记忆生成一个事实，也不能引用一个
不存在的证据 ID。

## 10. `model_config` 的其他常用配置

以下配置不一定都适合 ABook，但理解它们有助于阅读 Pydantic 代码。

### `validate_assignment`

默认情况下，普通模型创建后重新赋值不一定会再次验证。可以开启：

```python
class Settings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    timeout_seconds: int
```

这样后续执行 `settings.timeout_seconds = "bad"` 时也会触发验证。对于需要可变的
配置对象，这比完全冻结更合适。

### `validate_default`

让默认值也参与验证：

```python
class Settings(BaseModel):
    model_config = ConfigDict(validate_default=True)

    timeout_seconds: int = "30"
```

### `str_strip_whitespace`

让字符串输入自动去除首尾空白：

```python
class UserInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str
```

如果空白处理有业务含义，不应全局开启，而应在具体字段或业务边界显式处理。

### `strict`

开启严格模式，减少类型自动转换：

```python
class StrictPayload(BaseModel):
    model_config = ConfigDict(strict=True)

    count: int
```

外部输入、协议字段和安全敏感数据适合考虑严格模式；普通配置则应根据兼容性需求
决定是否使用。

### `arbitrary_types_allowed`

允许字段使用未被 Pydantic 详细建模的 Python 类型。只有确实需要保存第三方对象
或运行时对象时才应启用。对于需要 JSON 持久化的数据，优先定义明确的嵌套模型，
而不是放宽类型检查。

## 11. 配置的继承和覆盖

父模型的配置可以被子模型继承或覆盖：

```python
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(StrictModel):
    evidence_id: str
```

`Evidence` 默认继承 `extra="forbid"`。如果子类重新定义：

```python
class FlexibleEvidence(StrictModel):
    model_config = ConfigDict(extra="allow")

    evidence_id: str
```

则子类的配置会覆盖父类对应配置。项目中应避免随意覆盖基础模型的安全约束，
除非子类确实有不同的数据协议。

## 12. 常见误区

### 把模型当作普通字典

错误写法：

```python
record["content"]
```

应使用：

```python
record.content
```

需要字典时再显式调用：

```python
record.model_dump()
```

### 以为类型注解本身会验证数据

```python
class Payload:
    count: int
```

普通 Python 类中的注解主要是说明，不能自动阻止错误数据。继承
`BaseModel` 后，Pydantic 才会在模型边界执行验证。

### 使用 `extra="ignore"` 掩盖拼写错误

如果输入字段拼写错误，而配置是 `ignore`，错误字段可能被静默丢弃。对于 Runtime
状态、工具参数和模型结构化输出，通常应优先使用 `extra="forbid"`。

### 把 `frozen=True` 当作深度冻结

它主要阻止模型属性重新赋值，并不自动让所有内部容器变成不可变对象。需要真正的
不可变数据时，应同时设计字段类型和复制策略。

### 直接把模型对象写入 JSON

应先调用：

```python
json_data = model.model_dump()
```

然后交给 JSON 库或项目的持久化层处理。

## 13. ABook 的使用约定

在这个项目中，可以遵循以下判断：

1. 外部输入、模型输出和持久化快照应定义为 `BaseModel`。
2. 有明确字段协议的数据模型优先使用 `extra="forbid"`。
3. 工具结果、证据记录和事实引用这类不可篡改数据使用 `frozen=True`。
4. 长度、数量和取值范围使用 `Field`、`Literal` 或显式校验限制。
5. 进入存储层前先完成 Pydantic 验证。
6. 保存 JSON 时使用 `model_dump()`，读取 JSON 时使用 `model_validate()`。
7. 不要把 `BaseModel` 当作业务逻辑容器；模型负责数据结构和验证，领域行为仍应
   放在对应的 Runtime、存储层或服务函数中。

简化地说，ABook 使用 Pydantic 模型是为了建立清晰的数据边界：

```text
    不可信输入
        -> BaseModel 验证
        -> 结构化、受约束的对象
        -> 业务逻辑或持久化
```
