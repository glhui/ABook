# datetime 标准库

- 区分无时区时间与带时区时间；跨系统交换时优先使用带时区的 `datetime`。
- 使用 `datetime.fromisoformat` 和 `isoformat` 处理 ISO 8601，除非需求明确指定其他格式。
- 时间间隔使用 `timedelta` 表达，不手工拼接秒数与日期字段。
- 测试中使用固定输入，不依赖系统当前时间或本地时区。
