# chunky FAQ

## 安装问题

### Q: 为什么在某个 conda 环境中安装了 chunky，但在其他环境甚至非 conda 终端中也能执行 chunky 命令？

**原因**：

pip 默认安装到**当前 Python 对应的 site-packages/bin 目录**。如果你在 conda base 环境中用 pip 安装，chunky 会被安装到：

```
/opt/homebrew/Caskroom/miniforge/base/bin/chunky
```

由于 conda 默认配置 `auto_activate_base: True`，base 环境始终在 PATH 中，因此**无论激活哪个 conda 环境**，都能找到这个命令。

**验证方法**：
```bash
which chunky
# 输出类似: /opt/homebrew/Caskroom/miniforge/base/bin/chunky

conda config --show | grep auto_activate_base
# 如果是 True，说明 base 环境始终自动激活
```

**解决方案**：

1. **只让 chunky 在特定环境中可用**：
   ```bash
   conda activate 你的环境
   pip install -e ~/Desktop/play_code/chunky
   ```

2. **卸载 base 环境的 chunky**：
   ```bash
   conda activate base
   pip uninstall chunky
   
   # 然后在目标环境中安装
   conda activate 你的环境
   pip install -e ~/Desktop/play_code/chunky
   ```

3. **禁用 auto_activate_base**（可选）：
   ```bash
   conda config --set auto_activate_base False
   ```
   注意：这会让 base 环境不会自动激活，需要手动 `conda activate base`。

---

## 使用问题

### Q: chunky build 卡在 "Step 3-6/6 Processing chunks..." 不动

**可能原因**：

1. **Milvus 连接问题**：配置中 `use_lite: true` 但本地 Milvus-Lite 无法启动
2. **网络问题**：使用 API 方式的 embedding/LLM 但网络超时

**解决方案**：

```bash
# 检查配置
chunky config --list

# 如果使用 Milvus Docker，确保：
# 1. docker-compose.yml 存在
# 2. Milvus 容器正在运行
docker ps | grep milvus

# 配置使用 Milvus Docker
chunky milvus config
# 选择 "No" (不用 Lite)
# URI: localhost:19530
```

### Q: 如何切换到 ChromaDB？

```bash
chunky chroma --collection 你的集合名
```

或手动修改 `~/.config/chunky/config.yaml`：
```yaml
vector_store_type: chroma
```

### Q: 如何配置 vLLM embedding 模型？

```bash
chunky embedding config
# 选择 3 (vllm)
# Model name: e5-mistral-7b-instruct
# API base URL: http://localhost:8000
# API key: (根据需要填写)
```
