# 亚马逊广告AI看板

> 一句话定位：把“广告 + 产品经营 + 生命周期”等数据整理成 **运营能抓重点** 的结构化输出；AI 只做解释与写报告，不参与算数。

## 目标与范围（Build in public v1）
- **当前目标（L0）**：只输出建议与清单，不修改广告
- **核心价值**：把多源报表统一口径，输出可执行的运营入口（Dashboard/Watchlists/Action Board）
- **范围外**：自动投放执行（L2）、广告 API 直连、利润精算（需更完整成本数据）

---

## 快速开始

### 1) 环境准备
- 推荐 Python `3.11~3.12`
- 建议使用虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 数据放置（默认结构）
```
（目前数据来源是赛狐ERP）

data/
  input/
    广告数据/            # 广告-报告下载 SP/SB/SD 报表（按日）
    产品分析/            # 数据-产品分析（按日）
    产品映射/            # 商品-商品列表
  output/
```

### 3) 一键运行
```bash
python main.py
```

### 4) 查看结果
- 入口文件：`data/output/<时间戳>/START_HERE.html`
- 每店铺聚焦版：`data/output/<时间戳>/<shop>/reports/dashboard.html`

---

## 依赖列表（建议版本）
- `pandas>=2.2,<3`
- `numpy>=2,<3`
- `openpyxl>=3.1,<4`
- `matplotlib>=3.8,<4`
- `seaborn>=0.13,<1`

---

## 常用命令示例

- **仅生成必要文件（更快更轻）**
```bash
python main.py --input-dir data/input --out-dir data/output --output-profile minimal
```

- **不生成报告 HTML/Markdown（只要 CSV/JSON）**
```bash
python main.py --input-dir data/input --out-dir data/output --no-report
```

- **只跑指定店铺**
```bash
python main.py --input-dir data/input --out-dir data/output --only-shop US
```

---

## AI 报告（可选）
默认不调用模型，避免误耗 token。需要时：
1) 复制 `.env.example` → `.env` 并填写：
   - `LLM_PROVIDER`
   - `LLM_API_KEY`
   - `LLM_MODEL`
2) 运行：
```bash
python main.py --input-dir data/input --out-dir data/output --ai-report
```

只想生成提示词不调用模型：
```bash
python main.py --input-dir data/input --out-dir data/output --ai-prompt-only
```

---

## 配置说明
- 运营策略：`config/ops_policy.json`
- 总选项档位：`config/ops_profile.json`（建议日常只改这里）

---

## Build in public v1 对齐清单
- ✅ 项目定位清晰（L0 输出建议、不改广告）
- ✅ 可运行（依赖 + 命令 + 默认目录）
- ✅ 输出入口明确（START_HERE.html / dashboard.html）
- ✅ 隐私边界清晰（.gitignore 统一管理）

---

## 故障排查
- **依赖安装失败**：优先使用虚拟环境再安装依赖
- **无输出/找不到报表**：确认 `--input-dir` 指向正确目录，且文件命名包含 SP/SB/SD + 报表关键词
- **Excel 读取警告**：`openpyxl` 的样式警告可忽略，不影响读取

---

## 许可与声明
- 本项目为离线分析工具，输出结果仅供运营决策参考
- 如需自动化投放或写回广告，请评估权限、审计与风控后再进入 L1/L2 阶段

## 作者与联系
- 博客：https://amzalysis.com/
- 公众号：跨境Ai视界

## 许可证
本项目采用 MIT License，详见 `LICENSE`。

