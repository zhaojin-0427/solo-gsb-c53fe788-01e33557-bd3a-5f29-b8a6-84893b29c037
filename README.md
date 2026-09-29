# 印前色彩软打样台（Prepress Color Soft-Proofing Station）

本地运行的 Web 应用：上传 **RGB / CMYK 的 TIFF、PNG 或 JPEG** 图像，选择**源 ICC 配置、印刷 ICC 配置、显示 ICC 配置**与渲染意图，由后端通过 **LittleCMS（Pillow ImageCms）** 完成全链路色彩转换，并在浏览器中进行显示器软打样、像素级 **CIEDE2000 色差**分析与区域取色。

- 后端：Python 3.12 · FastAPI · SQLite · Pillow ImageCms（LittleCMS 2）· NumPy
- 前端：原生 HTML / CSS / JavaScript（无框架、无构建步骤）
- 部署：Docker Compose 一键启动

---

## 一键启动

需要安装 Docker（含 Compose v2）。

```bash
docker compose up -d --build
```

启动后浏览器访问：

> **http://localhost:8000**

停止：

```bash
docker compose down
```

数据（上传的图像任务、ICC 配置、SQLite 数据库）保存在 Docker 卷 `softproof-data`（容器内 `/data`）中，容器重启后仍然存在。

### 不使用 Docker（本地开发）

```bash
pip install -r requirements.txt
# 还需要系统安装 LittleCMS 2 共享库，例如 Debian/Ubuntu：
# sudo apt-get install liblcms2-2
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 功能与色彩流程

### 处理链路

对每一张上传图像，后端依次执行（全部由 LittleCMS 完成转换，**绝不使用浏览器默认色彩解释**）：

1. **源图显示预览**：源像素 `源配置 → 显示配置`；
2. **沿 源 → 印刷 → Lab 回算色值**（Lab 由 LittleCMS 以 **TYPE_Lab_DBL 原生 double 高精度**直接输出，不经 ICC v2 8 位 PCS 字节量化）：
   - 源像素 `源配置 → 印刷配置` 得到设备 **CMYK** 值；
   - CMYK `印刷配置 → Lab PCS（double）` 得到印刷后 Lab；
   - 同时源像素 `源配置 → Lab PCS（double）` 得到原始 Lab；
3. **软打样图**：印刷后 Lab（double）`→ 显示配置`（按所选显示渲染意图，呈现仍使用所选显示 ICC）；
4. 对 `Lab_源` 与 `Lab_印刷后` 两份高精度 Lab 逐像素计算 **CIEDE2000（ΔE00）**；
5. 生成：
   - **ΔE 热图**（黑→蓝→青→绿→黄→红伪彩色）；
   - **热图叠加图**（半透明叠加在软打样上）；
   - **超阈值二值遮罩**（ΔE00 ≥ 阈值的像素为白色）；
   - 均值 / P95 / 最大 ΔE00 与**超阈值面积比例**统计。

热图、遮罩、统计、框选区域的 Lab/色差值与 `report.json` 全部取自同一份逐像素 double Lab / ΔE00 结果，因此**阈值设在像素实际 ΔE00 附近时不会因额外的 8 位量化翻转临界像素的遮罩归属**。

输出给浏览器的 PNG 均嵌入 ICC 配置，浏览器不会按未标记数据自行解释：**源图显示预览与软打样嵌入所选择的显示配置**（像素本身位于该显示空间）；**ΔE 热图、遮罩与叠加图是无物理色度含义的合成可视化，嵌入 sRGB** 以保证颜色稳定呈现。

### 渲染意图

源→印刷、Lab→显示两个阶段分别可选：

- 感知（Perceptual）
- **相对色度（Relative Colorimetric，默认）**
- 饱和度（Saturation）
- 绝对色度（Absolute Colorimetric）

### 区域框选

在任一预览图上**鼠标拖拽框选矩形区域**，右侧面板显示该区域：

- 源 Lab 与印刷后 Lab 的均值/最小/最大；
- 印刷 **CMYK（C,M,Y,K 百分比，0% 无墨 / 100% 满版）**均值与最大；
- 区域 ΔE00 均值 / P95 / 最大 / 超阈值面积比例；
- 5×5 采样网格的各阶段数值。

### 导出

每个任务可导出（单个或一键 ZIP 打包）：

- 源图显示预览 PNG、软打样 PNG、ΔE 热图 PNG、热图叠加 PNG、超阈值遮罩 PNG；
- `report.json`：**绑定原图 SHA-256、三份配置的名称/类/色彩空间/SHA-256 摘要、渲染意图与阈值参数、统计结果**。

### 确定性（重复输入结果一致）

任务 ID = 原图哈希 + 三份配置哈希 + 两个渲染意图 + 阈值 的 SHA-256。相同输入会命中同一任务目录与同一报告；报告不含时间戳，ZIP 条目时间固定为 1980-01-01，因此**重复输入产生字节级一致的结果**。

---

## 严格的配置/通道校验

以下情况后端一律返回 `422` 并拒绝处理（不会按浏览器默认色彩空间解释图像）：

| 情形 | 处理 |
| --- | --- |
| 未选择/找不到所需 ICC 配置 | 拒绝 |
| ICC 文件损坏、缺少 `acsp` 签名、lcms 无法打开 | 拒绝（上传配置时即拒绝） |
| 印刷配置不是 CMYK 输出配置（非 `prtr/spac` + `CMYK`） | 拒绝 |
| 显示配置不是 RGB 显示器配置（非 `mntr` + `RGB`） | 拒绝 |
| 图像通道数/色彩空间与源配置不匹配（例如 RGB 图配 CMYK 源） | 拒绝 |
| 勾选“使用内嵌配置”但图像无内嵌 ICC，或内嵌 ICC 无效 | 拒绝 |
| 16 位、索引色（P/LA/I 等）、GIF/WebP 等非支持格式 | 拒绝，请先在印前软件中转换并嵌入 ICC |
| RGBA 图 | 透明通道合成到白色后按 3 通道 RGB 处理（唯一的规范化操作） |
| 图像最长边超过上限（默认 3000px） | 拒绝 |

支持的输入容器：**TIFF / PNG / JPEG**；支持的图像色彩空间：**8 位 RGB、RGBA、CMYK、灰度 L**（灰度需搭配灰度源配置）。

---

## 内置与自定义 ICC 配置

首次启动时应用自动生成：

- `sRGB IEC61966-2.1 (built-in)`：默认可选的源 / 显示配置；
- `CIELAB identity (built-in)`：流水线内部使用的 Lab PCS 配置（ICC v2）；
- `Soft Proofing Demo SWOP-like CMYK (synthetic)`：**合成的、自包含的合法 CMYK 输出配置**，让开箱即可演示完整 RGB→CMYK→显示链路。其色域模型是简化模型（互补油墨 + sRGB 色度），**仅供演示，不可用于生产**。

生产使用时，在页面“**上传自定义 ICC 配置**”处上传真实的印刷机/打样机 CMYK 配置（如 GRACoL / SWOP / FOGRA `.icc/.icm`），或把文件放入数据目录（见下）。也可直接上传显示器校准后的 RGB 配置作为显示配置。

上传的配置按角色存放（容器内 `/data`）：

```
/data/profiles/source/    # 源（RGB/CMYK/灰度）
/data/profiles/press/     # 印刷（CMYK）
/data/profiles/display/   # 显示（RGB）
/data/profiles/builtin/   # 内置自动生成，勿手动修改
```

将 `.icc/.icm` 文件放进对应目录后刷新页面即可识别。

---

## HTTP API 摘要

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| GET | `/api/profiles` | 按 source/press/display 分组列出可用配置 |
| POST | `/api/profiles/{source\|press\|display}` | 上传并严格校验 ICC 配置（multipart `file`） |
| POST | `/api/jobs` | 上传图像并执行软打样（multipart 表单，见下） |
| GET | `/api/jobs` | 最近任务列表（SQLite） |
| GET | `/api/jobs/{id}` | 任务 JSON 报告 |
| GET | `/api/jobs/{id}/images/{name}.png` | preview / softproof / heatmap / overlay / mask |
| GET | `/api/jobs/{id}/report` | 下载 report.json |
| GET | `/api/jobs/{id}/export.zip` | 全部 PNG + JSON 打包下载 |
| POST | `/api/jobs/{id}/region` | 区域分析，body：`{x,y,width,height}` |

`POST /api/jobs` 表单字段：

- `image`：图像文件；
- `source_profile` / `press_profile` / `display_profile`：配置 ID；
- `source_intent` / `proof_intent`：`perceptual | relative_colorimetric | saturation | absolute_colorimetric`；
- `threshold`：ΔE00 阈值（默认 3.0）；
- `use_embedded`：`true/false`，使用图像内嵌 ICC 作为源配置。

---

## 实现说明

- **色差公式**：`app/colorscience.py` 中为矢量化 CIEDE2000，已对照 Sharma 等 (2005) 标准测试数据验证（最大偏差 < 1e-3）。
- **Lab 精度**：源色与印后色通过 `app/lcms_native.py` 以 ctypes 直接调用系统 `liblcms2` 的 `TYPE_Lab_DBL` 变换，CIELAB 以**原生 double** 进入 CIEDE2000，不经过 Pillow 8 位 LAB 图像或 ICC v2 PCS 字节量化；区域取色时 Lab 数组以 float32 落盘（差异 < 1e-5 ΔE00）。软打样仍走 `TYPE_Lab_DBL → 所选显示 ICC（8 位 RGB）`。
- **合成 CMYK 配置**：`app/lcms_native.py` 通过 ctypes 调用系统 `liblcms2` 的 `cmsCreateProfilePlaceholder` / CLUT16 pipeline API 生成 17³/17⁴ 网格的合法 ICC v2.4 CMYK 配置，保证开箱演示可用。
- **环境变量**：
  - `SPD_DATA_DIR`（默认 `./data`）：数据根目录；
  - `SPD_MAX_DIMENSION`（默认 `3000`）：允许的图像最长边像素；
  - `SPD_MAX_UPLOAD_MB`（默认 `50`）：上传大小上限。

## 目录结构

```
.
├── app/
│   ├── main.py          # FastAPI 路由
│   ├── pipeline.py      # 色彩流水线、导出
│   ├── colorscience.py  # Lab 解码、CIEDE2000、热图
│   ├── profiles.py      # ICC 配置管理与严格校验
│   ├── regions.py       # 框选区域统计
│   ├── database.py      # SQLite
│   ├── lcms_native.py   # liblcms2 ctypes 绑定与内置配置生成
│   ├── config.py
│   └── static/          # 前端
├── tests/               # pytest 端到端测试
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### 运行测试

```bash
pip install -r requirements.txt pytest httpx
pytest -q
```
