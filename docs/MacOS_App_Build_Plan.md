# Chunky macOS App 构建方案

基于 CoPaw 桌面应用调研的完整构建指南。

---

## 目录

1. [技术栈选择](#1-技术栈选择)
2. [项目结构](#2-项目结构)
3. [前端开发-console](#3-前端开发-console)
4. [后端集成](#4-后端集成)
5. [electron-配置](#5-electron-配置)
6. [构建与打包](#6-构建与打包)
7. [github-actions-cicd](#7-github-actions-cicd)
8. [快速开始](#8-快速开始)
9. [常见问题](#9-常见问题)

---

## 1. 技术栈选择

### 核心框架

| 层级 | 技术 | 说明 |
|------|------|------|
| **桌面框架** | Electron | 成熟稳定，跨平台支持好 |
| **前端框架** | React 18 + TypeScript | 类型安全，生态丰富 |
| **构建工具** | Vite 6 | 快速开发，热更新 |
| **UI 组件库** | Ant Design 5 | 企业级组件 |
| **后端** | Python + typer | 现有 chunky CLI |

### 选择 Electron 而非 Tauri 的原因

- ✅ CoPaw 成功案例，验证可行
- ✅ Node.js 生态成熟，electron-builder 打包方便
- ✅ Python 与 Electron 解耦，后端独立运行
- ✅ 支持 macOS App Store 分发

---

## 2. 项目结构

```
chunky/
├── src/                          # Python 后端 (现有)
│   └── chunky/
│       ├── cli/                  # CLI 入口
│       ├── core/                 # 核心功能
│       ├── parsers/              # 文档解析
│       ├── embedding/            # Embedding
│       ├── vector_store/         # 向量存储
│       └── utils/                # 工具函数
├── console/                      # Electron 前端 (新建)
│   ├── public/                   # 静态资源
│   ├── src/                     # React 源码
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── components/          # UI 组件
│   │   ├── pages/              # 页面
│   │   ├── hooks/             # React Hooks
│   │   ├── services/          # API 调用
│   │   └── styles/            # 样式
│   ├── electron/              # Electron 主进程
│   │   ├── main.ts           # 主进程入口
│   │   ├── preload.ts        # 预加载脚本
│   │   └── ipc.ts            # IPC 通信
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── electron-builder.yml
├── desktop/                    # 桌面应用相关 (新建)
│   ├── build/                 # 构建资源
│   │   ├── icon.icns         # macOS 图标
│   │   └── entitlements.mac.plist
│   └── scripts/              # 构建脚本
│       └── build-macos.sh
├── pyproject.toml              # Python 项目配置
└── docs/
    └── MacOS_App_Build_Plan.md
```

---

## 3. 前端开发 (Console)

### 3.1 初始化前端项目

```bash
cd ~/Desktop/play_code/chunky
mkdir -p console
cd console

# 初始化 npm 项目
npm init -y

# 安装依赖
npm install react@18 react-dom@18 react-router-dom@7
npm install antd@5 antd-style@3 @ant-design/x-markdown
npm install ahooks@3 i18next react-i18next lucide-react
npm install @dnd-kit/core @dnd-kit/sortable
npm install react-markdown remark-gfm

# 开发依赖
npm install -D typescript@5.8 @types/react@18 @types/react-dom@18
npm install -D @vitejs/plugin-react@4 vite@6
npm install -D eslint@9 prettier@3 globals@16
```

### 3.2 package.json 配置

```json
{
  "name": "chunky-console",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite --host",
    "build": "tsc -b && vite build",
    "build:prod": "tsc -b && vite build --mode production",
    "preview": "vite preview",
    "lint": "eslint .",
    "format": "prettier --write ."
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "antd": "^5.29.1",
    "lucide-react": "^0.562.0"
  },
  "devDependencies": {
    "typescript": "~5.8.3",
    "vite": "^6.3.5",
    "@vitejs/plugin-react": "^4.4.1"
  }
}
```

### 3.3 Vite 配置 (vite.config.ts)

```typescript
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiBaseUrl = env.BASE_URL ?? "";

  return {
    define: {
      BASE_URL: JSON.stringify(apiBaseUrl),
      MOBILE: false,
    },
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      host: "0.0.0.0",
      port: 5173,
    },
    build: {
      // 输出到 chunky Python 包的 console 目录
      outDir: path.resolve(__dirname, "../src/chunky/console"),
      emptyOutDir: true,
    },
  };
});
```

### 3.4 前端页面设计

#### 主要页面

1. **Dashboard** - 知识库概览、统计信息
2. **Build** - 文档构建界面、上传文件
3. **Search** - 混合搜索界面
4. **Collections** - 集合管理
5. **Settings** - 配置管理

#### 组件结构

```
src/
├── components/
│   ├── Layout/           # 布局组件 (Sidebar, Header, Content)
│   ├── Build/            # 构建相关 (FileUploader, ProgressBar)
│   ├── Search/            # 搜索相关 (SearchBar, ResultList)
│   ├── Collections/      # 集合管理 (CollectionList, CollectionCard)
│   └── Settings/         # 设置相关 (ConfigForm, ModelSelector)
├── pages/
│   ├── Dashboard.tsx
│   ├── Build.tsx
│   ├── Search.tsx
│   ├── Collections.tsx
│   └── Settings.tsx
├── hooks/
│   ├── useChunky.ts      # chunky CLI 调用
│   ├── useConfig.ts      # 配置管理
│   └── useSearch.ts      # 搜索功能
└── services/
    └── chunkyApi.ts      # IPC 通信封装
```

---

## 4. 后端集成

### 4.1 Python Web 服务 (新增)

在 `src/chunky/cli/` 下新增 web_server.py：

```python
"""Chunky Web Server for Desktop App"""
import uvicorn
import typer
from pathlib import Path

app = typer.Typer(help="Chunky Web Server")

@app.command()
def main(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
):
    """Start the Chunky web server."""
    uvicorn.run(
        "chunky.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )

if __name__ == "__main__":
    app()
```

### 4.2 FastAPI 服务 (新增)

在 `src/chunky/api/` 下创建：

```python
"""Chunky REST API for Desktop App"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import subprocess
import json

app = FastAPI(title="Chunky API")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BuildRequest(BaseModel):
    directory: str
    collection: Optional[str] = None
    test_mode: bool = False

class SearchRequest(BaseModel):
    query: str
    collection: str
    top_k: int = 5

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.post("/api/build")
async def build_knowledge_base(req: BuildRequest):
    """触发知识库构建"""
    cmd = ["chunky", "build", "--dir", req.directory]
    if req.collection:
        cmd.extend(["--collection", req.collection])
    if req.test_mode:
        cmd.append("--test")

    result = subprocess.run(cmd, capture_output=True, text=True)
    return {"success": result.returncode == 0, "output": result.stdout, "error": result.stderr}

@app.post("/api/search")
async def search(req: SearchRequest):
    """执行混合搜索"""
    cmd = ["chunky", "search", req.query, "--collection", req.collection, "-k", str(req.top_k)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {"success": result.returncode == 0, "output": result.stdout, "error": result.stderr}

@app.get("/api/collections")
async def list_collections():
    """列出所有集合"""
    cmd = ["chunky", "collections", "list"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {"success": result.returncode == 0, "output": result.stdout}

@app.get("/api/config")
async def get_config():
    """获取配置"""
    cmd = ["chunky", "config", "--list"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {"success": result.returncode == 0, "output": result.stdout}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

---

## 5. Electron 配置

### 5.1 Electron 主进程 (electron/main.ts)

```typescript
import { app, BrowserWindow, ipcMain, Menu, shell } from "electron";
import path from "path";
import { spawn, execSync } from "child_process";

// 保持窗口引用
let mainWindow: BrowserWindow | null = null;
let chunkyProcess: ReturnType<typeof spawn> | null = null;

const isDev = process.env.NODE_ENV === "development";

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: "Chunky",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // 加载内容
  if (isDev) {
    mainWindow.loadURL("http://localhost:5173");
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, "../console/index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// 创建应用菜单
function createMenu() {
  const template: Electron.MenuItemConstructorOptions[] = [
    {
      label: "File",
      submenu: [
        { label: "Open Folder...", accelerator: "CmdOrCtrl+O", click: () => {} },
        { type: "separator" },
        { role: "quit" },
      ],
    },
    {
      label: "Edit",
      submenu: [
        { role: "undo" },
        { role: "redo" },
        { type: "separator" },
        { role: "cut" },
        { role: "copy" },
        { role: "paste" },
      ],
    },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
      ],
    },
    {
      label: "Window",
      submenu: [
        { role: "minimize" },
        { role: "zoom" },
        { role: "close" },
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// IPC 通信：执行 chunky 命令
ipcMain.handle("chunky:execute", async (_event, args: string[]) => {
  return new Promise((resolve, reject) => {
    const proc = spawn("chunky", args, { shell: true });
    let stdout = "";
    let stderr = "";

    proc.stdout?.on("data", (data) => { stdout += data.toString(); });
    proc.stderr?.on("data", (data) => { stderr += data.toString(); });

    proc.on("close", (code) => {
      resolve({ code, stdout, stderr });
    });

    proc.on("error", (err) => {
      reject(err);
    });
  });
});

// IPC 通信：获取配置
ipcMain.handle("chunky:getConfig", async () => {
  try {
    const result = execSync("chunky config --list", { encoding: "utf-8" });
    return { success: true, output: result };
  } catch (e: any) {
    return { success: false, error: e.message };
  }
});

// 启动
app.whenReady().then(() => {
  createMenu();
  createWindow();
});

app.on("window-all-closed", () => {
  if (chunkyProcess) {
    chunkyProcess.kill();
  }
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
```

### 5.2 预加载脚本 (electron/preload.ts)

```typescript
import { contextBridge, ipcRenderer } from "electron";

// 暴露安全的 API 给渲染进程
contextBridge.exposeInMainWorld("chunky", {
  execute: (args: string[]) => ipcRenderer.invoke("chunky:execute", args),
  getConfig: () => ipcRenderer.invoke("chunky:getConfig"),
});
```

### 5.3 TypeScript 类型声明

```typescript
// src/types/chunky.d.ts
declare global {
  interface Window {
    chunky: {
      execute: (args: string[]) => Promise<{ code: number; stdout: string; stderr: string }>;
      getConfig: () => Promise<{ success: boolean; output?: string; error?: string }>;
    };
  }
}

export {};
```

### 5.4 electron-builder 配置 (electron-builder.yml)

```yaml
appId: com.chunky.app
productName: Chunky
copyright: Copyright © 2024 Chunky

directories:
  output: dist-electron
  buildResources: build

files:
  - src/chunky/**/*
  - console/**/*
  - "!node_modules/**/*"
  - node_modules/electron/**/*
  - node_modules/electron-builder/**/*

mac:
  category: public.app-category.productivity
  target:
    - target: dmg
      arch: [x64, arm64]
  artifactName: ${productName}-${version}-macOS.${ext}

dmg:
  contents:
    - x: 130
      y: 220
    - x: 410
      y: 220
      type: link
      path: /Applications

asar: true
```

### 5.5 electron-builder 安装

```bash
cd console
npm install -D electron@31 electron-builder@24
```

---

## 6. 构建与打包

### 6.1 本地开发

```bash
# 终端 1: 启动前端开发服务器
cd console
npm run dev

# 终端 2: 启动 Electron
cd console
npx electron .
```

### 6.2 本地打包

```bash
# 构建前端
cd console
npm run build

# 复制到 Python 包
mkdir -p ../src/chunky/console
cp -R dist/. ../src/chunky/console/

# 安装 Python 包
cd ..
pip install -e .

# 打包 Electron 应用
cd console
npm run build:electron
```

### 6.3 打包脚本 (package.json)

```json
{
  "scripts": {
    "dev": "vite --host",
    "build": "tsc -b && vite build",
    "build:electron": "electron-builder --mac",
    "build:electron:dir": "electron-builder --mac --dir",
    "postinstall": "electron-builder install-app-deps"
  }
}
```

### 6.4 macOS 应用图标

1. 准备 1024x1024 PNG 图片
2. 使用 iconutil 转换为 .icns:

```bash
# 在 macOS 上执行
mkdir -p desktop/build
# 使用预览.app 将 PNG 转为 icns
iconutil -c icns icon.png -o desktop/build/icon.icns
```

---

## 7. GitHub Actions CI/CD

### 7.1 桌面应用构建 Workflow

```yaml
# .github/workflows/desktop.yml
name: Desktop App Build

on:
  push:
    branches: [main, master]
    tags:
      - "v*"
  pull_request:
    branches: [main, master]

jobs:
  build-macos:
    runs-on: macos-latest
    strategy:
      matrix:
        include:
          - arch: x64
            artifact_name: "Chunky-${{ github.ref_name }}-macOS-x64.dmg"
          - arch: arm64
            artifact_name: "Chunky-${{ github.ref_name }}-macOS-arm64.dmg"

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Build Console Frontend
        run: |
          cd console
          npm ci
          npm run build

      - name: Copy Console to Package
        run: |
          mkdir -p src/chunky/console
          cp -R console/dist/. src/chunky/console/

      - name: Install Python Package
        run: |
          pip install -e .

      - name: Build Electron App
        run: |
          cd console
          npx electron-builder --mac ${{ matrix.arch == 'arm64' && '--arm64' || '' }}

      - name: Upload Artifact
        uses: actions/upload-artifact@v4
        with:
          name: ${{ matrix.artifact_name }}
          path: console/dist-electron/*.dmg

  release:
    needs: build-macos
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest

    steps:
      - name: Download Artifacts
        uses: actions/download-artifact@v4

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: "*/*.dmg"
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

---

## 8. 快速开始

### 8.1 克隆项目

```bash
git clone https://github.com/NJUxlj/chunky.git
cd chunky
```

### 8.2 构建前端

```bash
cd console
npm install
npm run build
```

### 8.3 集成到 Python 包

```bash
# 创建 console 目录
mkdir -p ../src/chunky/console

# 复制构建产物
cp -R dist/. ../src/chunky/console/
```

### 8.4 安装 Python 包

```bash
cd ..
pip install -e .
```

### 8.5 运行桌面应用

```bash
cd console
npx electron .
```

### 8.6 或者使用 Web 模式

```bash
chunky web
# 然后在浏览器打开 http://localhost:5173
```

---

## 9. 常见问题

### Q1: macOS 无法打开未签名的应用？

```
# 临时允许
xattr -cr /Applications/Chunky.app

# 或者在系统设置中手动允许
# 系统偏好设置 -> 安全性与隐私 -> 仍然打开
```

### Q2: Electron 打包后找不到 chunky 命令？

确保在应用启动时将 chunky 添加到 PATH，或者使用绝对路径调用：

```typescript
// 在 main.ts 中
const chunkyPath = isDev
  ? "/usr/local/bin/chunky"  // 开发模式
  : path.join(process.resourcesPath, "chunky");  // 生产模式
```

### Q3: 如何处理 Python 环境？

Electron 应用应该自带 Python 环境或者要求用户安装：

```yaml
# electron-builder.yml
extraMetadata:
  main: build/main.js
mac:
  category: public.app-category.productivity
  # 要求用户安装 Python 3.10+
  target:
    - dmg
```

### Q4: 如何支持 Apple Silicon (M1/M2)？

```bash
# 构建时指定架构
npx electron-builder --mac --arm64

# 或者同时构建两种架构
npx electron-builder --mac
# 自动生成 x64 和 arm64 两个版本
```

### Q5: 应用启动后如何保持后台运行？

```typescript
// main.ts
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// 防止退出（macOS 特性）
app.on("window-all-closed", () => {
  // 不退出，保持在 Dock 中
});
```

---

## 参考资料

- [CoPaw 源码仓库](https://github.com/agentscope-ai/CoPaw)
- [Electron 官方文档](https://www.electronjs.org/docs)
- [electron-builder 文档](https://www.electron.build/)
- [Vite 配置](https://vitejs.dev/config/)
- [Ant Design React](https://ant.design/components/overview)