# Release Guide - Telegram RAG System

This guide covers different ways to release and distribute your Telegram RAG System.

## 📦 **Release Options**

### **1. GitHub Release (Recommended)**

**Best for**: Open source distribution, community contributions, and version tracking.

```bash
# 1. Prepare the release
git add .
git commit -m "Release v2.0.0: Qt interface and enhanced features"
git tag v2.0.0
git push origin main --tags

# 2. Create GitHub release
# Go to GitHub → Releases → Create new release
# - Tag: v2.0.0
# - Title: "Telegram RAG System v2.0.0 - Qt Desktop Interface"
# - Description: Copy from CHANGELOG.md
# - Upload assets: source code zip/tar.gz
```

**GitHub Release Features:**
- ✅ **Version tracking** with git tags
- ✅ **Release notes** from CHANGELOG.md
- ✅ **Download statistics** and analytics
- ✅ **Community features** (issues, discussions, wiki)
- ✅ **Free hosting** for open source projects

### **2. PyPI Package Distribution**

**Best for**: Easy installation via `pip install`.

```bash
# 1. Install build tools
pip install build twine

# 2. Build the package
python -m build

# 3. Upload to TestPyPI (testing)
python -m twine upload --repository testpypi dist/*

# 4. Test installation
pip install --index-url https://test.pypi.org/simple/ telegram-rag-openvino

# 5. Upload to PyPI (production)
python -m twine upload dist/*
```

**PyPI Benefits:**
- ✅ **Easy installation**: `pip install telegram-rag-openvino`
- ✅ **Dependency management** handled automatically
- ✅ **Version management** with semantic versioning
- ✅ **Discovery** through PyPI search

### **3. Docker Distribution**

**Best for**: Consistent environment and easy deployment.

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    qt6-base-dev \
    libqt6gui6 \
    libqt6widgets6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Install the application
RUN pip install -e .

# Expose port for Gradio interface
EXPOSE 7860

# Command to run the application
CMD ["telegram-rag-gradio"]
```

Build and distribute:
```bash
# Build image
docker build -t telegram-rag-openvino:2.0.0 .

# Push to Docker Hub
docker tag telegram-rag-openvino:2.0.0 yourusername/telegram-rag-openvino:2.0.0
docker push yourusername/telegram-rag-openvino:2.0.0

# Or push to GitHub Container Registry
docker tag telegram-rag-openvino:2.0.0 ghcr.io/yourusername/telegram-rag-openvino:2.0.0
docker push ghcr.io/yourusername/telegram-rag-openvino:2.0.0
```

### **4. Executable Distribution**

**Best for**: Non-technical users who don't want to install Python.

Using PyInstaller:
```bash
# Install PyInstaller
pip install pyinstaller

# Create executable for Qt interface
pyinstaller --onefile --windowed \
    --add-data "requirements.txt:." \
    --add-data "llm_config.py:." \
    --name TelegramRAG \
    telegram_rag_qt.py

# Create executable for Gradio interface  
pyinstaller --onefile \
    --add-data "requirements.txt:." \
    --add-data "llm_config.py:." \
    --name TelegramRAG-Web \
    telegram_rag_gradio.py
```

Distribution package structure:
```
TelegramRAG-v2.0.0/
├── TelegramRAG.exe (or TelegramRAG on Linux/Mac)
├── TelegramRAG-Web.exe
├── env.example
├── README.md
├── LICENSE
└── docs/
```

### **5. Conda Package**

**Best for**: Data science community and conda users.

Create `meta.yaml`:
```yaml
{% set name = "telegram-rag-openvino" %}
{% set version = "2.0.0" %}

package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  path: ..

build:
  number: 0
  script: {{ PYTHON }} -m pip install . -vv
  entry_points:
    - telegram-rag = run_qt_app:main
    - telegram-rag-qt = telegram_rag_qt:main
    - telegram-rag-gradio = telegram_rag_gradio:main

requirements:
  host:
    - python >=3.8
    - pip
    - setuptools
  run:
    - python >=3.8
    - openvino >=2024.0.0
    - pyside6 >=6.6.0
    - langchain >=0.1.0
    # ... other dependencies

test:
  imports:
    - telegram_rag_qt
    - telegram_rag_gradio
  commands:
    - telegram-rag --help

about:
  home: https://github.com/yourusername/telegram-rag-openvino
  license: Apache-2.0
  license_file: LICENSE
  summary: Advanced Telegram message analysis using OpenVINO and RAG
```

## 🚀 **Recommended Release Strategy**

### **Phase 1: Development Release**
1. **GitHub Release**: For developers and early adopters
2. **TestPyPI**: For testing package distribution
3. **Documentation**: Complete README and docs

### **Phase 2: Community Release**
1. **PyPI Package**: Easy `pip install` for Python users
2. **Docker Images**: For containerized deployments
3. **Conda Package**: For data science community

### **Phase 3: End-User Release**
1. **Executables**: For non-technical users
2. **App Stores**: If applicable (Microsoft Store, etc.)
3. **Enterprise**: Custom packages for enterprise users

## 📋 **Release Checklist**

### **Pre-Release**
- [ ] Update version numbers in `setup.py` and `pyproject.toml`
- [ ] Update `CHANGELOG.md` with new features and fixes
- [ ] Update `README.md` with any new instructions
- [ ] Test installation from clean environment
- [ ] Test both Qt and Gradio interfaces
- [ ] Test with different Python versions (3.8-3.12)
- [ ] Update documentation and examples

### **Release Process**
- [ ] Create git tag with version number
- [ ] Push tag to GitHub
- [ ] Create GitHub release with changelog
- [ ] Build and upload to PyPI (optional)
- [ ] Build and push Docker images (optional)
- [ ] Update conda-forge recipe (optional)
- [ ] Announce on relevant channels

### **Post-Release**
- [ ] Monitor for issues and bug reports
- [ ] Update documentation based on user feedback
- [ ] Plan next release features
- [ ] Engage with community contributions

## 📊 **Distribution Comparison**

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **GitHub** | Free, version control, community | Requires git knowledge | Developers |
| **PyPI** | Easy install, dependency management | Python-only | Python users |
| **Docker** | Consistent environment | Requires Docker | Deployment |
| **Executable** | No dependencies | Large file size | End users |
| **Conda** | Popular in data science | Additional maintenance | Data scientists |

## 🎯 **Quick Start for Users**

After release, users can install via:

```bash
# Method 1: PyPI (if published)
pip install telegram-rag-openvino

# Method 2: GitHub
git clone https://github.com/yourusername/telegram-rag-openvino
cd telegram-rag-openvino
pip install -r requirements.txt

# Method 3: Docker
docker run -it telegram-rag-openvino:2.0.0

# Method 4: Download executable
# Download from GitHub releases, extract, and run
```

Choose the release method that best fits your target audience and distribution goals! 