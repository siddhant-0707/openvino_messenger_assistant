# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_all, collect_submodules, collect_dynamic_libs

datas = [
    ('docs/images/OpenVINO_logo.png', 'docs/images'),
    ('examples', 'examples'),
    ('src', 'src'),  # include project sources so imports work in frozen app
]
binaries = []
hiddenimports = [
    # Project-local modules that are sometimes imported dynamically
    'telegram_rag_gradio',
    'telegram_rag_integration',
    'telegram_ingestion',
    'ov_langchain_helper',
    'llm_config',
    'npu_models',
    'genai_helper',
    'cmd_helper',
    'pip_helper',
    'gradio_helper',
    'notebook_utils',
]

# One-shot robust collection for key packages
pkgs = [
    # UI / Qt
    'PySide6',
    # OpenVINO stack
    'openvino', 'openvino_genai', 'openvino_tokenizers',
    # NLP / HF
    'transformers', 'tokenizers', 'huggingface_hub',
    # LangChain split packages
    'langchain', 'langchain_core', 'langchain_community', 'langchain_text_splitters',
    # Vector stores / numerical libs
    'faiss', 'numpy', 'scipy',
    # Gradio stack & deps
    'gradio', 'gradio_client', 'safehttpx', 'httpx', 'httpcore', 'anyio', 'starlette', 'h11', 'websockets',
    # Config / async / misc
    'pydantic', 'pydantic_core', 'typing_extensions', 'annotated_types', 'dotenv', 'nest_asyncio',
    # Telegram client
    'telethon',
    # Parsing helpers
    'newspaper', 'bs4', 'lxml',
    'groovy',
]
for p in pkgs:
    try:
        c_d, c_b, c_h = collect_all(p)
        datas += c_d; binaries += c_b; hiddenimports += c_h
    except Exception:
        pass


# Ensure OpenVINO GenAI compiled extension and DLLs are bundled
try:
    hiddenimports += collect_submodules('openvino_genai')
    binaries += collect_dynamic_libs('openvino_genai')
except Exception:
    pass

# Ensure core OpenVINO runtime DLLs are bundled
try:
    hiddenimports += collect_submodules('openvino')
    binaries += collect_dynamic_libs('openvino')
except Exception:
    pass

# Tokenizers for OpenVINO may also ship native libs
try:
    hiddenimports += collect_submodules('openvino_tokenizers')
    binaries += collect_dynamic_libs('openvino_tokenizers')
except Exception:
    pass

a = Analysis(
    ['run_qt_app.py'],
    pathex=['/home/sidd/Documents/GitHub/openvino_messenger_assistant', '/home/sidd/Documents/GitHub/openvino_messenger_assistant/src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TelegramRAG',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TelegramRAG',
)
