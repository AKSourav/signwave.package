# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Collect all necessary data for packages
datas = [('modelislv17.p', '.')]
binaries = []
hiddenimports = [
    'uvicorn.logging',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'mediapipe',
    'mediapipe.python',
    'fastapi',
    'starlette',
    'pydantic',
    'websockets',
    'websockets.legacy',
    'websockets.legacy.client',
    'websockets.legacy.server',
    'typing_extensions',
    'opencv-python',
    'numpy',
    'pickle',
    'base64',
    'json',
    'sklearn',
    'sklearn.base',
    'sklearn.utils',
    'sklearn.utils.validation',
    'sklearn.preprocessing',
    'sklearn.model_selection',
    'sklearn.tree',
    'sklearn.ensemble',
    'sklearn.linear_model',
    'sklearn.pipeline',
    'sklearn.feature_selection',
    'sklearn.neighbors',
    'sklearn.metrics',
    'sklearn.svm',
    'joblib',
    'torch',
    'torch.nn',
    'torch.utils',
    'torch.utils.data',
    'torchvision',
] + collect_submodules('mediapipe')

# Collect additional dependencies for MediaPipe
mediapipe_ret = collect_all('mediapipe')
datas += mediapipe_ret[0]
binaries += mediapipe_ret[1]
hiddenimports += mediapipe_ret[2]

# Collect additional dependencies for OpenCV
opencv_ret = collect_all('cv2')
datas += opencv_ret[0]
binaries += opencv_ret[1]
hiddenimports += opencv_ret[2]

# Collect additional dependencies for FastAPI
fastapi_ret = collect_all('fastapi')
datas += fastapi_ret[0]
binaries += fastapi_ret[1]
hiddenimports += fastapi_ret[2]

# Collect additional dependencies for Scikit-learn
sklearn_ret = collect_all('sklearn')
datas += sklearn_ret[0]
binaries += sklearn_ret[1]
hiddenimports += sklearn_ret[2]

# Collect PyTorch dependencies
torch_ret = collect_all('torch')
datas += torch_ret[0]
binaries += torch_ret[1]
hiddenimports += torch_ret[2]

# Add environment variables
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SignLanguageApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Create a directory for additional files if needed
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SignLanguageApp'
)