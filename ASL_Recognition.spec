# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os
import sys

block_cipher = None

# Create runtime hook for SSL
with open('ssl_hook.py', 'w') as f:
    f.write('''
import os
import ssl
import sys

def ssl_hook():
    if sys.platform == 'darwin':
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        
ssl_hook()
''')

# Collect all necessary dependencies
hidden_imports = [
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.main',
    'sklearn',
    'sklearn.neighbors',
    'sklearn.ensemble',
    'sklearn.tree',
    'mediapipe',
    'mediapipe.python',
    'fastapi',
    'fastapi.applications',
    'fastapi.routing',
    'fastapi.responses',
    'cv2',
    'numpy',
    'starlette',
    'starlette.routing',
    'starlette.middleware',
    'anyio',
    'email',
    'email.mime',
    'email.mime.text',
    'typing_extensions',
    'asyncio',
    '_ssl',
    'ssl',
    'certifi'
]

# Collect all mediapipe data files
mediapipe_datas = collect_data_files('mediapipe')

a = Analysis(
    ['main.py'],  # Your main script
    pathex=[],
    binaries=[],
    datas=[
        ('model1.p', '.'),  # Your model file
        *mediapipe_datas,  # MediaPipe data files
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['ssl_hook.py'],  # Add the SSL hook
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove conflicting libraries
def remove_from_list(source_list, filenames):
    return [item for item in source_list if not any(
        item[0].endswith(filename) for filename in filenames
    )]

a.binaries = remove_from_list(a.binaries, [
    'libcrypto.1.1.dylib',
    'libcrypto.3.dylib',
    'libssl.1.1.dylib',
    'libssl.3.dylib',
    'libpython3.7m.dylib',
    'libpython3.8.dylib',
    'libpython3.9.dylib',
    'libpython3.10.dylib',
])

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # Changed to True to create a directory instead of single file
    name='ASL_Recognition',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True
    # icon='app_icon.ico'
)

# Create a directory containing all dependencies
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ASL_Recognition'
)

# For macOS, create a .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,  # Changed from exe to coll
        name='ASL_Recognition.app',
        # icon='app_icon.ico',
        bundle_identifier='com.aslrecognition.app',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'LSBackgroundOnly': 'False',
            'NSRequiresAquaSystemAppearance': 'False',
            'CFBundleShortVersionString': '1.0.0',
        },
    )