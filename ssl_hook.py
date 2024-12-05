
import os
import ssl
import sys

def ssl_hook():
    if sys.platform == 'darwin':
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        
ssl_hook()
