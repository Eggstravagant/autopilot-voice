"""
Build step: download the model and voices from the kokoro-onnx release and check their SHA-256, so a
broken or tampered download fails the build instead of shipping. Usage: python fetch_model.py fp16|int8
"""
import hashlib
import sys
import time
import urllib.request

BASE = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/'
FILES = {
    'kokoro-v1.0.fp16.onnx': 'c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204',
    'kokoro-v1.0.int8.onnx': '6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb',
    'voices-v1.0.bin': 'bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d',
}


def fetch(name, want, tries=4):
    for attempt in range(1, tries + 1):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(BASE + name, timeout=60) as r, open(name, 'wb') as out:
                while chunk := r.read(1 << 20):
                    h.update(chunk)
                    out.write(chunk)
            if h.hexdigest() != want:
                raise ValueError(f'{name}: checksum {h.hexdigest()} is not {want}')
            print(f'{name}: ok', flush=True)
            return
        except Exception as e:  # network blips on a build machine: try again, then fail the build
            print(f'{name}: attempt {attempt} failed: {e}', flush=True)
            if attempt == tries:
                raise
            time.sleep(3 * attempt)


if __name__ == '__main__':
    variant = sys.argv[1] if len(sys.argv) > 1 else 'fp16'
    model = f'kokoro-v1.0.{variant}.onnx'
    if model not in FILES:
        sys.exit(f'unknown model {variant!r}: use fp16 or int8')
    for name in (model, 'voices-v1.0.bin'):
        fetch(name, FILES[name])
