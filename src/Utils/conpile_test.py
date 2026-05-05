import os
import subprocess

path_env = os.environ.get('PATH', '')
if 'Visual Studio' in path_env:
    print("✅ PyCharm sees Visual Studio variables!")
else:
    print("❌ PyCharm DOES NOT see Visual Studio variables.")

try:

    result = subprocess.run(['cl'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("✅ SUCCESS: Compiler found!")
    print(result.stdout.split('\n')[0])
except FileNotFoundError:
    print("❌ ERROR: 'cl.exe' not found.")