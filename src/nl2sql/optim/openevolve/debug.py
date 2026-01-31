import os
import sys
from pathlib import Path
import openevolve

# 1. Locate the library source to understand its expectations
package_dir = Path(openevolve.__file__).parent
print(f"--- DIAGNOSTIC ---")
print(f"OpenEvolve installed at: {package_dir}")

# 2. Simulate the path joining logic
# This mimics what typical config loaders do
config_template_dir = "/home/sagemaker-user/nl2sql/src/nl2sql/optim/openevolve/templates"
user_file = "debug.txt"

path_obj = Path(config_template_dir) / user_file
print(f"\nChecking Path: {path_obj}")
print(f"Exists? : {path_obj.exists()}")
print(f"Is File?: {path_obj.is_file()}")

# 3. Check for internal template restrictions
# Some versions of OpenEvolve have a hardcoded list of allowed extensions
print(f"\n--- CHECKING LIBRARY DEFAULTS ---")
try:
    from openevolve.prompt import templates
    print("Found internal templates module.")
    # If the library uses an Enum or Registry, we might see it here
    print(f"Dir of templates module: {dir(templates)}")
except ImportError:
    print("Could not import openevolve.prompt.templates directly.")

print("\n--- NEXT STEP ---")
if path_obj.exists():
    print("✅ File system is fine. The issue is inside OpenEvolve's loader (likely extension filter).")
    print("👉 Try renaming 'debug.txt' to 'debug.j2' and updating config.yaml")
else:
    print("❌ Path is wrong. Fix the path in config.yaml.")