import os
import importlib

# Get the directory of the current file (__init__.py)
package_dir = os.path.dirname(__file__)

# Dynamically import all Python files in this package
for file in os.listdir(package_dir):
    if file.endswith(".py") and file != "__init__.py":
        module_name = f"{__name__}.{file[:-3]}"  # Convert to module path
        importlib.import_module(module_name)
