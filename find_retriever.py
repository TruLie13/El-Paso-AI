import pkgutil
import importlib
import inspect

# The class name we are searching for
TARGET_CLASS = "SelfQueryingRetriever"

# The packages to search within
PACKAGES_TO_SEARCH = ['langchain',
                      'langchain_experimental', 'langchain_community']

print(f"Searching for class '{TARGET_CLASS}' in {PACKAGES_TO_SEARCH}...")

found = False
for package_name in PACKAGES_TO_SEARCH:
    try:
        package = importlib.import_module(package_name)

        # Walk through all modules and submodules in the package
        for _, module_name, _ in pkgutil.walk_packages(package.__path__, package_name + '.'):
            try:
                module = importlib.import_module(module_name)
                # Check all members of the module
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and name == TARGET_CLASS:
                        print("\n--- FOUND! ---")
                        print("The class was found in the following module:")
                        print(f"Module: {module.__name__}")
                        print(
                            "\nUse this exact import statement in your ask.py script:")
                        print(f"from {module.__name__} import {TARGET_CLASS}")
                        print("--------------")
                        found = True
            except Exception:
                # Ignore modules that can't be imported for any reason
                continue
    except ImportError:
        print(f"Could not import the package '{package_name}'. Skipping.")
        continue

if not found:
    print(
        f"\nCould not find the class '{TARGET_CLASS}' in any of the specified packages.")
