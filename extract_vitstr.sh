#!/bin/bash

# Extract ViTSTR architecture code from doctr
# Run this on the machine where doctr is installed

echo "Extracting ViTSTR source code from doctr..."
echo ""

python3 << 'EOF'
import doctr.models.recognition.vitstr.pytorch as vitstr_module
import inspect
import sys

try:
    # Get the ViTSTR class source
    print("=" * 80)
    print("ViTSTR CLASS SOURCE")
    print("=" * 80)
    vitstr_source = inspect.getsource(vitstr_module.ViTSTR)
    print(vitstr_source)

    print("\n" + "=" * 80)
    print("VITSTR_SMALL FACTORY FUNCTION")
    print("=" * 80)
    factory_source = inspect.getsource(vitstr_module.vitstr_small)
    print(factory_source)

    print("\n" + "=" * 80)
    print("MODULE IMPORTS AND DEPENDENCIES")
    print("=" * 80)
    import doctr.models.recognition.vitstr.pytorch
    print("File location:", doctr.models.recognition.vitstr.pytorch.__file__)

    # Try to get other classes if they exist
    print("\n" + "=" * 80)
    print("ALL CLASSES IN MODULE")
    print("=" * 80)
    for name, obj in inspect.getmembers(vitstr_module):
        if inspect.isclass(obj) and obj.__module__ == 'doctr.models.recognition.vitstr.pytorch':
            print(f"\nClass: {name}")
            print(inspect.getsource(obj))

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo ""
echo "Done! Copy the output above to a file to review the architecture."
