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

    # Get all classes in the module
    print("\n" + "=" * 80)
    print("ALL CLASSES IN MODULE")
    print("=" * 80)
    for name, obj in inspect.getmembers(vitstr_module):
        if inspect.isclass(obj) and obj.__module__ == 'doctr.models.recognition.vitstr.pytorch':
            print(f"\nClass: {name}")
            print(inspect.getsource(obj))

    # Get utility functions
    print("\n" + "=" * 80)
    print("UTILITY FUNCTIONS IN MODULE")
    print("=" * 80)
    for name, obj in inspect.getmembers(vitstr_module):
        if inspect.isfunction(obj) and obj.__module__ == 'doctr.models.recognition.vitstr.pytorch':
            # Skip private/internal functions
            if not name.startswith('_') or name in ['_bf16_to_float32', '_vitstr']:
                print(f"\nFunction: {name}")
                print(inspect.getsource(obj))

    # Get VIT backbone
    print("\n" + "=" * 80)
    print("VIT_S BACKBONE FUNCTION")
    print("=" * 80)
    try:
        from doctr.models.backbones.vit import vit_s
        print(inspect.getsource(vit_s))
    except Exception as e:
        print(f"Could not extract vit_s: {e}")

    # Get _ViTSTR base class if it exists
    print("\n" + "=" * 80)
    print("_VITSTR BASE CLASS")
    print("=" * 80)
    try:
        base_class = vitstr_module._ViTSTR
        print(inspect.getsource(base_class))
    except Exception as e:
        print(f"Could not extract _ViTSTR base class: {e}")

    # Get imports at module level
    print("\n" + "=" * 80)
    print("MODULE IMPORTS AND DEPENDENCIES")
    print("=" * 80)
    import doctr.models.recognition.vitstr.pytorch
    print("File location:", doctr.models.recognition.vitstr.pytorch.__file__)
    print("\nCheck these doctr imports that need replacing:")
    print("- from doctr.models.utils import load_pretrained_params")
    print("- from doctr.models.backbones.vit import vit_s")
    print("- Doctr base classes (_ViTSTR, _ViTSTRPostProcessor)")

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

echo ""
echo "Done! Copy the output above to a file to review the architecture."
