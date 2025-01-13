# Test file to verify imports
import sys
print("Python Path:", sys.path)

try:
    from src.api import endpoints
    print("Successfully imported endpoints")
except Exception as e:
    print("Error importing endpoints:", str(e))

try:
    from src.models import model
    print("Successfully imported model")
except Exception as e:
    print("Error importing model:", str(e))

try:
    from src.monitoring import model_monitoring
    print("Successfully imported model_monitoring")
except Exception as e:
    print("Error importing model_monitoring:", str(e))