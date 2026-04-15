import sys

sys.path.insert(0, ".")
from tetrahedron_memory.router import create_app
import uvicorn

app = create_app(dimension=3, precision="fast")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8199)
