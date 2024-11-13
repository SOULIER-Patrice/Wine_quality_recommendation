from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import predict, model


app = FastAPI(
    title="WineAPP",
    version="beta 1.0",
)

# CORS ---
origins = [
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)
# ---

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Add endpoinds
app.include_router(predict.router)
app.include_router(model.router)