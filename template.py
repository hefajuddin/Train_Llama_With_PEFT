import os, sys
from pathlib import Path
import logging

listOfFiles=[
    f"src/pretrained_model.py",
    f"src/tokenized_model.py",
    f"src/collator.py",
    f"data/context.py",
    f"data/dataset.py",
    f"templates/index.html",
    ".env",
    "config.py",
    "requirements.txt",
    "app.py",
    "trainer.py", 
    "test_gpu.py", 
    "upload.py",
]

for path in listOfFiles:
    filepath=Path(path)
    filedir, filename=os.path.split(path)

    if filedir!="":
        os.makedirs(filedir, exist_ok=True)

    if(not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "w") as f:
            pass

    else:
        logging.info("file is already present at :{filepath}" )

