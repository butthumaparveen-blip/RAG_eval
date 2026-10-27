from retriever_py import load_vectorstore
import json
store=load_vectorstore()
from pathlib import Path
data=store._collection.get(include=["documents","metadatas"])
dumb=[
    {"id":i,"text":data,"meta":meta}
    for i,data,meta in zip(data["ids"],data["documents"],data["metadatas"])
]
dumb.sort(key= lambda c:(str(c["meta"].get("page")),c["id"]))
output_file = Path.cwd() / "chunk_data.json"
with open(output_file,"w",encoding="utf-8") as f:
    json.dump(dumb,f,indent=4,ensure_ascii=False)

print(f"{len(dumb)}")