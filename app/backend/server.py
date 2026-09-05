from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import base64
import os
import subprocess
import re

app=FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

@app.websocket("/ws")
async def websocket_endpoint(websocket:WebSocket):
    await websocket.accept()
    try:
        data=await websocket.receive_bytes()
        bin_path=os.path.abspath("temp_input.bin")
        with open(bin_path,"wb") as f:
            f.write(data)
        scratch_dir=os.path.abspath("../../")
        subprocess.run(["make","MAIN_SRC=scripts/visualize.cpp","TARGET=visualize"],cwd=scratch_dir)
        process=subprocess.Popen(["./visualize",bin_path],cwd=scratch_dir,stdout=subprocess.PIPE,text=True,bufsize=1)
        for line in process.stdout:
            line=line.strip()
            if "Saved Feature Map:" in line:
                parts=line.split("Saved Feature Map:")
                if len(parts)>1:
                    pgm_path=os.path.join(scratch_dir,parts[1].strip())
                    if os.path.exists(pgm_path):
                        with open(pgm_path,"rb") as f:
                            b64=base64.b64encode(f.read()).decode('utf-8')
                            await websocket.send_json({"type":"layer","data":b64})
            match=re.search(r"PREDICTION:\s*([a-zA-Z]+)",line)
            if match:
                await websocket.send_json({"type":"prediction","data":match.group(1)})
        process.wait()
        await websocket.send_json({"type":"done"})
        await websocket.close()
    except WebSocketDisconnect:
        pass
