Link : https://huggingface.co/google/mt5-small

### To install : 
- `cd ../../Models;git clone https://huggingface.co/google/mt5-small;cd ../scripts/google_mt5-small`

### To Run :
- `python3 run_google_mt5_small_cpu.py` (for cpu version doesn't work)
- `python3 run_google_mt5_small_gpu.py` (for gpu version to be tested)

### Server option
- `pip install flask` (library to make server)
- run server `python3 run_google_mt5_small_cpu_server.py` (to be tested)
- run javascript example `node run_JS_client.js` (to be tested)