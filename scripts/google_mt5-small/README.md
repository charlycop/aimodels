Link : https://huggingface.co/google/mt5-small

# mT5 have to be finetuned to be used, it cannot be used out of the box for translation.
Please see explanation [here](https://stackoverflow.com/questions/76040850/can-mt5-model-on-huggingface-be-used-for-machine-translation)

### To install : 
- `cd ../../Models;git clone https://huggingface.co/google/mt5-small;cd ../scripts/google_mt5-small`

### To Run :
- `python3 run_google_mt5_small_cpu.py` (for cpu version doesn't work)
- `python3 run_google_mt5_small_gpu.py` (for gpu version to be tested)

### Server option
- `pip install flask` (library to make server)
- run server `python3 run_google_mt5_small_cpu_server.py` (to be tested)
- run javascript example `node run_JS_client.js` (to be tested)