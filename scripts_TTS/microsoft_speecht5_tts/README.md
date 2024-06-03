Link : https://huggingface.co/microsoft/speecht5_tts

### To install : 
- `cd ../../Models;git clone https://huggingface.co/microsoft/speecht5_tts;cd ../scripts_TTS/microsoft_speecht5_tts`

### Before running :
- `pip install --upgrade pip`
- `pip install --upgrade transformers sentencepiece datasets[audio]`

### To Run :
- `python3 run_microsoft_speecht5_tts_cpu.py`
- `python3 run_microsoft_speecht5_tts_gpu.py`