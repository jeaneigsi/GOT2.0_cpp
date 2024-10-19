# GOT-OCR-Inference

Research on accelerating the **GOT-OCR** project deployment, supporting multiple languages on cpu

## Research 1:
- [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Release-exe (GOT weights)](https://huggingface.co/kaifeise/GOT-gguf/tree/main)

## Releases:
- **Base SDK package**: [Download here](https://pan.baidu.com/s/10Lo-yY_ZNW7gs0Gd9hiaMw) (Code password: ie4n)
- **Update package**: [Download here](https://pan.baidu.com/s/1pw2JRQZjBZYo4UU-7UNuhQ) (Code: 5x3d)

### Instructions to install from cli:
1. Download and extract the **Base SDK package**.
2. Download the **Update package** and extract it to replace/overwrite the files in the base package.
3. **Double-click `启动.bat`** to start the application.

```bash
pip install llama-cpp-python
```
**Code Usage**:
The following code snippet is a basic demonstration for testing if the model embedding works properly:  
```bash
<|im_start|>system
You should follow the instructions carefully and explain your answers in detail.<|im_end|>
<|im_start|>user
<img></img>
OCR: <|im_end|><|im_start|>assistant
```

`Notes`:
GOT-OCR2.0 deployment acceleration research was conducted using `llama-cpp-python`.
The source code and documentation for llama-cpp-python and llama were referenced to implement possible inference solutions.
No official documentation exists for embedding custom vectors, and this implementation is based on the available knowledge.  

#### Model Quantization:
The quantized version of the model is provided here, but it is not guaranteed to be completely correct since it’s based on the official model's quantization. Some layers of the GOT model may have been included in the quantization by mistake.

Quantized model weights: Download here (Code: 3zop)
If you want to perform the quantization yourself, refer to the modified convert_hf_to_gguf.py script. Make sure to update the config.json file as follows:

```bash
"architectures": [
  "GOTQwenForCausalLM"
]
```
change to :
```bash
"architectures": [
  "Qwen2ForCausalLM"
]
```
This change is necessary to avoid errors when the quantization script attempts to locate the model architecture type.

**Mention**
This fork is based of original repos from [GOT-OCR-Inference](https://github.com/1694439208/GOT-OCR-Inference). Kudos to him for gguf implementation
