From [official docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

## 0. What I am assuming

* Fresh DGX Spark with NVIDIA’s DGX Base OS image (Ubuntu aarch64) and CUDA 13 preinstalled. ([NVIDIA][1])
* You run commands in a shell on the Spark, as a normal user with `sudo`.
* You want CUDA backend (not Vulkan, HIP, etc).

If you already messed with CUDA or installed `nvcc` via `apt`, I will call that out in a later “pitfalls” section.

---

## 1. Verify the base environment

Open a terminal and check:

```bash
uname -m
cat /etc/os-release | head
nvidia-smi
nvcc --version
```

On a healthy Spark you should see:

* `uname -m` → `aarch64`
* `nvidia-smi` shows a GB10 GPU with about 119–120 GB memory
* `nvcc --version` shows CUDA 13 and lives in `/usr/local/cuda/bin/nvcc` ([NVIDIA Developer Forums][2])

If `which nvcc` prints `/usr/bin/nvcc`, that means you installed an old toolkit from `apt`. Remove it eventually, or explicitly point CMake at the correct compiler later (I will mention how).

---

## 2. Install build tools and libcurl

DGX OS already has most things, but run this to be safe:

```bash
sudo apt update

sudo apt install -y \
  git build-essential cmake ninja-build \
  libcurl4-openssl-dev
```

`libcurl4-openssl-dev` is needed if you want to use the convenient Hugging Face download flags (`-hf` / `--hf-repo`). ([GitHub][3])

---

## 3. Get llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

This is exactly what the official build guide says. ([GitHub][3])

---

## 4. Configure CMake for CUDA on DGX Spark

On Spark, CUDA 13 already supports Blackwell and compute capability 12.1 (GB10 is `sm_121`). ([NVIDIA Developer Forums][4])
So you do not need to pass custom `CMAKE_CUDA_ARCHITECTURES` unless something is broken.

### 4.1 Simple recommended configuration

From the NVIDIA DGX Spark tutorial, adapted slightly to enable curl: ([NVIDIA Developer Forums][2])

```bash
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON
```

That gives you:

* CUDA backend enabled
* Curl support enabled so you can use `-hf repo:variant` instead of manually downloading GGUF files

If you want to be explicit about the CUDA compiler (for example if you accidentally have an old `/usr/bin/nvcc` lying around), you can do:

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

Do **not** install a second CUDA from `apt` on Spark. The dev forum thread you saw is literally someone breaking their setup that way and then fighting with CMake to pick the right `nvcc`. ([NVIDIA Developer Forums][2])

> Optional: you can add `-G Ninja` if you want Ninja instead of Makefiles.

---

## 5. Build with all 20 CPU cores

Spark’s GB10 has 20 CPU cores. Use them.

```bash
cmake --build build --config Release -j 20
```

This is the exact pattern NVIDIA’s own tutorial uses. ([NVIDIA Developer Forums][2])

After this you should have binaries in `build/bin`, for example:

* `build/bin/llama-cli`
* `build/bin/llama-server`
* `build/bin/llama-bench`

Check:

```bash
./build/bin/llama-server --version
```

You should see a line about CUDA devices, something like:

```text
ggml_cuda_init: found 1 CUDA devices:
Device 0: NVIDIA GB10, compute capability 12.1, ...
```

If you do not see CUDA mentioned, the build did not pick up `GGML_CUDA=ON`.

---

## 6. Get a GGUF model

You have two choices:

### 6.1 Let llama.cpp pull from Hugging Face for you (recommended)

Thanks to `LLAMA_CURL=ON`, you can use the `-hf` shorthand. The upstream docs use this pattern: ([GitHub][5])

For example, to run Meta Llama 3.1 8B Instruct Q4_K_M: ([Hugging Face][6])

```bash
./build/bin/llama-server \
  -hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M \
  --host 0.0.0.0 \
  --port 5000 \
  -c 8192 \
  --no-mmap \
  -fa 1 \
  -ngl 999
```

What this does:

* `-hf repo:Q4_K_M` – ask llama.cpp to download the `Q4_K_M` variant from that repo
* `--no-mmap` – on Spark this shortens load time for huge models; DGX tutorial recommends it for big Qwen3-235B as well ([NVIDIA Developer Forums][2])
* `-fa 1` – enable flash attention explicitly
* `-ngl 999` – offload all layers to GPU (on CUDA this means “use GPU as much as possible”) ([NVIDIA Developer Forums][2])

Then open `http://<spark-hostname>:5000` from your browser.

### 6.2 Manual download

If you prefer to manage models yourself:

```bash
mkdir -p ~/models
cd ~/models

# example: download one GGUF file
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

Then run:

```bash
cd ~/llama.cpp

./build/bin/llama-server \
  --model ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 5000 \
  -c 8192 \
  --no-mmap \
  -fa 1 \
  -ngl 999
```

---

## 7. Using the 128 GB unified memory properly

Spark’s 128 GB is unified LPDDR5x shared between GB10’s CPU and GPU. ([Arm][7])
For llama.cpp that mostly behaves like “huge VRAM” plus some paging.

Key points:

1. **Unified memory toggle**

   llama.cpp uses CUDA Unified Memory when you set:

   ```bash
   export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
   ```

   The docs describe this env var explicitly: Unified Memory lets it swap into system RAM instead of hard crashing on OOM. ([GitHub][3])

   On Spark, this is useful when:

   * You push into crazy territory like Qwen3 235B or 120B+ models
   * You want very long contexts (`-c 64k` or more) on big models

   But: once you really oversubscribe, tokens per second will drop. Use it as an escape hatch, not the default for every tiny model.

2. **Always avoid `--mmap` on huge models**

   On Spark, the DGX tutorial recommends `--no-mmap` for large GGUF because mapping a 100+ GB file through page cache plus Unified Memory is slower to load. ([NVIDIA Developer Forums][2])

3. **Monitor memory and thermals**

   People have reported the Spark throttling under continuous heavy load. ([Tom's Hardware][8])

   While you test, run in another terminal:

   ```bash
   watch -n 2 nvidia-smi
   ```

   and keep an eye on temperature and power draw. If you see frequent reboots or hard throttling, the bottleneck is the chassis cooling, not llama.cpp.

---

## 8. Performance tuning for llama.cpp on Spark

Once it is working, here are the knobs that actually matter:

### 8.1 CUDA specific

* `-ngl 999`
  Force all layers to GPU. Already suggested above.

* `-fa 1`
  Enable flash attention. On Spark this is important for throughput.

* Leave `GGML_CUDA_FORCE_MMQ` and `GGML_CUDA_FORCE_CUBLAS` unset at first.
  llama.cpp picks between custom MMQ kernels and cuBLAS depending on tensor core support. Manual forcing is only worth it if you benchmark carefully. ([Reddit][9])

### 8.2 Threads and batch size

For interactive chat:

```bash
-t 10 -b 64
```

* `-t` number of CPU threads. Use around the performance core count (Spark has 10 Cortex X925 + 10 A725; using 10–14 threads is usually enough). ([Medium][10])
* `-b` batch size. 64 is a good starting point. Above 128 you only really benefit for batched serving, not single-user.

For server style with multiple clients, move toward:

```bash
-t 16 -b 128
```

but check memory usage. 128 GB sounds huge, but 200B models with 32k context plus high batch will still chew through it.

### 8.3 Context size

* Small chat: `-c 8192` or `-c 16384`
* Heavy reasoning or long RAG: `-c 32768` or higher if the model supports it

Remember: context length directly increases both memory and latency per token.

---

## 9. Common pitfalls on DGX Spark

These are things I would actively avoid:

1. **Installing another CUDA with `apt`**

   That is how people end up with a broken `/usr/bin/nvcc` that CMake picks up instead of the official `/usr/local/cuda/bin/nvcc`. ([NVIDIA Developer Forums][2])

2. **Mixing old llama.cpp with very new CUDA**

   For Blackwell (including GB10) you want at least CUDA 12.8 support in your stack. CUDA 13 is already there on Spark and llama.cpp is actively tested on Blackwell hardware. ([NVIDIA Developer Forums][11])
   Do not pin to an ancient llama.cpp commit.

3. **Ignoring errors from the build**

   If `cmake` or `cmake --build` emits anything about unsupported compute capability, you probably have the wrong nvcc. Fix that first; do not brute force with random flags.

---

## 10. Minimal checklist

If you want a “just do these and you should be fine” summary:

```bash
# 1. Make sure we are on the right nvcc
which nvcc
nvcc --version   # should be CUDA 13.x from /usr/local/cuda

# 2. Install tools
sudo apt update
sudo apt install -y git build-essential cmake ninja-build libcurl4-openssl-dev

# 3. Clone llama.cpp
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# 4. Configure for CUDA
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON

# 5. Build (20 cores)
cmake --build build --config Release -j 20

# 6. (Optional) enable unified memory for very large models
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1