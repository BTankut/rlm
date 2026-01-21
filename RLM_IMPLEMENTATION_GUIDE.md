# RLM (Recursive Language Models) Implementation Guide

## Claude Code CLI ile Lokal LLM Üzerinde Geliştirme

Bu doküman, "Recursive Language Models" makalesindeki (Zhang et al., 2025) yaklaşımı lokal bir LLM üzerinde implemente etmek için kapsamlı bir rehberdir.

---

## Temel Konsept

RLM'in ana fikri: **Uzun prompt'ları doğrudan LLM'e vermek yerine, harici bir ortamın (environment) parçası olarak ele al ve LLM'in bu ortamla programatik olarak etkileşmesine izin ver.**

```
┌─────────────────────────────────────────────────────────────┐
│  Geleneksel Yaklaşım:                                       │
│  [Çok Uzun Prompt] ──────▶ [LLM] ──────▶ [Cevap]           │
│  ❌ Context rot, performans düşüşü                          │
├─────────────────────────────────────────────────────────────┤
│  RLM Yaklaşımı:                                             │
│  [Uzun Prompt] ──▶ [REPL Environment'a Yükle]              │
│                           │                                 │
│                           ▼                                 │
│                    [LLM Kod Yazar]                          │
│                           │                                 │
│                           ▼                                 │
│              [Kod Çalışır, Sonuç LLM'e Döner]              │
│                           │                                 │
│                           ▼                                 │
│                    [İteratif Tekrar]                        │
│                           │                                 │
│                           ▼                                 │
│                   [FINAL() ile Cevap]                       │
│  ✅ 10M+ token işlenebilir, context rot minimize            │
└─────────────────────────────────────────────────────────────┘
```

---

## Ön Gereksinimler

### Donanım
- **Minimum:** 8GB VRAM (qwen3:8b için)
- **Önerilen:** 12-16GB VRAM (qwen3:14b veya codestral:22b için)
- **CPU-only:** Mümkün ama yavaş (qwen3:4b)

### Yazılım
```bash
# Ollama kurulumu
curl -fsSL https://ollama.com/install.sh | sh

# Model indirme (VRAM'e göre seç)
ollama pull qwen3:14b      # 12GB+ VRAM
ollama pull qwen3:8b       # 8GB VRAM
ollama pull deepseek-coder-v2:16b  # Alternatif

# Ollama'yı başlat
ollama serve
```

### Proje Dizini
```bash
mkdir rlm-project && cd rlm-project
git init
claude  # Claude Code CLI'ı başlat
```

---

# PHASE 1: MVP (Kısa Vade)

**Süre:** 1-2 gün  
**Hedef:** Temel RLM döngüsünü çalıştır, 50K token context ile basit soru-cevap

## 1.1 Proje Yapısını Oluştur

Claude Code'a ver:

```
Aşağıdaki yapıda bir Python projesi oluştur:

rlm/
├── __init__.py
├── config.py           # Konfigürasyon
├── llm/
│   ├── __init__.py
│   └── client.py       # Ollama API client
├── core/
│   ├── __init__.py
│   ├── executor.py     # Python REPL executor
│   ├── parser.py       # Code block ve FINAL parser
│   └── state.py        # Variable state management
├── rlm.py              # Ana orchestrator
└── main.py             # Entry point

Ayrıca:
- requirements.txt (openai, python-dotenv)
- .env.example (OLLAMA_BASE_URL, MODEL_NAME)
- .gitignore
```

## 1.2 Konfigürasyon

Claude Code'a ver:

```
config.py dosyasını oluştur:

Gerekli ayarlar:
- OLLAMA_BASE_URL: http://localhost:11434/v1
- ROOT_MODEL: qwen3:14b (veya sistemdeki model)
- SUB_MODEL: Şimdilik ROOT_MODEL ile aynı
- MAX_ITERATIONS: 20 (sonsuz döngü koruması)
- MAX_OUTPUT_CHARS: 10000 (LLM'e dönen output limiti)
- MAX_CONTEXT_CHARS_DISPLAY: 500 (context preview için)
- EXECUTION_TIMEOUT: 30 (saniye)

.env dosyasından okusun, varsayılanlar olsun.
```

## 1.3 LLM Client

Claude Code'a ver:

```
llm/client.py için:

OpenAI SDK kullanarak Ollama-compatible client yaz.

Fonksiyon:
def call_llm(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7
) -> str

Özellikler:
- Ollama'nın OpenAI-compatible endpoint'ini kullan
- Streaming opsiyonel (şimdilik kapalı)
- Basit error handling
- Token sayısını logla (debug için)
```

## 1.4 Code Executor

Claude Code'a ver:

```
core/executor.py için:

Güvenli(ish) bir Python code executor yaz.

Fonksiyon:
def execute_code(
    code: str,
    globals_dict: dict,
    timeout: int = 30
) -> ExecutionResult

ExecutionResult dataclass:
- stdout: str
- stderr: str  
- success: bool
- error: str | None

Özellikler:
- exec() kullan, globals_dict'i paylaş (variable persistence)
- stdout'u yakala (io.StringIO ile)
- Exception handling
- Timeout (signal veya threading ile)

ÖNEMLİ: Bu MVP için basit exec() yeterli. 
Production'da RestrictedPython veya Docker sandbox gerekecek.
```

## 1.5 Parser

Claude Code'a ver:

```
core/parser.py için:

LLM çıktısını parse eden fonksiyonlar yaz.

1. extract_code_blocks(text: str) -> list[str]
   - ```repl veya ```python bloklarını çıkar
   - Birden fazla blok olabilir

2. detect_final_answer(text: str) -> tuple[bool, str | None]
   - FINAL(cevap) pattern'ini yakala
   - Cevabı döndür

3. detect_final_var(text: str) -> tuple[bool, str | None]
   - FINAL_VAR(variable_name) pattern'ini yakala
   - Variable adını döndür

Regex kullan, edge case'lere dikkat et:
- FINAL() içinde parantez olabilir
- Multiline cevaplar olabilir
- Kod bloğu içinde FINAL olmamalı (yanlış pozitif)
```

## 1.6 State Manager

Claude Code'a ver:

```
core/state.py için:

REPL environment state'ini yöneten class yaz.

class REPLState:
    def __init__(self):
        self.globals = {}  # Persistent variables
        self.history = []  # Execution history
    
    def initialize_context(self, context: str, context_type: str = "string"):
        # context değişkenini globals'a ekle
        # context metadata'sını hesapla (length, chunks vs.)
    
    def get_context_info(self) -> dict:
        # LLM'e verilecek context özeti
        # - total_length
        # - type
        # - preview (ilk N karakter)
    
    def add_llm_query_function(self, llm_client):
        # llm_query() fonksiyonunu globals'a ekle
        # Bu Phase 2'de aktif olacak, şimdilik placeholder
```

## 1.7 System Prompt

**KRİTİK:** Makale Appendix D'deki prompt'u kullan.

Claude Code'a ver:

```
llm/prompts.py dosyası oluştur.

İçinde RLM_SYSTEM_PROMPT değişkeni olsun.

Prompt içeriği (Appendix D'den, SUB-CALL KISMI OLMADAN - Phase 1 için):

---
You are tasked with answering a query with associated context. You can access, 
transform, and analyze this context interactively in a REPL environment.
You will be queried iteratively until you provide a final answer.

Your context is a {context_type} with {context_total_length} total characters.

The REPL environment is initialized with:
1. A 'context' variable that contains extremely important information about your query. 
   You should check the content of the 'context' variable to understand what you are 
   working with. Make sure you look through it sufficiently as you answer your query.
2. The ability to use 'print()' statements to view the output of your REPL code and 
   continue your reasoning.

You will only be able to see truncated outputs from the REPL environment.

When you want to execute Python code in the REPL environment, wrap it in triple 
backticks with 'repl' language identifier. For example:

```repl
chunk = context[:10000]
print(chunk)
```

You can use regex, string operations, and any standard Python to analyze the context.

IMPORTANT: When you are done, provide your final answer inside a FINAL() function.
Example: FINAL(The answer is 42)

Or use FINAL_VAR(variable_name) to return a variable from the REPL environment.

Think step by step. Execute code to explore the context. 
Answer the original query explicitly in your final answer.
---

Bu prompt'u format string olarak yaz, {context_type} ve {context_total_length} 
placeholder'ları olsun.
```

## 1.8 Ana Orchestrator

Claude Code'a ver:

```
rlm.py için:

RLM class'ını yaz. Bu ana orchestrator.

class RLM:
    def __init__(self, config=None):
        self.config = config or default_config
        self.llm_client = LLMClient()
        self.state = None
    
    def run(self, query: str, context: str) -> str:
        # 1. State'i initialize et
        # 2. System prompt'u hazırla (context metadata ile)
        # 3. İteratif döngü başlat
        # 4. Her iterasyonda:
        #    a. LLM'i çağır
        #    b. Response'u parse et
        #    c. FINAL varsa → döndür
        #    d. Code block varsa → execute et
        #    e. Output'u truncate et
        #    f. Conversation history'e ekle
        #    g. Max iteration check
        # 5. Max iteration'a ulaşıldıysa hata veya son state'i döndür

Conversation history'i şu formatta tut:
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": query},
    {"role": "assistant", "content": llm_response_1},
    {"role": "user", "content": f"REPL Output:\n{truncated_output}"},
    {"role": "assistant", "content": llm_response_2},
    ...
]

Her iteration'ı logla (debug için).
```

## 1.9 Test

Claude Code'a ver:

```
main.py için basit bir test yaz:

1. Örnek context oluştur (50K karakter):
   - Lorem ipsum veya gerçek bir metin
   - İçine "gizli" bir bilgi yerleştir (needle in haystack)
   
2. RLM'i çalıştır:
   - Query: "Context içinde gizli mesaj ne?"
   - Sonucu yazdır
   
3. Debug bilgileri:
   - Kaç iteration sürdü
   - Hangi kod blokları çalıştı
   - Final cevap ne

Ayrıca test/ dizini altında pytest testleri yaz:
- test_parser.py (code block extraction, FINAL detection)
- test_executor.py (basic code execution)
- test_rlm.py (integration test)
```

## 1.10 Phase 1 Tamamlanma Kriterleri

- [ ] Proje yapısı oluşturuldu
- [ ] Ollama'ya bağlanabiliyor
- [ ] Basit kod execute edebiliyor
- [ ] FINAL() parse edebiliyor
- [ ] 50K token context ile needle-in-haystack çalışıyor
- [ ] Testler geçiyor

---

# PHASE 2: Recursive Sub-Calls (Orta Vade)

**Süre:** 3-5 gün  
**Hedef:** llm_query() ile recursive çağrı desteği, OOLONG benzeri task

## 2.1 llm_query Fonksiyonu

Claude Code'a ver:

```
core/state.py dosyasını güncelle.

llm_query fonksiyonunu implement et:

def create_llm_query_function(llm_client, model: str, max_chars: int = 500000):
    """
    REPL environment içinde kullanılacak llm_query fonksiyonu oluşturur.
    
    Bu fonksiyon:
    - Sub-LLM çağrısı yapar
    - Context'i prompt ile birleştirir
    - Sonucu string olarak döndürür
    """
    
    def llm_query(prompt: str) -> str:
        # Max chars kontrolü
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars] + "\n[TRUNCATED]"
        
        response = llm_client.call([
            {"role": "user", "content": prompt}
        ], model=model)
        
        return response
    
    return llm_query

Bu fonksiyonu REPLState.globals'a ekle:
self.globals['llm_query'] = create_llm_query_function(...)
```

## 2.2 System Prompt Güncellemesi

Claude Code'a ver:

```
llm/prompts.py dosyasını güncelle.

Yeni prompt (Appendix D'nin tam versiyonu):

---
You are tasked with answering a query with associated context. You can access, 
transform, and analyze this context interactively in a REPL environment that 
can recursively query sub-LLMs, which you are strongly encouraged to use as 
much as possible. You will be queried iteratively until you provide a final answer.

Your context is a {context_type} with {context_total_length} total characters.

The REPL environment is initialized with:
1. A 'context' variable that contains extremely important information about your query.
2. A 'llm_query' function that allows you to query an LLM (that can handle around 
   500K chars) inside your REPL environment.
3. The ability to use 'print()' statements to view the output of your REPL code.

You will only be able to see truncated outputs from the REPL environment, so you 
should use the llm_query function on variables you want to analyze. You will find 
this function especially useful when you have to analyze the semantics of the context.

Example strategy:
1. Look at the context structure
2. Figure out a chunking strategy  
3. Break up the context into smart chunks
4. Query an LLM per chunk with a particular question
5. Save answers to a buffer
6. Query an LLM with all the buffers to produce your final answer

Example code:

```repl
query = "Find all mentions of 'magic number' in the context"
chunk_size = len(context) // 10
answers = []

for i in range(10):
    start = i * chunk_size
    end = start + chunk_size if i < 9 else len(context)
    chunk = context[start:end]
    
    answer = llm_query(f"Answer this: {query}\n\nContext chunk:\n{chunk}")
    answers.append(answer)
    print(f"Chunk {i}: {answer[:100]}...")

final = llm_query(f"Aggregate these answers: {answers}")
print(final)
```

Remember: Your sub-LLMs are powerful -- they can fit around 500K characters.
Don't be afraid to put a lot of context into them.

IMPORTANT: When done, use FINAL(your answer) or FINAL_VAR(variable_name).
---
```

## 2.3 Cost ve Call Tracking

Claude Code'a ver:

```
Yeni dosya: core/tracking.py

Token ve call takibi için:

class UsageTracker:
    def __init__(self):
        self.root_calls = 0
        self.sub_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_history = []
    
    def log_call(self, call_type: str, input_tokens: int, output_tokens: int):
        # call_type: "root" veya "sub"
        # Her çağrıyı kaydet
    
    def get_summary(self) -> dict:
        # Özet istatistikler döndür
    
    def estimate_cost(self, price_per_1k_input: float, price_per_1k_output: float) -> float:
        # Maliyet tahmini (API kullanırsan)

LLM client'ı bu tracker'ı kullansın.
```

## 2.4 Sub-Call Limiti

**ÖNEMLİ (Appendix A'dan):** Qwen modeli çok fazla sub-call yapma eğiliminde. Bunu kontrol et.

Claude Code'a ver:

```
config.py'ye ekle:
- MAX_SUB_CALLS: 100 (varsayılan, task'a göre ayarlanabilir)
- SUB_CALL_WARNING_THRESHOLD: 50

core/state.py'de llm_query fonksiyonunu güncelle:

Sub-call sayacı ekle. Limit aşılırsa:
1. Uyarı mesajı döndür
2. Veya exception fırlat

Ayrıca, Qwen modeli için system prompt'a şu uyarıyı ekle (Appendix D'den):

"IMPORTANT: Be very careful about using 'llm_query' as it incurs high runtime costs. 
Always batch as much information as reasonably possible into each call (aim for 
around ~200k characters per call). For example, if you have 1000 lines of information 
to process, it's much better to split into chunks of 5 and call 'llm_query' on each 
chunk (200 calls total) rather than making 1000 individual calls."
```

## 2.5 OOLONG-Style Test

Claude Code'a ver:

```
tests/test_oolong_style.py oluştur.

OOLONG benchmark'ı simüle et:
- Çok sayıda satır (örn: 1000 satır)
- Her satır: "Date: X | User: Y | Question: Z"
- Her sorunun semantik bir kategorisi var (ama yazılmamış)
- Kategoriler: numeric value, entity, location, description, abbreviation, human being

Test query:
"Yukarıdaki veride 'numeric value' kategorisindeki soru sayısı, 
'location' kategorisindeki soru sayısından fazla mı, az mı, eşit mi?"

Bu test:
1. Her satırı semantik olarak kategorize etmeli (llm_query ile)
2. Sayıları toplamalı
3. Karşılaştırma yapmalı

Beklenen davranış:
- RLM chunklara ayırmalı
- Her chunk için llm_query çağırmalı
- Sonuçları aggregate etmeli
```

## 2.6 Output Truncation İyileştirmesi

Claude Code'a ver:

```
rlm.py'de output truncation'ı iyileştir:

Şu anki: İlk N karakter
Yeni: Akıllı truncation

def smart_truncate(output: str, max_chars: int = 10000) -> str:
    if len(output) <= max_chars:
        return output
    
    # İlk %40 + son %40 + ortada "[TRUNCATED X chars]"
    first_part = int(max_chars * 0.4)
    last_part = int(max_chars * 0.4)
    
    middle_chars = len(output) - first_part - last_part
    
    return (
        output[:first_part] + 
        f"\n\n[... TRUNCATED {middle_chars} characters ...]\n\n" +
        output[-last_part:]
    )
```

## 2.7 Phase 2 Tamamlanma Kriterleri

- [ ] llm_query fonksiyonu çalışıyor
- [ ] Sub-call tracking aktif
- [ ] Sub-call limiti var
- [ ] OOLONG-style test geçiyor
- [ ] 100K+ token context işlenebiliyor
- [ ] Chunking + aggregation pattern çalışıyor

---

# PHASE 3: Production Hardening (Uzun Vade)

**Süre:** 1-2 hafta  
**Hedef:** Güvenlik, performans, observability

## 3.1 Sandbox Güvenliği

Claude Code'a ver:

```
core/sandbox.py oluştur.

İki seviye güvenlik:

### Seviye 1: RestrictedPython (basit)
- Tehlikeli built-in'leri kaldır (eval, exec, open, __import__)
- File system erişimini engelle
- Network erişimini engelle

### Seviye 2: Docker Container (gelişmiş)
- Her execution ayrı container'da
- Resource limitleri (CPU, memory, time)
- Network isolation
- Read-only filesystem (context hariç)

Şimdilik Seviye 1'i implement et:

from RestrictedPython import compile_restricted, safe_builtins

def create_restricted_globals(context, llm_query_func):
    return {
        '__builtins__': safe_builtins,
        'context': context,
        'llm_query': llm_query_func,
        'print': print,  # Güvenli print
        # İzin verilen modüller
        're': __import__('re'),
        'json': __import__('json'),
        'collections': __import__('collections'),
    }

def execute_restricted(code: str, globals_dict: dict) -> ExecutionResult:
    byte_code = compile_restricted(code, '<rlm>', 'exec')
    exec(byte_code, globals_dict)
    # ...
```

## 3.2 Async Execution

Claude Code'a ver:

```
Sub-call'ları paralel çalıştırmak için async support ekle.

llm/async_client.py:

import asyncio
import aiohttp

class AsyncLLMClient:
    async def call(self, messages, model):
        # Async HTTP call
        
    async def batch_call(self, prompts: list[str], model: str) -> list[str]:
        # Birden fazla prompt'u paralel çağır
        tasks = [self.call([{"role": "user", "content": p}], model) for p in prompts]
        return await asyncio.gather(*tasks)

core/state.py'de:

async def llm_query_batch(prompts: list[str]) -> list[str]:
    # Batch processing için
    # RLM bu fonksiyonu loop içinde kullanabilir

Not: Ollama'nın paralel request desteğini kontrol et.
Çoğu lokal setup'ta sequential daha stabil olabilir.
```

## 3.3 Caching

Claude Code'a ver:

```
core/cache.py oluştur.

Aynı prompt'a aynı cevabı vermemek için basit cache:

import hashlib
from functools import lru_cache

class LLMCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get_key(self, prompt: str, model: str) -> str:
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, prompt: str, model: str) -> str | None:
        key = self.get_key(prompt, model)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, prompt: str, model: str, response: str):
        if len(self.cache) >= self.max_size:
            # LRU eviction (basit versiyon: rastgele sil)
            self.cache.pop(next(iter(self.cache)))
        key = self.get_key(prompt, model)
        self.cache[key] = response

LLM client'ta cache kullan:
- Aynı sub-query tekrar gelirse cache'ten döndür
- Özellikle chunked processing'te faydalı
```

## 3.4 Logging ve Observability

Claude Code'a ver:

```
core/logging.py oluştur.

Structured logging:

import logging
import json
from datetime import datetime

class RLMLogger:
    def __init__(self, name: str = "rlm"):
        self.logger = logging.getLogger(name)
        self.run_id = None
    
    def start_run(self, query: str, context_length: int):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log("run_start", {
            "query": query[:100],
            "context_length": context_length
        })
    
    def log_iteration(self, iteration: int, code: str, output: str):
        self.log("iteration", {
            "iteration": iteration,
            "code_length": len(code),
            "output_length": len(output),
            "code_preview": code[:200]
        })
    
    def log_sub_call(self, prompt_length: int, response_length: int):
        self.log("sub_call", {
            "prompt_length": prompt_length,
            "response_length": response_length
        })
    
    def log_final(self, answer: str, iterations: int, sub_calls: int):
        self.log("run_end", {
            "answer_preview": answer[:100],
            "total_iterations": iterations,
            "total_sub_calls": sub_calls
        })
    
    def log(self, event: str, data: dict):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "event": event,
            **data
        }
        self.logger.info(json.dumps(entry))

Dosyaya ve console'a yazsın.
Opsiyonel: SQLite'a da kaydet (analiz için).
```

## 3.5 Error Recovery

Claude Code'a ver:

```
rlm.py'de robust error handling ekle:

class RLMError(Exception):
    pass

class MaxIterationsError(RLMError):
    pass

class ExecutionError(RLMError):
    pass

class SubCallLimitError(RLMError):
    pass

run() metodunda:

1. Code execution hatası:
   - Hatayı LLM'e göster
   - "REPL Error: {error}\nPlease fix your code."
   - Retry (max 3 kez aynı hata için)

2. LLM API hatası:
   - Exponential backoff ile retry
   - 3 başarısız denemeden sonra fail

3. Timeout:
   - Partial output'u göster
   - LLM'e timeout bilgisi ver

4. Max iteration:
   - Son state'i döndür (best effort)
   - Veya explicit fail

5. Sub-call limit:
   - Uyarı ver, devam et
   - Veya soft limit aşımında stop
```

## 3.6 Benchmark Suite

Claude Code'a ver:

```
benchmarks/ dizini oluştur.

Makaleden benchmark'ları adapte et:

### S-NIAH (Single Needle in a Haystack)
benchmarks/s_niah.py:
- Rastgele metin içine tek bir "needle" yerleştir
- Farklı context boyutlarında test et (8K, 16K, 32K, 64K, 128K, 256K)
- Accuracy ölç

### OOLONG-Simple
benchmarks/oolong_simple.py:
- Semantik kategorilendirme gerektiren aggregation
- Linear complexity: O(N) processing

### OOLONG-Pairs  
benchmarks/oolong_pairs.py:
- Pair-wise reasoning gerektiren task
- Quadratic complexity: O(N²) processing
- Makaledeki 20 task'ı implement et (Appendix E.1)

### CodeQA-Simple
benchmarks/code_qa.py:
- Basit bir codebase ver
- "Bu fonksiyon ne yapıyor?" tarzı sorular

Her benchmark için:
- Otomatik veri üretimi
- Scoring fonksiyonu
- Baseline (direct LLM) karşılaştırması
```

## 3.7 CLI Interface

Claude Code'a ver:

```
cli.py oluştur.

Komut satırından kullanım:

# Basit query
python cli.py --query "Özet çıkar" --context-file document.txt

# Benchmark çalıştır
python cli.py benchmark --name s_niah --sizes 8k,16k,32k

# Interactive mode
python cli.py interactive

# Config override
python cli.py --model qwen3:8b --max-iterations 30 --query "..."

argparse veya click kullan.
```

## 3.8 Web UI (Opsiyonel)

Claude Code'a ver:

```
Basit bir Gradio veya Streamlit UI:

ui/app.py:

Özellikler:
- Context yükleme (paste veya file upload)
- Query girme
- Real-time iteration görüntüleme
- Code execution output'larını gösterme
- Final answer
- Usage statistics

Gradio tercih edilir (daha basit).
```

## 3.9 Phase 3 Tamamlanma Kriterleri

- [ ] Sandbox güvenliği aktif
- [ ] Logging ve tracking çalışıyor
- [ ] Error recovery robust
- [ ] Benchmark suite hazır
- [ ] CLI kullanılabilir
- [ ] 1M+ token context test edildi
- [ ] Performance baseline'lar ölçüldü

---

# Emergent Patterns ve Best Practices

Makaleden öğrenilen pattern'lar (Section 3.1):

## Pattern 1: Code ile Filtreleme

```python
# LLM, model priors kullanarak regex ile arama yapıyor
import re
keywords = ["festival", "La Union", "beauty pageant"]
for kw in keywords:
    matches = re.findall(f".*{kw}.*", context, re.IGNORECASE)
    print(f"{kw}: {len(matches)} matches")
```

## Pattern 2: Chunk + Sub-call + Aggregate

```python
# Uzun context'i parçala, her parçayı analiz et, sonuçları birleştir
chunks = [context[i:i+100000] for i in range(0, len(context), 100000)]
results = []
for i, chunk in enumerate(chunks):
    result = llm_query(f"Analyze this chunk:\n{chunk}")
    results.append(result)
    print(f"Chunk {i}: {result[:50]}...")

final = llm_query(f"Combine these analyses:\n{results}")
```

## Pattern 3: Answer Verification

```python
# Cevabı bulduktan sonra doğrula
candidate_answer = "Maria Dalmacio"
verification = llm_query(f"""
Verify this answer: {candidate_answer}
Against this context: {relevant_section}
Is this correct? Why?
""")
print(verification)
```

## Pattern 4: Variable'da Output Biriktirme

```python
# Uzun output için variable kullan, FINAL_VAR ile döndür
all_pairs = []
for i in range(len(users)):
    for j in range(i+1, len(users)):
        if check_condition(users[i], users[j]):
            all_pairs.append((users[i], users[j]))

final_result = "\n".join([f"({a}, {b})" for a, b in all_pairs])
# Sonra: FINAL_VAR(final_result)
```

---

# Dikkat Edilmesi Gerekenler (Appendix A'dan)

## 1. Model-Specific Prompt Tuning
- Aynı prompt farklı modellerde farklı çalışır
- Qwen için sub-call uyarısı gerekli
- Test et ve ayarla

## 2. Kod Yeteneği Kritik
- Küçük modeller (< 7B) zorlanabilir
- Kod üretimi için optimize edilmiş model tercih et

## 3. Thinking Models ve Token Limiti
- Reasoning token'ları output'u aşabilir
- Max output token'ı yüksek tut

## 4. Async Sub-calls
- Sequential implementasyon yavaş
- Async önemli ama complexity ekler

## 5. FINAL Detection Brittle
- Model bazen planı FINAL olarak verebilir
- Robust parsing gerekli

---

# Hızlı Referans: Claude Code Komutları

```bash
# Projeyi başlat
claude "Phase 1'i implement et: [bu dokümandan ilgili kısmı yapıştır]"

# Test çalıştır
claude "Testleri çalıştır ve hataları düzelt"

# Debug
claude "Bu hata oluşuyor: [hata mesajı]. Düzelt."

# Refactor
claude "executor.py'yi async yap"

# Benchmark
claude "S-NIAH benchmark'ını çalıştır, sonuçları raporla"

# Yeni özellik
claude "Caching ekle"
```

---

# Versiyon Geçmişi

| Versiyon | Tarih | Değişiklik |
|----------|-------|------------|
| 0.1 | Phase 1 | MVP, basit REPL |
| 0.2 | Phase 2 | Sub-calls, tracking |
| 0.3 | Phase 3 | Sandbox, async, benchmarks |
| 1.0 | Stable | Production ready |

---

# Kaynaklar

- **Makale:** Zhang et al., "Recursive Language Models", arXiv:2512.24601v1
- **Appendix D:** System Prompts
- **Appendix E:** OOLONG-Pairs Tasks
- **Appendix A:** Negative Results / What Didn't Work

---

*Bu doküman Claude Code CLI ile iteratif olarak kullanılmak üzere tasarlanmıştır.*
