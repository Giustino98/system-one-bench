# system-one-bench

Un benchmark riproducibile per confrontare un modello **System One** come Jev / TypeSafe AI con modelli locali e cloud. Include sia la classificazione BANKING77 sia **BBH Logical Deduction** con 3, 5 e 7 oggetti.

## Ipotesi da testare

Jev è un classificatore decisionale zero-shot: riceve stato e opzioni e restituisce una scelta tipizzata, con probabilità se l'endpoint le espone. Un LLM locale, invece, genera autoregressivamente un'etichetta. L'ipotesi non è che uno vinca sempre, ma che:

- Jev possa offrire bassa latenza, output strutturato e confidenze calibrate su intenti semanticamente semplici;
- un Qwen locale possa risultare più economico a volume, privato/offline e competitivo in accuratezza;
- task che richiedono più passaggi di ragionamento possano ridurre il vantaggio del modello non autoregressivo.

Il benchmark BBH verifica in modo più diretto se il test-time compute sequenziale acquista valore aumentando la complessità da 3 → 5 → 7 oggetti. Jev, Qwen e Gemini ricevono lo stesso problema, le stesse alternative e la stessa istruzione, senza few-shot. Qwen e Gemini usano i rispettivi meccanismi nativi di thinking; Jev resta la baseline System One.

## Cosa contiene

```text
src/system_one_bench/
  adapters/       # TypeSafe diretto, Jev via OpenRouter e baseline MLX/Qwen
  dataset.py      # loader separati per classificazione e BBH multiple-choice
  metrics.py      # qualità, latenza, calibrazione e costo
  persistence.py  # metadata.json, predictions.jsonl, summary.json
  runner.py       # guardrail dry-run e orchestrazione
configs/          # esperimenti dichiarativi, sicuri per default
tests/            # parsing e formule delle metriche
```

Ogni configurazione è prima **parsata e validata** (campi sconosciuti inclusi), così un run non parte con YAML ambiguo. Le risposte raw sono escluse dai risultati per default, utile per ridurre la persistenza accidentale di dati sensibili.

## Setup — Mac Apple Silicon da 16 GB

Il progetto usa Python 3.14.6 gestito da [uv](https://docs.astral.sh/uv/), fissato in `.python-version`; non usa il Python globale. Dal repository:

```bash
cd system-one-bench
make setup
```

Per BANKING77 il checkpoint iniziale resta `mlx-community/Qwen2.5-3B-Instruct-4bit`, una scelta prudente per 16 GB di memoria unificata. La configurazione locale predefinita usa l'istanza MLX già caricata in LM Studio:

```bash
lms get https://huggingface.co/mlx-community/Qwen2.5-3B-Instruct-4bit --mlx --yes
lms load qwen2.5-3b-instruct --context-length 4096
lms server start
curl http://127.0.0.1:1234/v1/models
```

LM Studio deve mostrare l'identificatore `qwen2.5-3b-instruct`. L'adapter MLX diretto (`kind: mlx_qwen`) rimane disponibile per esecuzioni senza server. Non è una promessa di prestazioni: osserva memoria, velocità e validità dell'output sul pilot prima di provare un checkpoint più grande.

BANKING77 viene letto da una conversione Parquet verificata e fissata a commit, perché `datasets` 4.x non esegue più il vecchio `banking77.py`. I pilot con `limit` applicano uno shuffle deterministico basato su `run.seed`, così non selezionano solo la prima classe del dataset.

Comandi principali:

```bash
make check          # lint, format-check, typing e test
make plan-local     # valida e mostra il piano senza inferenza
make run-local      # esegue Qwen; richiede dry_run: false
make plan-jev       # valida e mostra il piano Jev senza costi
make run-jev        # esegue Jev; richiede chiave e dry_run: false
make plan-bbh-qwen  # valida il piano BBH/Qwen senza caricare il 14B
make plan-bbh-jev   # valida il piano BBH/Jev senza chiamate a pagamento
make plan-bbh-gemini # valida il piano BBH/Gemini senza chiamate API
```

## BBH Logical Deduction: Jev, Qwen3 e Gemini 3.8 Flash

I due YAML usano esclusivamente:

- `logical_deduction_three_objects`
- `logical_deduction_five_objects`
- `logical_deduction_seven_objects`

Il loader legge `lighteval/bbh`, mirror Hugging Face dei dati BIG-Bench Hard, fissato alla revisione `1b61f099fcbf9e55691ef8cc6b4b8fb431dae097`. Il limite è applicato **per task**, quindi `limit: 3` produce nove esempi. Input, choices e indice target vengono convertiti in un unico domain model multiple-choice con chiavi canoniche `A/B/C/...`.

La baseline locale usa direttamente `mlx-lm` con il checkpoint MLX 4-bit già scaricato da LM Studio in `~/.lmstudio/models/lmstudio-community/Qwen3-14B-MLX-4bit`; LM Studio non deve essere in esecuzione e il modello non viene riscaricato. Il chat template riceve `enable_thinking=true`. La generazione resta libera fino a `</think>`, poi un logits processor ammette soltanto un JSON finale `{"choice":"A"}` con enum specifico dell'esempio. Reasoning e risposta strutturata restano quindi separati nella stessa inferenza.

Su un Mac Apple Silicon con 16 GB il 14B 4-bit è vicino al limite pratico: chiudi applicazioni pesanti e lascia il modello scaricato da LM Studio prima di avviare il benchmark nativo. La configurazione usa sampling raccomandato per il thinking (`temperature: 0.6`, `top_p: 0.95`, `top_k: 20`) e un tetto di 8.192 output token. Il tetto non forza il modello a consumare tutti i token, ma evita di troncare prematuramente i casi più complessi.

I comandi `plan` non eseguono inferenze. Controlla sempre il valore corrente di `run.dry_run` prima dei comandi `run`; la configurazione Qwen locale è pronta per il run completo (`limit: null`, `dry_run: false`):

```bash
make plan-bbh-qwen
make plan-bbh-jev
make plan-bbh-gemini
```

Quando vuoi eseguire, assicurati che LM Studio non tenga il modello in memoria e poi lancia:

```bash
make run-bbh-qwen  # MLX nativo, nessuna API key o server
make run-bbh-jev   # usa OPENROUTER_API_KEY da .env
make run-bbh-gemini # usa GEMINI_API_KEY da .env
```

### Pausare e riprendere un run locale

Per liberare CPU/GPU o riavviare, interrompi il comando con `Ctrl-C`: ogni sample completato resta nell'append-only `predictions.jsonl`. Per riprendere senza duplicare o ricalcolare tali sample, usa lo stesso YAML e la directory originale:

```bash
make resume-bbh-qwen RESUME_DIR=results/20260923T085321Z-mlx_qwen
```

Il resume verifica configurazione, ordine delle choices e dataset rigenerato contro `metadata.json`; l'eventuale sample interrotto a metà viene rieseguito una sola volta.

Ogni record conserva task, choices, risposta, latenza, finish reason e raw output; per Qwen conserva anche il reasoning separato, per Jev probabilità, usage e costo quando restituiti dal provider. Nel run Qwen `continue_on_error: true`: timeout, errori HTTP o payload non validi vengono salvati come record `__error__`, contati come errori e il benchmark passa al campione successivo. Il summary viene quindi prodotto anche in presenza di singoli fallimenti.

### Gemini 3.8 Flash

`configs/bbh-logical-deduction-gemini-3.8-flash.yaml` chiama direttamente la Gemini API con `gemini-3.8-flash`, API key nell'header `x-goog-api-key`, thinking level `medium` e thought summaries persistiti separatamente dalla risposta finale. Il YAML usa lo stesso dataset, task e istruzione di Qwen. La richiesta impone `responseMimeType: application/json` e uno JSON Schema con il solo campo `choice`, vincolato alle alternative dell'esempio: il parser valuta quel campo, non testo libero o markdown.

Per preparare il run:

```bash
cp .env.example .env
# inserisci GEMINI_API_KEY=... in .env
make plan-bbh-gemini
```

Poi imposta esplicitamente `run.dry_run: false` nel YAML Gemini e lancia `make run-bbh-gemini`. Il prezzo iniziale inserito nel YAML è $0,75 / milione token input e $3,75 / milione token output; verifica il listino Google prima di un benchmark esteso, perché i token di thinking sono conteggiati come output dal runner.

## Accesso a Jev e costi

L'accesso TypeSafe diretto è ancora in early access. Il canale immediatamente utilizzabile è quindi OpenRouter, tramite il suo Decisions endpoint alpha. Sono disponibili configurazioni separate per non confondere la latenza nativa TypeSafe con quella osservata attraverso il gateway.

Prezzi verificati il 21 settembre 2026:

- Jev 1.13: **$0,042 per milione di token input**, output gratuito;
- OpenRouter Standard: stesso prezzo di inferenza, più **5,5% sull'acquisto dei crediti**;
- acquisto minimo OpenRouter: **$5 di crediti**; la commissione minima pubblicata è **$0,80**, quindi il primo pagamento tipico è $5,80;
- nessun abbonamento o minimo mensile per Standard pay-as-you-go.

Il costo computazionale dell'esperimento è molto inferiore al top-up minimo. Come stima iniziale, assumendo 500–1.000 token input per richiesta (messaggio più 77 criteri):

- pilot da 25 esempi: circa **$0,0005–$0,0011**;
- test completo da 3.080 esempi: circa **$0,065–$0,13**.

Questa è una proiezione, non un preventivo: dopo il pilot il runner salva token e costo restituiti dal provider, da cui si ricava il costo preciso del full run.

Tutti i YAML hanno `run.dry_run: true`; il comando `run` rifiuta esplicitamente di partire finché non viene cambiato in `false`. Le chiavi restano nel file `.env`, escluso da Git:

```bash
cp .env.example .env
# modifica .env e inserisci una sola delle chiavi:
# OPENROUTER_API_KEY=sk-or-v1-...
# TYPESAFE_API_KEY=...  # quando arriverà l'accesso diretto
```

Il target `make run-jev` carica automaticamente `.env` tramite `uv`. Se esegui la CLI
direttamente, usa `uv run --env-file .env system-one-bench run <config>`.

Non incollare mai una chiave nei YAML, nel codice o nei risultati.

### Creare la chiave OpenRouter

1. Crea o accedi all'account su [openrouter.ai](https://openrouter.ai/).
2. Apri [Credits](https://openrouter.ai/settings/credits), acquista il minimo di $5 e disattiva Auto Recharge se vuoi un tetto rigido.
3. Apri [API Keys](https://openrouter.ai/settings/keys), crea una chiave chiamata `system-one-bench` e imposta un limite piccolo (per esempio $1) se l'interfaccia lo consente.
4. Copia la chiave una sola volta in `.env` come `OPENROUTER_API_KEY=...`.
5. Lascia `limit: 25` e `dry_run: true` finché il piano non è stato controllato.

L'endpoint OpenRouter Decisions è ancora alpha e arrotonda le probabilità a due decimali. Per questo i risultati vengono identificati come `jev_openrouter`; quando TypeSafe abiliterà l'accesso diretto, ripeteremo lo stesso protocollo con `banking77-jev.yaml`.

## Workflow consigliato quando saremo pronti

1. Ispeziona sempre la configurazione, senza rete né inferenza:

   ```bash
   uv run system-one-bench validate-config configs/banking77-jev-openrouter.yaml
   uv run system-one-bench plan configs/banking77-jev-openrouter.yaml
   ```

2. Fai un pilot identico da 25 esempi. Imposta `dry_run: false` nel YAML locale, aggiungi la chiave solo all'ambiente e verifica `results/<run-id>/predictions.jsonl` per output non validi.

3. Esegui un run per modello, con stesso split, stesso limite e stesse 77 label. Per Jev aggiorna prima `input_per_million_usd` / `output_per_million_usd` con il pricing effettivo.

   ```bash
   make run-jev
   make run-local
   ```

4. Rimuovi `limit` solo dopo il pilot. Conserva config e `metadata.json` accanto a ogni risultato, così le comparazioni rimangono auditabili.

## Metriche pianificate

| Metrica | Significato | Disponibilità |
| --- | --- | --- |
| Accuracy | quota di intenti corretti | tutti i modelli |
| Accuracy per task | accuratezza separata per 3, 5 e 7 oggetti | BBH |
| Macro-F1 | media F1 con peso uguale alle 77 classi | tutti i modelli |
| Error rate | quota di richieste fallite ma registrate senza fermare il run | tutti i modelli |
| p50 / p95 latency | mediana e coda lunga per richiesta | tutti i modelli |
| Costo stimato | costo provider, oppure token osservati × rate YAML | solo con usage completo |
| Multiclass Brier score | qualità probabilistica; minore è meglio | solo quando arrivano probabilità per label |
| ECE (10 bin) | distanza tra confidence e accuratezza; minore è meglio | solo quando arrivano probabilità per label |

La baseline MLX non inventa pseudo-probabilità: Brier ed ECE restano `null` finché non si aggiunge un metodo probabilistico comparabile. Questo evita un confronto di calibrazione fuorviante.

## Qualità e limiti sperimentali

- Su BANKING77 i modelli vedono identiche label canoniche (`cash_withdrawal`, ecc.) con descrizioni ottenute sostituendo gli underscore. Su BBH vedono lo stesso testo e le stesse alternative originali.
- Qwen usa una grammatica JSON specifica per esempio soltanto dopo la chiusura di `</think>`. Un errore di generazione o parsing viene salvato con prefisso `__error__:` e contato sia nell'`invalid_output_rate` sia nell'`error_rate`, senza interrompere il run.
- L'adapter Jev conserva il payload della risposta solo se `persist_raw_responses: true`. I summary non contengono chiavi.
- La latenza è end-to-end osservata dal processo client; include rete per Jev e non è un benchmark server-side.
- OpenRouter aggiunge un hop di rete e restituisce probabilità arrotondate: latenza e calibrazione vanno tenute separate da una futura esecuzione TypeSafe diretta.
- Il confronto non equipara compute o energia: Qwen usa sampling e può consumare fino al budget di generazione, mentre Jev restituisce una decisione in un singolo passaggio. È precisamente il trade-off misurato, ma va considerato leggendo latenza e costo.

## Sviluppo e controlli locali

Il target aggregato è:

```bash
make check
```

Sono disponibili anche `make lint`, `make format-check`, `make typecheck` e `make test`.

Licenza: MIT.
