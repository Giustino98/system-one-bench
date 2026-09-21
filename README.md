# system-one-bench

Un benchmark riproducibile per confrontare un modello **System One** come Jev / TypeSafe AI con modelli locali. Include sia la classificazione BANKING77 sia **BBH Logical Deduction** con 3, 5 e 7 oggetti.

## Ipotesi da testare

Jev è un classificatore decisionale zero-shot: riceve stato e opzioni e restituisce una scelta tipizzata, con probabilità se l'endpoint le espone. Un LLM locale, invece, genera autoregressivamente un'etichetta. L'ipotesi non è che uno vinca sempre, ma che:

- Jev possa offrire bassa latenza, output strutturato e confidenze calibrate su intenti semanticamente semplici;
- un Qwen locale possa risultare più economico a volume, privato/offline e competitivo in accuratezza;
- task che richiedono più passaggi di ragionamento possano ridurre il vantaggio del modello non autoregressivo.

Il benchmark BBH verifica in modo più diretto se il test-time compute sequenziale di Qwen3 acquista valore aumentando la complessità da 3 → 5 → 7 oggetti. Jev e Qwen ricevono lo stesso problema, le stesse alternative e la stessa istruzione, senza few-shot; la sola differenza intenzionale è il thinking autoregressivo nativo di Qwen3.

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
```

## BBH Logical Deduction: Jev contro Qwen3 thinking

I due YAML usano esclusivamente:

- `logical_deduction_three_objects`
- `logical_deduction_five_objects`
- `logical_deduction_seven_objects`

Il loader legge `lighteval/bbh`, mirror Hugging Face dei dati BIG-Bench Hard, fissato alla revisione `1b61f099fcbf9e55691ef8cc6b4b8fb431dae097`. Il limite è applicato **per task**, quindi `limit: 3` produce nove esempi. Input, choices e indice target vengono convertiti in un unico domain model multiple-choice con chiavi canoniche `A/B/C/...`.

La baseline locale usa `mlx-community/Qwen3-14B-4bit` (conversione MLX 4-bit di `Qwen/Qwen3-14B`) direttamente tramite MLX-LM. `thinking: true` viene passato al chat template come `enable_thinking=True`: il reasoning non è simulato con “think step by step”. Il testo prima di `</think>` viene conservato separatamente, mentre il punteggio usa soltanto una risposta finale esatta nel formato `ANSWER: X`. Il parser non fa fuzzy matching.

Su un Mac Apple Silicon con 16 GB il 14B 4-bit è vicino al limite pratico: chiudi applicazioni pesanti e non caricare contemporaneamente lo stesso checkpoint in LM Studio. Il primo avvio scarica il modello; assicurati di avere spazio libero adeguato. La configurazione usa sampling raccomandato per il thinking (`temperature: 0.6`, `top_p: 0.95`, `top_k: 20`) e 1.024 token massimi, non il vecchio decoding greedy da 12 token di BANKING77.

Entrambe le configurazioni sono sicure per default:

```bash
make plan-bbh-qwen
make plan-bbh-jev
```

Quando vuoi eseguire lo smoke test, cambia consapevolmente `run.dry_run` a `false` nel relativo YAML e lancia:

```bash
make run-bbh-qwen  # locale, nessuna API key
make run-bbh-jev   # usa OPENROUTER_API_KEY da .env
```

Non aumentare/rimuovere `limit` prima di avere confrontato gli artefatti smoke. Ogni record conserva task, choices, risposta, latenza e raw output; per Qwen conserva anche il reasoning separato, per Jev probabilità, usage e costo quando restituiti dal provider.

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
| p50 / p95 latency | mediana e coda lunga per richiesta | tutti i modelli |
| Costo stimato | costo provider, oppure token osservati × rate YAML | solo con usage completo |
| Multiclass Brier score | qualità probabilistica; minore è meglio | solo quando arrivano probabilità per label |
| ECE (10 bin) | distanza tra confidence e accuratezza; minore è meglio | solo quando arrivano probabilità per label |

La baseline MLX non inventa pseudo-probabilità: Brier ed ECE restano `null` finché non si aggiunge un metodo probabilistico comparabile. Questo evita un confronto di calibrazione fuorviante.

## Qualità e limiti sperimentali

- Su BANKING77 i modelli vedono identiche label canoniche (`cash_withdrawal`, ecc.) con descrizioni ottenute sostituendo gli underscore. Su BBH vedono lo stesso testo e le stesse alternative originali.
- Il Qwen è forzato a generare una label esatta. Un output fuori vocabolario viene salvato con prefisso `__invalid__:` e contato come errore, senza interrompere o correggere semanticamente il run; il summary espone anche `invalid_output_rate`.
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
