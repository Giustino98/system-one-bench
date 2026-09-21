# system-one-bench

Un benchmark riproducibile per verificare, anziché assumere, quando un modello **System One** come Jev / TypeSafe AI abbia un trade-off migliore di un modello locale su una classificazione reale: il dataset [BANKING77](https://huggingface.co/datasets/PolyAI/banking77).

## Ipotesi da testare

Jev è un classificatore decisionale zero-shot: riceve stato e opzioni e restituisce una scelta tipizzata, con probabilità se l'endpoint le espone. Un LLM locale, invece, genera autoregressivamente un'etichetta. L'ipotesi non è che uno vinca sempre, ma che:

- Jev possa offrire bassa latenza, output strutturato e confidenze calibrate su intenti semanticamente semplici;
- un Qwen locale possa risultare più economico a volume, privato/offline e competitivo in accuratezza;
- task che richiedono più passaggi di ragionamento possano ridurre il vantaggio del modello non autoregressivo.

Il repository misura questa ipotesi su un task controllato. Non trasforma un confronto su BANKING77 in una prova generale di “reasoning”: è un primo esperimento, estendibile con dataset multi-hop e dati di dominio.

## Cosa contiene

```text
src/system_one_bench/
  adapters/       # TypeSafe diretto, Jev via OpenRouter e baseline MLX/Qwen
  dataset.py      # loader Hugging Face e mapping stabile delle 77 label
  metrics.py      # qualità, latenza, calibrazione e costo
  persistence.py  # metadata.json, predictions.jsonl, summary.json
  runner.py       # guardrail dry-run e orchestrazione
configs/          # esperimenti dichiarativi, sicuri per default
tests/            # parsing e formule delle metriche
```

Ogni configurazione è prima **parsata e validata** (campi sconosciuti inclusi), così un run non parte con YAML ambiguo. Le risposte raw sono escluse dai risultati per default, utile per ridurre la persistenza accidentale di dati sensibili.

## Setup — Mac Apple Silicon da 16 GB

Serve Python 3.11+ e, consigliato, [uv](https://docs.astral.sh/uv/). Dal repository:

```bash
cd system-one-bench
uv venv --python 3.11
uv sync --extra dev
```

Per la baseline locale MLX, installa anche l'extra dedicato:

```bash
uv sync --extra dev --extra local
```

Il checkpoint iniziale è `mlx-community/Qwen2.5-3B-Instruct-4bit`: è una scelta prudente per 16 GB di memoria unificata. Non è una promessa di prestazioni; osserva memoria, velocità e validità dell'output sul pilot prima di provare un checkpoint più grande.

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

Prima del run esporta la chiave dalla shell; il file `.env` non viene caricato automaticamente:

```bash
set -a
source .env
set +a
```

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
   uv run system-one-bench run configs/banking77-jev-openrouter.yaml
   uv run system-one-bench run configs/banking77-mlx-qwen.yaml
   ```

4. Rimuovi `limit` solo dopo il pilot. Conserva config e `metadata.json` accanto a ogni risultato, così le comparazioni rimangono auditabili.

## Metriche pianificate

| Metrica | Significato | Disponibilità |
| --- | --- | --- |
| Accuracy | quota di intenti corretti | tutti i modelli |
| Macro-F1 | media F1 con peso uguale alle 77 classi | tutti i modelli |
| p50 / p95 latency | mediana e coda lunga per richiesta | tutti i modelli |
| Costo stimato | costo provider, oppure token osservati × rate YAML | solo con usage completo |
| Multiclass Brier score | qualità probabilistica; minore è meglio | solo quando arrivano probabilità per label |
| ECE (10 bin) | distanza tra confidence e accuratezza; minore è meglio | solo quando arrivano probabilità per label |

La baseline MLX non inventa pseudo-probabilità: Brier ed ECE restano `null` finché non si aggiunge un metodo probabilistico comparabile. Questo evita un confronto di calibrazione fuorviante.

## Qualità e limiti sperimentali

- I modelli vedono identiche label canoniche (`cash_withdrawal`, ecc.); non ci sono descrizioni curate per classe. Un secondo protocollo potrà introdurre una label-card uguale per entrambi, dichiarandolo chiaramente.
- Il Qwen è forzato a generare una label esatta. Un output fuori vocabolario fa fallire il run invece di essere silenziosamente corretto: è un segnale utile per l'analisi.
- L'adapter Jev conserva il payload della risposta solo se `persist_raw_responses: true`. I summary non contengono chiavi.
- La latenza è end-to-end osservata dal processo client; include rete per Jev e non è un benchmark server-side.
- OpenRouter aggiunge un hop di rete e restituisce probabilità arrotondate: latenza e calibrazione vanno tenute separate da una futura esecuzione TypeSafe diretta.

## Sviluppo e controlli locali

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

Licenza: MIT.
