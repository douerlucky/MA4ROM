# Results

This directory is the compact public index of the paper's final reported
experiments. Intermediate mappings, prompts, generation logs, candidate lists,
checkpoints, failed runs, repair diagnostics, and repeated copies were removed.

## Evidence levels

The files intentionally distinguish three evidence levels:

1. **Canonical final artefacts** — final mapping, generated ontology, aggregate
   F1, and compact query metrics. This level is provided for MA4ROM Table 2 and
   the Table 6 FK-removal experiment.
2. **Audited compact summaries** — run counters and final scores retained after
   removing paths, logs, and intermediate files. Table 6 FK recovery uses this
   level.
3. **Reported-only summaries** — values transcribed from the paper when the
   audited archives did not contain coherent raw provenance. These rows are
   explicitly labelled and must not be described as local reruns.

## Layout

```text
results/
├── baselines/llm4vkg/       # Table 2 reported-only comparison values
├── paper/table2/             # MA4ROM DS-V4F and GPT-4o final artefacts
├── paper/table3/             # Multi-agent ablation summary + caveats
├── paper/table4/             # Context-retrieval summary + provenance gap
├── paper/table5/             # OP ablation summary + provenance gap
├── paper/table6/             # FK-removal artefacts and FK-recovery summary
└── checksums.sha256          # SHA-256 for every other results file
```

## Source/version boundary

The DeepSeek V4-Flash Table 2 artefacts and Table 6 FK-removal artefacts came
from frozen paper snapshot
`36382fe941480958f4186543998fc195f39f4862`. Each configuration was executed
once. The GPT-4o comparison was also one execution per configuration and has
source fingerprint
`1a8278d1adce9408dea47cf63647dc18ec9243849d17f9d9fd39a0f8468f5438`.

They are **not** claimed to be reruns of the current default-branch source and
are **not** presented as multi-run statistics. The current source tree contains
later general implementation corrections.

The audited local LLM4VKG checkout did not contain its original baseline raw
outputs. Its values are therefore comparison-only paper transcriptions under
`baselines/llm4vkg/`; MA4ROM output directories were not relabelled as baseline
evidence.

## Canonical scenario files

Each artefact-backed scenario contains:

- `mapping.ttl`: final R2RML mapping;
- `ontology.ttl`: generated ontology used by the evaluator;
- `f1.txt`: aggregate precision, recall, and F1;
- `per_query_metrics.json`: compact query-level metrics when retained by the
  original evaluator. The frozen NPD evaluator output did not retain this file.

The Table 2 macro F1 values are 0.7994 for DeepSeek V4-Flash, 0.7948 for
GPT-4o, and a reported 0.6073 for LLM4VKG. These are descriptive averages over
the listed single executions, not repeated-run estimates.
