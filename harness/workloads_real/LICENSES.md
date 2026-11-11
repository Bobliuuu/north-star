# Licenses of the prompts in `realtraffic.jsonl`

The repo's MIT license (top-level `LICENSE`) covers the code only. Each prompt in
`realtraffic.jsonl` is an excerpt of a public dataset and stays under that dataset's
own terms (also recorded per row in the `license` and `source` fields).

| Dataset (`source`) | Rows | License |
|---|---|---|
| HuggingFaceH4/ultrachat_200k | 9 | MIT |
| abisee/cnn_dailymail 3.0.0 | 9 | Apache-2.0 |
| glaiveai/glaive-function-calling-v2 | 9 | Apache-2.0 |
| nuprl/CanItEdit | 9 | MIT |
| openai/openai_humaneval | 9 | MIT |
| emozilla/quality | 9 | CC-BY-4.0 |
| google-research-datasets/mbpp | 9 | CC-BY-4.0 |
| rajpurkar/squad_v2 | 9 | CC-BY-SA-4.0 (share-alike applies to these rows) |
| ccdv/govreport-summarization | 18 | US government reports, public domain |
| EdinburghNLP/xsum | 9 | **No license declared** on the Hugging Face dataset card (`unknown`); BBC news articles, used here as short benchmark excerpts. Remove these rows if you need clear redistribution rights. |

Attribution: cite the datasets above when redistributing these rows. Row counts are
from the file at the time of writing; the `source` field is authoritative.
