# Retained partial attempt

The first real-corpus index build, verified load, 24 deterministic protocol runs,
and independent formula checks completed. No final smoke summary was produced:
the live worker adapter failed while reading the original corpus JSONL with
`str.splitlines()`:

`json.decoder.JSONDecodeError: Unterminated string starting at: line 1 column 14 (char 13)`

The corpus has 4,936 newline-delimited records but 4,946 `splitlines()` segments
(2 embedded U+2028 and 8 U+2029). Chunks have 10,313 records but 10,318 segments
(1 U+2028 and 4 U+2029). These are valid JSON string characters and original
inputs were not modified. The repair uses the adapter's supported whole-JSON
array input mode with semantically identical, hash-bound local transport copies.
No provider-worker file was edited. The original index and partial outputs are
retained here; the final run rebuilt the exact requested index destination.
