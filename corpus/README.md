# Put your documents here

`ingest -dir corpus` indexes every `.md`, `.markdown`, `.txt`, `.docx`,
`.pptx` and `.xlsx` file under this directory, recursively. Subdirectories are fine — the path relative to `-dir`
becomes the document id, so `handbook/billing.md` is indexed under that name.

Chunking is heading-aware, so documents that use `#`/`##` headings chunk along
their real section boundaries and each chunk carries its heading path. Plain
text works too, it just falls back to paragraph packing.

Office files are converted to markdown first, so their own structure drives the
same chunking: Word heading styles become headings (with `Title` as the parent
of `Heading 1`), each PowerPoint slide becomes a section, and each Excel sheet
becomes a section with rows as pipe-separated lines. Word tables keep their
columns rather than being flattened into prose.

The pre-2007 binary formats (`.doc`, `.ppt`, `.xls`) are not readable — they are
not ZIP archives. Re-save them as `.docx`/`.pptx`/`.xlsx`; you get a clear error
saying so rather than a silent skip.

Ingest is idempotent: chunk ids are derived from the document id and position,
so re-running overwrites in place instead of duplicating. A full re-index after
changing the embedding model or chunk size is therefore safe to repeat.

Check how a document will be split before indexing it — this needs no services
and no API keys:

```bash
./ask-my-docs chunks -dir corpus          # ids, sizes, section paths
./ask-my-docs chunks -dir corpus -text    # ...and the chunk bodies
```

Those printed ids are what `eval/golden.jsonl` has to reference.

This file is a placeholder and is itself indexable — delete it once you add
real documents, or leave it and ignore the handful of chunks it contributes.
