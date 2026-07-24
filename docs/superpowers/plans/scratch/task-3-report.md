# Task 3 Report: review_server preserves segment_levels

## Status: DONE

## Summary
Added two tests to `tests/test_autopresort_review_server.py` (verbatim from the brief).
Both pass on the existing implementation; no production code changed.

## Production code changed?
No. `auto_presort/store.py` and `auto_presort/review_server.py` were untouched.
`update_doc` already loads the full doc, edits only segments/approved, and saves
the full doc — so `segment_levels` survives both round-trips as designed.

## Step 2 command + output
Command:
```
conda run --no-capture-output -n AR python -m pytest tests/test_autopresort_review_server.py -v -k "segment_levels"
```
Result:
```
tests/test_autopresort_review_server.py::test_api_doc_returns_segment_levels PASSED [ 50%]
tests/test_autopresort_review_server.py::test_save_preserves_segment_levels PASSED [100%]
======================= 2 passed, 6 deselected in 1.07s =======================
```

## Step 4 command + output
Command:
```
conda run --no-capture-output -n AR python -m pytest tests/ -v -k "autopresort"
```
Result:
```
====================== 67 passed, 96 deselected in 6.75s ======================
```
All autopresort backend tests green, including the two new ones. No regressions.

## Concerns
None.
