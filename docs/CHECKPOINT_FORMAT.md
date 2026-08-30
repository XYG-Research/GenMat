# Checkpoint format

## Supported versions

- Format 1: legacy checkpoint with no `format_version` field.
- Format 2: current versioned checkpoint with provenance metadata.

`genmat.checkpoints.load_model_checkpoint` validates required fields, vocabulary
consistency, model configuration, exact state-dict compatibility, and an optional
expected SHA256 before returning a model.

## Required model fields

```text
model_state
model_config
vocab
id_to_token
tokenize_config
pad_id
```

Format 2 additionally writes:

```text
format_version = 2
metadata.genmat_version # canonical writer version
metadata.genim_version  # retained serialized compatibility key
metadata.created_at_utc
metadata.training_seed
metadata.training_steps
metadata.train_size
metadata.validation_size
metadata.validation_fraction
metadata.last_batch_train_loss
metadata.validation_loss
metadata.element_feature_set
metadata.dataset_name
metadata.dataset_sha256
metadata.dataset_provenance
```

Dataset/token statistics and symmetry statistics remain top-level fields for
legacy compatibility.

## Element-feature compatibility

Legacy model configurations omit `element_feature_set` and therefore resolve to
`legacy4`, preserving the historical linear-layer shape. New training defaults
to `periodic8`. Changing this field on an existing checkpoint is invalid because
it changes the learned projection dimensions; strict loading rejects the mismatch.

## Distribution

Weights should be immutable release assets rather than Git blobs. Publish:

1. the exact filename and byte size;
2. SHA256;
3. model and tokenizer configuration;
4. training-data provenance and license/terms;
5. intended chemistry domain and benchmark report.

`download_checkpoint` writes to a temporary file, verifies the optional hash,
and atomically replaces the cache target only after success.
