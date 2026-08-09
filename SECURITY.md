# Security policy

## Supported version

Security fixes target the latest GenIM release and the current main branch.

## Reporting

Report vulnerabilities privately through the repository owner's GitHub
security advisory channel. Do not include credentials, private datasets, or
untrusted checkpoint files in a public issue.

## Checkpoint safety

Model checkpoints are executable-risk inputs. GenIM loads supported PyTorch
checkpoints with `weights_only=True`, validates their primitive schema and
tensor state, and can require an expected SHA256. Only use checkpoints from a
trusted source with a published checksum. Never weaken these checks merely to
load an unknown file.

GenIM's safe loader reduces pickle-related risk; it does not prove that a model
is scientifically trustworthy or free from adversarial behavior.
