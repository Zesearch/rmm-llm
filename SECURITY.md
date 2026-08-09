# Security

Do not commit Hugging Face, Weights & Biases, cluster, or cloud credentials.
Pass credentials through the environment or your cluster's secret manager.

If a credential was committed to an earlier private working directory, revoke
and rotate it before publishing this repository. Removing the string from the
latest commit is not sufficient because it may remain in Git history.

