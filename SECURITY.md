# Security

Do not commit API keys, database passwords, `.env` files, generated
`ontology.properties`, logs, or experiment output directories.

MA4ROM reads LLM and PostgreSQL credentials only from environment variables.
If a credential is accidentally committed, revoke it immediately and remove it
from every published Git object; deleting it in a later commit is insufficient.

Please report security issues privately to the repository owner before opening
a public issue.

