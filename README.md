# GitHub Knowledge Graph

A tool to analyze GitHub repositories and create a knowledge graph using Neo4j, optimized for AI coding agents.

## Features

- Extracts repository structure (files, directories, functions, classes)
- Analyzes commit history and author relationships
- Creates relationships between code elements
- Exports to LLM-readable format for AI coding agents
- Identifies API patterns, database connections, and error handling

## Setup

### 1. Environment Variables

Create a `.env` file or set these environment variables:

```bash
# GitHub Personal Access Token
# Get this from: https://github.com/settings/tokens
export GITHUB_TOKEN="your_github_token_here"

# Neo4j Database Credentials
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_neo4j_password_here"
```

### 2. Neo4j Database

Start a Neo4j instance using Docker:

```bash
docker run \
    --name neo4j \
    -p7474:7474 -p7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/your_password_here \
    neo4j:latest
```

### 3. Install Dependencies

```bash
uv sync
```

## Usage

1. Update the `REPO_URL` in `main.py` to point to your target repository
2. Run the script:

```bash
uv run main.py
```

3. The script will:
   - Clear any existing graph data
   - Fetch repository structure from GitHub API
   - Parse Python files for functions and classes
   - Clone the repository to analyze commit history
   - Create a knowledge graph in Neo4j
   - Export to `knowledge_graph_export.txt` for AI analysis

## Output

The tool generates:
- **Neo4j Graph Database**: Interactive graph with nodes and relationships
- **LLM-readable Export**: Text file optimized for AI coding agents with:
  - Project overview and entry points
  - Code architecture (functions, classes, relationships)
  - API patterns and interfaces
  - Error handling patterns
  - Development activity and hotspots
  - Codebase metrics and complexity analysis

## Configuration

Edit these variables in `main.py`:

- `REPO_URL`: GitHub API URL for the repository to analyze
- `NEO4J_URI`: Neo4j connection string (default: bolt://localhost:7687)
- `LOCAL_REPO_PATH`: Temporary directory for cloning (default: ./temp_repo)
- `OUTPUT_FILE`: Path for LLM-readable export (default: ./knowledge_graph_export.txt)

## Example Repository URLs

```python
# Public repositories
REPO_URL = "https://api.github.com/repos/openai/codex"
REPO_URL = "https://api.github.com/repos/microsoft/vscode"
REPO_URL = "https://api.github.com/repos/python/cpython"

# Private repositories (requires appropriate token permissions)
REPO_URL = "https://api.github.com/repos/your-org/your-private-repo"
```

## Troubleshooting

### Common Issues

1. **Environment Variables Not Set**
   ```
   ValueError: GITHUB_TOKEN environment variable is required
   ```

   Solution: Set the required environment variables as shown above.

2. **Neo4j Connection Failed**
   ```
   ServiceUnavailable: Failed to establish connection
   ```

   Solution: Ensure Neo4j is running and accessible at the configured URI.

3. **GitHub API Rate Limits**
   ```
   403 Forbidden
   ```

   Solution: Use a GitHub Personal Access Token with appropriate permissions.

4. **Repository Access Denied**
   ```
   404 Not Found
   ```

   Solution: Ensure the repository exists and your token has access to it.

## License

MIT License