import requests
from neo4j import GraphDatabase
from git import Repo
import os
import ast
import base64
from urllib.parse import urlparse
import shutil

from dotenv import load_dotenv

load_dotenv()

# Configuration from environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Configuration constants
REPO_URL = "https://api.github.com/repos/openai/codex"  # Example repo
NEO4J_URI = "bolt://localhost:7687"  # Docker Neo4j instance
LOCAL_REPO_PATH = "./temp_repo"  # Temporary path for cloning
OUTPUT_FILE = "./knowledge_graph_export.txt"  # LLM-readable output file

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# GitHub API headers
headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}


def clear_graph(tx):
    """Clear existing nodes and relationships in the graph."""
    tx.run("MATCH (n) DETACH DELETE n")


def add_directory(tx, path):
    """Add a directory node to Neo4j."""
    tx.run("MERGE (d:Directory {path: $path}) " "SET d.type = 'directory'", path=path)


def add_file(tx, path):
    """Add a file node to Neo4j."""
    tx.run("MERGE (f:File {path: $path}) " "SET f.type = 'file'", path=path)


def add_contains(tx, parent, child):
    """Add a CONTAINS relationship between a directory/file and a file/function/class."""
    tx.run(
        "MATCH (p {path: $parent}) "
        "MATCH (c {path: $child}) "
        "MERGE (p)-[:CONTAINS]->(c)",
        parent=parent,
        child=child,
    )


def add_function(tx, name, file_path):
    """Add a function node to Neo4j."""
    func_path = f"{file_path}::{name}"  # Unique identifier
    tx.run(
        "MERGE (func:Function {path: $path}) "
        "SET func.name = $name, func.file = $file_path, func.type = 'function'",
        path=func_path,
        name=name,
        file_path=file_path,
    )


def add_class(tx, name, file_path):
    """Add a class node to Neo4j."""
    class_path = f"{file_path}::{name}"  # Unique identifier
    tx.run(
        "MERGE (c:Class {path: $path}) "
        "SET c.name = $name, c.file = $file_path, c.type = 'class'",
        path=class_path,
        name=name,
        file_path=file_path,
    )


def add_commit(tx, commit_id, message, timestamp):
    """Add a commit node to Neo4j."""
    tx.run(
        "MERGE (c:Commit {id: $commit_id}) "
        "SET c.message = $message, c.timestamp = $timestamp, c.type = 'commit'",
        commit_id=commit_id,
        message=message,
        timestamp=timestamp,
    )


def add_author(tx, name, email):
    """Add an author node to Neo4j."""
    tx.run(
        "MERGE (a:Author {name: $name}) " "SET a.email = $email, a.type = 'author'",
        name=name,
        email=email,
    )


def add_authored(tx, author, commit_id):
    """Add an AUTHORED relationship between an author and a commit."""
    tx.run(
        "MATCH (a:Author {name: $author}) "
        "MATCH (c:Commit {id: $commit_id}) "
        "MERGE (a)-[:AUTHORED]->(c)",
        author=author,
        commit_id=commit_id,
    )


def add_modified(tx, commit_id, file_path):
    """Add a MODIFIED relationship between a commit and a file."""
    tx.run(
        "MATCH (c:Commit {id: $commit_id}) "
        "MATCH (f:File {path: $file_path}) "
        "MERGE (c)-[:MODIFIED]->(f)",
        commit_id=commit_id,
        file_path=file_path,
    )


def fetch_repo_tree(repo_url):
    """Fetch the repository's file tree using GitHub API."""
    response = requests.get(f"{repo_url}/git/trees/main?recursive=1", headers=headers)
    response.raise_for_status()
    return response.json()["tree"]


def fetch_file_content(repo_owner, repo_name, path):
    """Fetch the content of a file from GitHub."""
    response = requests.get(
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}",
        headers=headers,
    )
    response.raise_for_status()
    content = response.json()
    if "content" in content:
        return base64.b64decode(content["content"]).decode("utf-8", errors="ignore")
    return ""


def clone_repo(repo_url, local_path):
    """Clone the repository locally to access commit history."""
    if os.path.exists(local_path):
        shutil.rmtree(local_path)

    # Convert API URL to git clone URL
    # From: https://api.github.com/repos/owner/repo
    # To: https://github.com/owner/repo.git
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.split("/")
    if len(path_parts) >= 4 and path_parts[1] == "repos":
        owner = path_parts[2]
        repo_name = path_parts[3]
        git_url = f"https://github.com/{owner}/{repo_name}.git"
    else:
        raise ValueError(f"Invalid GitHub API URL format: {repo_url}")

    print(f"Cloning from: {git_url}")
    return Repo.clone_from(git_url, local_path)


def parse_python_file(content, file_path, session):
    """Parse Python file content to extract functions and classes."""
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                session.execute_write(add_function, node.name, file_path)
                session.execute_write(
                    add_contains, file_path, f"{file_path}::{node.name}"
                )
                print(f"Added function: {node.name} in {file_path}")
            elif isinstance(node, ast.ClassDef):
                session.execute_write(add_class, node.name, file_path)
                session.execute_write(
                    add_contains, file_path, f"{file_path}::{node.name}"
                )
                print(f"Added class: {node.name} in {file_path}")
    except SyntaxError as e:
        print(f"Failed to parse {file_path}: {e}")


def process_repository():
    """Traverse the repository, parse Python files, and build the knowledge graph in Neo4j."""
    # Clear existing graph
    with driver.session() as session:
        session.execute_write(clear_graph)
        print("Cleared existing graph.")

    # Fetch repository tree
    print("Fetching repository tree...")
    tree = fetch_repo_tree(REPO_URL)
    repo_owner, repo_name = urlparse(REPO_URL).path.split("/")[2:4]

    # Process directories, files, and Python file contents
    with driver.session() as session:
        for item in tree:
            path = item["path"]
            if item["type"] == "tree":
                session.execute_write(add_directory, path)
                print(f"Added directory: {path}")
            elif item["type"] == "blob":
                session.execute_write(add_file, path)
                print(f"Added file: {path}")
                # Parse Python files for functions and classes
                if path.endswith(".py"):
                    content = fetch_file_content(repo_owner, repo_name, path)
                    parse_python_file(content, path, session)

            # Add CONTAINS relationships for directory structure
            if "/" in path:
                parent_dir = "/".join(path.split("/")[:-1])
                if parent_dir:
                    session.execute_write(add_contains, parent_dir, path)
                    print(f"Added CONTAINS: {parent_dir} -> {path}")

    # Clone repo to access commit history
    print("Cloning repository locally...")
    repo = clone_repo(REPO_URL, LOCAL_REPO_PATH)

    # Process commits (limit to recent 10 for example)
    with driver.session() as session:
        for commit in list(repo.iter_commits())[:10]:
            commit_id = commit.hexsha
            author_name = commit.author.name
            author_email = commit.author.email
            message = commit.message.strip()
            timestamp = commit.committed_datetime.isoformat()

            # Add commit and author
            session.execute_write(add_commit, commit_id, message, timestamp)
            session.execute_write(add_author, author_name, author_email)
            session.execute_write(add_authored, author_name, commit_id)
            print(f"Added commit: {commit_id} by {author_name}")

            # Add MODIFIED relationships for files
            for file_path in commit.stats.files:
                session.execute_write(add_modified, commit_id, file_path)
                print(f"Added MODIFIED: {commit_id} -> {file_path}")


def export_to_llm_readable_format(output_file):
    """Export the knowledge graph to an LLM-readable text format optimized for AI coding agents."""
    with driver.session() as session:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Repository Knowledge Graph\n")
            f.write("# Optimized for AI Coding Agent Analysis\n\n")

            # Export project overview and entry points
            f.write("## Project Overview\n\n")

            # Identify configuration and entry point files
            config_files = session.run(
                """
                MATCH (f:File) 
                WHERE f.path =~ '.*\\.(json|yaml|yml|toml|ini|cfg|conf|py)$' 
                AND (f.path CONTAINS 'config' OR f.path CONTAINS 'setup' OR 
                     f.path CONTAINS 'requirements' OR f.path CONTAINS 'package' OR
                     f.path =~ '.*main\\.py$' OR f.path =~ '.*__init__\\.py$')
                RETURN f.path 
                ORDER BY f.path
                """
            ).data()

            f.write("### Configuration & Entry Points:\n")
            for config_record in config_files:
                f.write(f"- {config_record['f.path']}\n")

            # Export repository structure (simplified)
            f.write("\n### Key Directories:\n")
            directories = session.run(
                """
                MATCH (d:Directory) 
                WHERE NOT d.path CONTAINS '__pycache__' AND NOT d.path CONTAINS '.git'
                RETURN d.path 
                ORDER BY d.path
                LIMIT 20
                """
            ).data()

            for dir_record in directories:
                f.write(f"- {dir_record['d.path']}\n")

            # Export code structure with enhanced function information
            f.write("\n## Code Architecture\n\n")

            # Enhanced Functions with signatures and relationships
            f.write("### Functions & Methods:\n")
            functions = session.run(
                """
                MATCH (func:Function)
                OPTIONAL MATCH (f:File {path: func.file})
                RETURN func.name, func.file, func.path
                ORDER BY func.file, func.name
                """
            ).data()

            current_file = None
            for func_record in functions:
                file_path = func_record["func.file"]
                func_name = func_record["func.name"]

                if file_path != current_file:
                    current_file = file_path
                    f.write(f"\n**{file_path}:**\n")

                f.write(f"  - {func_name}()\n")

            # Enhanced Classes
            f.write("\n### Classes & Objects:\n")
            classes = session.run(
                """
                MATCH (cls:Class)
                RETURN cls.name, cls.file
                ORDER BY cls.file, cls.name
                """
            ).data()

            current_file = None
            for cls_record in classes:
                file_path = cls_record["cls.file"]
                cls_name = cls_record["cls.name"]

                if file_path != current_file:
                    current_file = file_path
                    f.write(f"\n**{file_path}:**\n")

                f.write(f"  - class {cls_name}\n")

            # NEW: File Dependencies and Import Relationships
            f.write("\n## Dependencies & Import Graph\n\n")

            # Python files and their potential imports (basic analysis)
            python_files = session.run(
                """
                MATCH (f:File) 
                WHERE f.path ENDS WITH '.py'
                RETURN f.path
                ORDER BY f.path
                """
            ).data()

            f.write("### Python Module Structure:\n")
            for py_file in python_files:
                file_path = py_file["f.path"]
                # Convert file path to module name
                module_name = file_path.replace("/", ".").replace(".py", "")
                if module_name.endswith(".__init__"):
                    module_name = module_name[:-9]  # Remove .__init__
                f.write(f"- {module_name} ({file_path})\n")

            # NEW: COMPREHENSIVE RELATIONSHIPS SECTION
            f.write("\n## Code Relationships & Dependencies\n\n")

            # File -> Functions/Classes relationships
            f.write("### File Contents (What's defined where):\n")
            file_contents = session.run(
                """
                MATCH (f:File)-[:CONTAINS]->(item)
                WHERE item:Function OR item:Class
                WITH f.path as file_path, 
                     collect(CASE WHEN item:Function THEN 'func:' + item.name ELSE 'class:' + item.name END) as items
                WHERE size(items) > 0
                RETURN file_path, items
                ORDER BY file_path
                """
            ).data()

            for file_content in file_contents:
                file_path = file_content["file_path"]
                items = file_content["items"]
                f.write(f"\n**{file_path}:**\n")
                for item in items:
                    if item.startswith("func:"):
                        f.write(f"  📄 Function: {item[5:]}\n")
                    elif item.startswith("class:"):
                        f.write(f"  🏗️  Class: {item[6:]}\n")

            # Function/Class -> File relationships (reverse lookup)
            f.write("\n### Code Element Locations (Where to find things):\n")

            # Functions by file
            functions_by_file = session.run(
                """
                MATCH (func:Function)
                RETURN func.file as file_path, collect(func.name) as functions
                ORDER BY file_path
                """
            ).data()

            f.write("\n#### Functions by File:\n")
            for func_file in functions_by_file:
                file_path = func_file["file_path"]
                functions = func_file["functions"]
                if functions:
                    f.write(f"- **{file_path}**: {', '.join(functions)}\n")

            # Classes by file
            classes_by_file = session.run(
                """
                MATCH (cls:Class)
                RETURN cls.file as file_path, collect(cls.name) as classes
                ORDER BY file_path
                """
            ).data()

            f.write("\n#### Classes by File:\n")
            for class_file in classes_by_file:
                file_path = class_file["file_path"]
                classes = class_file["classes"]
                if classes:
                    f.write(f"- **{file_path}**: {', '.join(classes)}\n")

            # Directory structure with code elements
            f.write("\n### Directory-based Code Organization:\n")
            dir_structure = session.run(
                """
                MATCH (d:Directory)-[:CONTAINS]->(f:File)
                OPTIONAL MATCH (f)-[:CONTAINS]->(func:Function)
                OPTIONAL MATCH (f)-[:CONTAINS]->(cls:Class)
                WITH d.path as dir_path, f.path as file_path, 
                     count(DISTINCT func) as func_count, 
                     count(DISTINCT cls) as class_count
                WHERE func_count > 0 OR class_count > 0
                RETURN dir_path, collect({file: file_path, functions: func_count, classes: class_count}) as files
                ORDER BY dir_path
                """
            ).data()

            for dir_info in dir_structure:
                dir_path = dir_info["dir_path"]
                files = dir_info["files"]
                if files:
                    f.write(f"\n**📁 {dir_path}/:**\n")
                    for file_info in files:
                        file_path = file_info["file"]
                        func_count = file_info["functions"]
                        class_count = file_info["classes"]
                        f.write(
                            f"  - {file_path}: {func_count} functions, {class_count} classes\n"
                        )

            # Cross-file relationships and potential dependencies
            f.write("\n### Potential Cross-File Dependencies:\n")
            f.write("*Note: Based on naming patterns and file structure analysis*\n\n")

            # Files that might import each other (based on naming patterns)
            potential_imports = session.run(
                """
                MATCH (f1:File), (f2:File)
                WHERE f1.path ENDS WITH '.py' AND f2.path ENDS WITH '.py' AND f1 <> f2
                AND (
                    f1.path CONTAINS split(f2.path, '.')[0] OR
                    f2.path CONTAINS split(f1.path, '.')[0] OR
                    split(f1.path, '/')[0] = split(f2.path, '/')[0]
                )
                RETURN f1.path as file1, f2.path as file2
                ORDER BY f1.path, f2.path
                LIMIT 20
                """
            ).data()

            if potential_imports:
                f.write("#### Likely Import Relationships:\n")
                for import_rel in potential_imports:
                    file1 = import_rel["file1"]
                    file2 = import_rel["file2"]
                    f.write(f"- {file1} ↔️ {file2}\n")
            else:
                f.write("#### No obvious import patterns detected from file names.\n")

            # Function/Class name analysis for relationships
            f.write("\n### Code Element Relationships:\n")

            # Functions that might call each other (same file)
            same_file_functions = session.run(
                """
                MATCH (f:File)-[:CONTAINS]->(func1:Function)
                MATCH (f)-[:CONTAINS]->(func2:Function)
                WHERE func1 <> func2
                RETURN f.path as file_path, collect(func1.name) as functions
                ORDER BY f.path
                """
            ).data()

            f.write("#### Functions in Same Files (Potential Call Relationships):\n")
            for same_file in same_file_functions:
                file_path = same_file["file_path"]
                functions = same_file["functions"]
                if len(functions) > 1:
                    f.write(
                        f"- **{file_path}**: {' → '.join(functions[:5])}{'...' if len(functions) > 5 else ''}\n"
                    )

            # Classes and their potential methods
            class_methods = session.run(
                """
                MATCH (f:File)-[:CONTAINS]->(cls:Class)
                MATCH (f)-[:CONTAINS]->(func:Function)
                RETURN cls.name as class_name, cls.file as file_path, collect(func.name) as potential_methods
                ORDER BY cls.file, cls.name
                """
            ).data()

            f.write("\n#### Classes and Potential Methods:\n")
            for class_method in class_methods:
                class_name = class_method["class_name"]
                file_path = class_method["file_path"]
                methods = class_method["potential_methods"]
                if methods:
                    f.write(
                        f"- **{class_name}** ({file_path}): {', '.join(methods[:5])}{'...' if len(methods) > 5 else ''}\n"
                    )

            # Summary of relationships for AI agent
            f.write("\n### Relationship Summary for AI Agent:\n")

            total_contains = session.run(
                "MATCH ()-[:CONTAINS]->() RETURN count(*) as count"
            ).single()["count"]
            files_with_code = session.run(
                """
                MATCH (f:File)-[:CONTAINS]->(item)
                WHERE item:Function OR item:Class
                RETURN count(DISTINCT f) as count
                """
            ).single()["count"]

            f.write(f"- **Total containment relationships**: {total_contains}\n")
            f.write(f"- **Files containing code elements**: {files_with_code}\n")
            f.write(
                f"- **Key insight**: Use file paths to understand module structure\n"
            )
            f.write(f"- **Key insight**: Functions in same file likely interact\n")
            f.write(
                f"- **Key insight**: Directory structure indicates logical grouping\n"
            )

            # NEW: API and Interface Patterns
            f.write("\n## API Patterns & Interfaces\n\n")

            # Identify potential API endpoints and public interfaces
            api_functions = session.run(
                """
                MATCH (func:Function)
                WHERE func.name STARTS WITH 'api_' OR 
                      func.name STARTS WITH 'get_' OR 
                      func.name STARTS WITH 'post_' OR 
                      func.name STARTS WITH 'put_' OR 
                      func.name STARTS WITH 'delete_' OR
                      func.name CONTAINS 'endpoint' OR
                      func.name CONTAINS 'handler'
                RETURN func.name, func.file
                ORDER BY func.file, func.name
                """
            ).data()

            f.write("### Potential API Functions:\n")
            if api_functions:
                current_file = None
                for api_func in api_functions:
                    file_path = api_func["func.file"]
                    func_name = api_func["func.name"]

                    if file_path != current_file:
                        current_file = file_path
                        f.write(f"\n**{file_path}:**\n")

                    f.write(f"  - {func_name}()\n")
            else:
                f.write("No obvious API patterns detected.\n")

            # NEW: Database and External Service Connections
            f.write("\n### Database & External Connections:\n")
            db_functions = session.run(
                """
                MATCH (func:Function)
                WHERE func.name CONTAINS 'db' OR 
                      func.name CONTAINS 'database' OR 
                      func.name CONTAINS 'sql' OR 
                      func.name CONTAINS 'query' OR
                      func.name CONTAINS 'connect' OR
                      func.name CONTAINS 'session'
                RETURN func.name, func.file
                ORDER BY func.file, func.name
                """
            ).data()

            if db_functions:
                current_file = None
                for db_func in db_functions:
                    file_path = db_func["func.file"]
                    func_name = db_func["func.name"]

                    if file_path != current_file:
                        current_file = file_path
                        f.write(f"\n**{file_path}:**\n")

                    f.write(f"  - {func_name}()\n")
            else:
                f.write("No obvious database connection patterns detected.\n")

            # NEW: Error Handling and Critical Functions
            f.write("\n## Error Handling & Critical Functions\n\n")

            error_functions = session.run(
                """
                MATCH (func:Function)
                WHERE func.name CONTAINS 'error' OR 
                      func.name CONTAINS 'exception' OR 
                      func.name CONTAINS 'handle' OR 
                      func.name CONTAINS 'validate' OR
                      func.name CONTAINS 'check' OR
                      func.name STARTS WITH 'try_'
                RETURN func.name, func.file
                ORDER BY func.file, func.name
                """
            ).data()

            f.write("### Error Handling Functions:\n")
            if error_functions:
                current_file = None
                for error_func in error_functions:
                    file_path = error_func["func.file"]
                    func_name = error_func["func.name"]

                    if file_path != current_file:
                        current_file = file_path
                        f.write(f"\n**{file_path}:**\n")

                    f.write(f"  - {func_name}()\n")
            else:
                f.write("No obvious error handling patterns detected.\n")

            # Simplified commit history (only recent summary)
            f.write("\n## Recent Development Activity\n\n")
            recent_commits = session.run(
                """
                MATCH (c:Commit)
                RETURN c.message, c.timestamp
                ORDER BY c.timestamp DESC
                LIMIT 5
                """
            ).data()

            f.write("### Last 5 Commits:\n")
            for commit_record in recent_commits:
                message = commit_record["c.message"]
                timestamp = commit_record["c.timestamp"]
                f.write(f"- {timestamp}: {message}\n")

            # Enhanced file modification patterns (focus on hotspots)
            f.write("\n### Development Hotspots:\n")
            file_changes = session.run(
                """
                MATCH (c:Commit)-[:MODIFIED]->(f:File)
                WHERE f.path ENDS WITH '.py'
                RETURN f.path, count(c) as modification_count
                ORDER BY modification_count DESC
                LIMIT 10
                """
            ).data()

            f.write("Most frequently modified Python files:\n")
            for change_record in file_changes:
                file_path = change_record["f.path"]
                count = change_record["modification_count"]
                f.write(f"- {file_path} ({count} modifications)\n")

            # Essential statistics for coding agents
            f.write("\n## Codebase Metrics\n\n")

            # Count by file type
            file_types = session.run(
                """
                MATCH (f:File)
                WITH f.path as path
                WITH CASE 
                    WHEN path ENDS WITH '.py' THEN 'Python'
                    WHEN path ENDS WITH '.js' THEN 'JavaScript'
                    WHEN path ENDS WITH '.json' THEN 'JSON'
                    WHEN path ENDS WITH '.yaml' OR path ENDS WITH '.yml' THEN 'YAML'
                    WHEN path ENDS WITH '.md' THEN 'Markdown'
                    WHEN path ENDS WITH '.txt' THEN 'Text'
                    ELSE 'Other'
                END as file_type
                RETURN file_type, count(*) as count
                ORDER BY count DESC
                """
            ).data()

            f.write("### File Types:\n")
            for file_type_record in file_types:
                file_type = file_type_record["file_type"]
                count = file_type_record["count"]
                f.write(f"- {file_type}: {count} files\n")

            # Code complexity indicators
            complexity_stats = session.run(
                """
                MATCH (f:File)
                OPTIONAL MATCH (f)-[:CONTAINS]->(func:Function)
                OPTIONAL MATCH (f)-[:CONTAINS]->(cls:Class)
                WITH f.path as file_path, count(DISTINCT func) as func_count, count(DISTINCT cls) as class_count
                WHERE func_count > 0 OR class_count > 0
                RETURN file_path, func_count, class_count, (func_count + class_count) as total_complexity
                ORDER BY total_complexity DESC
                LIMIT 10
                """
            ).data()

            f.write("\n### Code Complexity (Top 10 files by function/class count):\n")
            for complexity_record in complexity_stats:
                file_path = complexity_record["file_path"]
                func_count = complexity_record["func_count"]
                class_count = complexity_record["class_count"]
                f.write(
                    f"- {file_path}: {func_count} functions, {class_count} classes\n"
                )

            # Summary for AI agent
            f.write("\n## AI Agent Summary\n\n")
            f.write("### Key Insights for Code Understanding:\n")

            total_files = session.run(
                "MATCH (f:File) RETURN count(f) as count"
            ).single()["count"]
            total_functions = session.run(
                "MATCH (func:Function) RETURN count(func) as count"
            ).single()["count"]
            total_classes = session.run(
                "MATCH (cls:Class) RETURN count(cls) as count"
            ).single()["count"]

            f.write(f"- Total files: {total_files}\n")
            f.write(f"- Total functions: {total_functions}\n")
            f.write(f"- Total classes: {total_classes}\n")
            f.write(
                f"- Average functions per file: {total_functions/max(1, total_files):.1f}\n"
            )

            f.write("\n### Recommended Starting Points for Code Analysis:\n")
            f.write("1. Check configuration files for setup requirements\n")
            f.write("2. Examine main.py or __init__.py files for entry points\n")
            f.write("3. Review file contents and relationships for code structure\n")
            f.write("4. Analyze API functions for external interfaces\n")
            f.write("5. Study error handling patterns for robustness\n")
            f.write(
                "6. Use directory-based organization to understand logical grouping\n"
            )


def main():
    try:
        process_repository()
        print("Knowledge graph successfully created in Neo4j.")

        # Export to LLM-readable format
        print(f"Exporting knowledge graph to {OUTPUT_FILE}...")
        export_to_llm_readable_format(OUTPUT_FILE)
        print(f"Export completed. LLM-readable file saved as: {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        driver.close()
        # Clean up local repo
        if os.path.exists(LOCAL_REPO_PATH):
            shutil.rmtree(LOCAL_REPO_PATH)


if __name__ == "__main__":
    main()
