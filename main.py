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


def add_import(tx, from_file, to_file, import_name, import_type="module"):
    """Add an IMPORTS relationship between files."""
    tx.run(
        "MERGE (imp:Import {name: $import_name, from_file: $from_file, to_file: $to_file, type: $import_type}) "
        "WITH imp "
        "MATCH (f1:File {path: $from_file}) "
        "MATCH (f2:File {path: $to_file}) "
        "MERGE (f1)-[:IMPORTS {name: $import_name, type: $import_type}]->(f2) "
        "MERGE (f1)-[:CONTAINS]->(imp) "
        "MERGE (imp)-[:REFERENCES]->(f2)",
        from_file=from_file,
        to_file=to_file,
        import_name=import_name,
        import_type=import_type,
    )


def add_function_call(tx, caller_path, callee_name, caller_file):
    """Add a CALLS relationship between functions."""
    tx.run(
        "MATCH (caller {path: $caller_path}) "
        "MERGE (call:FunctionCall {caller: $caller_path, callee: $callee_name, file: $caller_file}) "
        "MERGE (caller)-[:CALLS {function: $callee_name}]->(call)",
        caller_path=caller_path,
        callee_name=callee_name,
        caller_file=caller_file,
    )


def add_inheritance(tx, child_class_path, parent_class_name, file_path):
    """Add an INHERITS relationship between classes."""
    tx.run(
        "MATCH (child:Class {path: $child_class_path}) "
        "MERGE (parent:Class {name: $parent_class_name, file: $file_path}) "
        "MERGE (child)-[:INHERITS]->(parent)",
        child_class_path=child_class_path,
        parent_class_name=parent_class_name,
        file_path=file_path,
    )


def add_method(tx, method_name, class_path, file_path):
    """Add a method node and relationship to its class."""
    method_path = f"{class_path}::{method_name}"
    tx.run(
        "MERGE (method:Method {path: $method_path}) "
        "SET method.name = $method_name, method.class = $class_path, method.file = $file_path, method.type = 'method' "
        "WITH method "
        "MATCH (cls:Class {path: $class_path}) "
        "MERGE (cls)-[:HAS_METHOD]->(method)",
        method_path=method_path,
        method_name=method_name,
        class_path=class_path,
        file_path=file_path,
    )


def add_decorator(tx, decorated_path, decorator_name, file_path):
    """Add a DECORATED_BY relationship."""
    tx.run(
        "MATCH (decorated {path: $decorated_path}) "
        "MERGE (decorator:Decorator {name: $decorator_name, file: $file_path}) "
        "MERGE (decorated)-[:DECORATED_BY]->(decorator)",
        decorated_path=decorated_path,
        decorator_name=decorator_name,
        file_path=file_path,
    )


def add_variable(tx, var_name, scope_path, file_path, var_type="variable"):
    """Add a variable node and relationship to its scope."""
    var_path = f"{scope_path}::{var_name}"
    tx.run(
        "MERGE (var:Variable {path: $var_path}) "
        "SET var.name = $var_name, var.scope = $scope_path, var.file = $file_path, var.type = $var_type "
        "WITH var "
        "MATCH (scope {path: $scope_path}) "
        "MERGE (scope)-[:DEFINES]->(var)",
        var_path=var_path,
        var_name=var_name,
        scope_path=scope_path,
        file_path=file_path,
        var_type=var_type,
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
    """Parse Python file content to extract functions, classes, and all relationships."""
    try:
        tree = ast.parse(content)

        # Track current class context for methods
        current_class = None
        current_class_path = None

        # Extract imports first
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.name
                    # Try to resolve to file path (basic heuristic)
                    potential_file = import_name.replace(".", "/") + ".py"
                    session.execute_write(
                        add_import, file_path, potential_file, import_name, "module"
                    )
                    print(f"Added import: {file_path} -> {import_name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module
                    potential_file = module_name.replace(".", "/") + ".py"
                    for alias in node.names:
                        import_name = alias.name
                        session.execute_write(
                            add_import,
                            file_path,
                            potential_file,
                            f"{module_name}.{import_name}",
                            "from_import",
                        )
                        print(
                            f"Added from import: {file_path} -> {module_name}.{import_name}"
                        )

        # Process top-level nodes in order to maintain context
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_path = f"{file_path}::{node.name}"
                current_class = node.name
                current_class_path = class_path

                # Add class
                session.execute_write(add_class, node.name, file_path)
                session.execute_write(add_contains, file_path, class_path)
                print(f"Added class: {node.name} in {file_path}")

                # Add inheritance relationships
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        parent_name = base.id
                        session.execute_write(
                            add_inheritance, class_path, parent_name, file_path
                        )
                        print(f"Added inheritance: {node.name} -> {parent_name}")
                    elif isinstance(base, ast.Attribute):
                        # Handle module.ClassName inheritance
                        parent_name = (
                            ast.unparse(base)
                            if hasattr(ast, "unparse")
                            else str(base.attr)
                        )
                        session.execute_write(
                            add_inheritance, class_path, parent_name, file_path
                        )
                        print(f"Added inheritance: {node.name} -> {parent_name}")

                # Add decorators
                for decorator in node.decorator_list:
                    decorator_name = (
                        ast.unparse(decorator)
                        if hasattr(ast, "unparse")
                        else str(decorator)
                    )
                    session.execute_write(
                        add_decorator, class_path, decorator_name, file_path
                    )
                    print(f"Added decorator: {decorator_name} on class {node.name}")

                # Process class methods
                for class_node in node.body:
                    if isinstance(class_node, ast.FunctionDef):
                        method_path = f"{class_path}::{class_node.name}"

                        # Add method
                        session.execute_write(
                            add_method, class_node.name, class_path, file_path
                        )
                        print(f"Added method: {class_node.name} in class {node.name}")

                        # Add method decorators
                        for decorator in class_node.decorator_list:
                            decorator_name = (
                                ast.unparse(decorator)
                                if hasattr(ast, "unparse")
                                else str(decorator)
                            )
                            session.execute_write(
                                add_decorator, method_path, decorator_name, file_path
                            )
                            print(
                                f"Added decorator: {decorator_name} on method {class_node.name}"
                            )

                        # Extract function calls within method
                        _extract_function_calls(
                            class_node, method_path, file_path, session
                        )

                        # Extract variables within method
                        _extract_variables(class_node, method_path, file_path, session)

                    elif isinstance(class_node, ast.Assign):
                        # Class attributes
                        for target in class_node.targets:
                            if isinstance(target, ast.Name):
                                session.execute_write(
                                    add_variable,
                                    target.id,
                                    class_path,
                                    file_path,
                                    "class_attribute",
                                )
                                print(
                                    f"Added class attribute: {target.id} in {node.name}"
                                )

                current_class = None
                current_class_path = None

            elif isinstance(node, ast.FunctionDef):
                func_path = f"{file_path}::{node.name}"

                # Add function
                session.execute_write(add_function, node.name, file_path)
                session.execute_write(add_contains, file_path, func_path)
                print(f"Added function: {node.name} in {file_path}")

                # Add decorators
                for decorator in node.decorator_list:
                    decorator_name = (
                        ast.unparse(decorator)
                        if hasattr(ast, "unparse")
                        else str(decorator)
                    )
                    session.execute_write(
                        add_decorator, func_path, decorator_name, file_path
                    )
                    print(f"Added decorator: {decorator_name} on function {node.name}")

                # Extract function calls within function
                _extract_function_calls(node, func_path, file_path, session)

                # Extract variables within function
                _extract_variables(node, func_path, file_path, session)

            elif isinstance(node, ast.Assign):
                # Module-level variables
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        session.execute_write(
                            add_variable,
                            target.id,
                            file_path,
                            file_path,
                            "module_variable",
                        )
                        print(f"Added module variable: {target.id} in {file_path}")

    except SyntaxError as e:
        print(f"Failed to parse {file_path}: {e}")


def _extract_function_calls(func_node, func_path, file_path, session):
    """Extract function calls from within a function or method."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                # Simple function call: func_name()
                callee_name = node.func.id
                session.execute_write(
                    add_function_call, func_path, callee_name, file_path
                )
                print(f"Added function call: {func_path} -> {callee_name}")
            elif isinstance(node.func, ast.Attribute):
                # Method call: obj.method() or module.func()
                callee_name = (
                    ast.unparse(node.func)
                    if hasattr(ast, "unparse")
                    else f"{node.func.attr}"
                )
                session.execute_write(
                    add_function_call, func_path, callee_name, file_path
                )
                print(f"Added method call: {func_path} -> {callee_name}")


def _extract_variables(func_node, func_path, file_path, session):
    """Extract variable assignments from within a function or method."""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    session.execute_write(
                        add_variable, var_name, func_path, file_path, "local_variable"
                    )
                    print(f"Added local variable: {var_name} in {func_path}")
                elif isinstance(target, ast.Attribute):
                    # Instance variables: self.var = value
                    if isinstance(target.value, ast.Name) and target.value.id == "self":
                        var_name = target.attr
                        session.execute_write(
                            add_variable,
                            var_name,
                            func_path,
                            file_path,
                            "instance_variable",
                        )
                        print(f"Added instance variable: {var_name} in {func_path}")
        elif isinstance(node, ast.AnnAssign) and node.target:
            # Type annotated assignments: var: int = 5
            if isinstance(node.target, ast.Name):
                var_name = node.target.id
                session.execute_write(
                    add_variable, var_name, func_path, file_path, "annotated_variable"
                )
                print(f"Added annotated variable: {var_name} in {func_path}")


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

            # NEW: DETAILED RELATIONSHIP ANALYSIS
            f.write("\n## Detailed Relationship Analysis\n\n")

            # Import relationships
            f.write("### Import Dependencies:\n")
            import_relationships = session.run(
                """
                MATCH (f1:File)-[r:IMPORTS]->(f2:File)
                RETURN f1.path as from_file, f2.path as to_file, r.name as import_name, r.type as import_type
                ORDER BY from_file, to_file
                """
            ).data()

            if import_relationships:
                current_file = None
                for import_rel in import_relationships:
                    from_file = import_rel["from_file"]
                    to_file = import_rel["to_file"]
                    import_name = import_rel["import_name"]
                    import_type = import_rel["import_type"]

                    if from_file != current_file:
                        current_file = from_file
                        f.write(f"\n**{from_file}:**\n")

                    f.write(
                        f"  - imports `{import_name}` from {to_file} ({import_type})\n"
                    )
            else:
                f.write("No import relationships detected.\n")

            # Function call relationships
            f.write("\n### Function Call Relationships:\n")
            function_calls = session.run(
                """
                MATCH (caller)-[r:CALLS]->(call:FunctionCall)
                RETURN caller.path as caller_path, r.function as callee_name, call.file as file_path
                ORDER BY caller_path, callee_name
                """
            ).data()

            if function_calls:
                current_caller = None
                for call_rel in function_calls:
                    caller_path = call_rel["caller_path"]
                    callee_name = call_rel["callee_name"]
                    file_path = call_rel["file_path"]

                    if caller_path != current_caller:
                        current_caller = caller_path
                        f.write(f"\n**{caller_path}:**\n")

                    f.write(f"  - calls `{callee_name}()`\n")
            else:
                f.write("No function call relationships detected.\n")

            # Class inheritance relationships
            f.write("\n### Class Inheritance Hierarchy:\n")
            inheritance_relationships = session.run(
                """
                MATCH (child:Class)-[:INHERITS]->(parent:Class)
                RETURN child.name as child_name, child.file as child_file, 
                       parent.name as parent_name, parent.file as parent_file
                ORDER BY child_file, child_name
                """
            ).data()

            if inheritance_relationships:
                for inherit_rel in inheritance_relationships:
                    child_name = inherit_rel["child_name"]
                    child_file = inherit_rel["child_file"]
                    parent_name = inherit_rel["parent_name"]
                    parent_file = inherit_rel["parent_file"]
                    f.write(
                        f"- **{child_name}** ({child_file}) inherits from **{parent_name}** ({parent_file})\n"
                    )
            else:
                f.write("No inheritance relationships detected.\n")

            # Class methods relationships
            f.write("\n### Class Methods:\n")
            class_method_relationships = session.run(
                """
                MATCH (cls:Class)-[:HAS_METHOD]->(method:Method)
                RETURN cls.name as class_name, cls.file as class_file, 
                       collect(method.name) as methods
                ORDER BY class_file, class_name
                """
            ).data()

            if class_method_relationships:
                for class_methods in class_method_relationships:
                    class_name = class_methods["class_name"]
                    class_file = class_methods["class_file"]
                    methods = class_methods["methods"]
                    f.write(f"\n**{class_name}** ({class_file}):\n")
                    for method in methods:
                        f.write(f"  - {method}()\n")
            else:
                f.write("No class method relationships detected.\n")

            # Decorator relationships
            f.write("\n### Decorators:\n")
            decorator_relationships = session.run(
                """
                MATCH (decorated)-[:DECORATED_BY]->(decorator:Decorator)
                RETURN decorated.path as decorated_path, decorator.name as decorator_name
                ORDER BY decorated_path
                """
            ).data()

            if decorator_relationships:
                current_decorated = None
                for decorator_rel in decorator_relationships:
                    decorated_path = decorator_rel["decorated_path"]
                    decorator_name = decorator_rel["decorator_name"]

                    if decorated_path != current_decorated:
                        current_decorated = decorated_path
                        f.write(f"\n**{decorated_path}:**\n")

                    f.write(f"  - @{decorator_name}\n")
            else:
                f.write("No decorator relationships detected.\n")

            # Variable relationships
            f.write("\n### Variables and Scope:\n")
            variable_relationships = session.run(
                """
                MATCH (scope)-[:DEFINES]->(var:Variable)
                RETURN scope.path as scope_path, var.name as var_name, var.type as var_type
                ORDER BY scope_path, var_type, var_name
                """
            ).data()

            if variable_relationships:
                current_scope = None
                for var_rel in variable_relationships:
                    scope_path = var_rel["scope_path"]
                    var_name = var_rel["var_name"]
                    var_type = var_rel["var_type"]

                    if scope_path != current_scope:
                        current_scope = scope_path
                        f.write(f"\n**{scope_path}:**\n")

                    type_emoji = {
                        "module_variable": "🌐",
                        "class_attribute": "🏗️",
                        "instance_variable": "🔧",
                        "local_variable": "📍",
                        "annotated_variable": "📝",
                    }.get(var_type, "📄")

                    f.write(f"  - {type_emoji} {var_name} ({var_type})\n")
            else:
                f.write("No variable relationships detected.\n")

            # Relationship statistics
            f.write("\n### Relationship Statistics:\n")

            # Count different relationship types
            relationship_stats = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) as relationship_type, count(r) as count
                ORDER BY count DESC
                """
            ).data()

            f.write("**Relationship counts by type:**\n")
            for stat in relationship_stats:
                rel_type = stat["relationship_type"]
                count = stat["count"]
                f.write(f"- {rel_type}: {count}\n")

            # Total nodes by type
            node_stats = session.run(
                """
                MATCH (n)
                RETURN labels(n)[0] as node_type, count(n) as count
                ORDER BY count DESC
                """
            ).data()

            f.write("\n**Node counts by type:**\n")
            for stat in node_stats:
                node_type = stat["node_type"]
                count = stat["count"]
                f.write(f"- {node_type}: {count}\n")

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
            total_methods = session.run(
                "MATCH (method:Method) RETURN count(method) as count"
            ).single()["count"]
            total_imports = session.run(
                "MATCH ()-[:IMPORTS]->() RETURN count(*) as count"
            ).single()["count"]
            total_function_calls = session.run(
                "MATCH ()-[:CALLS]->() RETURN count(*) as count"
            ).single()["count"]
            total_inheritance = session.run(
                "MATCH ()-[:INHERITS]->() RETURN count(*) as count"
            ).single()["count"]
            total_decorators = session.run(
                "MATCH ()-[:DECORATED_BY]->() RETURN count(*) as count"
            ).single()["count"]
            total_variables = session.run(
                "MATCH (var:Variable) RETURN count(var) as count"
            ).single()["count"]

            f.write(f"- **Total files**: {total_files}\n")
            f.write(f"- **Total functions**: {total_functions}\n")
            f.write(f"- **Total classes**: {total_classes}\n")
            f.write(f"- **Total methods**: {total_methods}\n")
            f.write(f"- **Total imports**: {total_imports}\n")
            f.write(f"- **Total function calls**: {total_function_calls}\n")
            f.write(f"- **Total inheritance relationships**: {total_inheritance}\n")
            f.write(f"- **Total decorators**: {total_decorators}\n")
            f.write(f"- **Total variables**: {total_variables}\n")
            f.write(
                f"- **Average functions per file**: {total_functions/max(1, total_files):.1f}\n"
            )
            f.write(
                f"- **Average methods per class**: {total_methods/max(1, total_classes):.1f}\n"
            )

            f.write("\n### Relationship Density Analysis:\n")

            # Calculate relationship density
            if total_files > 0:
                import_density = total_imports / total_files
                call_density = total_function_calls / max(1, total_functions)
                f.write(
                    f"- **Import density**: {import_density:.2f} imports per file\n"
                )
                f.write(f"- **Call density**: {call_density:.2f} calls per function\n")

                if total_classes > 0:
                    inheritance_ratio = total_inheritance / total_classes
                    f.write(
                        f"- **Inheritance ratio**: {inheritance_ratio:.2f} inheritance relationships per class\n"
                    )

            f.write("\n### Code Architecture Insights:\n")

            # Most connected files (by import relationships)
            most_imported = session.run(
                """
                MATCH (f:File)<-[:IMPORTS]-()
                RETURN f.path as file_path, count(*) as import_count
                ORDER BY import_count DESC
                LIMIT 3
                """
            ).data()

            if most_imported:
                f.write("**Most imported files (potential core modules):**\n")
                for imported in most_imported:
                    f.write(
                        f"- {imported['file_path']} ({imported['import_count']} imports)\n"
                    )

            # Most complex functions (by call count)
            most_calling = session.run(
                """
                MATCH (func)-[:CALLS]->()
                RETURN func.path as func_path, count(*) as call_count
                ORDER BY call_count DESC
                LIMIT 3
                """
            ).data()

            if most_calling:
                f.write("\n**Most complex functions (by call count):**\n")
                for calling in most_calling:
                    f.write(
                        f"- {calling['func_path']} ({calling['call_count']} calls)\n"
                    )

            # Classes with most methods
            largest_classes = session.run(
                """
                MATCH (cls:Class)-[:HAS_METHOD]->()
                RETURN cls.name as class_name, cls.file as class_file, count(*) as method_count
                ORDER BY method_count DESC
                LIMIT 3
                """
            ).data()

            if largest_classes:
                f.write("\n**Largest classes (by method count):**\n")
                for large_class in largest_classes:
                    f.write(
                        f"- {large_class['class_name']} in {large_class['class_file']} ({large_class['method_count']} methods)\n"
                    )

            f.write("\n### Recommended Starting Points for Code Analysis:\n")
            f.write(
                "1. **Configuration files**: Check setup requirements and dependencies\n"
            )
            f.write("2. **Entry points**: Examine main.py or __init__.py files\n")
            f.write(
                "3. **Import graph**: Follow import relationships to understand module dependencies\n"
            )
            f.write(
                "4. **Core classes**: Start with classes that have the most methods or inheritance\n"
            )
            f.write(
                "5. **Function calls**: Trace call relationships to understand execution flow\n"
            )
            f.write(
                "6. **API patterns**: Look for functions with API-like naming patterns\n"
            )
            f.write(
                "7. **Error handling**: Study error handling patterns for robustness\n"
            )
            f.write(
                "8. **Decorators**: Understand cross-cutting concerns through decorator usage\n"
            )
            f.write(
                "9. **Variable scope**: Analyze variable definitions to understand data flow\n"
            )
            f.write(
                "10. **Directory structure**: Use logical grouping to understand architecture\n"
            )

            f.write("\n### Neo4j Query Patterns for LLM Analysis:\n")
            f.write("**Essential Cypher queries for code understanding:**\n\n")

            f.write("```cypher\n")
            f.write("// Find all functions in a specific file\n")
            f.write(
                "MATCH (f:File {path: 'your_file.py'})-[:CONTAINS]->(func:Function)\n"
            )
            f.write("RETURN func.name\n\n")

            f.write("// Find all imports for a file\n")
            f.write("MATCH (f:File {path: 'your_file.py'})-[r:IMPORTS]->(target)\n")
            f.write("RETURN r.name, target.path, r.type\n\n")

            f.write("// Find function call chain\n")
            f.write("MATCH (func:Function)-[:CALLS*1..3]->(call)\n")
            f.write("WHERE func.name = 'your_function'\n")
            f.write("RETURN func.path, call.callee\n\n")

            f.write("// Find class hierarchy\n")
            f.write("MATCH (child:Class)-[:INHERITS*1..3]->(parent:Class)\n")
            f.write("RETURN child.name, parent.name\n\n")

            f.write("// Find all methods of a class\n")
            f.write(
                "MATCH (cls:Class {name: 'YourClass'})-[:HAS_METHOD]->(method:Method)\n"
            )
            f.write("RETURN method.name\n\n")

            f.write("// Find variables in scope\n")
            f.write("MATCH (scope)-[:DEFINES]->(var:Variable)\n")
            f.write("WHERE scope.path CONTAINS 'your_function'\n")
            f.write("RETURN var.name, var.type\n")
            f.write("```\n")

            f.write("\n### Integration with Neo4j MCP Server:\n")
            f.write(
                "This knowledge graph is optimized for use with Neo4j MCP (Model Context Protocol) servers.\n"
            )
            f.write("The rich relationship structure enables:\n\n")
            f.write(
                "- **Code navigation**: Follow relationships to understand code structure\n"
            )
            f.write(
                "- **Impact analysis**: Find what depends on a specific function or class\n"
            )
            f.write(
                "- **Refactoring support**: Identify all usages before making changes\n"
            )
            f.write(
                "- **Architecture understanding**: Visualize module dependencies and call graphs\n"
            )
            f.write(
                "- **Code generation**: Use patterns and relationships to generate similar code\n"
            )
            f.write("- **Bug hunting**: Trace execution paths and data flow\n")
            f.write(
                "- **Documentation**: Auto-generate documentation from relationships\n"
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
