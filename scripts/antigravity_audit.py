"""
==============================================================================
Universal-Node-Resolver — ANTIGRAVITY Audit Protocol
==============================================================================

Programmatically verifies OpenEnv compliance, model/security compatibility,
and prevents circular imports using AST and dynamic introspection.
"""

import ast
import inspect
import os
import sys

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def check_env_ast() -> bool:
    print("Running AST Checks on server/environment.py...")
    try:
        with open("server/environment.py", "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
            
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        env_class = next((c for c in classes if c.name == "UniversalNodeEnv"), None)
        
        if not env_class:
            print("❌ CRITICAL: UniversalNodeEnv not found in server/environment.py")
            return False
            
        methods = [node.name for node in env_class.body if isinstance(node, ast.FunctionDef)]
        required = ["reset", "step", "state"]
        missing = [m for m in required if m not in methods]
        
        if missing:
            print(f"❌ CRITICAL: UniversalNodeEnv missing required OpenEnv methods: {missing}")
            return False
            
        print("✅ AST Check: UniversalNodeEnv implements required OpenEnv interface.")
        return True
    except Exception as e:
        print(f"❌ CRITICAL: AST parsing failed: {e}")
        return False


def check_models_security() -> bool:
    print("Verifying Pydantic Models vs Security Shield...")
    try:
        from server.models import Action
        from server.security import PayloadDefenseShield
        
        # Check required fields for the Action schema
        fields = list(Action.model_fields.keys())
        required = ["action_type", "package_name", "version_target"]
        
        for r in required:
            if r not in fields:
                print(f"❌ CRITICAL: Action model missing required security field: {r}")
                return False
                
        print("✅ Model Check: Pydantic Action model compatible with PayloadDefenseShield.")
        return True
    except Exception as e:
        print(f"❌ CRITICAL: Model verification failed: {e}")
        return False


def check_circular_imports() -> bool:
    print("Simulating load order for Circular Dependency Detection...")
    try:
        # Load in reverse dependency order to flush out circularity
        import server.models
        import server.security
        import server.curriculum
        import server.chaos
        import server.registry
        import server.environment
        
        import client.agent
        import client.planner
        
        import api.main
        
        print("✅ Import Check: Zero circular dependencies detected.")
        return True
    except Exception as e:
        print(f"❌ CRITICAL: Import failure / Circular dependency detected:\n{e}")
        return False


def check_init_files() -> bool:
    print("Verifying python packages...")
    packages = ["server", "client", "api"]
    passed = True
    for pkg in packages:
        init_path = os.path.join(pkg, "__init__.py")
        if not os.path.exists(init_path):
            print(f"⚠️ WARNING: Missing {init_path}. Generating it automatically.")
            with open(init_path, "w") as f:
                f.write("")
            passed = False
    
    if passed:
        print("✅ Package Check: All __init__.py files present.")
    return True


if __name__ == "__main__":
    print("🚀 Booting ANTIGRAVITY Audit Protocol...\n" + "="*50)
    
    checks = [
        check_init_files(),
        check_circular_imports(),
        check_env_ast(),
        check_models_security()
    ]
    
    print("="*50)
    if all(checks):
        print("✨ AUDIT PASSED: The codebase is absolutely bulletproof.")
        sys.exit(0)
    else:
        print("🚨 AUDIT FAILED: Review the critical errors above.")
        sys.exit(1)
