#!/usr/bin/env python3
"""
Quick test runner for DeepEP - runs a subset of configurations
"""

import os
import sys
import json
import subprocess
import re
import time
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Quick test configurations (subset)
INTRANODE_CONFIGS = {
    "num_tokens": [2048, 4096],
    "hidden": [5120, 7168],
    "num_topk": [4, 8],
    "num_experts": [128, 256],
}

LOW_LATENCY_CONFIGS = {
    "num_tokens": [64, 128],
    "hidden": [5120, 7168],
    "num_topk": [4, 8],
    "num_experts": [144, 288],
}

# Import functions from comprehensive runner
from comprehensive_test_runner import (
    run_command, parse_intranode_output, parse_low_latency_output,
    average_list, FIXED_PARAMS
)

def run_quick_tests():
    """Run a quick subset of tests"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"artifact_evaluation/quick_test_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Test 1: Basic intranode
    print("Test 1: Basic intranode configuration")
    cmd = "source .venv/bin/activate && PYTHONPATH=. python tests/test_intranode.py --num-tokens 2048 --hidden 7168 --num-topk 8 --num-experts 256"
    stdout, stderr, returncode = run_command(cmd, timeout=120)
    
    if returncode == 0:
        parsed = parse_intranode_output(stdout)
        results.append({
            "test": "intranode_basic",
            "success": True,
            "parsed": parsed
        })
        print(f"✓ Success: FP8={parsed.get('dispatch_fp8_gbps', 0):.1f} GB/s")
    else:
        print(f"✗ Failed: {stderr[:100]}")
        results.append({"test": "intranode_basic", "success": False})
    
    # Test 2: Basic low-latency
    print("\nTest 2: Basic low-latency configuration")
    cmd = "source .venv/bin/activate && PYTHONPATH=. python tests/test_low_latency.py --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288"
    stdout, stderr, returncode = run_command(cmd, timeout=120)
    
    if returncode == 0:
        parsed = parse_low_latency_output(stdout)
        # Average the lists
        keys_to_average = [k for k in parsed.keys() if isinstance(parsed[k], list) and parsed[k]]
        for key in keys_to_average:
            parsed[f"{key}_avg"] = sum(parsed[key]) / len(parsed[key])
        results.append({
            "test": "low_latency_basic",
            "success": True,
            "parsed": parsed
        })
        avg_gbps = parsed.get('dispatch_combine_gbps_avg', 0)
        print(f"✓ Success: Dispatch+Combine={avg_gbps:.1f} GB/s")
    else:
        print(f"✗ Failed: {stderr[:100]}")
        results.append({"test": "low_latency_basic", "success": False})
    
    # Save results
    with open(f"{output_dir}/quick_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nQuick tests completed. Results saved to {output_dir}")
    return all(r.get("success", False) for r in results)

if __name__ == "__main__":
    success = run_quick_tests()
    sys.exit(0 if success else 1)