#!/usr/bin/env python3
"""
Comprehensive test runner for DeepEP artifact evaluation
Runs multiple test configurations and collects results for analysis
"""

import os
import sys
import json
import subprocess
import re
import time
from datetime import datetime
from typing import Dict, List, Any
import itertools

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test configurations
INTRANODE_CONFIGS = {
    "num_tokens": [1024, 2048, 4096, 8192],
    "hidden": [4096, 5120, 7168, 8192],
    "num_topk": [2, 4, 8, 16],
    "num_experts": [64, 128, 256, 512],
}

LOW_LATENCY_CONFIGS = {
    "num_tokens": [32, 64, 128, 256],
    "hidden": [4096, 5120, 7168, 8192],
    "num_topk": [2, 4, 8, 16],
    "num_experts": [64, 144, 288, 576],
    "disable_nvlink": [False, True],
}

# Fixed parameters
FIXED_PARAMS = {
    "num_processes": 8,  # Use all 8 GPUs
}

def run_command(cmd: str, timeout: int = 600) -> tuple[str, str, int]:
    """Run a command and return stdout, stderr, and return code"""
    try:
        # Use bash explicitly for source command
        result = subprocess.run(
            ["/bin/bash", "-c", cmd], capture_output=True, text=True, timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout} seconds", -1

def parse_intranode_output(output: str) -> Dict[str, Any]:
    """Parse output from test_intranode.py"""
    results = {
        "layout_kernel_ms": None,
        "dispatch_fp8_gbps": None,
        "dispatch_fp8_us": None,
        "dispatch_bf16_gbps": None,
        "dispatch_bf16_us": None,
        "combine_gbps": None,
        "combine_us": None,
        "tests_passed": False,
    }
    
    # Parse layout kernel performance
    layout_match = re.search(r"\[layout\] Kernel performance: ([\d.]+) ms", output)
    if layout_match:
        results["layout_kernel_ms"] = float(layout_match.group(1))
    
    # Parse best dispatch FP8
    fp8_match = re.search(r"Best dispatch \(FP8\):.*?([\d.]+) GB/s.*?t: ([\d.]+) us", output)
    if fp8_match:
        results["dispatch_fp8_gbps"] = float(fp8_match.group(1))
        results["dispatch_fp8_us"] = float(fp8_match.group(2))
    
    # Parse best dispatch BF16
    bf16_match = re.search(r"Best dispatch \(BF16\):.*?([\d.]+) GB/s.*?t: ([\d.]+) us", output)
    if bf16_match:
        results["dispatch_bf16_gbps"] = float(bf16_match.group(1))
        results["dispatch_bf16_us"] = float(bf16_match.group(2))
    
    # Parse best combine
    combine_match = re.search(r"Best combine:.*?([\d.]+) GB/s.*?t: ([\d.]+) us", output)
    if combine_match:
        results["combine_gbps"] = float(combine_match.group(1))
        results["combine_us"] = float(combine_match.group(2))
    
    # Check if all tests passed
    results["tests_passed"] = "passed" in output and "failed" not in output.lower()
    
    return results

def parse_low_latency_output(output: str) -> Dict[str, Any]:
    """Parse output from test_low_latency.py"""
    results = {
        "buffer_size_mb": None,
        "dispatch_combine_gbps": [],
        "dispatch_combine_us": [],
        "dispatch_gbps": [],
        "dispatch_us": [],
        "combine_gbps": [],
        "combine_us": [],
        "dispatch_send_us": [],
        "dispatch_recv_us": [],
        "combine_send_us": [],
        "combine_recv_us": [],
    }
    
    # Parse buffer size
    buffer_match = re.search(r"Allocating buffer size: ([\d.]+) MB", output)
    if buffer_match:
        results["buffer_size_mb"] = float(buffer_match.group(1))
    
    # Parse dispatch + combine bandwidth
    for match in re.finditer(r"Dispatch \+ combine bandwidth: ([\d.]+) GB/s, avg_t=([\d.]+) us", output):
        results["dispatch_combine_gbps"].append(float(match.group(1)))
        results["dispatch_combine_us"].append(float(match.group(2)))
    
    # Parse individual dispatch/combine
    for match in re.finditer(r"Dispatch bandwidth: ([\d.]+) GB/s, avg_t=([\d.]+) us", output):
        results["dispatch_gbps"].append(float(match.group(1)))
        results["dispatch_us"].append(float(match.group(2)))
    
    for match in re.finditer(r"Combine bandwidth: ([\d.]+) GB/s, avg_t=([\d.]+) us", output):
        results["combine_gbps"].append(float(match.group(1)))
        results["combine_us"].append(float(match.group(2)))
    
    # Parse send/recv times
    for match in re.finditer(r"Dispatch send/recv time: ([\d.]+) \+ ([\d.]+) us", output):
        results["dispatch_send_us"].append(float(match.group(1)))
        results["dispatch_recv_us"].append(float(match.group(2)))
    
    for match in re.finditer(r"Combine send/recv time: ([\d.]+) \+ ([\d.]+) us", output):
        results["combine_send_us"].append(float(match.group(1)))
        results["combine_recv_us"].append(float(match.group(2)))
    
    return results

def average_list(lst: List[float]) -> float:
    """Calculate average of a list, return None if empty"""
    return sum(lst) / len(lst) if lst else None

def run_test_suite(test_type: str, configs: Dict, output_dir: str) -> List[Dict[str, Any]]:
    """Run a suite of tests with different configurations"""
    results = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all parameter combinations
    param_names = list(configs.keys())
    param_values = [configs[name] for name in param_names]
    
    total_tests = 1
    for values in param_values:
        total_tests *= len(values)
    
    print(f"\nRunning {test_type} tests: {total_tests} configurations")
    print("=" * 60)
    
    test_count = 0
    for values in itertools.product(*param_values):
        test_count += 1
        config = dict(zip(param_names, values))
        config.update(FIXED_PARAMS)
        
        # Build command
        cmd_parts = [
            "source .venv/bin/activate &&",
            "PYTHONPATH=.",
            f"python tests/{test_type}.py"
        ]
        
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    cmd_parts.append(f"--{key.replace('_', '-')}")
            else:
                cmd_parts.append(f"--{key.replace('_', '-')} {value}")
        
        cmd = " ".join(cmd_parts)
        
        print(f"\n[{test_count}/{total_tests}] Running: {test_type}")
        print(f"Config: {config}")
        
        # Run test
        start_time = time.time()
        stdout, stderr, returncode = run_command(cmd, timeout=300)
        duration = time.time() - start_time
        
        # Parse results
        if test_type == "test_intranode":
            parsed = parse_intranode_output(stdout)
        else:  # test_low_latency
            parsed = parse_low_latency_output(stdout)
            # Average the list values
            for key in parsed:
                if isinstance(parsed[key], list):
                    parsed[f"{key}_avg"] = average_list(parsed[key])
        
        # Store results
        result = {
            "test_type": test_type,
            "config": config,
            "duration_sec": duration,
            "returncode": returncode,
            "timestamp": datetime.now().isoformat(),
            "parsed_results": parsed,
        }
        
        results.append(result)
        
        # Save intermediate results
        with open(f"{output_dir}/{test_type}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        if returncode == 0:
            print(f"✓ Test completed in {duration:.1f}s")
            if test_type == "test_intranode":
                if parsed["dispatch_fp8_gbps"]:
                    print(f"  FP8 Dispatch: {parsed['dispatch_fp8_gbps']:.1f} GB/s")
                if parsed["dispatch_bf16_gbps"]:
                    print(f"  BF16 Dispatch: {parsed['dispatch_bf16_gbps']:.1f} GB/s")
            else:
                if parsed.get("dispatch_combine_gbps_avg"):
                    print(f"  Dispatch+Combine: {parsed['dispatch_combine_gbps_avg']:.1f} GB/s")
        else:
            print(f"✗ Test failed with code {returncode}")
            if stderr:
                print(f"  Error: {stderr[:200]}...")
    
    return results

def main():
    """Main test runner"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"artifact_evaluation/test_results_{timestamp}"
    
    print("DeepEP Comprehensive Test Suite")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now()}")
    
    all_results = {}
    
    # Run intranode tests
    print("\n\n=== INTRANODE TESTS ===")
    intranode_results = run_test_suite(
        "test_intranode", 
        INTRANODE_CONFIGS, 
        output_dir
    )
    all_results["intranode"] = intranode_results
    
    # Run low-latency tests
    print("\n\n=== LOW-LATENCY TESTS ===")
    low_latency_results = run_test_suite(
        "test_low_latency", 
        LOW_LATENCY_CONFIGS, 
        output_dir
    )
    all_results["low_latency"] = low_latency_results
    
    # Save all results
    with open(f"{output_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nAll tests completed!")
    print(f"Results saved to: {output_dir}")
    print(f"End time: {datetime.now()}")
    
    # Print summary statistics
    print("\n=== SUMMARY ===")
    print(f"Intranode tests: {len(intranode_results)}")
    print(f"Low-latency tests: {len(low_latency_results)}")
    
    successful_intra = sum(1 for r in intranode_results if r["returncode"] == 0)
    successful_ll = sum(1 for r in low_latency_results if r["returncode"] == 0)
    
    print(f"Successful tests: {successful_intra + successful_ll}/{len(intranode_results) + len(low_latency_results)}")

if __name__ == "__main__":
    main()