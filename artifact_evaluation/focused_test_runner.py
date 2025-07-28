#!/usr/bin/env python3
"""
Focused test runner for DeepEP - runs meaningful parameter sweeps
"""

import os
import sys
import json
import time
from datetime import datetime
import itertools

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from comprehensive runner
from comprehensive_test_runner import (
    run_command, parse_intranode_output, parse_low_latency_output,
    average_list, FIXED_PARAMS
)

# Focused test configurations - sweep one parameter at a time
INTRANODE_SWEEPS = {
    "token_sweep": {
        "fixed": {"hidden": 7168, "num_topk": 8, "num_experts": 256},
        "sweep": {"num_tokens": [512, 1024, 2048, 4096, 8192, 16384]}
    },
    "hidden_sweep": {
        "fixed": {"num_tokens": 4096, "num_topk": 8, "num_experts": 256},
        "sweep": {"hidden": [2048, 4096, 5120, 6144, 7168, 8192]}
    },
    "topk_sweep": {
        "fixed": {"num_tokens": 4096, "hidden": 7168, "num_experts": 256},
        "sweep": {"num_topk": [1, 2, 4, 8, 16, 32]}
    },
    "expert_sweep": {
        "fixed": {"num_tokens": 4096, "hidden": 7168, "num_topk": 8},
        "sweep": {"num_experts": [32, 64, 128, 256, 512, 1024]}
    }
}

LOW_LATENCY_SWEEPS = {
    "token_sweep": {
        "fixed": {"hidden": 7168, "num_topk": 8, "num_experts": 288},
        "sweep": {"num_tokens": [16, 32, 64, 128, 256, 512]}
    },
    "hidden_sweep": {
        "fixed": {"num_tokens": 128, "num_topk": 8, "num_experts": 288},
        "sweep": {"hidden": [2048, 4096, 5120, 6144, 7168, 8192]}
    },
    "topk_sweep": {
        "fixed": {"num_tokens": 128, "hidden": 7168, "num_experts": 288},
        "sweep": {"num_topk": [1, 2, 4, 8, 16, 32]}
    },
    "expert_sweep": {
        "fixed": {"num_tokens": 128, "hidden": 7168, "num_topk": 8},
        "sweep": {"num_experts": [36, 72, 144, 288, 576]}
    },
    "nvlink_comparison": {
        "fixed": {"num_tokens": 128, "hidden": 7168, "num_topk": 8, "num_experts": 288},
        "sweep": {"disable_nvlink": [False, True]}
    }
}

def run_parameter_sweep(test_type: str, sweep_name: str, config: dict, output_dir: str):
    """Run a parameter sweep test"""
    results = []
    fixed_params = config["fixed"].copy()
    fixed_params.update(FIXED_PARAMS)
    
    sweep_param = list(config["sweep"].keys())[0]
    sweep_values = config["sweep"][sweep_param]
    
    print(f"\n{test_type} - {sweep_name}")
    print(f"Sweeping {sweep_param}: {sweep_values}")
    print(f"Fixed params: {fixed_params}")
    print("-" * 60)
    
    for value in sweep_values:
        # Build command
        params = fixed_params.copy()
        params[sweep_param] = value
        
        cmd_parts = [
            "source .venv/bin/activate &&",
            "PYTHONPATH=.",
            f"python tests/{test_type}.py"
        ]
        
        for key, val in params.items():
            if isinstance(val, bool):
                if val:
                    cmd_parts.append(f"--{key.replace('_', '-')}")
            else:
                cmd_parts.append(f"--{key.replace('_', '-')} {val}")
        
        cmd = " ".join(cmd_parts)
        
        print(f"\n{sweep_param}={value}...", end='', flush=True)
        
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
            keys_to_average = [k for k in parsed.keys() if isinstance(parsed[k], list) and parsed[k]]
            for key in keys_to_average:
                parsed[f"{key}_avg"] = sum(parsed[key]) / len(parsed[key])
        
        # Store results
        result = {
            "sweep_name": sweep_name,
            "sweep_param": sweep_param,
            "sweep_value": value,
            "fixed_params": fixed_params,
            "duration_sec": duration,
            "returncode": returncode,
            "parsed_results": parsed,
        }
        
        results.append(result)
        
        # Print result
        if returncode == 0:
            if test_type == "test_intranode":
                metric = parsed.get('dispatch_bf16_gbps', 0)
                print(f" ✓ {metric:.1f} GB/s ({duration:.1f}s)")
            else:
                metric = parsed.get('dispatch_combine_gbps_avg', 0)
                print(f" ✓ {metric:.1f} GB/s ({duration:.1f}s)")
        else:
            print(f" ✗ Failed")
    
    return results

def main():
    """Main test runner"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"artifact_evaluation/focused_test_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("DeepEP Focused Parameter Sweep Tests")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now()}")
    
    all_results = {
        "metadata": {
            "timestamp": timestamp,
            "start_time": datetime.now().isoformat(),
            "num_gpus": FIXED_PARAMS["num_processes"]
        },
        "intranode": {},
        "low_latency": {}
    }
    
    # Run intranode sweeps
    print("\n\n=== INTRANODE PARAMETER SWEEPS ===")
    for sweep_name, sweep_config in INTRANODE_SWEEPS.items():
        results = run_parameter_sweep("test_intranode", sweep_name, sweep_config, output_dir)
        all_results["intranode"][sweep_name] = results
        
        # Save intermediate results
        with open(f"{output_dir}/intranode_{sweep_name}.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Run low-latency sweeps
    print("\n\n=== LOW-LATENCY PARAMETER SWEEPS ===")
    for sweep_name, sweep_config in LOW_LATENCY_SWEEPS.items():
        results = run_parameter_sweep("test_low_latency", sweep_name, sweep_config, output_dir)
        all_results["low_latency"][sweep_name] = results
        
        # Save intermediate results
        with open(f"{output_dir}/low_latency_{sweep_name}.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Save all results
    all_results["metadata"]["end_time"] = datetime.now().isoformat()
    with open(f"{output_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nAll tests completed!")
    print(f"Results saved to: {output_dir}")
    print(f"End time: {datetime.now()}")
    
    # Print summary
    total_tests = sum(len(sweep) for sweep in all_results["intranode"].values())
    total_tests += sum(len(sweep) for sweep in all_results["low_latency"].values())
    
    successful = 0
    for test_type in ["intranode", "low_latency"]:
        for sweep_results in all_results[test_type].values():
            successful += sum(1 for r in sweep_results if r["returncode"] == 0)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total tests run: {total_tests}")
    print(f"Successful: {successful}/{total_tests}")
    
    return output_dir

if __name__ == "__main__":
    output_dir = main()
    print(f"\nNext step: Run visualization script on {output_dir}")