#!/usr/bin/env python3
"""
Visualization script for DeepEP test results
Creates comprehensive plots with subplots for analysis
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import pandas as pd

def load_results(results_dir: str):
    """Load all results from a test directory"""
    with open(os.path.join(results_dir, "all_results.json"), "r") as f:
        return json.load(f)

def extract_sweep_data(sweep_results):
    """Extract data from sweep results for plotting"""
    sweep_values = []
    metrics = {
        'dispatch_fp8_gbps': [],
        'dispatch_bf16_gbps': [],
        'combine_gbps': [],
        'dispatch_combine_gbps_avg': [],
        'dispatch_gbps_avg': [],
        'combine_gbps_avg': [],
    }
    
    for result in sweep_results:
        if result['returncode'] == 0:
            sweep_values.append(result['sweep_value'])
            parsed = result['parsed_results']
            
            for metric, values in metrics.items():
                value = parsed.get(metric, None)
                values.append(value if value is not None else np.nan)
    
    return sweep_values, metrics

def create_intranode_plots(intranode_data, output_dir):
    """Create plots for intranode test results"""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Token sweep plot
    ax1 = fig.add_subplot(gs[0, 0])
    sweep_vals, metrics = extract_sweep_data(intranode_data['token_sweep'])
    if sweep_vals:
        ax1.plot(sweep_vals, metrics['dispatch_fp8_gbps'], 'o-', label='FP8 Dispatch', linewidth=2, markersize=8)
        ax1.plot(sweep_vals, metrics['dispatch_bf16_gbps'], 's-', label='BF16 Dispatch', linewidth=2, markersize=8)
        ax1.plot(sweep_vals, metrics['combine_gbps'], '^-', label='Combine', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Tokens', fontsize=12)
        ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax1.set_title('Performance vs Token Count', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
    
    # Hidden dimension sweep
    ax2 = fig.add_subplot(gs[0, 1])
    sweep_vals, metrics = extract_sweep_data(intranode_data['hidden_sweep'])
    if sweep_vals:
        ax2.plot(sweep_vals, metrics['dispatch_fp8_gbps'], 'o-', label='FP8 Dispatch', linewidth=2, markersize=8)
        ax2.plot(sweep_vals, metrics['dispatch_bf16_gbps'], 's-', label='BF16 Dispatch', linewidth=2, markersize=8)
        ax2.plot(sweep_vals, metrics['combine_gbps'], '^-', label='Combine', linewidth=2, markersize=8)
        ax2.set_xlabel('Hidden Dimension', fontsize=12)
        ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax2.set_title('Performance vs Hidden Dimension', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Top-k sweep
    ax3 = fig.add_subplot(gs[1, 0])
    sweep_vals, metrics = extract_sweep_data(intranode_data['topk_sweep'])
    if sweep_vals:
        ax3.plot(sweep_vals, metrics['dispatch_fp8_gbps'], 'o-', label='FP8 Dispatch', linewidth=2, markersize=8)
        ax3.plot(sweep_vals, metrics['dispatch_bf16_gbps'], 's-', label='BF16 Dispatch', linewidth=2, markersize=8)
        ax3.plot(sweep_vals, metrics['combine_gbps'], '^-', label='Combine', linewidth=2, markersize=8)
        ax3.set_xlabel('Top-K Value', fontsize=12)
        ax3.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax3.set_title('Performance vs Top-K Selection', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)
    
    # Expert count sweep
    ax4 = fig.add_subplot(gs[1, 1])
    sweep_vals, metrics = extract_sweep_data(intranode_data['expert_sweep'])
    if sweep_vals:
        ax4.plot(sweep_vals, metrics['dispatch_fp8_gbps'], 'o-', label='FP8 Dispatch', linewidth=2, markersize=8)
        ax4.plot(sweep_vals, metrics['dispatch_bf16_gbps'], 's-', label='BF16 Dispatch', linewidth=2, markersize=8)
        ax4.plot(sweep_vals, metrics['combine_gbps'], '^-', label='Combine', linewidth=2, markersize=8)
        ax4.set_xlabel('Number of Experts', fontsize=12)
        ax4.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax4.set_title('Performance vs Expert Count', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log', base=2)
    
    # Bandwidth comparison heatmap
    ax5 = fig.add_subplot(gs[2:, :])
    
    # Collect all bandwidth data
    all_configs = []
    all_bandwidths = []
    
    for sweep_name, sweep_data in intranode_data.items():
        for result in sweep_data:
            if result['returncode'] == 0:
                config_str = f"{sweep_name}_{result['sweep_param']}={result['sweep_value']}"
                bf16_bw = result['parsed_results'].get('dispatch_bf16_gbps', 0)
                if bf16_bw:
                    all_configs.append(config_str)
                    all_bandwidths.append(bf16_bw)
    
    if all_configs:
        # Create horizontal bar chart
        y_pos = np.arange(len(all_configs))
        ax5.barh(y_pos, all_bandwidths, color=plt.cm.viridis(np.array(all_bandwidths)/max(all_bandwidths)))
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(all_configs, fontsize=10)
        ax5.set_xlabel('BF16 Dispatch Bandwidth (GB/s)', fontsize=12)
        ax5.set_title('Bandwidth Comparison Across All Configurations', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(all_bandwidths):
            ax5.text(v + 2, i, f'{v:.1f}', va='center')
    
    fig.suptitle('Intranode Test Results - Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intranode_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_low_latency_plots(low_latency_data, output_dir):
    """Create plots for low-latency test results"""
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Token sweep plot
    ax1 = fig.add_subplot(gs[0, 0])
    sweep_vals, metrics = extract_sweep_data(low_latency_data['token_sweep'])
    if sweep_vals:
        ax1.plot(sweep_vals, metrics['dispatch_combine_gbps_avg'], 'o-', label='Dispatch+Combine', linewidth=2, markersize=8, color='red')
        ax1.plot(sweep_vals, metrics['dispatch_gbps_avg'], 's-', label='Dispatch Only', linewidth=2, markersize=8, color='blue')
        ax1.plot(sweep_vals, metrics['combine_gbps_avg'], '^-', label='Combine Only', linewidth=2, markersize=8, color='green')
        ax1.set_xlabel('Number of Tokens', fontsize=12)
        ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax1.set_title('Low-Latency Performance vs Token Count', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Hidden dimension sweep
    ax2 = fig.add_subplot(gs[0, 1])
    sweep_vals, metrics = extract_sweep_data(low_latency_data['hidden_sweep'])
    if sweep_vals:
        ax2.plot(sweep_vals, metrics['dispatch_combine_gbps_avg'], 'o-', label='Dispatch+Combine', linewidth=2, markersize=8, color='red')
        ax2.plot(sweep_vals, metrics['dispatch_gbps_avg'], 's-', label='Dispatch Only', linewidth=2, markersize=8, color='blue')
        ax2.plot(sweep_vals, metrics['combine_gbps_avg'], '^-', label='Combine Only', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Hidden Dimension', fontsize=12)
        ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax2.set_title('Low-Latency Performance vs Hidden Dimension', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Top-k sweep
    ax3 = fig.add_subplot(gs[1, 0])
    sweep_vals, metrics = extract_sweep_data(low_latency_data['topk_sweep'])
    if sweep_vals:
        ax3.plot(sweep_vals, metrics['dispatch_combine_gbps_avg'], 'o-', label='Dispatch+Combine', linewidth=2, markersize=8, color='red')
        ax3.plot(sweep_vals, metrics['dispatch_gbps_avg'], 's-', label='Dispatch Only', linewidth=2, markersize=8, color='blue')
        ax3.plot(sweep_vals, metrics['combine_gbps_avg'], '^-', label='Combine Only', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Top-K Value', fontsize=12)
        ax3.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax3.set_title('Low-Latency Performance vs Top-K Selection', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Expert count sweep
    ax4 = fig.add_subplot(gs[1, 1])
    sweep_vals, metrics = extract_sweep_data(low_latency_data['expert_sweep'])
    if sweep_vals:
        ax4.plot(sweep_vals, metrics['dispatch_combine_gbps_avg'], 'o-', label='Dispatch+Combine', linewidth=2, markersize=8, color='red')
        ax4.plot(sweep_vals, metrics['dispatch_gbps_avg'], 's-', label='Dispatch Only', linewidth=2, markersize=8, color='blue')
        ax4.plot(sweep_vals, metrics['combine_gbps_avg'], '^-', label='Combine Only', linewidth=2, markersize=8, color='green')
        ax4.set_xlabel('Number of Experts', fontsize=12)
        ax4.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax4.set_title('Low-Latency Performance vs Expert Count', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Latency analysis
    ax5 = fig.add_subplot(gs[2, :])
    
    # Extract latency data
    all_configs = []
    dispatch_latencies = []
    combine_latencies = []
    
    for sweep_name, sweep_data in low_latency_data.items():
        if sweep_name == 'nvlink_comparison':
            continue
        for result in sweep_data:
            if result['returncode'] == 0:
                config_str = f"{result['sweep_param']}={result['sweep_value']}"
                parsed = result['parsed_results']
                if parsed.get('dispatch_combine_us_avg'):
                    all_configs.append(config_str)
                    dispatch_latencies.append(parsed.get('dispatch_us_avg', 0))
                    combine_latencies.append(parsed.get('combine_us_avg', 0))
    
    if all_configs:
        x = np.arange(len(all_configs))
        width = 0.35
        
        ax5.bar(x - width/2, dispatch_latencies, width, label='Dispatch Latency', color='skyblue')
        ax5.bar(x + width/2, combine_latencies, width, label='Combine Latency', color='lightcoral')
        
        ax5.set_xlabel('Configuration', fontsize=12)
        ax5.set_ylabel('Latency (microseconds)', fontsize=12)
        ax5.set_title('Latency Breakdown by Configuration', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(all_configs, rotation=45, ha='right')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # NVLink comparison
    ax6 = fig.add_subplot(gs[3, :])
    nvlink_data = low_latency_data.get('nvlink_comparison', [])
    nvlink_enabled = None
    nvlink_disabled = None
    
    for result in nvlink_data:
        if result['returncode'] == 0:
            if result['sweep_value'] == False:  # NVLink enabled
                nvlink_enabled = result['parsed_results'].get('dispatch_combine_gbps_avg', 0)
            else:  # NVLink disabled
                nvlink_disabled = result['parsed_results'].get('dispatch_combine_gbps_avg', 0)
    
    if nvlink_enabled:
        categories = ['NVLink Enabled', 'NVLink Disabled']
        values = [nvlink_enabled, nvlink_disabled if nvlink_disabled else 0]
        colors = ['green', 'red']
        
        bars = ax6.bar(categories, values, color=colors, alpha=0.7)
        ax6.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax6.set_title('Impact of NVLink on Low-Latency Performance', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                        f'{val:.1f}', ha='center', va='bottom', fontsize=12)
    
    # Summary statistics
    ax7 = fig.add_subplot(gs[4, :])
    ax7.axis('off')
    
    # Calculate summary stats
    summary_text = "Low-Latency Mode Summary Statistics\n" + "="*50 + "\n\n"
    
    total_tests = sum(len(sweep) for sweep in low_latency_data.values())
    successful = sum(1 for sweep in low_latency_data.values() 
                    for result in sweep if result['returncode'] == 0)
    
    summary_text += f"Total tests run: {total_tests}\n"
    summary_text += f"Successful tests: {successful} ({successful/total_tests*100:.1f}%)\n\n"
    
    # Best configurations
    best_bandwidth = 0
    best_config = ""
    
    for sweep_name, sweep_data in low_latency_data.items():
        for result in sweep_data:
            if result['returncode'] == 0:
                bw = result['parsed_results'].get('dispatch_combine_gbps_avg', 0)
                if bw > best_bandwidth:
                    best_bandwidth = bw
                    best_config = f"{sweep_name}: {result['sweep_param']}={result['sweep_value']}"
    
    summary_text += f"Best configuration: {best_config}\n"
    summary_text += f"Peak bandwidth: {best_bandwidth:.1f} GB/s\n"
    
    ax7.text(0.1, 0.5, summary_text, fontsize=12, family='monospace', 
             verticalalignment='center', transform=ax7.transAxes)
    
    fig.suptitle('Low-Latency Test Results - Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'low_latency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plots(all_data, output_dir):
    """Create comparison plots between intranode and low-latency modes"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Bandwidth comparison
    ax = axes[0, 0]
    
    # Get best bandwidths from each mode
    intranode_bw = []
    low_latency_bw = []
    
    for result in all_data['intranode']['token_sweep']:
        if result['returncode'] == 0:
            bw = result['parsed_results'].get('dispatch_bf16_gbps', 0)
            if bw > 0:
                intranode_bw.append(bw)
    
    for result in all_data['low_latency']['token_sweep']:
        if result['returncode'] == 0:
            bw = result['parsed_results'].get('dispatch_combine_gbps_avg', 0)
            if bw > 0:
                low_latency_bw.append(bw)
    
    if intranode_bw and low_latency_bw:
        data = [intranode_bw, low_latency_bw]
        bp = ax.boxplot(data, labels=['Intranode', 'Low-Latency'], patch_artist=True)
        
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax.set_title('Bandwidth Distribution Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Scaling analysis
    ax = axes[0, 1]
    
    # Token scaling comparison
    intra_tokens = []
    intra_bw = []
    ll_tokens = []
    ll_bw = []
    
    for result in all_data['intranode']['token_sweep']:
        if result['returncode'] == 0:
            intra_tokens.append(result['sweep_value'])
            intra_bw.append(result['parsed_results'].get('dispatch_bf16_gbps', 0))
    
    for result in all_data['low_latency']['token_sweep']:
        if result['returncode'] == 0:
            ll_tokens.append(result['sweep_value'])
            ll_bw.append(result['parsed_results'].get('dispatch_combine_gbps_avg', 0))
    
    if intra_tokens and ll_tokens:
        ax.plot(intra_tokens, intra_bw, 'o-', label='Intranode', linewidth=2, markersize=8)
        ax.plot(ll_tokens, ll_bw, 's-', label='Low-Latency', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Tokens', fontsize=12)
        ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax.set_title('Token Scaling Comparison', fontsize=14, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Efficiency analysis
    ax = axes[1, 0]
    
    # Calculate efficiency (bandwidth per token)
    if intra_tokens and ll_tokens:
        intra_eff = [bw/tok for tok, bw in zip(intra_tokens, intra_bw) if bw > 0]
        ll_eff = [bw/tok for tok, bw in zip(ll_tokens, ll_bw) if bw > 0]
        
        ax.plot(intra_tokens[:len(intra_eff)], intra_eff, 'o-', label='Intranode', linewidth=2, markersize=8)
        ax.plot(ll_tokens[:len(ll_eff)], ll_eff, 's-', label='Low-Latency', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Tokens', fontsize=12)
        ax.set_ylabel('Bandwidth per Token (GB/s/token)', fontsize=12)
        ax.set_title('Communication Efficiency Analysis', fontsize=14, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create comparison table
    table_data = []
    table_data.append(['Metric', 'Intranode', 'Low-Latency'])
    
    # Peak bandwidth
    peak_intra = max(intra_bw) if intra_bw else 0
    peak_ll = max(ll_bw) if ll_bw else 0
    table_data.append(['Peak Bandwidth (GB/s)', f'{peak_intra:.1f}', f'{peak_ll:.1f}'])
    
    # Average bandwidth
    avg_intra = np.mean(intra_bw) if intra_bw else 0
    avg_ll = np.mean(ll_bw) if ll_bw else 0
    table_data.append(['Avg Bandwidth (GB/s)', f'{avg_intra:.1f}', f'{avg_ll:.1f}'])
    
    # Token range
    if intra_tokens and ll_tokens:
        table_data.append(['Token Range', f'{min(intra_tokens)}-{max(intra_tokens)}', 
                          f'{min(ll_tokens)}-{max(ll_tokens)}'])
    
    # Success rate
    intra_success = sum(1 for r in all_data['intranode']['token_sweep'] if r['returncode'] == 0)
    ll_success = sum(1 for r in all_data['low_latency']['token_sweep'] if r['returncode'] == 0)
    intra_total = len(all_data['intranode']['token_sweep'])
    ll_total = len(all_data['low_latency']['token_sweep'])
    
    table_data.append(['Success Rate', f'{intra_success}/{intra_total}', f'{ll_success}/{ll_total}'])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style the header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Performance Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    fig.suptitle('Intranode vs Low-Latency Mode Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mode_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main(results_dir: str):
    """Main visualization function"""
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    
    output_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating intranode plots...")
    create_intranode_plots(results['intranode'], output_dir)
    
    print("Creating low-latency plots...")
    create_low_latency_plots(results['low_latency'], output_dir)
    
    print("Creating comparison plots...")
    create_comparison_plots(results, output_dir)
    
    print(f"\nVisualization complete! Plots saved to: {output_dir}")
    print("Generated files:")
    print("  - intranode_analysis.png")
    print("  - low_latency_analysis.png")
    print("  - mode_comparison.png")
    
    return output_dir

if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Find the most recent results directory
        artifact_dir = "artifact_evaluation"
        dirs = [d for d in os.listdir(artifact_dir) if d.startswith("focused_test_results_")]
        if dirs:
            results_dir = os.path.join(artifact_dir, sorted(dirs)[-1])
        else:
            print("No results directory found!")
            sys.exit(1)
    
    output_dir = main(results_dir)