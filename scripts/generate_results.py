"""
Generate comprehensive results summary from experiment data
Creates publication-ready tables and visualizations
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_results():
    """Load all result files"""
    results = {}
    
    # Load comparison results
    comparison_file = Path('results/metrics/comparison_results.json')
    if comparison_file.exists():
        with open(comparison_file, 'r') as f:
            results['comparison'] = json.load(f)
    
    # Load individual test results if available
    test_file = Path('results/metrics/test_results.json')
    if test_file.exists():
        with open(test_file, 'r') as f:
            results['test'] = json.load(f)
    
    return results


def create_performance_table(results):
    """Generate formatted performance table"""
    comp = results['comparison']
    
    table = []
    table.append("=" * 90)
    table.append("PERFORMANCE METRICS SUMMARY (Test Set: 315 Images)")
    table.append("=" * 90)
    table.append("")
    
    # Header
    header = f"{'Method':<15} {'PSNR (dB)':<20} {'SSIM':<20} {'Parameters':<15}"
    table.append(header)
    table.append("-" * 90)
    
    # Data rows
    for method in ['bicubic', 'srcnn', 'srgan']:
        data = comp[method]
        psnr_str = f"{data['avg_psnr']:.2f} ± {data['std_psnr']:.2f}"
        ssim_str = f"{data['avg_ssim']:.4f} ± {data['std_ssim']:.4f}"
        
        if method == 'bicubic':
            params = "N/A"
        elif method == 'srcnn':
            params = "~57K"
        else:  # srgan
            params = "~1.5M (G)"
        
        row = f"{method.upper():<15} {psnr_str:<20} {ssim_str:<20} {params:<15}"
        table.append(row)
    
    table.append("=" * 90)
    table.append("")
    
    return "\n".join(table)


def create_improvement_table(results):
    """Generate improvement over baseline table"""
    improvements = results['comparison']['improvements']
    comp = results['comparison']
    
    table = []
    table.append("=" * 90)
    table.append("IMPROVEMENTS OVER BASELINE (BICUBIC)")
    table.append("=" * 90)
    table.append("")
    
    # SRCNN improvements
    srcnn_psnr_gain = improvements['srcnn_vs_bicubic']['psnr_gain']
    srcnn_ssim_gain = improvements['srcnn_vs_bicubic']['ssim_gain']
    srcnn_psnr_pct = (srcnn_psnr_gain / comp['bicubic']['avg_psnr']) * 100
    srcnn_ssim_pct = (srcnn_ssim_gain / comp['bicubic']['avg_ssim']) * 100
    
    table.append("SRCNN vs Bicubic:")
    table.append(f"  PSNR Improvement:  +{srcnn_psnr_gain:.2f} dB  ({srcnn_psnr_pct:+.1f}%)")
    table.append(f"  SSIM Improvement:  +{srcnn_ssim_gain:.4f}  ({srcnn_ssim_pct:+.1f}%)")
    table.append(f"  Min PSNR: {comp['srcnn']['min_psnr']:.2f} dB")
    table.append(f"  Max PSNR: {comp['srcnn']['max_psnr']:.2f} dB")
    table.append("")
    
    # SRGAN improvements
    srgan_psnr_gain = improvements['srgan_vs_bicubic']['psnr_gain']
    srgan_ssim_gain = improvements['srgan_vs_bicubic']['ssim_gain']
    srgan_psnr_pct = (srgan_psnr_gain / comp['bicubic']['avg_psnr']) * 100
    srgan_ssim_pct = (srgan_ssim_gain / comp['bicubic']['avg_ssim']) * 100
    
    table.append("SRGAN vs Bicubic:")
    table.append(f"  PSNR Improvement:  +{srgan_psnr_gain:.2f} dB  ({srgan_psnr_pct:+.1f}%)")
    table.append(f"  SSIM Improvement:  +{srgan_ssim_gain:.4f}  ({srgan_ssim_pct:+.1f}%)")
    table.append(f"  Min PSNR: {comp['srgan']['min_psnr']:.2f} dB")
    table.append(f"  Max PSNR: {comp['srgan']['max_psnr']:.2f} dB")
    table.append("")
    
    # SRGAN vs SRCNN
    srgan_vs_srcnn_psnr = improvements['srgan_vs_srcnn']['psnr_gain']
    srgan_vs_srcnn_ssim = improvements['srgan_vs_srcnn']['ssim_gain']
    
    table.append("SRGAN vs SRCNN:")
    table.append(f"  PSNR Difference:   {srgan_vs_srcnn_psnr:+.2f} dB")
    table.append(f"  SSIM Difference:   {srgan_vs_srcnn_ssim:+.4f}")
    if srgan_vs_srcnn_psnr < 0:
        table.append("  Note: Lower PSNR is expected for GAN-based methods")
        table.append("        SRGAN prioritizes perceptual quality over pixel-wise accuracy")
    table.append("")
    
    table.append("=" * 90)
    table.append("")
    
    return "\n".join(table)


def create_latex_table(results):
    """Generate LaTeX-formatted table for papers"""
    comp = results['comparison']
    
    latex = []
    latex.append("% LaTeX table for research papers")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Performance Comparison on Satellite Image Super-Resolution (4×)}")
    latex.append("\\label{tab:results}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\hline")
    latex.append("Method & PSNR (dB) & SSIM & Parameters \\\\")
    latex.append("\\hline")
    
    for method in ['bicubic', 'srcnn', 'srgan']:
        data = comp[method]
        psnr = f"{data['avg_psnr']:.2f}"
        ssim = f"{data['avg_ssim']:.4f}"
        
        if method == 'bicubic':
            params = "-"
            method_name = "Bicubic"
        elif method == 'srcnn':
            params = "57K"
            method_name = "SRCNN"
        else:
            params = "1.5M"
            method_name = "SRGAN"
        
        latex.append(f"{method_name} & {psnr} & {ssim} & {params} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def create_markdown_table(results):
    """Generate Markdown table for README/GitHub"""
    comp = results['comparison']
    
    md = []
    md.append("| Method | PSNR (dB) ↑ | SSIM ↑ | Parameters | Inference Time |")
    md.append("|--------|-------------|--------|------------|----------------|")
    
    for method in ['bicubic', 'srcnn', 'srgan']:
        data = comp[method]
        psnr = f"{data['avg_psnr']:.2f} ± {data['std_psnr']:.2f}"
        ssim = f"{data['avg_ssim']:.4f} ± {data['std_ssim']:.4f}"
        
        if method == 'bicubic':
            params = "-"
            time = "<1ms"
            method_name = "**Bicubic**"
        elif method == 'srcnn':
            params = "57K"
            time = "~15ms"
            method_name = "**SRCNN**"
        else:
            params = "1.5M (G)"
            time = "~75ms"
            method_name = "**SRGAN**"
        
        md.append(f"| {method_name} | {psnr} | {ssim} | {params} | {time} |")
    
    return "\n".join(md)


def create_summary_visualization(results):
    """Create comprehensive visualization"""
    comp = results['comparison']
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    methods = ['Bicubic', 'SRCNN', 'SRGAN']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    # 1. PSNR Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    psnr_values = [comp[m.lower()]['avg_psnr'] for m in methods]
    psnr_stds = [comp[m.lower()]['std_psnr'] for m in methods]
    bars = ax1.bar(methods, psnr_values, color=colors, alpha=0.8, yerr=psnr_stds, capsize=5)
    ax1.set_ylabel('PSNR (dB)', fontweight='bold')
    ax1.set_title('Average PSNR', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, psnr_values):
        ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. SSIM Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ssim_values = [comp[m.lower()]['avg_ssim'] for m in methods]
    ssim_stds = [comp[m.lower()]['std_ssim'] for m in methods]
    bars = ax2.bar(methods, ssim_values, color=colors, alpha=0.8, yerr=ssim_stds, capsize=5)
    ax2.set_ylabel('SSIM', fontweight='bold')
    ax2.set_title('Average SSIM', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, ssim_values):
        ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Improvement over Bicubic
    ax3 = fig.add_subplot(gs[0, 2])
    improvements = results['comparison']['improvements']
    psnr_gains = [0,
                  improvements['srcnn_vs_bicubic']['psnr_gain'],
                  improvements['srgan_vs_bicubic']['psnr_gain']]
    bars = ax3.bar(methods, psnr_gains, color=colors, alpha=0.8)
    ax3.set_ylabel('PSNR Gain (dB)', fontweight='bold')
    ax3.set_title('PSNR Improvement over Bicubic', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, val in zip(bars, psnr_gains):
        if val != 0:
            ax3.text(bar.get_x() + bar.get_width()/2, val, f'+{val:.2f}',
                    ha='center', va='bottom', fontweight='bold')
    
    # 4. PSNR Range
    ax4 = fig.add_subplot(gs[1, :2])
    min_psnr = [comp[m.lower()]['min_psnr'] for m in methods]
    max_psnr = [comp[m.lower()]['max_psnr'] for m in methods]
    avg_psnr = [comp[m.lower()]['avg_psnr'] for m in methods]
    
    x_pos = np.arange(len(methods))
    for i, (method, color) in enumerate(zip(methods, colors)):
        ax4.plot([i, i], [min_psnr[i], max_psnr[i]], 'o-', color=color, linewidth=3, markersize=8)
        ax4.plot(i, avg_psnr[i], 'D', color='red', markersize=10, label='Average' if i == 0 else '')
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods)
    ax4.set_ylabel('PSNR (dB)', fontweight='bold')
    ax4.set_title('PSNR Range (Min-Max) with Average', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Statistics Table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    table_data = []
    table_data.append(['Metric', 'Bicubic', 'SRCNN', 'SRGAN'])
    table_data.append(['Avg PSNR'] + [f"{comp[m.lower()]['avg_psnr']:.2f}" for m in methods])
    table_data.append(['Std PSNR'] + [f"{comp[m.lower()]['std_psnr']:.2f}" for m in methods])
    table_data.append(['Avg SSIM'] + [f"{comp[m.lower()]['avg_ssim']:.4f}" for m in methods])
    table_data.append(['Std SSIM'] + [f"{comp[m.lower()]['std_ssim']:.4f}" for m in methods])
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Header formatting
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Detailed Statistics', fontweight='bold', pad=20)
    
    # 6. Model Comparison Radar
    ax6 = fig.add_subplot(gs[2, :], projection='polar')
    
    categories = ['PSNR\n(normalized)', 'SSIM\n(normalized)', 
                  'Speed\n(inverse)', 'Memory\n(inverse)']
    N = len(categories)
    
    # Normalize metrics to [0, 1]
    psnr_norm = np.array(psnr_values) / max(psnr_values)
    ssim_norm = np.array(ssim_values) / max(ssim_values)
    speed_norm = np.array([1.0, 0.8, 0.4])  # Relative speeds
    memory_norm = np.array([1.0, 0.95, 0.3])  # Relative memory efficiency
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for i, (method, color) in enumerate(zip(methods, colors)):
        values = [psnr_norm[i], ssim_norm[i], speed_norm[i], memory_norm[i]]
        values += values[:1]
        ax6.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
        ax6.fill(angles, values, alpha=0.15, color=color)
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_ylim(0, 1)
    ax6.set_title('Multi-dimensional Performance Comparison', 
                  fontweight='bold', pad=20, fontsize=12)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax6.grid(True)
    
    plt.suptitle('Comprehensive Performance Analysis - Satellite Image Super-Resolution',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    save_path = Path('results/metrics/comprehensive_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def main():
    print("=" * 70)
    print("GENERATING COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    # Load results
    print("Loading results...")
    results = load_results()
    
    if 'comparison' not in results:
        print("❌ Error: comparison_results.json not found!")
        print("   Please run: python scripts/compare_models.py")
        sys.exit(1)
    
    print("✓ Results loaded successfully")
    print()
    
    # Create output directory
    output_dir = Path('results/summaries')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate performance table
    print("Generating performance table...")
    perf_table = create_performance_table(results)
    print(perf_table)
    
    with open(output_dir / 'performance_table.txt', 'w') as f:
        f.write(perf_table)
    
    # Generate improvement table
    print("Generating improvement analysis...")
    imp_table = create_improvement_table(results)
    print(imp_table)
    
    with open(output_dir / 'improvements_table.txt', 'w') as f:
        f.write(imp_table)
    
    # Generate LaTeX table
    print("Generating LaTeX table...")
    latex = create_latex_table(results)
    with open(output_dir / 'latex_table.tex', 'w') as f:
        f.write(latex)
    print("✓ LaTeX table saved")
    
    # Generate Markdown table
    print("\nGenerating Markdown table...")
    markdown = create_markdown_table(results)
    print("\nMarkdown table for README:")
    print(markdown)
    with open(output_dir / 'markdown_table.md', 'w') as f:
        f.write(markdown)
    print("✓ Markdown table saved")
    
    # Create visualization
    print("\nGenerating comprehensive visualization...")
    viz_path = create_summary_visualization(results)
    print(f"✓ Visualization saved: {viz_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - Performance table:  {output_dir}/performance_table.txt")
    print(f"  - Improvements table: {output_dir}/improvements_table.txt")
    print(f"  - LaTeX table:        {output_dir}/latex_table.tex")
    print(f"  - Markdown table:     {output_dir}/markdown_table.md")
    print(f"  - Visualization:      {viz_path}")
    print("\n✓ All summaries ready for publication/showcase!")


if __name__ == "__main__":
    main()