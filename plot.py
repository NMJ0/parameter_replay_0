
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 14
rcParams['figure.dpi'] = 100
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1
rcParams['axes.linewidth'] = 1.2
rcParams['grid.linewidth'] = 0.8
rcParams['lines.linewidth'] = 2.0
rcParams['lines.markersize'] = 8

COLORS = {
    'primary': '#1f77b4',      # Professional blue
    'secondary': '#ff7f0e',    # Professional orange  
    'accent': '#2ca02c',       # Professional green
    'highlight': '#d62728',    # Professional red
    'purple': '#9467bd',       # Professional purple
    'brown': '#8c564b',        # Professional brown
}



def plot_parameter_importance(percent_list, accuracy_list, task_id, 
                              save_path=None, show_plot=True):
   
    # Create figure with optimal size for publications (typically 3.5" or 7" width)
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='white')
    
    # Convert to numpy arrays for easier manipulation
    x = np.array(percent_list)
    y = np.array(accuracy_list)
    
    # Plot filled area under curve (subtle)
    ax.fill_between(x, 0, y, alpha=0.15, color=COLORS['primary'], 
                     linewidth=0, label='_nolegend_')
    
    # Main line plot
    line = ax.plot(x, y, 
                   color=COLORS['primary'], 
                   linewidth=2.5, 
                   marker='o', 
                   markersize=8,
                   markerfacecolor='white',
                   markeredgewidth=2.5,
                   markeredgecolor=COLORS['primary'],
                   label='Test Accuracy',
                   zorder=3)
    
    # Add subtle grid
    ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.8, color='gray', zorder=0)
    ax.set_axisbelow(True)
    
    # Highlight optimal point (maximum accuracy)
    max_idx = np.argmax(y)
    max_x, max_y = x[max_idx], y[max_idx]
    
    ax.scatter([max_x], [max_y], 
              s=200, marker='*', 
              color=COLORS['highlight'], 
              edgecolor='white',
              linewidth=1.5, 
              zorder=5,
              label=f'Optimal: {max_y:.1f}%')
    
    # Add value annotations for key points (first, optimal, last)
    key_points = [0, 1,2,3,4,5,6,7,8,9]
    for idx in key_points:
        ax.annotate(f'{y[idx]:.1f}%',
                   xy=(x[idx], y[idx]),
                   xytext=(0, 10 if idx != max_idx else 15),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   fontweight='bold' if idx == max_idx else 'normal',
                   bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='white',
                           edgecolor=COLORS['primary'] if idx == max_idx else 'gray',
                           linewidth=1.5 if idx == max_idx else 1,
                           alpha=0.9),
                   zorder=4)
    
    # Set labels and title
    ax.set_xlabel('Retained Parameters (%)', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_title(f'Task {task_id}: Parameter Importance Analysis', 
                fontweight='bold', pad=15)
    
    # Set axis limits with some padding
    ax.set_xlim(0, max(x) + 1)
    ax.set_ylim(0, min(105, max(y) * 1.15))
    
    # Set x-axis ticks
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(val)}' for val in x])
    
    # Add legend
    legend = ax.legend(loc='lower right', 
                      frameon=True, 
                      fancybox=False,
                      edgecolor='gray',
                      framealpha=0.95)
    
    # Style the spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('gray')
        ax.spines[spine].set_linewidth(1.2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        # Support multiple formats
        for fmt in ['.pdf', '.png', '.eps']:
            if save_path.endswith(fmt):
                plt.savefig(save_path, format=fmt[1:], 
                           dpi=300 if fmt == '.png' else None,
                           bbox_inches='tight', 
                           facecolor='white',
                           edgecolor='none')
                print(f"✓ Figure saved: {save_path}")
                break
        else:
            # Default to PDF if no extension
            save_path_pdf = save_path if save_path.endswith('.pdf') else save_path + '.pdf'
            plt.savefig(save_path_pdf, format='pdf', 
                       bbox_inches='tight', facecolor='white')
            print(f"✓ Figure saved: {save_path_pdf}")
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig, ax


