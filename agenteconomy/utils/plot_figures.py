"""
Economic Simulation Data Visualization Tool

Usage:
    from agenteconomy.utils.plot_figures import EconomyPlotter
    
    plotter = EconomyPlotter("output/monthly_records/run_20260202_054039")
    plotter.plot_gdp()
    plotter.plot_inflation()
    plotter.plot_phillips_curve()
    plotter.plot_beveridge_curve()  # Unemployment vs Vacancy
    plotter.plot_gini_coefficient()
    plotter.plot_labor_market()
    plotter.plot_government()
    plotter.plot_all()
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import numpy as np

# Use default fonts (no Chinese required)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


class EconomyPlotter:
    """Economic Simulation Data Visualizer"""
    
    def __init__(self, data_dir: str):
        """
        Initialize the plotter
        
        Args:
            data_dir: Directory path containing JSON files
        """
        self.data_dir = Path(data_dir)
        self.data: List[Dict[str, Any]] = []
        self._load_data()
    
    def _load_data(self):
        """Load all JSON data files"""
        json_files = sorted(self.data_dir.glob("*.json"))
        
        for f in json_files:
            with open(f, 'r', encoding='utf-8') as fp:
                record = json.load(fp)
                self.data.append(record)
        
        # Sort by econ_month
        self.data.sort(key=lambda x: x.get("econ_month", 0))
        print(f"Loaded {len(self.data)} months of data")
    
    def _get_series(self, key_path: str) -> Tuple[List[int], List[float]]:
        """
        获取指定路径的时间序列数据
        
        Args:
            key_path: 点分隔的键路径，如 "macro.nominal_gdp"
            
        Returns:
            (months, values) 元组
        """
        months = []
        values = []
        
        keys = key_path.split(".")
        for record in self.data:
            month = record.get("econ_month", 0)
            
            # 递归获取嵌套值
            value = record
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    value = None
                    break
            
            if value is not None:
                months.append(month)
                values.append(float(value))
        
        return months, values
    
    def plot_gdp(self, save_path: Optional[str] = None):
        """Plot GDP analysis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Nominal GDP and Real GDP
        ax1 = axes[0, 0]
        months, nominal_gdp = self._get_series("macro.nominal_gdp")
        _, real_gdp = self._get_series("macro.real_gdp")
        
        if months:
            ax1.plot(months, nominal_gdp, 'b-o', label='Nominal GDP', markersize=4)
            ax1.plot(months, real_gdp, 'r-s', label='Real GDP', markersize=4)
            ax1.set_xlabel('Month')
            ax1.set_ylabel('GDP ($)')
            ax1.set_title('GDP Levels')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. GDP Growth Rate
        ax2 = axes[0, 1]
        _, growth_rate = self._get_series("macro.gdp_growth_rate")
        _, real_growth = self._get_series("macro.real_gdp_growth_rate")
        
        if months and growth_rate:
            # Filter out None values
            valid_months = [m for m, g in zip(months, growth_rate) if g is not None]
            valid_growth = [g * 100 for g in growth_rate if g is not None]
            valid_real = [g * 100 for g in real_growth if g is not None]
            
            if valid_months:
                ax2.bar([m - 0.2 for m in valid_months], valid_growth, width=0.4, label='Nominal Growth', alpha=0.7)
                ax2.bar([m + 0.2 for m in valid_months], valid_real, width=0.4, label='Real Growth', alpha=0.7)
                ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax2.set_xlabel('Month')
                ax2.set_ylabel('Growth Rate (%)')
                ax2.set_title('GDP Growth Rate')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # 3. GDP Composition
        ax3 = axes[1, 0]
        _, consumption_rate = self._get_series("macro.consumption_rate")
        _, investment_rate = self._get_series("macro.investment_rate")
        _, government_rate = self._get_series("macro.government_rate")
        
        if months and consumption_rate:
            ax3.stackplot(months, 
                         [c * 100 for c in consumption_rate],
                         [i * 100 for i in investment_rate],
                         [g * 100 for g in government_rate],
                         labels=['Consumption', 'Investment', 'Government'],
                         alpha=0.7)
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Share (%)')
            ax3.set_title('GDP Expenditure Composition')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
        
        # 4. Labor Share
        ax4 = axes[1, 1]
        _, labor_share = self._get_series("macro.labor_share")
        
        if months and labor_share:
            ax4.plot(months, [l * 100 for l in labor_share], 'g-o', markersize=4)
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Labor Share (%)')
            ax4.set_title('Labor Income Share of GDP')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        else:
            save_path = self.data_dir / "gdp_analysis.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.close()
        return fig
    
    def plot_inflation(self, save_path: Optional[str] = None):
        """Plot inflation rate charts"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Price Index
        ax1 = axes[0]
        months, _ = self._get_series("macro.nominal_gdp")
        
        # Get price index
        price_indices = []
        for record in self.data:
            pi = record.get("macro", {}).get("price_index", {})
            idx = pi.get("index") if isinstance(pi, dict) else None
            price_indices.append(idx)
        
        valid_data = [(m, p) for m, p in zip(months, price_indices) if p is not None]
        if valid_data:
            vm, vp = zip(*valid_data)
            ax1.plot(vm, vp, 'b-o', markersize=4)
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Price Index')
            ax1.set_title('Consumer Price Index (CPI)')
            ax1.grid(True, alpha=0.3)
        
        # 2. Inflation Rate
        ax2 = axes[1]
        _, inflation = self._get_series("macro.inflation_rate")
        
        valid_infl = [(m, i * 100) for m, i in zip(months, inflation) if i is not None]
        if valid_infl:
            vm, vi = zip(*valid_infl)
            colors = ['green' if v >= 0 else 'red' for v in vi]
            ax2.bar(vm, vi, color=colors, alpha=0.7)
            ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax2.axhline(y=2, color='orange', linestyle='--', linewidth=1, label='2% Target')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Inflation Rate (%)')
            ax2.set_title('Monthly Inflation Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            save_path = self.data_dir / "inflation_analysis.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        print(f"Chart saved to {save_path}")
        plt.close()
        return fig
    
    def plot_phillips_curve(self, save_path: Optional[str] = None):
        """Plot Phillips Curve (Unemployment-Inflation relationship)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        months, _ = self._get_series("macro.nominal_gdp")
        _, employment_rate = self._get_series("labor_market.employment_rate")
        _, inflation = self._get_series("macro.inflation_rate")
        
        # Calculate unemployment rate
        unemployment_rates = []
        inflation_rates = []
        labels = []
        
        for i, (m, emp, infl) in enumerate(zip(months, employment_rate, inflation)):
            if emp is not None and infl is not None:
                unemp = (1 - emp) * 100  # Unemployment = 1 - Employment
                infl_pct = infl * 100
                unemployment_rates.append(unemp)
                inflation_rates.append(infl_pct)
                labels.append(f"M{m}")
        
        if unemployment_rates and inflation_rates:
            # Scatter plot + time path
            scatter = ax.scatter(unemployment_rates, inflation_rates, 
                               c=range(len(unemployment_rates)), 
                               cmap='viridis', s=100, alpha=0.7)
            
            # Connect points with line
            ax.plot(unemployment_rates, inflation_rates, 'k-', alpha=0.3)
            
            # Annotate months
            for i, (x, y, label) in enumerate(zip(unemployment_rates, inflation_rates, labels)):
                ax.annotate(label, (x, y), textcoords="offset points", 
                           xytext=(5, 5), fontsize=8, alpha=0.7)
            
            # Add trend line
            if len(unemployment_rates) > 2:
                z = np.polyfit(unemployment_rates, inflation_rates, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(unemployment_rates), max(unemployment_rates), 100)
                ax.plot(x_line, p(x_line), 'r--', label=f'Trend Line (slope={z[0]:.2f})', alpha=0.7)
            
            ax.set_xlabel('Unemployment Rate (%)', fontsize=12)
            ax.set_ylabel('Inflation Rate (%)', fontsize=12)
            ax.set_title('Phillips Curve', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, label='Time Progression')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            save_path = self.data_dir / "phillips_curve.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        print(f"Chart saved to {save_path}")
        plt.close()
        return fig
    
    def plot_gini_coefficient(self, save_path: Optional[str] = None):
        """Plot Gini coefficient chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        months, gini = self._get_series("household.aggregate.assets_distribution.gini")
        
        if months and gini:
            ax.plot(months, gini, 'purple', marker='o', markersize=6, linewidth=2)
            ax.axhline(y=0.4, color='orange', linestyle='--', label='Warning Level (0.4)')
            ax.axhline(y=0.5, color='red', linestyle='--', label='High Inequality (0.5)')
            
            ax.fill_between(months, 0, gini, alpha=0.3, color='purple')
            
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Gini Coefficient', fontsize=12)
            ax.set_title('Wealth Inequality (Gini Coefficient)', fontsize=14)
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            save_path = self.data_dir / "gini_coefficient.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        print(f"Chart saved to {save_path}")
        plt.close()
        return fig
    
    def plot_labor_market(self, save_path: Optional[str] = None):
        """Plot labor market indicators"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        months, _ = self._get_series("macro.nominal_gdp")
        
        # 1. Employment Rate
        ax1 = axes[0, 0]
        _, employment_rate = self._get_series("labor_market.employment_rate")
        if months and employment_rate:
            ax1.plot(months, [e * 100 for e in employment_rate], 'g-o', markersize=4)
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Employment Rate (%)')
            ax1.set_title('Employment Rate')
            ax1.grid(True, alpha=0.3)
        
        # 2. Average Wage
        ax2 = axes[0, 1]
        _, avg_wage = self._get_series("labor_market.average_wage")
        if months and avg_wage:
            ax2.plot(months, avg_wage, 'b-o', markersize=4)
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Average Wage ($)')
            ax2.set_title('Monthly Average Wage')
            ax2.grid(True, alpha=0.3)
        
        # 3. Total Wages
        ax3 = axes[1, 0]
        _, total_wage = self._get_series("labor_market.total_wage_gross")
        _, total_wage_net = self._get_series("labor_market.total_wage_net")
        if months and total_wage:
            ax3.plot(months, total_wage, 'b-o', label='Gross Wages', markersize=4)
            ax3.plot(months, total_wage_net, 'g-s', label='Net Wages', markersize=4)
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Total Wages ($)')
            ax3.set_title('Total Wage Bill')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Labor Utilization
        ax4 = axes[1, 1]
        _, total_labor = self._get_series("labor_market.total_labor")
        _, employed_labor = self._get_series("labor_market.employed_labor")
        if months and total_labor:
            ax4.bar([m - 0.2 for m in months], total_labor, width=0.4, label='Total Labor', alpha=0.7)
            ax4.bar([m + 0.2 for m in months], employed_labor, width=0.4, label='Employed', alpha=0.7)
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Labor (person-hours)')
            ax4.set_title('Labor Supply and Demand')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            save_path = self.data_dir / "labor_market.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        print(f"Chart saved to {save_path}")
        plt.close()
        return fig
    
    def plot_beveridge_curve(self, save_path: Optional[str] = None):
        """Plot Beveridge Curve (Unemployment-Vacancy relationship)
        
        The Beveridge curve shows the inverse relationship between 
        unemployment rate and job vacancy rate.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        months, _ = self._get_series("macro.nominal_gdp")
        _, unemployment_rate = self._get_series("labor_market.unemployment_rate")
        _, job_fill_rate = self._get_series("labor_market.job_fill_rate")
        
        # Also try to get total jobs and matched jobs for more accurate vacancy calculation
        _, total_job_positions = self._get_series("labor_market.total_job_positions")
        _, total_matched_jobs = self._get_series("labor_market.total_matched_jobs")
        
        unemployment_rates = []
        vacancy_rates = []
        labels = []
        
        for i, m in enumerate(months):
            unemp = unemployment_rate[i] if i < len(unemployment_rate) else None
            
            # Calculate vacancy rate: prefer using positions data if available
            vacancy = None
            if i < len(total_job_positions) and i < len(total_matched_jobs):
                total_pos = total_job_positions[i]
                matched = total_matched_jobs[i]
                if total_pos and total_pos > 0:
                    # Vacancy rate = unfilled positions / total positions
                    vacancy = (total_pos - matched) / total_pos
            
            # Fallback: use job_fill_rate
            if vacancy is None and i < len(job_fill_rate) and job_fill_rate[i] is not None:
                vacancy = 1 - job_fill_rate[i]
            
            if unemp is not None and vacancy is not None:
                unemployment_rates.append(unemp * 100)
                vacancy_rates.append(vacancy * 100)
                labels.append(f"M{m}")
        
        if unemployment_rates and vacancy_rates:
            # Scatter plot with time progression
            scatter = ax.scatter(unemployment_rates, vacancy_rates, 
                               c=range(len(unemployment_rates)), 
                               cmap='plasma', s=120, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            # Connect points with line to show time path
            ax.plot(unemployment_rates, vacancy_rates, 'k-', alpha=0.3, linewidth=1)
            
            # Annotate months
            for i, (x, y, label) in enumerate(zip(unemployment_rates, vacancy_rates, labels)):
                ax.annotate(label, (x, y), textcoords="offset points", 
                           xytext=(5, 5), fontsize=9, alpha=0.8)
            
            # Add trend line if enough data points
            if len(unemployment_rates) > 2:
                # Filter out any extreme values
                valid_pairs = [(u, v) for u, v in zip(unemployment_rates, vacancy_rates) 
                              if 0 <= u <= 100 and 0 <= v <= 100]
                if len(valid_pairs) > 2:
                    valid_u, valid_v = zip(*valid_pairs)
                    z = np.polyfit(valid_u, valid_v, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(valid_u), max(valid_u), 100)
                    ax.plot(x_line, p(x_line), 'r--', 
                           label=f'Trend Line (slope={z[0]:.3f})', alpha=0.7, linewidth=2)
            
            # Add reference diagonal (45-degree line)
            max_val = max(max(unemployment_rates), max(vacancy_rates))
            min_val = min(min(unemployment_rates), min(vacancy_rates))
            ax.plot([min_val, max_val], [min_val, max_val], 'g:', 
                   alpha=0.5, label='45° Reference Line')
            
            ax.set_xlabel('Unemployment Rate (%)', fontsize=12)
            ax.set_ylabel('Job Vacancy Rate (%)', fontsize=12)
            ax.set_title('Beveridge Curve\n(Unemployment vs. Job Vacancy)', fontsize=14)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar for time progression
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Time Progression', fontsize=10)
            
            # Add explanatory text
            ax.text(0.02, 0.98, 
                   'Note: Points moving down-left indicate\nimproving labor market efficiency',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'Insufficient data for Beveridge Curve\n\n'
                   'Required: unemployment_rate, job_fill_rate\n'
                   'or total_job_positions/total_matched_jobs',
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('Beveridge Curve (No Data Available)', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            save_path = self.data_dir / "beveridge_curve.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        print(f"Chart saved to {save_path}")
        plt.close()
        return fig
    
    def plot_okun_law(self, save_path: Optional[str] = None):
        """Plot Okun's Law (Unemployment change vs. GDP growth relationship)
        
        Okun's Law states that for every 1% increase in unemployment above 
        the natural rate, GDP falls by approximately 2-3% below potential.
        
        We plot: GDP Growth Rate (Y-axis) vs. Change in Unemployment Rate (X-axis)
        A negative slope indicates the Okun's Law relationship.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        months, _ = self._get_series("macro.nominal_gdp")
        _, unemployment_rate = self._get_series("labor_market.unemployment_rate")
        _, real_gdp_growth = self._get_series("macro.real_gdp_growth_rate")
        _, nominal_gdp_growth = self._get_series("macro.gdp_growth_rate")
        
        # Use real GDP growth if available, otherwise use nominal
        gdp_growth = real_gdp_growth if any(g is not None for g in real_gdp_growth) else nominal_gdp_growth
        
        # Calculate unemployment change (ΔU)
        delta_unemployment = []
        gdp_growth_values = []
        labels = []
        
        for i in range(1, len(months)):  # Start from 1 to calculate change
            if i < len(unemployment_rate) and i-1 < len(unemployment_rate):
                prev_unemp = unemployment_rate[i-1]
                curr_unemp = unemployment_rate[i]
                growth = gdp_growth[i] if i < len(gdp_growth) else None
                
                if prev_unemp is not None and curr_unemp is not None and growth is not None:
                    # Convert to percentage points
                    delta_u = (curr_unemp - prev_unemp) * 100
                    delta_unemployment.append(delta_u)
                    gdp_growth_values.append(growth * 100 if abs(growth) < 1 else growth)
                    labels.append(f"M{months[i]}")
        
        if delta_unemployment and gdp_growth_values:
            # Scatter plot with time progression
            scatter = ax.scatter(delta_unemployment, gdp_growth_values, 
                               c=range(len(delta_unemployment)), 
                               cmap='viridis', s=120, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            # Connect points with line to show time path
            ax.plot(delta_unemployment, gdp_growth_values, 'k-', alpha=0.3, linewidth=1)
            
            # Annotate months
            for i, (x, y, label) in enumerate(zip(delta_unemployment, gdp_growth_values, labels)):
                ax.annotate(label, (x, y), textcoords="offset points", 
                           xytext=(5, 5), fontsize=9, alpha=0.8)
            
            # Add regression line (Okun's Law fit)
            if len(delta_unemployment) > 2:
                z = np.polyfit(delta_unemployment, gdp_growth_values, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(delta_unemployment), max(delta_unemployment), 100)
                ax.plot(x_line, p(x_line), 'r--', 
                       label=f'Okun Fit: g = {z[1]:.2f} + ({z[0]:.2f})×ΔU', alpha=0.7, linewidth=2)
                
                # Calculate Okun coefficient (typical is around -2)
                okun_coefficient = abs(z[0]) if z[0] != 0 else 0
                ax.text(0.02, 0.02, 
                       f"Okun Coefficient: {z[0]:.2f}\n"
                       f"(Classical value: ≈-2)",
                       transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            # Add reference lines
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_xlabel('Change in Unemployment Rate (percentage points)', fontsize=12)
            ax.set_ylabel('Real GDP Growth Rate (%)', fontsize=12)
            ax.set_title("Okun's Law\n(GDP Growth vs. Unemployment Change)", fontsize=14)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar for time progression
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Time Progression', fontsize=10)
            
            # Add explanatory text
            ax.text(0.98, 0.98, 
                   "Okun's Law: When unemployment rises,\nGDP growth tends to fall (negative slope)",
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, "Insufficient data for Okun's Law\n\n"
                   'Required: unemployment_rate (consecutive months)\n'
                   'and real_gdp_growth_rate or gdp_growth_rate',
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title("Okun's Law (No Data Available)", fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            save_path = self.data_dir / "okun_law.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        print(f"Chart saved to {save_path}")
        plt.close()
        return fig
    
    def plot_government(self, save_path: Optional[str] = None):
        """Plot government fiscal indicators"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        months, _ = self._get_series("macro.nominal_gdp")
        
        # 1. Tax Composition
        ax1 = axes[0]
        _, income_tax = self._get_series("government.personal_income_tax")
        _, consume_tax = self._get_series("government.consume_tax")
        _, corporate_tax = self._get_series("government.corporate_tax")
        
        if months and income_tax:
            ax1.stackplot(months, income_tax, consume_tax, corporate_tax,
                         labels=['Personal Income Tax', 'Consumption Tax', 'Corporate Tax'],
                         alpha=0.7)
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Tax Revenue ($)')
            ax1.set_title('Tax Composition')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
        
        # 2. Redistribution
        ax2 = axes[1]
        _, redistribution = self._get_series("government.redistribution_total")
        _, total_tax = self._get_series("government.total_tax")
        
        if months and redistribution:
            ax2.plot(months, total_tax, 'b-o', label='Total Tax', markersize=4)
            ax2.plot(months, redistribution, 'g-s', label='Redistribution', markersize=4)
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Amount ($)')
            ax2.set_title('Government Revenue & Expenditure')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            save_path = self.data_dir / "government_finance.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        print(f"Chart saved to {save_path}")
        plt.close()
        return fig
    
    def plot_all(self, output_dir: Optional[str] = None):
        """Generate all charts"""
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = self.data_dir
        
        print("=" * 50)
        print("Generating all charts...")
        print("=" * 50)
        
        self.plot_gdp(output_path / "01_gdp_analysis.png")
        self.plot_inflation(output_path / "02_inflation_analysis.png")
        self.plot_phillips_curve(output_path / "03_phillips_curve.png")
        self.plot_beveridge_curve(output_path / "04_beveridge_curve.png")
        self.plot_okun_law(output_path / "05_okun_law.png")
        self.plot_gini_coefficient(output_path / "06_gini_coefficient.png")
        self.plot_labor_market(output_path / "07_labor_market.png")
        self.plot_government(output_path / "08_government_finance.png")
        
        print("=" * 50)
        print(f"All charts saved to {output_path}")
        print("=" * 50)
    
    def get_summary_df(self):
        """Get summary data as DataFrame (requires pandas)"""
        try:
            import pandas as pd
        except ImportError:
            print("pandas required: pip install pandas")
            return None
        
        records = []
        for d in self.data:
            record = {
                "month": d.get("econ_month"),
                "preheat": d.get("preheat"),
                "nominal_gdp": d.get("macro", {}).get("nominal_gdp"),
                "real_gdp": d.get("macro", {}).get("real_gdp"),
                "gdp_growth": d.get("macro", {}).get("gdp_growth_rate"),
                "inflation_rate": d.get("macro", {}).get("inflation_rate"),
                "employment_rate": d.get("labor_market", {}).get("employment_rate"),
                "average_wage": d.get("labor_market", {}).get("average_wage"),
                "gini": d.get("household", {}).get("aggregate", {}).get("assets_distribution", {}).get("gini"),
                "total_tax": d.get("government", {}).get("total_tax"),
                "consumption_rate": d.get("macro", {}).get("consumption_rate"),
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        return df


# ==================== Usage Example ====================

if __name__ == "__main__":
    import sys
    
    # Use the latest run record by default
    records_dir = Path("output/monthly_records")
    if records_dir.exists():
        runs = sorted(records_dir.iterdir(), reverse=True)
        if runs:
            latest_run = runs[0]
            print(f"Using latest run record: {latest_run}")
            
            plotter = EconomyPlotter(str(latest_run))
            
            if len(plotter.data) > 1:
                plotter.plot_all()
            else:
                print("Insufficient data (only 1 month), cannot generate meaningful time series charts")
                print("Please run more months of simulation and try again")
        else:
            print("No run records found")
    else:
        print("Directory output/monthly_records not found")

