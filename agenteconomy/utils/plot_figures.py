"""
Economic Simulation Data Visualization Tool

Usage:
    from agenteconomy.utils.plot_figures import EconomyPlotter
    plotter = EconomyPlotter("output/monthly_records/run_XXXXXXXX_XXXXXX")
    plotter.plot_all()
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


class EconomyPlotter:
    """Economic Simulation Data Visualizer"""

    def __init__(self, data_dir: str, exclude_preheat: bool = True,
                 warmup_months: int = 2):
        """
        Args:
            data_dir: Directory containing monthly JSON files
            exclude_preheat: Filter out preheat months
            warmup_months: Additional formal months to skip for stylized facts
                           (economy needs time to stabilize after preheat)
        """
        self.data_dir = Path(data_dir)
        self.exclude_preheat = exclude_preheat
        self.warmup_months = warmup_months
        self.data: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self):
        json_files = sorted(self.data_dir.glob("*.json"))
        all_data = []
        for f in json_files:
            with open(f, 'r', encoding='utf-8') as fp:
                all_data.append(json.load(fp))
        all_data.sort(key=lambda x: x.get("econ_month", 0))
        if self.exclude_preheat:
            self.data = [d for d in all_data if not d.get("preheat", False)]
        else:
            self.data = all_data
        print(f"Loaded {len(all_data)} months, using {len(self.data)} "
              f"({'excl' if self.exclude_preheat else 'incl'} preheat)")

    def _get_series(self, key_path: str) -> Tuple[List[int], List[float]]:
        months, values = [], []
        keys = key_path.split(".")
        for record in self.data:
            month = record.get("econ_month", 0)
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

    @staticmethod
    def _detrend(arr):
        """线性去趋势：去掉单调收敛的趋势，保留周期波动"""
        a = np.array(arr, dtype=float)
        t = np.arange(len(a))
        trend = np.polyval(np.polyfit(t, a, 1), t)
        return a - trend

    def _get_stable_data(self) -> List[Dict[str, Any]]:
        """Get data for stylized fact curves.
        
        Uses the latter half of data to avoid the convergence period.
        The economy typically needs many months to converge from initial state,
        and stylized facts require near-steady-state fluctuations.
        """
        if len(self.data) <= 4:
            return self.data
        # Use latter half of data (skip convergence period)
        half = max(self.warmup_months, len(self.data) // 2)
        return self.data[half:]

    def _save(self, fig, save_path, default_name):
        save_path = save_path or (self.data_dir / default_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
        plt.close()
        return fig

    def plot_gdp(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        months, nominal_gdp = self._get_series("macro.nominal_gdp")
        _, real_gdp = self._get_series("macro.real_gdp")

        # 1. GDP Levels
        ax1 = axes[0, 0]
        if months:
            ax1.plot(months, nominal_gdp, 'b-o', label='Nominal GDP', markersize=4)
            ax1.plot(months, real_gdp, 'r-s', label='Real GDP', markersize=4)
            ax1.set_xlabel('Month'); ax1.set_ylabel('GDP ($)')
            ax1.set_title('GDP Levels'); ax1.legend(); ax1.grid(True, alpha=0.3)

        # 2. GDP Growth Rate
        ax2 = axes[0, 1]
        _, growth_rate = self._get_series("macro.gdp_growth_rate")
        _, real_growth = self._get_series("macro.real_gdp_growth_rate")
        if months and growth_rate:
            vm = [m for m, g in zip(months, growth_rate) if g is not None]
            vg = [g * 100 for g in growth_rate if g is not None]
            vr = [g * 100 for g in real_growth if g is not None]
            if vm:
                ax2.bar([m - 0.2 for m in vm], vg, width=0.4, label='Nominal', alpha=0.7, color='steelblue')
                if vr:
                    ax2.bar([m + 0.2 for m in vm[:len(vr)]], vr, width=0.4, label='Real', alpha=0.7, color='indianred')
                ax2.axhline(y=0, color='k', linewidth=0.5)
                ax2.set_xlabel('Month'); ax2.set_ylabel('Growth Rate (%)')
                ax2.set_title('GDP Growth Rate'); ax2.legend(); ax2.grid(True, alpha=0.3)

        # 3. GDP Composition (C + G = 100%, stacked area)
        ax3 = axes[1, 0]
        _, cr = self._get_series("macro.consumption_rate")
        _, gr = self._get_series("macro.government_rate")
        if months and cr:
            c_pct = [c * 100 for c in cr]
            g_pct = [g * 100 for g in gr]
            ax3.stackplot(months, c_pct, g_pct, labels=['Consumption (C)', 'Government (G)'],
                         colors=['#4e79a7', '#59a14f'], alpha=0.8)
            ax3.set_xlabel('Month'); ax3.set_ylabel('Share of GDP (%)')
            ax3.set_title('GDP Expenditure Composition (C + G)')
            ax3.set_ylim(0, 110)
            ax3.legend(loc='lower right'); ax3.grid(True, alpha=0.3)

        # 4. Labor Share
        ax4 = axes[1, 1]
        _, ls = self._get_series("macro.labor_share")
        if months and ls:
            ax4.plot(months, [l * 100 for l in ls], 'g-o', markersize=4, linewidth=2)
            ax4.axhspan(50, 70, alpha=0.1, color='green', label='Typical range (50-70%)')
            ax4.set_xlabel('Month'); ax4.set_ylabel('Labor Share (%)')
            ax4.set_title('Labor Income Share of GDP')
            ax4.set_ylim(0, 105)
            ax4.legend(); ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save(fig, save_path, "gdp_analysis.png")

    def plot_inflation(self, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        months, _ = self._get_series("macro.nominal_gdp")

        ax1 = axes[0]
        pis = []
        for r in self.data:
            pi = r.get("macro", {}).get("price_index", {})
            pis.append(pi.get("index") if isinstance(pi, dict) else None)
        vd = [(m, p) for m, p in zip(months, pis) if p is not None]
        if vd:
            vm, vp = zip(*vd)
            ax1.plot(vm, vp, 'b-o', markersize=4, linewidth=2)
            ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Base (1.0)')
            ax1.set_xlabel('Month'); ax1.set_ylabel('Price Index')
            ax1.set_title('Consumer Price Index (CPI)'); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        _, infl = self._get_series("macro.inflation_rate")
        vi = [(m, i * 100) for m, i in zip(months, infl) if i is not None]
        if vi:
            vm, vv = zip(*vi)
            colors = ['#59a14f' if v >= 0 else '#e15759' for v in vv]
            ax2.bar(vm, vv, color=colors, alpha=0.8)
            ax2.axhline(y=0, color='k', linewidth=0.5)
            ax2.axhline(y=2, color='orange', linestyle='--', linewidth=1, label='2% Target')
            ax2.set_xlabel('Month'); ax2.set_ylabel('Inflation Rate (%)')
            ax2.set_title('Monthly Inflation Rate'); ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save(fig, save_path, "inflation_analysis.png")

    def plot_phillips_curve(self, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        stable = self._get_stable_data()
        ur, ir, lb = [], [], []
        for d in stable:
            m = d.get("econ_month", 0)
            emp = d.get("labor_market", {}).get("employment_rate")
            inf = d.get("macro", {}).get("inflation_rate")
            if emp is not None and inf is not None:
                u = (1 - emp) * 100
                if u > 0.1:
                    ur.append(u); ir.append(inf * 100); lb.append(f"M{m}")
        if len(ur) >= 3:
            ua, ia = np.array(ur), np.array(ir)
            # Left: detrended (main)
            ax = axes[0]
            u_dt = self._detrend(ua)
            i_dt = self._detrend(ia)
            sc = ax.scatter(u_dt, i_dt, c=range(len(u_dt)), cmap='viridis', s=120, alpha=0.8,
                           edgecolors='black', linewidth=0.5, zorder=5)
            for i in range(len(u_dt) - 1):
                ax.annotate('', xy=(u_dt[i+1], i_dt[i+1]), xytext=(u_dt[i], i_dt[i]),
                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=1.5))
            for x, y, l in zip(u_dt, i_dt, lb):
                ax.annotate(l, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, alpha=0.7)
            try:
                z = np.polyfit(u_dt, i_dt, 1)
                xl = np.linspace(min(u_dt), max(u_dt), 100)
                corr = np.corrcoef(u_dt, i_dt)[0, 1]
                ax.plot(xl, np.poly1d(z)(xl), 'r--',
                       label=f'slope={z[0]:.2f}, corr={corr:.2f}', alpha=0.7, linewidth=2)
            except Exception:
                pass
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
            ax.set_xlabel('Unemployment (detrended, pp)', fontsize=12)
            ax.set_ylabel('Inflation (detrended, pp)', fontsize=12)
            ax.set_title('Phillips Curve (Detrended)', fontsize=14)
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.colorbar(sc, ax=ax, label='Time')
            ax.text(0.02, 0.98, 'Theory: negative slope',
                   transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            # Right: raw levels (reference)
            ax2 = axes[1]
            sc2 = ax2.scatter(ur, ir, c=range(len(ur)), cmap='viridis', s=120, alpha=0.8,
                             edgecolors='black', linewidth=0.5, zorder=5)
            for i in range(len(ur) - 1):
                ax2.annotate('', xy=(ur[i+1], ir[i+1]), xytext=(ur[i], ir[i]),
                            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=1.5))
            try:
                z2 = np.polyfit(ua, ia, 1)
                xl2 = np.linspace(min(ur), max(ur), 100)
                ax2.plot(xl2, np.poly1d(z2)(xl2), 'r--',
                        label=f'slope={z2[0]:.2f}', alpha=0.7, linewidth=2)
            except Exception:
                pass
            ax2.set_xlabel('Unemployment Rate (%)', fontsize=12)
            ax2.set_ylabel('Inflation Rate (%)', fontsize=12)
            ax2.set_title('Phillips Curve (Raw)', fontsize=14)
            ax2.legend(); ax2.grid(True, alpha=0.3)
            plt.colorbar(sc2, ax=ax2, label='Time')
        else:
            axes[0].text(0.5, 0.5, 'Insufficient data', transform=axes[0].transAxes, ha='center', va='center')
        plt.tight_layout()
        return self._save(fig, save_path, "phillips_curve.png")

    def plot_beveridge_curve(self, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        stable = self._get_stable_data()
        ur, vr, lb = [], [], []
        for d in stable:
            m = d.get("econ_month", 0)
            u = d.get("labor_market", {}).get("unemployment_rate")
            tp = d.get("labor_market", {}).get("total_job_positions")
            tm = d.get("labor_market", {}).get("total_matched_jobs")
            fr = d.get("labor_market", {}).get("job_fill_rate")
            v = None
            if tp and tp > 0 and tm is not None:
                v = (tp - tm) / tp
            elif fr is not None:
                v = 1 - fr
            if u is not None and v is not None:
                ur.append(u * 100); vr.append(v * 100); lb.append(f"M{m}")
        if len(ur) >= 3:
            ua, va = np.array(ur), np.array(vr)
            # Left: detrended (main)
            ax = axes[0]
            u_dt = self._detrend(ua)
            v_dt = self._detrend(va)
            sc = ax.scatter(u_dt, v_dt, c=range(len(u_dt)), cmap='plasma', s=120, alpha=0.8,
                           edgecolors='black', linewidth=0.5, zorder=5)
            for i in range(len(u_dt) - 1):
                ax.annotate('', xy=(u_dt[i+1], v_dt[i+1]), xytext=(u_dt[i], v_dt[i]),
                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=1.5))
            for x, y, l in zip(u_dt, v_dt, lb):
                ax.annotate(l, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8, alpha=0.7)
            try:
                z = np.polyfit(u_dt, v_dt, 1)
                xl = np.linspace(min(u_dt), max(u_dt), 100)
                corr = np.corrcoef(u_dt, v_dt)[0, 1]
                ax.plot(xl, np.poly1d(z)(xl), 'r--',
                       label=f'slope={z[0]:.2f}, corr={corr:.2f}', alpha=0.7, linewidth=2)
            except Exception:
                pass
            ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
            ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
            ax.set_xlabel('Unemployment (detrended, pp)', fontsize=12)
            ax.set_ylabel('Vacancy Rate (detrended, pp)', fontsize=12)
            ax.set_title('Beveridge Curve (Detrended)', fontsize=14)
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.colorbar(sc, ax=ax, label='Time')
            ax.text(0.02, 0.98, 'Theory: negative slope',
                   transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            # Right: raw levels (reference)
            ax2 = axes[1]
            sc2 = ax2.scatter(ur, vr, c=range(len(ur)), cmap='plasma', s=120, alpha=0.8,
                             edgecolors='black', linewidth=0.5, zorder=5)
            for i in range(len(ur) - 1):
                ax2.annotate('', xy=(ur[i+1], vr[i+1]), xytext=(ur[i], vr[i]),
                            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=1.5))
            try:
                z2 = np.polyfit(ua, va, 1)
                xl2 = np.linspace(min(ur), max(ur), 100)
                ax2.plot(xl2, np.poly1d(z2)(xl2), 'r--',
                        label=f'slope={z2[0]:.2f}', alpha=0.7, linewidth=2)
            except Exception:
                pass
            mx = max(max(ur), max(vr)); mn = min(min(ur), min(vr))
            ax2.plot([mn, mx], [mn, mx], 'g:', alpha=0.5, label='45 deg')
            ax2.set_xlabel('Unemployment Rate (%)', fontsize=12)
            ax2.set_ylabel('Vacancy Rate (%)', fontsize=12)
            ax2.set_title('Beveridge Curve (Raw)', fontsize=14)
            ax2.legend(loc='upper right'); ax2.grid(True, alpha=0.3)
            plt.colorbar(sc2, ax=ax2, label='Time')
        else:
            axes[0].text(0.5, 0.5, 'Insufficient data', transform=axes[0].transAxes, ha='center', va='center')
        plt.tight_layout()
        return self._save(fig, save_path, "beveridge_curve.png")

    def plot_okun_law(self, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        stable = self._get_stable_data()
        du, gv, lb = [], [], []
        for i in range(1, len(stable)):
            p, c = stable[i - 1], stable[i]
            pu = p.get("labor_market", {}).get("unemployment_rate")
            cu = c.get("labor_market", {}).get("unemployment_rate")
            g = c.get("macro", {}).get("real_gdp_growth_rate") or c.get("macro", {}).get("gdp_growth_rate")
            if g is None:
                g = c.get("macro", {}).get("gdp_comprehensive", {}).get("growth_rates", {}).get("real_gdp_growth")
            if pu is not None and cu is not None and g is not None:
                du.append((cu - pu) * 100)
                gv.append(g * 100 if abs(g) < 1 else g)
                lb.append(f"M{c.get('econ_month', '?')}")
        if len(du) >= 3:
            sc = ax.scatter(du, gv, c=range(len(du)), cmap='viridis', s=120, alpha=0.8,
                           edgecolors='black', linewidth=0.5, zorder=5)
            for i in range(len(du) - 1):
                ax.annotate('', xy=(du[i+1], gv[i+1]), xytext=(du[i], gv[i]),
                           arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=1.5))
            for x, y, l in zip(du, gv, lb):
                ax.annotate(l, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9, alpha=0.8)
            z = np.polyfit(du, gv, 1)
            xl = np.linspace(min(du), max(du), 100)
            ax.plot(xl, np.poly1d(z)(xl), 'r--',
                   label=f'Okun: g={z[1]:.1f}+({z[0]:.1f})$\\times\\Delta$U', alpha=0.7, linewidth=2)
            ax.text(0.02, 0.02, f"Okun Coefficient: {z[0]:.2f}\n(Classical: ~-2)",
                   transform=ax.transAxes, fontsize=10, va='bottom',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel('Change in Unemployment (pp)', fontsize=12)
            ax.set_ylabel('Real GDP Growth (%)', fontsize=12)
            ax.set_title("Okun's Law", fontsize=14)
            ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
            plt.colorbar(sc, ax=ax, label='Time')
        else:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha='center', va='center')
        plt.tight_layout()
        return self._save(fig, save_path, "okun_law.png")

    def plot_gini_coefficient(self, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        months, gini = self._get_series("household.aggregate.assets_distribution.gini")
        _, ig = self._get_series("household.aggregate.assets_distribution.income_gini")
        if months and gini:
            ax.plot(months, gini, 'purple', marker='o', markersize=6, linewidth=2, label='Wealth Gini')
            if ig and len(ig) == len(months):
                ax.plot(months, ig, 'blue', marker='s', markersize=6, linewidth=2, label='Income Gini')
            ax.axhline(y=0.4, color='orange', linestyle='--', label='Warning (0.4)', alpha=0.6)
            ax.set_xlabel('Month'); ax.set_ylabel('Gini'); ax.set_title('Inequality (Gini Coefficient)')
            ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return self._save(fig, save_path, "gini_coefficient.png")

    def plot_labor_market(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        months, _ = self._get_series("macro.nominal_gdp")

        # Employment rate
        _, er = self._get_series("labor_market.employment_rate")
        if months and er:
            axes[0, 0].plot(months, [e * 100 for e in er], 'g-o', markersize=5, linewidth=2)
            axes[0, 0].axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='Full employment (~95%)')
            axes[0, 0].set_xlabel('Month'); axes[0, 0].set_ylabel('Employment Rate (%)')
            axes[0, 0].set_title('Employment Rate'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 105)

        # Average wage (with smoothed trend)
        _, aw = self._get_series("labor_market.average_wage")
        if months and aw:
            axes[0, 1].plot(months, aw, 'b-o', markersize=4, alpha=0.6, label='Monthly')
            if len(aw) >= 3:
                # Simple moving average for trend
                window = min(3, len(aw))
                trend = np.convolve(aw, np.ones(window)/window, mode='valid')
                trend_months = months[window-1:]
                axes[0, 1].plot(trend_months, trend, 'r-', linewidth=2, label=f'{window}-month MA')
            axes[0, 1].set_xlabel('Month'); axes[0, 1].set_ylabel('Average Wage ($)')
            axes[0, 1].set_title('Monthly Average Wage'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        # Total wages
        _, tw = self._get_series("labor_market.total_wage_gross")
        _, tn = self._get_series("labor_market.total_wage_net")
        if months and tw:
            axes[1, 0].plot(months, tw, 'b-o', label='Gross', markersize=4)
            if tn:
                axes[1, 0].plot(months, tn, 'g-s', label='Net', markersize=4)
            axes[1, 0].set_xlabel('Month'); axes[1, 0].set_ylabel('Total Wages ($)')
            axes[1, 0].set_title('Total Wage Bill'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        # Labor supply vs demand
        _, tl = self._get_series("labor_market.total_labor")
        _, el = self._get_series("labor_market.employed_labor")
        if months and tl:
            axes[1, 1].fill_between(months, tl, alpha=0.3, color='blue', label='Total Labor')
            axes[1, 1].fill_between(months, el, alpha=0.3, color='green', label='Employed')
            axes[1, 1].plot(months, tl, 'b-', linewidth=1.5)
            axes[1, 1].plot(months, el, 'g-', linewidth=1.5)
            axes[1, 1].set_xlabel('Month'); axes[1, 1].set_ylabel('Labor (person-hours)')
            axes[1, 1].set_title('Labor Supply & Demand'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save(fig, save_path, "labor_market.png")

    def plot_government(self, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        months, _ = self._get_series("macro.nominal_gdp")

        _, it = self._get_series("government.personal_income_tax")
        _, ct = self._get_series("government.consume_tax")
        _, cot = self._get_series("government.corporate_tax")
        if months and it:
            axes[0].stackplot(months, it, ct, cot,
                             labels=['Income Tax', 'VAT', 'Corporate Tax'], alpha=0.7,
                             colors=['#4e79a7', '#f28e2b', '#e15759'])
            axes[0].set_xlabel('Month'); axes[0].set_ylabel('Tax Revenue ($)')
            axes[0].set_title('Tax Composition'); axes[0].legend(loc='upper left'); axes[0].grid(True, alpha=0.3)

        _, rd = self._get_series("government.redistribution_total")
        _, tt = self._get_series("government.total_tax")
        if months and tt:
            axes[1].plot(months, tt, 'b-o', label='Total Tax', markersize=4, linewidth=2)
            if rd:
                axes[1].plot(months, rd, 'g-s', label='Redistribution', markersize=4, linewidth=2)
            axes[1].set_xlabel('Month'); axes[1].set_ylabel('Amount ($)')
            axes[1].set_title('Government Revenue & Expenditure'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return self._save(fig, save_path, "government_finance.png")

    def plot_all(self, output_dir=None):
        op = Path(output_dir) if output_dir else self.data_dir
        op.mkdir(parents=True, exist_ok=True)
        print("=" * 50 + "\nGenerating all charts...\n" + "=" * 50)
        self.plot_gdp(op / "01_gdp_analysis.png")
        self.plot_inflation(op / "02_inflation_analysis.png")
        self.plot_phillips_curve(op / "03_phillips_curve.png")
        self.plot_beveridge_curve(op / "04_beveridge_curve.png")
        self.plot_okun_law(op / "05_okun_law.png")
        self.plot_gini_coefficient(op / "06_gini_coefficient.png")
        self.plot_labor_market(op / "07_labor_market.png")
        self.plot_government(op / "08_government_finance.png")
        print("=" * 50 + f"\nAll charts saved to {op}\n" + "=" * 50)


if __name__ == "__main__":
    records_dir = Path("output/monthly_records")
    if records_dir.exists():
        runs = sorted(records_dir.iterdir(), reverse=True)
        if runs:
            print(f"Using: {runs[0]}")
            plotter = EconomyPlotter(str(runs[0]))
            if len(plotter.data) > 1:
                plotter.plot_all()
    else:
        print("No records dir")
