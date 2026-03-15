"""
Simulation Configuration Module

Loads and manages simulation configuration from YAML files.
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

from agenteconomy.utils.logger import get_logger
from agenteconomy.center.Model import TaxBracket

logger = get_logger(name="simulation_config")


@dataclass
class SimulationConfig:
    """
    Simulation configuration class.
    
    Can be initialized with default values or loaded from a YAML file.
    """
    # Simulation parameters
    num_months: int = 12
    preheat_months: int = 0
    num_households: int = 1000
    num_firms: int = 100
    num_banks: int = 10
    num_government: int = 1
    num_laborhours: int = 100
    num_products: int = 100
    num_transactions: int = 100
    num_wages: int = 100
    num_innovations: int = 100
    num_research: int = 100
    num_investments: int = 100
    debug_logging: bool = True
    debug_max_households: int = 0
    debug_max_firms: int = 0
    local_record_dir: Optional[str] = None
    
    # Tax Policy
    enable_progressive_tax_system: bool = True
    income_tax_rate: List[TaxBracket] = field(default_factory=lambda: [
        TaxBracket(cutoff=0.0, rate=0.10),
        TaxBracket(cutoff=10000.0, rate=0.15),
        TaxBracket(cutoff=40000.0, rate=0.22),
        TaxBracket(cutoff=85000.0, rate=0.24),
        TaxBracket(cutoff=160000.0, rate=0.32),
        TaxBracket(cutoff=200000.0, rate=0.35)
    ])
    fallback_income_tax_rate: float = 0.2
    corporate_tax_rate: float = 0.21
    vat_rate: float = 0.08
    fica_tax_rate: float = 0.0765
    
    # Interest rate
    interest_rate: float = 0.005  # Annual rate
    
    # Demand logging configuration
    enable_month1_household_demand_logging_only: bool = False
    demand_logging_unlimited_stock_amount: float = 1e12
    demand_logging_output_dir: str = "output/month1_household_demand_3000sku"
    demand_logging_write_csv: bool = True
    demand_logging_run_employment_setup: bool = True
    demand_logging_pay_wages_before_consumption: bool = True
    
    # Concurrent configuration
    max_concurrent_tasks: int = 100
    max_llm_concurrent: int = 400

    # Checkpoint configuration
    checkpoint_interval: int = 1  # 每隔几个月保存一次 checkpoint（1=每月保存）
    checkpoint_output_dir: str = "output/checkpoints"  # checkpoint 保存目录
    checkpoint_compress: bool = True  # 是否压缩（gzip）

    # Firm initial capital configuration
    firm_initial_capital_multiplier: float = 1.5  # 企业初始资金 = 预期收入 * 此系数
    firm_min_initial_cash: float = 10000.0  # 企业最低初始资金

    # Category profit margins (optional, can be loaded from config)
    category_profit_margins: Optional[Dict[str, float]] = None
    
    # Alias for compatibility with existing code
    @property
    def gov_tax_brackets(self) -> List[TaxBracket]:
        """Alias for income_tax_rate for backward compatibility."""
        return self.income_tax_rate
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SimulationConfig':
        """
        Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            SimulationConfig instance with values loaded from YAML
            
        Example:
            config = SimulationConfig.from_yaml("config/config_normal.yaml")
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not data:
            logger.warning(f"YAML file {yaml_path} is empty, using defaults")
            return cls()
        
        # Extract simulation parameters
        sim_data = data.get('simulation', {})
        
        # Extract tax parameters
        tax_data = data.get('tax', {})
        
        # Extract demand logging parameters
        demand_data = data.get('demand list test', {})
        
        # Extract concurrent config
        concurrent_data = data.get('concurrent config', {})

        # Extract checkpoint config
        checkpoint_data = data.get('checkpoint', {})

        # Extract interest rate
        interest_data = tax_data.get('interest', {})
        
        # Parse income tax brackets
        income_tax_brackets = []
        if 'income_tax_rate' in tax_data:
            for bracket_data in tax_data['income_tax_rate']:
                income_tax_brackets.append(
                    TaxBracket(
                        cutoff=float(bracket_data.get('cutoff', 0)),
                        rate=float(bracket_data.get('rate', 0))
                    )
                )
        
        # Create config instance
        config = cls(
            # Simulation parameters
            num_months=sim_data.get('num_months', 12),
            preheat_months=sim_data.get('preheat_months', sim_data.get('warmup_months', 0)),
            num_households=sim_data.get('num_households', 100),
            num_firms=sim_data.get('num_firms', 100),
            num_banks=sim_data.get('num_banks', 10),
            num_government=sim_data.get('num_government', 1),
            num_laborhours=sim_data.get('num_laborhours', 100),
            num_products=sim_data.get('num_products', 100),
            num_transactions=sim_data.get('num_transactions', 100),
            num_wages=sim_data.get('num_wages', 100),
            num_innovations=sim_data.get('num_innovations', 100),
            num_research=sim_data.get('num_research', 100),
            num_investments=sim_data.get('num_investments', 100),
            debug_logging=bool(sim_data.get('debug_logging', True)),
            debug_max_households=int(sim_data.get('debug_max_households', 0) or 0),
            debug_max_firms=int(sim_data.get('debug_max_firms', 0) or 0),
            local_record_dir=sim_data.get('local_record_dir'),
            
            # Tax parameters
            enable_progressive_tax_system=True,  # Default to True if tax data exists
            income_tax_rate=income_tax_brackets if income_tax_brackets else [
                TaxBracket(cutoff=0.0, rate=0.10),
                TaxBracket(cutoff=10000.0, rate=0.15),
                TaxBracket(cutoff=40000.0, rate=0.22),
                TaxBracket(cutoff=85000.0, rate=0.24),
                TaxBracket(cutoff=160000.0, rate=0.32),
                TaxBracket(cutoff=200000.0, rate=0.35)
            ],
            corporate_tax_rate=float(tax_data.get('corporate_tax_rate', 0.21)),
            vat_rate=float(tax_data.get('vat_rate', 0.08)),
            fica_tax_rate=float(tax_data.get('fica_tax_rate', 0.0765)),
            
            # Interest rate
            interest_rate=float(interest_data.get('rate', 0.005)),
            
            # Demand logging
            enable_month1_household_demand_logging_only=demand_data.get('enable_month1_household_demand_logging_only', False),
            demand_logging_unlimited_stock_amount=float(demand_data.get('demand_logging_unlimited_stock_amount', 1e12)),
            demand_logging_output_dir=demand_data.get('demand_logging_output_dir', 'output/month1_household_demand_3000sku'),
            demand_logging_write_csv=demand_data.get('demand_logging_write_csv', True),
            demand_logging_run_employment_setup=demand_data.get('demand_logging_run_employment_setup', True),
            demand_logging_pay_wages_before_consumption=demand_data.get('demand_logging_pay_wages_before_consumption', True),
            
            # Concurrent configuration
            max_concurrent_tasks=concurrent_data.get('max_concurrent_tasks', 100),
            max_llm_concurrent=concurrent_data.get('max_llm_concurrent', 400),

            # Checkpoint configuration
            checkpoint_interval=int(checkpoint_data.get('checkpoint_interval', 1)),
            checkpoint_output_dir=checkpoint_data.get('checkpoint_output_dir', 'output/checkpoints'),
            checkpoint_compress=bool(checkpoint_data.get('checkpoint_compress', True)),
        )

        logger.info(f"Configuration loaded from {yaml_path}")
        logger.info(f"  - Simulation: {config.num_months} months, {config.num_households} households, {config.num_firms} firms")
        logger.info(f"  - Tax: Corporate {config.corporate_tax_rate:.1%}, VAT {config.vat_rate:.1%}")
        logger.info(f"  - Interest rate: {config.interest_rate:.1%} (annual)")
        logger.info(f"  - Checkpoint: interval={config.checkpoint_interval}, dir={config.checkpoint_output_dir}")

        return config
