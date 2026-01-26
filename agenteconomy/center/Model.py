"""
Economic Model Data Classes

This module defines all core data models for the agent-based economic simulation,
including assets, transactions, tax policies, jobs, and innovation tracking.
"""

from datetime import date, datetime
import time
from typing import Any, Callable, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Tax-Related Models
# =============================================================================

class TaxBracket(BaseModel):
    """
    A single tax bracket in a progressive tax system.
    
    Attributes:
        cutoff: Income level where this tax rate begins to apply
        rate: Tax rate (0-1) applied to income in this bracket
    """
    cutoff: float = Field(..., description="The cutoff point of the tax bracket")
    rate: float = Field(..., ge=0.0, le=1.0, description="The rate of the tax bracket")


class TaxPolicy(BaseModel):
    """
    Comprehensive tax policy with three core tax types.
    
    Attributes:
        income_tax_rate: Progressive income tax brackets for individuals 
        corporate_tax_rate: Corporate income tax rate (0-1) 
        vat_rate: Value-added tax rate (0-1)
    """
    income_tax_rate: List[TaxBracket] = Field(
        default_factory=lambda: [
            TaxBracket(cutoff=0.0, rate=0.10),
            TaxBracket(cutoff=10000.0, rate=0.15),
            TaxBracket(cutoff=40000.0, rate=0.22),
            TaxBracket(cutoff=85000.0, rate=0.24),
            TaxBracket(cutoff=160000.0, rate=0.32),
            TaxBracket(cutoff=200000.0, rate=0.35)
        ],
        description="Progressive income tax brackets for individuals"
    )
    corporate_tax_rate: float = Field(
        default=0.21,
        ge=0.0,
        le=1.0,
        description="Corporate tax rate (企业所得税)"
    )
    vat_rate: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Value-added tax rate (消费税)"
    )


# =============================================================================
# Asset-Related Models
# =============================================================================

class Asset(BaseModel):
    """
    Base class for all economic assets.
    
    Represents any tradable or holdable economic entity including money,
    goods, labor hours, and securities.
    """
    name: Optional[str] = Field(None, description="Name of the asset")
    asset_type: Literal['money', 'goods', 'labor_hour', 'security'] = Field(
        ...,
        description="Type of the asset"
    )
    classification: Optional[str] = Field(None, description="Classification of the asset")
    expiration_date: Optional[date] = Field(None, description="Expiration date of the asset")
    manufacturer: Optional[str] = Field(
        None,
        description="Manufacturer of the asset; could be None for labor hour and money"
    )
    price: Optional[float] = Field(
        None,
        gt=0,
        description="Price of the asset; could be None for labor hour and money"
    )
    amount: float = Field(..., ge=0, description="Amount of the asset")
    description: Optional[str] = Field(None, description="Description of the asset")


class Ledger(Asset):
    """
    Represents a monetary ledger/account.
    
    Tracks cash balances for agents. Allows negative amounts for firms
    to represent debt-based operations.
    """
    asset_type: Literal['money'] = Field(default='money', description="Type of the asset")
    amount: float = Field(..., description="Amount of money (can be negative for firms)")
    
    @classmethod
    def create(cls, agent_id: str, amount: float = 0.0) -> 'Ledger':
        """
        Create a new ledger for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            amount: Initial cash amount (can be negative for firms)
            
        Returns:
            New Ledger instance
        """
        return cls(agent_id=agent_id, asset_type='money', amount=amount)

class SavingsAccount(BaseModel):
    """
    # 储蓄账户
    代表家庭在银行的储蓄账户信息
    
    Attributes:
        account_id: 账户ID
        household_id: 家庭ID
        balance: 账户余额
        annual_interest_rate: 年利率
        last_interest_date: 上次计息日期
        created_month: 账户创建月份
    """
    account_id: str
    household_id: str
    balance: float = Field(default=0.0, ge=0.0)
    annual_interest_rate: float = Field(default=0.005, ge=0.0, le=1.0)  # 0.5% 年利率
    last_interest_date: Optional[int] = Field(default=None, description="上次计息的月份")
    created_month: int = Field(default=1, description="账户创建月份")

class LaborHour(BaseModel):
    """
    Represents labor hours offered by a household member.
    
    Tracks skills, abilities, availability, and employment status
    for head or spouse in a household.
    """
    agent_id: str = Field(..., description="ID of the agent providing labor")
    skill_profile: Optional[Dict[str, float]] = Field(None, description="Skill levels by skill type")
    ability_profile: Optional[Dict[str, float]] = Field(None, description="Ability levels by ability type")
    asset_type: Literal['labor_hour'] = Field(default='labor_hour', description="Type of the asset")
    total_hours: float = Field(..., gt=0, description="Total hours available")
    start_date: Optional[date] = Field(None, description="Start date")
    end_date: Optional[date] = Field(None, description="End date")
    lh_type: Literal['head', 'spouse'] = Field(default='head', description="Type of the labor hour")
    is_recurring: bool = Field(default=False, description="Whether the labor is recurring")
    cycle: Optional[Literal['daily', 'weekly', 'monthly']] = Field(
        None,
        description="If recurring, specify the cycle frequency"
    )
    template: str = Field(..., description="Template identifier for this labor type")
    daily_hours: Optional[float] = Field(None, description="Daily work hours (calculated if possible)")
    is_valid: bool = Field(default=True, description="Whether the labor hour is valid")
    job_title: Optional[str] = Field(None, description="The title of the job")
    job_SOC: Optional[str] = Field(None, description="The SOC code of the job")
    firm_id: Optional[str] = Field(None, description="The firm ID that posted the job")
    
    @model_validator(mode='after')
    def compute_daily_hours(self) -> 'LaborHour':
        """Calculate daily hours based on total hours and date range."""
        if self.is_recurring:
            self.daily_hours = None
        elif self.start_date and self.end_date:
            days = (self.end_date - self.start_date).days + 1
            if days <= 0:
                raise ValueError("End date must be after start date")
            self.daily_hours = self.total_hours / days
        else:
            self.daily_hours = None
        return self

    @classmethod
    def create(
        cls,
        agent_id: str,
        total_hours: float,
        template: str,
        skill_profile: Optional[Dict[str, float]] = None,
        ability_profile: Optional[Dict[str, float]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        is_recurring: bool = False,
        cycle: Optional[Literal['daily', 'weekly', 'monthly']] = None,
        lh_type: Literal['head', 'spouse'] = 'head'
    ) -> 'LaborHour':
        """
        Create a new labor hour offering.
        
        Args:
            agent_id: ID of the agent providing labor
            total_hours: Total hours available
            template: Template identifier
            skill_profile: Skills by type
            ability_profile: Abilities by type
            start_date: When labor availability starts
            end_date: When labor availability ends
            is_recurring: Whether labor repeats on a cycle
            cycle: Frequency of recurring labor
            lh_type: Whether this is for household head or spouse
            
        Returns:
            New LaborHour instance
            
        Raises:
            ValueError: If total_hours <= 0
        """
        if total_hours <= 0:
            raise ValueError("Total hours must be greater than zero")
        return cls(
            agent_id=agent_id,
            skill_profile=skill_profile,
            ability_profile=ability_profile,
            total_hours=total_hours,
            start_date=start_date,
            end_date=end_date,
            is_recurring=is_recurring,
            cycle=cycle,
            template=template,
            lh_type=lh_type
        )


class Product(Asset):
    """
    Represents a product available for purchase.
    
    Includes pricing, ownership, attributes, and nutrition/satisfaction data.
    """
    asset_type: Literal['products'] = Field(default='products', description="Type of the asset")
    product_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique product ID")
    name: str = Field(..., description="Name of the product")
    description: Optional[str] = Field(None, description="Description of the product")
    price: float = Field(..., gt=0, description="Current price of the product")
    base_price: Optional[float] = Field(
        default=None,
        gt=0,
        description="Original price used for unit cost calculation; stable even if price changes"
    )
    unit_cost: Optional[float] = Field(
        default=None,
        ge=0,
        description="Unit cost derived from base_price and industry gross margin"
    )
    owner_id: str = Field(..., description="ID of the owner/seller")
    brand: Optional[str] = Field(None, description="Brand of the product")
    attributes: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw attribute payload for the product"
    )
    is_food: Optional[bool] = Field(default=None, description="Whether the product is food")
    nutrition_supply: Optional[Dict[str, float]] = Field(
        default=None,
        description="Nutrition data when product is food"
    )
    satisfaction_attributes: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Satisfaction attributes for non-food products"
    )
    duration_months: Optional[int] = Field(
        default=None,
        description="Duration (months) product provides satisfaction/nutrition"
    )
    
    @classmethod
    def create(
        cls,
        name: str,
        price: float,
        owner_id: str,
        amount: float = 1.0,
        classification: Optional[str] = None,
        expiration_date: Optional[date] = None,
        manufacturer: Optional[str] = None,
        description: Optional[str] = None,
        brand: Optional[str] = None,
        product_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        is_food: Optional[bool] = None,
        nutrition_supply: Optional[Dict[str, float]] = None,
        satisfaction_attributes: Optional[Dict[str, Any]] = None,
        duration_months: Optional[int] = None,
        base_price: Optional[float] = None,
        unit_cost: Optional[float] = None
    ) -> 'Product':
        """
        Create a new product.
        
        Args:
            name: Product name
            price: Current selling price
            owner_id: ID of the owner/seller
            amount: Quantity available
            classification: Product classification
            expiration_date: When product expires
            manufacturer: Who manufactures the product
            description: Product description
            brand: Brand name
            product_id: Unique ID (auto-generated if None)
            attributes: Additional attributes
            is_food: Whether this is a food product
            nutrition_supply: Nutrition data for food
            satisfaction_attributes: Satisfaction data for non-food
            duration_months: How long product provides value
            base_price: Original/base price (defaults to price)
            unit_cost: Manufacturing cost per unit
            
        Returns:
            New Product instance
            
        Raises:
            ValueError: If price or base_price <= 0
        """
        if price <= 0:
            raise ValueError("Price must be greater than zero")
        if base_price is None:
            base_price = price
        elif base_price <= 0:
            raise ValueError("base_price must be greater than zero")
        
        return cls(
            name=name,
            asset_type='products',
            product_id=product_id,
            price=price,
            base_price=base_price,
            unit_cost=unit_cost,
            owner_id=owner_id,
            amount=amount,
            classification=classification,
            expiration_date=expiration_date,
            manufacturer=manufacturer,
            description=description,
            brand=brand,
            attributes=attributes,
            is_food=is_food,
            nutrition_supply=nutrition_supply,
            satisfaction_attributes=satisfaction_attributes,
            duration_months=duration_months
        )


# =============================================================================
# Inventory Management
# =============================================================================

class InventoryReservation(BaseModel):
    """
    Inventory reservation record for concurrent purchase handling.
    
    When a household selects products, inventory is immediately reserved
    to prevent race conditions. Purchase uses the reservation ID for confirmation.
    """
    reservation_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique reservation ID")
    buyer_id: str = Field(..., description="ID of the buyer")
    seller_id: str = Field(..., description="ID of the seller (product owner)")
    product_id: str = Field(..., description="ID of the product")
    product_name: str = Field(..., description="Name of the product")
    quantity: float = Field(..., gt=0, description="Reserved quantity")
    created_at: float = Field(default_factory=time.time, description="Creation timestamp")
    expires_at: float = Field(..., description="Expiration timestamp")
    status: Literal['active', 'confirmed', 'released', 'expired'] = Field(
        default='active',
        description="Reservation status"
    )
    
    @classmethod
    def create(
        cls,
        buyer_id: str,
        seller_id: str,
        product_id: str,
        product_name: str,
        quantity: float,
        timeout_seconds: float = 300
    ) -> 'InventoryReservation':
        """
        Create an inventory reservation.
        
        Args:
            buyer_id: ID of the buyer
            seller_id: ID of the seller
            product_id: ID of the product
            product_name: Name of the product
            quantity: Quantity to reserve
            timeout_seconds: Reservation timeout in seconds (default: 300 = 5 minutes)
            
        Returns:
            New InventoryReservation instance
        """
        now = time.time()
        return cls(
            buyer_id=buyer_id,
            seller_id=seller_id,
            product_id=product_id,
            product_name=product_name,
            quantity=quantity,
            created_at=now,
            expires_at=now + timeout_seconds
        )


# =============================================================================
# Job Market Models
# =============================================================================

class Job(BaseModel):
    """
    Represents a job posting by a firm.
    
    Includes requirements, compensation, and availability information.
    """
    job_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique job identifier")
    SOC: str = Field(..., description="Standard Occupational Classification code")
    title: str = Field(..., description="Job title")
    description: Optional[str] = Field(None, description="Job description")
    wage_per_hour: float = Field(..., description="Wage posted by the firm")
    required_skills: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Required skills for the job"
    )
    required_abilities: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Required abilities for the job"
    )
    firm_id: Optional[str] = Field(None, description="Firm that posted the job")
    is_valid: bool = Field(default=True, description="Whether the job is currently available")
    positions_available: int = Field(default=1, description="Number of positions available")
    hours_per_period: Optional[float] = Field(None, description="Hours per work period")

    @classmethod
    def create(
        cls,
        soc: str,
        title: str,
        wage_per_hour: float,
        firm_id: Optional[str] = None,
        description: Optional[str] = None,
        hours_per_period: Optional[float] = None,
        required_skills: Optional[Dict[str, Dict[str, float]]] = None,
        required_abilities: Optional[Dict[str, Dict[str, float]]] = None,
        job_id: Optional[str] = None
    ) -> 'Job':
        """
        Create a new job posting.
        
        Args:
            soc: Standard Occupational Classification code
            title: Job title
            wage_per_hour: Hourly wage
            firm_id: ID of hiring firm
            description: Job description
            hours_per_period: Expected hours per period
            required_skills: Required skill levels
            required_abilities: Required ability levels
            job_id: Unique ID (auto-generated if None)
            
        Returns:
            New Job instance
        """
        return cls(
            job_id=job_id or str(uuid4()),
            SOC=soc,
            title=title,
            wage_per_hour=wage_per_hour,
            firm_id=firm_id,
            description=description,
            hours_per_period=hours_per_period,
            required_skills=required_skills or {},
            required_abilities=required_abilities or {}
        )


class JobApplication(BaseModel):
    """
    Represents a job application from a household member.
    
    Tracks applicant information, expectations, and qualifications.
    """
    job_id: str = Field(..., description="ID of the job being applied for")
    household_id: str = Field(..., description="ID of the applying household")
    lh_type: Literal['head', 'spouse'] = Field(..., description="Type of labor hour (head or spouse)")
    expected_wage: float = Field(..., description="Expected wage by the job seeker")
    worker_skills: Dict[str, float] = Field(
        default_factory=dict,
        description="Worker's skill profile"
    )
    worker_abilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Worker's ability profile"
    )
    application_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the application was submitted"
    )
    month: int = Field(default=1, description="Month of the application")
    
    @classmethod
    def create(
        cls,
        job_id: str,
        household_id: str,
        lh_type: Literal['head', 'spouse'],
        expected_wage: float,
        worker_skills: Optional[Dict[str, float]] = None,
        worker_abilities: Optional[Dict[str, float]] = None,
        month: int = 1
    ) -> 'JobApplication':
        """
        Create a new job application.
        
        Args:
            job_id: ID of the job being applied for
            household_id: ID of the applying household
            lh_type: Whether head or spouse is applying
            expected_wage: Wage expectations
            worker_skills: Applicant's skills
            worker_abilities: Applicant's abilities
            month: Application month
            
        Returns:
            New JobApplication instance
        """
        return cls(
            job_id=job_id,
            household_id=household_id,
            lh_type=lh_type,
            expected_wage=expected_wage,
            worker_skills=worker_skills or {},
            worker_abilities=worker_abilities or {},
            month=month
        )


class MatchedJob(BaseModel):
    """
    Represents a successful job match between worker and employer.
    
    Contains the job details, agreed wage, and match quality metrics.
    """
    job: Job = Field(..., description="The matched job")
    average_wage: float = Field(..., description="Agreed/average wage")
    household_id: str = Field(..., description="ID of the matched household")
    lh_type: Literal['head', 'spouse'] = Field(..., description="Type of labor hour")
    firm_id: str = Field(..., description="ID of the hiring firm")
    skill_match_score: Optional[float] = Field(None, description="Quality of skill match (0-1)")
    
    @classmethod
    def create(
        cls,
        job: Job,
        average_wage: float,
        household_id: str,
        lh_type: Literal['head', 'spouse'],
        firm_id: str,
        skill_match_score: Optional[float] = None
    ) -> 'MatchedJob':
        """
        Create a matched job record.
        
        Args:
            job: The job being matched
            average_wage: Negotiated wage
            household_id: ID of matched household
            lh_type: Whether head or spouse
            firm_id: ID of hiring firm
            skill_match_score: Match quality score
            
        Returns:
            New MatchedJob instance
        """
        return cls(
            job=job,
            average_wage=average_wage,
            household_id=household_id,
            lh_type=lh_type,
            firm_id=firm_id,
            skill_match_score=skill_match_score
        )


class Wage(BaseModel):
    """
    Records wage payment for a specific period.
    """
    agent_id: str = Field(..., description="ID of the agent receiving wage")
    amount: float = Field(..., description="Wage amount")
    month: int = Field(..., description="Month of payment")
    
    @classmethod
    def create(cls, agent_id: str, amount: float, month: int) -> 'Wage':
        """
        Create a wage record.
        
        Args:
            agent_id: ID of wage recipient
            amount: Wage amount
            month: Payment month
            
        Returns:
            New Wage instance
        """
        return cls(agent_id=agent_id, amount=amount, month=month)


# =============================================================================
# Transaction Models
# =============================================================================

class TransactionStatus:
    """
    Enumeration of possible transaction states.
    
    Tracks the lifecycle of a transaction from creation to completion or failure.
    """
    PENDING: str = "pending"
    COMPLETED: str = "completed"
    FAILED: str = "failed"
    CANCELLED: str = "cancelled"


class Transaction(BaseModel):
    """
    Represents any economic transaction between agents.
    
    Includes monetary transfers, asset exchanges, labor payments, and various tax payments.
    Supports full transaction lifecycle tracking with status, timestamps, and metadata.
    """
    # === Core Transaction Fields ===
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique transaction ID")
    sender_id: str = Field(..., description="ID of the sender/payer")
    receiver_id: str = Field(..., description="ID of the receiver/payee")
    amount: float = Field(..., description="Monetary amount transferred")
    
    # === Asset and Labor Exchange ===
    assets: Optional[List[Any]] = Field(default_factory=list, description="Assets exchanged (e.g., Products)")
    labor_hours: List[LaborHour] = Field(default_factory=list, description="Labor hours exchanged")
    
    # === Transaction Type ===
    type: Literal[
        'purchase',           # Product purchase
        'interest',           # Interest payment
        'service',            # Service payment
        'redistribution',     # Government redistribution
        'consume_tax',        # Consumption/VAT tax
        'labor_tax',          # Labor income tax
        'fica_tax',           # FICA/social security tax
        'corporate_tax',      # Corporate income tax
        'labor_payment',      # Wage payment
        'inherent_market',    # Internal market transaction
        'government_procurement',  # Government purchase
        'transfer',           # Simple money transfer
        'product_sale',       # Product sale (manufacturer/retailer)
        'resource_purchase',  # Resource/input purchase
        'tax_collection',     # General tax collection
        'financial',          # Financial transaction
    ] = Field(default='purchase', description="Type of transaction")
    
    # === Lifecycle Tracking ===
    timestamp: datetime = Field(default_factory=datetime.now, description="When transaction was created")
    status: str = Field(default=TransactionStatus.PENDING, description="Current transaction status")
    error_message: Optional[str] = Field(None, description="Error message if transaction failed")
    
    # === Temporal Context ===
    month: Optional[int] = Field(default=0, description="Month number when transaction occurred")
    
    # === Extensibility ===
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional transaction metadata (product_id, quantity, price_per_unit, etc.)"
    )
    related_transaction_id: Optional[str] = Field(
        None,
        description="ID of related transaction (e.g., refund references original purchase)"
    )
    
    @classmethod
    def create_transfer(
        cls,
        sender_id: str,
        receiver_id: str,
        amount: float,
        month: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Transaction':
        """Create a simple money transfer transaction."""
        return cls(
            sender_id=sender_id,
            receiver_id=receiver_id,
            amount=amount,
            type='transfer',
            month=month,
            metadata=metadata or {},
            status=TransactionStatus.PENDING
        )
    
    @classmethod
    def create_product_transaction(
        cls,
        buyer_id: str,
        seller_id: str,
        amount: float,
        products: List[Any],
        month: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Transaction':
        """Create a product purchase transaction."""
        return cls(
            sender_id=buyer_id,
            receiver_id=seller_id,
            amount=amount,
            assets=products,
            type='purchase',
            month=month,
            metadata=metadata or {},
            status=TransactionStatus.PENDING
        )
    
    @classmethod
    def create_wage_payment(
        cls,
        employer_id: str,
        employee_id: str,
        amount: float,
        labor_hours: List[LaborHour],
        month: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Transaction':
        """Create a wage payment transaction."""
        return cls(
            sender_id=employer_id,
            receiver_id=employee_id,
            amount=amount,
            labor_hours=labor_hours,
            type='labor_payment',
            month=month,
            metadata=metadata or {},
            status=TransactionStatus.PENDING
        )
    
    @classmethod
    def create_tax_transaction(
        cls,
        payer_id: str,
        government_id: str,
        amount: float,
        tax_type: Literal['consume_tax', 'labor_tax', 'fica_tax', 'corporate_tax', 'tax_collection'],
        month: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Transaction':
        """Create a tax payment transaction."""
        return cls(
            sender_id=payer_id,
            receiver_id=government_id,
            amount=amount,
            type=tax_type,
            month=month,
            metadata=metadata or {},
            status=TransactionStatus.PENDING
        )
    
    @classmethod
    def create_resource_transaction(
        cls,
        buyer_id: str,
        seller_id: str,
        amount: float,
        resources: List[Any],
        month: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Transaction':
        """Create a resource/input purchase transaction."""
        return cls(
            sender_id=buyer_id,
            receiver_id=seller_id,
            amount=amount,
            assets=resources,
            type='resource_purchase',
            month=month,
            metadata=metadata or {},
            status=TransactionStatus.PENDING
        )

class PurchaseRecord(BaseModel):
    """
    Detailed record of a purchase transaction.
    
    Tracks what was bought, from whom, at what price, and when.
    """
    product_id: str = Field(..., description="ID of purchased product")
    product_name: str = Field(..., description="Name of purchased product")
    quantity: float = Field(..., description="Quantity purchased")
    price_per_unit: float = Field(..., description="Price per unit")
    total_spent: float = Field(..., description="Total amount spent")
    seller_id: str = Field(..., description="ID of the seller")
    tx_id: str = Field(..., description="Associated transaction ID")
    timestamp: datetime = Field(..., description="When purchase occurred")
    month: Optional[int] = Field(default=0, description="Month number")


class MiddlewareRegistry:
    """
    Registry for transaction middleware functions.
    
    Allows registration of callbacks that execute during specific transaction types,
    enabling extensible transaction processing logic.
    """
    
    def __init__(self):
        """Initialize empty middleware registry."""
        self.middlewares_by_type: Dict[str, List[Callable[[Transaction, Dict[str, float]], None]]] = {}

    def register(self, tx_type: str, func: Callable, tag: Optional[str] = None) -> None:
        """
        Register a middleware function for a transaction type.
        
        Args:
            tx_type: Type of transaction to apply middleware to
            func: Middleware function to execute
            tag: Optional tag for identifying/replacing middleware
        """
        if tag:
            # Remove any existing middleware with same tag
            self.middlewares_by_type[tx_type] = [
                f for f in self.middlewares_by_type.get(tx_type, [])
                if getattr(f, "_tag", None) != tag
            ]
            func._tag = tag  # type: ignore
        self.middlewares_by_type.setdefault(tx_type, []).append(func)

    def execute_all(self, tx_type: str, transaction: Transaction, ledger: Dict[str, Ledger]) -> None:
        """
        Execute all registered middleware for a transaction type.
        
        Args:
            tx_type: Type of transaction
            transaction: The transaction being processed
            ledger: Current ledger state
        """
        for mw in self.middlewares_by_type.get(tx_type, []):
            mw(transaction, ledger)


# =============================================================================
# Firm Innovation Models
# =============================================================================

class FirmInnovationConfig(BaseModel):
    """
    Configuration for a firm's innovation strategy.
    
    Determines how aggressive a firm is in pursuing innovation and
    what resources it allocates.
    """
    firm_id: str = Field(..., description="ID of the firm")
    innovation_strategy: Literal['encouraged', 'suppressed'] = Field(
        ...,
        description="Whether innovation is encouraged or suppressed"
    )
    labor_productivity_factor: float = Field(..., description="Labor productivity multiplier")
    profit_margin: Optional[float] = Field(None, description="Target profit margin")
    fund_share: float = Field(..., description="Share of funds allocated to innovation")


class FirmInnovationEvent(BaseModel):
    """
    Records an innovation event at a firm.
    
    Tracks changes in pricing, attributes, margins, or productivity
    resulting from innovation activities.
    """
    firm_id: str = Field(..., description="ID of the firm")
    innovation_type: Optional[Literal[
        'price',
        'attribute',
        'profit_margin',
        'labor_productivity_factor'
    ]] = Field(None, description="Type of innovation")
    month: int = Field(..., description="Month when innovation occurred")
    old_value: Optional[float] = Field(None, description="Value before innovation")
    new_value: Optional[float] = Field(None, description="Value after innovation")
    price_change: Optional[float] = Field(None, description="Change in price")
    attribute_change: Optional[float] = Field(None, description="Change in attributes")

    @classmethod
    def create(
        cls,
        firm_id: str,
        innovation_type: Optional[Literal[
            'price',
            'attribute',
            'profit_margin',
            'labor_productivity_factor'
        ]] = None,
        month: int = 0,
        old_value: Optional[float] = None,
        new_value: Optional[float] = None,
        price_change: Optional[float] = None,
        attribute_change: Optional[float] = None
    ) -> 'FirmInnovationEvent':
        """
        Create an innovation event record.
        
        Args:
            firm_id: ID of the innovating firm
            innovation_type: Type of innovation
            month: When innovation occurred
            old_value: Previous value
            new_value: New value
            price_change: Price delta
            attribute_change: Attribute delta
            
        Returns:
            New FirmInnovationEvent instance
        """
        return cls(
            firm_id=firm_id,
            innovation_type=innovation_type,
            month=month,
            old_value=old_value,
            new_value=new_value,
            price_change=price_change,
            attribute_change=attribute_change
        )


# =============================================================================
# Main Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    # Example: Calculate progressive income tax
    tax_policy = TaxPolicy(
        income_tax_rate=[
            TaxBracket(cutoff=0.0, rate=0.10),
            TaxBracket(cutoff=1000.0, rate=0.15),
            TaxBracket(cutoff=4000.0, rate=0.22),
            TaxBracket(cutoff=8500.0, rate=0.24),
            TaxBracket(cutoff=16000.0, rate=0.32),
            TaxBracket(cutoff=20000.0, rate=0.35)
        ],
        corporate_tax_rate=0.21,
        vat_rate=0.08
    )
    
    gross_wage = 4100
    total_tax = 0.0
    
    for i, bracket in enumerate(tax_policy.income_tax_rate):
        if gross_wage > bracket.cutoff:
            # Determine upper bound of this bracket
            if i + 1 < len(tax_policy.income_tax_rate):
                upper_bracket = tax_policy.income_tax_rate[i + 1].cutoff
            else:
                upper_bracket = float('inf')
            
            # Calculate taxable amount in this bracket
            taxable_in_bracket = min(gross_wage, upper_bracket) - bracket.cutoff
            total_tax += taxable_in_bracket * bracket.rate
        else:
            break
    
    print(f"Gross wage: ${gross_wage:.2f}")
    print(f"Total tax: ${total_tax:.2f}")
    print(f"Effective tax rate: {(total_tax / gross_wage * 100):.2f}%")
