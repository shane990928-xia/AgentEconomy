import ray
from typing import List, Dict, Optional, Tuple
from agenteconomy.center.Model import Job, MatchedJob, LaborHour, Wage
from agenteconomy.utils.logger import get_logger

@ray.remote
class LaborMarket:
    def __init__(self, economic_center=None):
        """
        Initialize LaborMarket
        
        Args:
            economic_center: Optional EconomicCenter Ray Actor for wage transfers
        """
        self.job_openings: List[Job] = [] # Store all job openings
        self.matched_jobs: List[MatchedJob] = []
        self.labor_hours: List[LaborHour] = [] # Store all labor hours
        self.job_applications: Dict[str, List[LaborHour]] = {} # job_id -> List[JobApplication]
        self.backup_candidates: Dict[str, List[Dict]] = {} # job_id -> List[backup_candidate_info]
        self.economic_center = economic_center
        self.logger = get_logger(name="LaborMarket")
        self.logger.info(f"LaborMarket initialized")

        # Records
        self.wage_history: List[Wage] = []
        
        # Default work parameters
        self.default_hours_per_week: float = 40.0
        self.default_weeks_per_month: float = 4.0
        
    def post_job(self, job: Job):
        """
        Post a job opening to the labor market.
        
        Args:
            job: Job object to be posted
        """
        self.job_openings.append(job)
        self.logger.info(f"Job {job.job_id} posted")
    
    def query_opening_jobs(self) -> List[Job]:
        return [job for job in self.job_openings if job.is_valid]
    
    def query_jobs_by_firm(self, firm_id: str) -> List[Job]:
        return [job for job in self.job_openings if job.firm_id == firm_id]

    def get_firm_job_snapshot(self, firm_id: str) -> Dict[str, int]:
        """
        Get current open positions by SOC for a firm.
        """
        positions_by_soc: Dict[str, int] = {}
        for job in self.job_openings:
            if job.firm_id != firm_id:
                continue
            if not job.is_valid or job.positions_available <= 0:
                continue
            positions_by_soc[job.SOC] = positions_by_soc.get(job.SOC, 0) + job.positions_available
        return positions_by_soc
    
    def query_jobs_by_soc(self, soc: str) -> List[Job]:
        return [job for job in self.job_openings if job.SOC == soc]
    
    def query_jobs_by_title(self, title: str) -> List[Job]:
        return [job for job in self.job_openings if job.title == title]
    
    def add_job_position(self, firm_id: str, job: Job):
        """
        Adds a job position to the market for a specific firm.
        If the job already exists, it increments the available positions.
        """
        for j in self.job_openings:
            if j.firm_id == firm_id and j.SOC == job.SOC:
                j.positions_available += max(1, int(job.positions_available or 1))
                j.is_valid = True
                return
        self.job_openings.append(job)

    def apply_job_plan(self, firm_id: str, jobs: List[Job]) -> Dict[str, int]:
        """
        Apply a job plan: add new roles or increase positions for existing roles.
        """
        added: Dict[str, int] = {}
        for job in jobs:
            if job.firm_id != firm_id:
                job.firm_id = firm_id
            if job.positions_available <= 0:
                continue
            self.add_job_position(firm_id, job)
            added[job.SOC] = added.get(job.SOC, 0) + int(job.positions_available)
        return added

    def align_job(self, household_id: str, job: Job, lh_type: str) -> Optional[Job]:
        """
        Aligns a job with a household, reducing the available positions.
        
        Args:
            household_id: ID of the household applying for the job
            job: Job object to align
            lh_type: Type of labor hour ('head' or 'spouse')
            
        Returns:
            Job object if alignment successful, None otherwise
        """
        for j in self.job_openings:
            if j.SOC == job.SOC and j.firm_id == job.firm_id and j.positions_available > 0:
                j.positions_available -= 1  # Decrease the number of available positions
                if j.positions_available <= 0:
                    j.is_valid = False
                self.matched_jobs.append(
                    MatchedJob.create(job=j, average_wage=j.wage_per_hour, household_id=household_id, lh_type=lh_type, firm_id=j.firm_id)
                )
                return j
        return None

    def summary(self) -> Dict[str, float]:
        """
        Summary of the labor market.
        
        Returns:
            Dictionary containing labor market statistics
        """
        total_job_positions = sum(job.positions_available for job in self.job_openings) + len(self.matched_jobs)
        
        # Safe division to avoid ZeroDivisionError
        total_labor_hours = len(self.labor_hours)
        total_matched = len(self.matched_jobs)
        
        employment_rate = total_matched / total_labor_hours if total_labor_hours > 0 else 0.0
        unemployment_rate = 1.0 - employment_rate
        
        average_wage = (
            sum(job.average_wage for job in self.matched_jobs) / total_matched
            if total_matched > 0 else 0.0
        )
        
        job_fill_rate = (
            total_matched / total_job_positions
            if total_job_positions > 0 else 0.0
        )
        
        return {
            "total_jobs": len(self.job_openings),
            "total_job_positions": total_job_positions,
            "total_matched_jobs": total_matched,
            "total_labor_hours": total_labor_hours,
            "total_job_applications": len(self.job_applications),
            "total_backup_candidates": len(self.backup_candidates),
            "employment_rate": employment_rate,
            "unemployment_rate": unemployment_rate,
            "average_wage": average_wage,
            "job_fill_rate": job_fill_rate,
        }

    def match_jobs(self, labor_hour: LaborHour) -> List[Job]:
        """
        Matches labor hours with available jobs.
        Returns top 3 best matching jobs sorted by matching loss (best match first).
        
        Args:
            labor_hour: LaborHour object containing worker's skills and abilities
            
        Returns:
            List of top 3 best matching Job objects, sorted by matching loss (best first)
        """
        # Check for None profiles
        if labor_hour.skill_profile is None or labor_hour.ability_profile is None:
            return []
        
        job_losses = []
        
        for job in self.job_openings:
            if job.is_valid and job.positions_available > 0:
                # Check if required profiles exist and are dictionaries
                if not isinstance(job.required_skills, dict) or not isinstance(job.required_abilities, dict):
                    continue
                    
                required_profile = [job.required_skills, job.required_abilities]
                worker_profile = [labor_hour.skill_profile, labor_hour.ability_profile]
                loss = self._compute_matching_loss(worker_profile, required_profile)
                
                if loss < 3000:
                    job_losses.append((job, loss))
        
        # 按损失排序（损失越小越好）
        job_losses.sort(key=lambda x: x[1])
        
        # 返回前3个最佳匹配的工作
        top_jobs = [job for job, loss in job_losses[:3]]
        return top_jobs
    
    def _compute_matching_loss(self, worker_profile: list, required_profile: list) -> float:
        """
        Compute matching loss between worker profile and job requirements.
        
        Args:
            worker_profile: List of [skill_profile, ability_profile] dictionaries
            required_profile: List of [required_skills, required_abilities] dictionaries
            
        Returns:
            Total matching loss (lower is better)
        """
        total_loss = 0.0

        # Validate input lengths
        if len(worker_profile) != len(required_profile):
            return float('inf')  # Invalid input

        for i in range(len(worker_profile)):
            # Check if profiles are dictionaries
            if not isinstance(required_profile[i], dict) or not isinstance(worker_profile[i], dict):
                continue
                
            for skill, req in required_profile[i].items():
                mean = req.get('mean')
                std = req.get('std')
                importance = req.get('importance', 1.0)

                if importance is None or importance <= 0:
                    continue
                if std is None or std <= 0:
                    continue
                if mean is None:
                    continue

                worker_value = worker_profile[i].get(skill, 0.0)

                distance = (worker_value - mean) / std
                
                if distance > 0:  
                    loss = importance * (distance ** 2) * 0.2
                else: 
                    loss = importance * (distance ** 2)

                total_loss += loss
        return total_loss

    # =========================================================================
    # Wage Calculation Methods
    # =========================================================================
    
    def set_economic_center(self, economic_center):
        """
        Set or update the economic center reference.
        
        Args:
            economic_center: EconomicCenter Ray Actor handle
        """
        self.economic_center = economic_center
        self.logger.info("EconomicCenter reference updated")
    
    def calculate_monthly_wage(
        self, 
        wage_per_hour: float, 
        hours_per_week: float = None,
        weeks_per_month: float = None
    ) -> float:
        """
        Calculate monthly wage from hourly wage.
        
        Args:
            wage_per_hour: Hourly wage rate
            hours_per_week: Work hours per week (default: 40)
            weeks_per_month: Work weeks per month (default: 4)
            
        Returns:
            Monthly gross wage
        """
        hours = hours_per_week if hours_per_week is not None else self.default_hours_per_week
        weeks = weeks_per_month if weeks_per_month is not None else self.default_weeks_per_month
        return wage_per_hour * hours * weeks
    
    def calculate_wage_for_matched_job(self, matched_job: MatchedJob) -> Dict:
        """
        Calculate wage details for a single matched job.
        
        Args:
            matched_job: MatchedJob instance
            
        Returns:
            Dict with wage details:
            {
                'household_id': str,
                'firm_id': str,
                'lh_type': str,
                'wage_per_hour': float,
                'hours_per_period': float,
                'monthly_gross_wage': float,
                'job_title': str,
                'soc': str
            }
        """
        job = matched_job.job
        hours_per_period = job.hours_per_period if job.hours_per_period else self.default_hours_per_week
        
        monthly_wage = self.calculate_monthly_wage(
            wage_per_hour=matched_job.average_wage,
            hours_per_week=hours_per_period
        )
        
        return {
            'household_id': matched_job.household_id,
            'firm_id': matched_job.firm_id,
            'lh_type': matched_job.lh_type,
            'wage_per_hour': matched_job.average_wage,
            'hours_per_period': hours_per_period,
            'monthly_gross_wage': monthly_wage,
            'job_title': job.title,
            'soc': job.SOC,
        }
    
    def calculate_all_wages(self) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Calculate wages for all matched jobs.
        
        Returns:
            Tuple of:
            - List of wage details for each matched job
            - Dict of firm_id -> total wage expense
        """
        wage_details = []
        firm_wage_totals: Dict[str, float] = {}
        
        for matched_job in self.matched_jobs:
            wage_info = self.calculate_wage_for_matched_job(matched_job)
            wage_details.append(wage_info)
            
            # Aggregate by firm
            firm_id = wage_info['firm_id']
            if firm_id not in firm_wage_totals:
                firm_wage_totals[firm_id] = 0.0
            firm_wage_totals[firm_id] += wage_info['monthly_gross_wage']
        
        return wage_details, firm_wage_totals
    
    def get_firm_labor_cost(self, firm_id: str) -> float:
        """
        Get total labor cost for a specific firm.
        
        Args:
            firm_id: Firm identifier
            
        Returns:
            Total monthly gross wage expense for the firm
        """
        total_cost = 0.0
        for matched_job in self.matched_jobs:
            if matched_job.firm_id == firm_id:
                wage_info = self.calculate_wage_for_matched_job(matched_job)
                total_cost += wage_info['monthly_gross_wage']
        return total_cost
    
    def get_firm_employees(self, firm_id: str) -> List[Dict]:
        """
        Get all employees (matched jobs) for a specific firm.
        
        Args:
            firm_id: Firm identifier
            
        Returns:
            List of wage details for each employee
        """
        employees = []
        for matched_job in self.matched_jobs:
            if matched_job.firm_id == firm_id:
                wage_info = self.calculate_wage_for_matched_job(matched_job)
                employees.append(wage_info)
        return employees
    
    def get_household_income(self, household_id: str) -> Dict:
        """
        Get total income for a specific household.
        
        Args:
            household_id: Household identifier
            
        Returns:
            Dict with household income details
        """
        total_income = 0.0
        jobs = []
        
        for matched_job in self.matched_jobs:
            if matched_job.household_id == household_id:
                wage_info = self.calculate_wage_for_matched_job(matched_job)
                total_income += wage_info['monthly_gross_wage']
                jobs.append(wage_info)
        
        return {
            'household_id': household_id,
            'total_monthly_income': total_income,
            'jobs': jobs,
            'job_count': len(jobs)
        }

    def process_monthly_wages(self, month: int) -> Dict:
        """
        Process wage payments for all matched jobs in a month.
        This method calculates wages and delegates actual transfers to EconomicCenter.
        
        Args:
            month: Current simulation month
            
        Returns:
            Dict with processing summary:
            {
                'month': int,
                'total_wages_paid': float,
                'total_employees': int,
                'wages_by_firm': {firm_id: total},
                'success': bool,
                'errors': List[str]
            }
        """
        wage_details, firm_totals = self.calculate_all_wages()
        
        total_paid = 0.0
        errors = []
        
        for wage_info in wage_details:
            try:
                # Record wage history in LaborMarket
                wage_record = Wage.create(
                    agent_id=wage_info['household_id'],
                    amount=wage_info['monthly_gross_wage'],
                    month=month
                )
                self.wage_history.append(wage_record)
                
                # Delegate actual transfer to EconomicCenter
                if self.economic_center is not None:
                    # Call EconomicCenter to process wage (handles taxes and ledger updates)
                    ray.get(self.economic_center.process_wage.remote(
                        month=month,
                        wage_hour=wage_info['wage_per_hour'],
                        household_id=wage_info['household_id'],
                        firm_id=wage_info['firm_id'],
                        hours_per_period=wage_info['hours_per_period'],
                        periods_per_month=self.default_weeks_per_month
                    ))
                
                total_paid += wage_info['monthly_gross_wage']
                
            except Exception as e:
                error_msg = f"Failed to process wage for {wage_info['household_id']}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        self.logger.info(
            f"Month {month}: Processed {len(wage_details)} wage payments, "
            f"total ${total_paid:.2f}"
        )
        
        return {
            'month': month,
            'total_wages_paid': total_paid,
            'total_employees': len(wage_details),
            'wages_by_firm': firm_totals,
            'success': len(errors) == 0,
            'errors': errors
        }

    async def step(self, month: int) -> Dict:
        """
        Execute monthly labor market step: process all wage payments.
        
        Args:
            month: Current simulation month
            
        Returns:
            Processing summary dict
        """
        return self.process_monthly_wages(month)
