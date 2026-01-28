import ray
from typing import List, Dict, Optional, Tuple, Any
from uuid import uuid4
from agenteconomy.center.Model import Job, MatchedJob, LaborHour, Wage, JobApplication
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
        self.job_applications: Dict[str, List[JobApplication]] = {} # job_id -> List[JobApplication]
        self.application_index: Dict[Tuple[str, str, str], JobApplication] = {}
        self.backup_candidates: Dict[str, List[Dict[str, Any]]] = {} # job_id -> List[backup_candidate_info]
        self.backup_cursor: Dict[str, int] = {}
        self.offers: Dict[str, Dict[str, Any]] = {}
        self.offers_by_job: Dict[str, List[str]] = {}
        self.offers_by_worker: Dict[Tuple[str, str], List[str]] = {}
        self.labor_index: Dict[Tuple[str, str], LaborHour] = {}
        self.matched_workers: set = set()
        self.economic_center = economic_center
        self.logger = get_logger(name="LaborMarket")
        self.logger.info(f"LaborMarket initialized")

        # Records
        self.wage_history: List[Wage] = []
        
        # Default work parameters
        self.default_hours_per_week: float = 40.0
        self.default_weeks_per_month: float = 4.0

    def _worker_key(self, household_id: str, lh_type: str) -> Tuple[str, str]:
        return (household_id, lh_type)

    def _register_labor_hour(self, labor_hour: LaborHour) -> None:
        key = self._worker_key(labor_hour.agent_id, labor_hour.lh_type)
        if key in self.labor_index:
            return
        self.labor_index[key] = labor_hour
        self.labor_hours.append(labor_hour)

    def register_labor_hours(self, labor_hours: List[LaborHour]) -> int:
        added = 0
        for labor_hour in labor_hours:
            key = self._worker_key(labor_hour.agent_id, labor_hour.lh_type)
            if key in self.labor_index:
                continue
            self._register_labor_hour(labor_hour)
            added += 1
        return added

    def get_labor_force_snapshot(self) -> Dict[str, Dict[str, int]]:
        snapshot: Dict[str, Dict[str, int]] = {}
        for labor_hour in self.labor_hours:
            household_id = labor_hour.agent_id
            entry = snapshot.setdefault(household_id, {"total": 0, "employed": 0, "unemployed": 0})
            entry["total"] += 1
            if not labor_hour.is_valid and labor_hour.firm_id is not None:
                entry["employed"] += 1
            else:
                entry["unemployed"] += 1
        return snapshot

    def _is_worker_available(self, household_id: str, lh_type: str) -> bool:
        key = self._worker_key(household_id, lh_type)
        if key in self.matched_workers:
            return False
        labor_hour = self.labor_index.get(key)
        if labor_hour is None:
            return True
        if not labor_hour.is_valid:
            return False
        if labor_hour.firm_id is not None:
            return False
        return True

    def _update_labor_status(self, household_id: str, lh_type: str, job: Job) -> None:
        key = self._worker_key(household_id, lh_type)
        labor_hour = self.labor_index.get(key)
        if labor_hour is not None:
            labor_hour.is_valid = False
            labor_hour.firm_id = job.firm_id
            labor_hour.job_title = job.title
            labor_hour.job_SOC = job.SOC

    def _get_job_by_id(self, job_id: str) -> Optional[Job]:
        for job in self.job_openings:
            if job.job_id == job_id:
                return job
        return None
        
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
                self.matched_workers.add(self._worker_key(household_id, lh_type))
                self._update_labor_status(household_id, lh_type, j)
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
        ranked = self.rank_jobs_for_labor(labor_hour, loss_threshold=3000.0)
        return [job for job, loss in ranked[:3]]

    def rank_jobs_for_labor(
        self,
        labor_hour: LaborHour,
        loss_threshold: float = 3000.0
    ) -> List[Tuple[Job, float]]:
        if labor_hour.skill_profile is None or labor_hour.ability_profile is None:
            return []

        job_losses: List[Tuple[Job, float]] = []
        for job in self.job_openings:
            if not job.is_valid or job.positions_available <= 0:
                continue
            if not isinstance(job.required_skills, dict) or not isinstance(job.required_abilities, dict):
                continue

            required_profile = [job.required_skills, job.required_abilities]
            worker_profile = [labor_hour.skill_profile, labor_hour.ability_profile]
            loss = self._compute_matching_loss(worker_profile, required_profile)
            if loss < loss_threshold:
                job_losses.append((job, loss))

        job_losses.sort(key=lambda x: x[1])
        return job_losses
    
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

    def submit_application(self, application: JobApplication, labor_hour: Optional[LaborHour] = None) -> bool:
        job = self._get_job_by_id(application.job_id)
        if job is None or not job.is_valid or job.positions_available <= 0:
            return False
        if not self._is_worker_available(application.household_id, application.lh_type):
            return False
        key = (application.job_id, application.household_id, application.lh_type)
        if key in self.application_index:
            return False
        if labor_hour is not None:
            self._register_labor_hour(labor_hour)
        self.job_applications.setdefault(application.job_id, []).append(application)
        self.application_index[key] = application
        return True

    def apply_for_jobs(
        self,
        labor_hour: LaborHour,
        month: int,
        max_apply: int = 3,
        loss_threshold: float = 3000.0
    ) -> List[JobApplication]:
        if max_apply <= 0:
            return []
        if not labor_hour.is_valid or labor_hour.firm_id is not None:
            return []
        if not self._is_worker_available(labor_hour.agent_id, labor_hour.lh_type):
            return []
        ranked = self.rank_jobs_for_labor(labor_hour, loss_threshold=loss_threshold)
        applications: List[JobApplication] = []
        for job, _loss in ranked[:max_apply]:
            application = JobApplication.create(
                job_id=job.job_id,
                household_id=labor_hour.agent_id,
                lh_type=labor_hour.lh_type,
                expected_wage=job.wage_per_hour,
                worker_skills=labor_hour.skill_profile,
                worker_abilities=labor_hour.ability_profile,
                month=month
            )
            if self.submit_application(application, labor_hour=labor_hour):
                applications.append(application)
        return applications

    def evaluate_applications(
        self,
        job_id: str,
        max_backups: int = 3
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        job = self._get_job_by_id(job_id)
        if job is None or not job.is_valid or job.positions_available <= 0:
            return [], []

        applications = self.job_applications.get(job_id, [])
        if not applications:
            return [], []

        candidates: List[Dict[str, Any]] = []
        required_profile = [job.required_skills or {}, job.required_abilities or {}]
        for app in applications:
            if not self._is_worker_available(app.household_id, app.lh_type):
                continue
            worker_profile = [app.worker_skills, app.worker_abilities]
            loss = self._compute_matching_loss(worker_profile, required_profile)
            if loss == float('inf'):
                continue
            score = loss
            if app.expected_wage > job.wage_per_hour:
                score += (app.expected_wage - job.wage_per_hour) * 0.5
            candidates.append({
                "household_id": app.household_id,
                "lh_type": app.lh_type,
                "loss": loss,
                "score": score,
                "expected_wage": app.expected_wage,
            })

        candidates.sort(key=lambda x: x["score"])
        positions = max(0, int(job.positions_available or 0))
        primary = candidates[:positions]
        backups = candidates[positions:]
        if max_backups > 0:
            backups = backups[:max_backups]
        return primary, backups

    def reset_matching_state(self) -> None:
        self.job_applications.clear()
        self.application_index.clear()
        self.backup_candidates.clear()
        self.backup_cursor.clear()
        self.offers.clear()
        self.offers_by_job.clear()
        self.offers_by_worker.clear()

    def _has_offer(self, job_id: str, household_id: str, lh_type: str) -> bool:
        for offer_id in self.offers_by_job.get(job_id, []):
            offer = self.offers.get(offer_id)
            if not offer:
                continue
            if offer["household_id"] == household_id and offer["lh_type"] == lh_type:
                return True
        return False

    def _create_offer(
        self,
        job: Job,
        candidate: Dict[str, Any],
        month: int,
        rank: int
    ) -> Optional[str]:
        household_id = candidate["household_id"]
        lh_type = candidate["lh_type"]
        if not self._is_worker_available(household_id, lh_type):
            return None
        if self._has_offer(job.job_id, household_id, lh_type):
            return None

        offer_id = str(uuid4())
        offer = {
            "offer_id": offer_id,
            "job_id": job.job_id,
            "firm_id": job.firm_id,
            "household_id": household_id,
            "lh_type": lh_type,
            "title": job.title,
            "soc": job.SOC,
            "wage_per_hour": job.wage_per_hour,
            "loss": candidate.get("loss"),
            "expected_wage": candidate.get("expected_wage"),
            "month": month,
            "rank": rank,
            "status": "pending",
        }
        self.offers[offer_id] = offer
        self.offers_by_job.setdefault(job.job_id, []).append(offer_id)
        self.offers_by_worker.setdefault(self._worker_key(household_id, lh_type), []).append(offer_id)
        return offer_id

    def make_offers(self, month: int, max_backups: int = 3, reset_existing: bool = False) -> Dict[str, int]:
        if reset_existing:
            self.offers.clear()
            self.offers_by_job.clear()
            self.offers_by_worker.clear()
            self.backup_candidates.clear()
            self.backup_cursor.clear()

        created = 0
        for job_id in list(self.job_applications.keys()):
            job = self._get_job_by_id(job_id)
            if job is None:
                continue
            primary, backups = self.evaluate_applications(job_id, max_backups=max_backups)
            self.backup_candidates[job_id] = backups
            self.backup_cursor[job_id] = 0
            for idx, candidate in enumerate(primary):
                if self._create_offer(job, candidate, month, rank=idx) is not None:
                    created += 1
        return {
            "offers_created": created,
            "jobs_with_offers": len(self.offers_by_job)
        }

    def get_offers_for_household(self, household_id: str, lh_type: Optional[str] = None) -> List[Dict[str, Any]]:
        offers: List[Dict[str, Any]] = []
        if lh_type is not None:
            keys = [self._worker_key(household_id, lh_type)]
        else:
            keys = [key for key in self.offers_by_worker.keys() if key[0] == household_id]
        for key in keys:
            for offer_id in self.offers_by_worker.get(key, []):
                offer = self.offers.get(offer_id)
                if offer and offer["status"] == "pending":
                    offers.append(offer)
        return offers

    def _pending_offers_for_job(self, job_id: str) -> List[str]:
        return [
            offer_id
            for offer_id in self.offers_by_job.get(job_id, [])
            if self.offers.get(offer_id, {}).get("status") == "pending"
        ]

    def _offer_next_backup(self, job_id: str, month: int) -> Optional[str]:
        job = self._get_job_by_id(job_id)
        if job is None or not job.is_valid or job.positions_available <= 0:
            return None
        if len(self._pending_offers_for_job(job_id)) >= int(job.positions_available or 0):
            return None

        backups = self.backup_candidates.get(job_id, [])
        idx = self.backup_cursor.get(job_id, 0)
        while idx < len(backups):
            candidate = backups[idx]
            idx += 1
            self.backup_cursor[job_id] = idx
            offer_id = self._create_offer(job, candidate, month, rank=idx)
            if offer_id is not None:
                return offer_id
        return None

    def _reject_offer(self, offer_id: str, trigger_backup: bool = True) -> bool:
        offer = self.offers.get(offer_id)
        if offer is None or offer["status"] != "pending":
            return False
        offer["status"] = "rejected"
        if trigger_backup:
            self._offer_next_backup(offer["job_id"], offer["month"])
        return True

    def reject_offer(self, offer_id: str) -> bool:
        return self._reject_offer(offer_id, trigger_backup=True)

    def _reject_other_offers(self, household_id: str, lh_type: str, keep_offer_id: str) -> int:
        rejected = 0
        for offer_id in self.offers_by_worker.get(self._worker_key(household_id, lh_type), []):
            if offer_id == keep_offer_id:
                continue
            if self._reject_offer(offer_id, trigger_backup=True):
                rejected += 1
        return rejected

    def _accept_offer(self, offer: Dict[str, Any]) -> bool:
        if offer["status"] != "pending":
            return False
        household_id = offer["household_id"]
        lh_type = offer["lh_type"]
        if not self._is_worker_available(household_id, lh_type):
            offer["status"] = "rejected"
            return False
        job = self._get_job_by_id(offer["job_id"])
        if job is None or not job.is_valid or job.positions_available <= 0:
            offer["status"] = "rejected"
            return False

        job.positions_available -= 1
        if job.positions_available <= 0:
            job.is_valid = False

        loss = offer.get("loss")
        skill_match_score = None
        if isinstance(loss, (int, float)) and loss >= 0:
            skill_match_score = 1.0 / (1.0 + loss)

        self.matched_jobs.append(
            MatchedJob.create(
                job=job,
                average_wage=job.wage_per_hour,
                household_id=household_id,
                lh_type=lh_type,
                firm_id=job.firm_id,
                skill_match_score=skill_match_score
            )
        )
        self.matched_workers.add(self._worker_key(household_id, lh_type))
        self._update_labor_status(household_id, lh_type, job)
        offer["status"] = "accepted"
        self._reject_other_offers(household_id, lh_type, offer["offer_id"])
        return True

    def accept_offer(self, offer_id: str) -> bool:
        offer = self.offers.get(offer_id)
        if offer is None:
            return False
        return self._accept_offer(offer)

    def resolve_offers(self, month: int, acceptance_policy: str = "best_loss") -> Dict[str, int]:
        accepted = 0
        rejected = 0
        rounds = 0
        prev_total = -1

        while True:
            rounds += 1
            pending_by_worker: Dict[Tuple[str, str], List[str]] = {}
            for worker_key, offer_ids in self.offers_by_worker.items():
                pending = [
                    offer_id for offer_id in offer_ids
                    if self.offers.get(offer_id, {}).get("status") == "pending"
                ]
                if pending:
                    pending_by_worker[worker_key] = pending

            if not pending_by_worker:
                break

            for _worker_key, offer_ids in pending_by_worker.items():
                chosen_id = self._select_offer(offer_ids, acceptance_policy)
                if chosen_id and self.accept_offer(chosen_id):
                    accepted += 1

            for _worker_key, offer_ids in pending_by_worker.items():
                for offer_id in offer_ids:
                    offer = self.offers.get(offer_id)
                    if offer and offer["status"] == "pending":
                        if self._reject_offer(offer_id, trigger_backup=True):
                            rejected += 1

            for job_id in list(self.backup_candidates.keys()):
                self._offer_next_backup(job_id, month)

            total_processed = accepted + rejected
            if total_processed == prev_total:
                break
            prev_total = total_processed

        return {
            "accepted": accepted,
            "rejected": rejected,
            "rounds": rounds
        }

    def _select_offer(self, offer_ids: List[str], acceptance_policy: str) -> Optional[str]:
        if not offer_ids:
            return None
        offers = [self.offers[offer_id] for offer_id in offer_ids if offer_id in self.offers]
        if not offers:
            return None
        if acceptance_policy == "highest_wage":
            offers.sort(key=lambda x: (-x.get("wage_per_hour", 0.0), x.get("loss", 0.0)))
        else:
            offers.sort(key=lambda x: (x.get("loss", 0.0), -x.get("wage_per_hour", 0.0)))
        return offers[0]["offer_id"]

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
