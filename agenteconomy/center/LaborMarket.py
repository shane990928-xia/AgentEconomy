import ray
from typing import List, Dict, Optional
from agenteconomy.center.Model import Job, MatchedJob, LaborHour
from agenteconomy.utils.logger import get_logger

@ray.remote
class LaborMarket:
    def __init__(self):
        self.job_openings: List[Job] = [] # Store all job openings
        self.matched_jobs: List[MatchedJob] = []
        self.labor_hours: List[LaborHour] = [] # Store all labor hours
        self.job_applications:Dict[str, List[LaborHour]] = {} # job_id -> List[JobApplication]
        self.backup_candidates:Dict[str, List[Dict]] = {} # job_id -> List[backup_candidate_info]

        self.logger = get_logger(name="LaborMarket")
        self.logger.info(f"LaborMarket initialized")

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
                j.positions_available += 1
                return
        self.job_openings.append(job)

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