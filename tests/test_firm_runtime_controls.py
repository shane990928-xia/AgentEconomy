import unittest

from agenteconomy.agent.firm import Firm
from agenteconomy.simulation.agent_loader import create_firms


class FirmRuntimeControlsTests(unittest.IsolatedAsyncioTestCase):
    async def test_post_jobs_does_not_call_llm_by_default(self):
        firm = Firm(
            firm_id="firm_no_llm",
            name="No LLM Firm",
            industry="missing_industry",
            labor_market=None,
        )

        jobs = await firm.post_jobs(period=1)

        self.assertEqual(jobs, [])


class FirmLoaderLimitTests(unittest.TestCase):
    def test_create_firms_honors_explicit_limit(self):
        firms = create_firms(limit=5)

        self.assertEqual(len(firms), 5)
        self.assertTrue(all(f.firm_id.startswith("ret_") for f in firms))

    def test_create_firms_limit_above_retail_keeps_mixed_structure(self):
        firms = create_firms(limit=8)

        self.assertEqual(len(firms), 8)
        self.assertEqual(sum(1 for f in firms if f.firm_id.startswith("ret_")), 5)
        self.assertTrue(any(f.firm_id.startswith("mfg_") for f in firms))
        self.assertTrue(any(f.firm_id.startswith("svc_") for f in firms))


if __name__ == "__main__":
    unittest.main()
