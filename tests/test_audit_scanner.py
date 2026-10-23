import unittest
import pandas as pd

from scanner import adjusted_plan_levels, build_exit_from_history


class CorporateActionStopTests(unittest.TestCase):
    def test_ex_dividend_does_not_produce_false_stop(self):
        history_prices = pd.DataFrame(
            {"Close": [100.0, 95.0, 94.0], "Adj Close": [96.0, 95.0, 94.0]},
            index=pd.to_datetime(["2026-10-20", "2026-10-22", "2026-10-23"]),
        )
        rec = {"code": "2330", "name": "範例", "action": "strong_buy",
               "close": 100.0, "stop": 97.0, "ma20": 80.0, "rec_date": "2026-10-20"}
        levels, verified = adjusted_plan_levels(rec, history_prices, "2026-10-23", 94.0)
        self.assertTrue(verified)
        self.assertAlmostEqual(levels["stop"], 93.12)
        today = pd.DataFrame({"code": ["2330"], "name": ["範例"], "close": [94.0],
                              "change_pct": [-1.0], "change": [-1.0]})
        days = [{"date": "2026-10-20", "tracked_pool": [rec], "closed": []}]
        trims, exits = build_exit_from_history([], days, "2026-10-23", today,
                                                histories={"2330": history_prices})
        self.assertEqual((len(trims), len(exits)), (0, 0))

    def test_no_adjustment_data_is_reported_as_unverified(self):
        rec = {"stop": 97.0, "tp1": 110.0, "rec_date": "2026-10-20"}
        values, verified = adjusted_plan_levels(rec, None, "2026-10-23", 94.0)
        self.assertFalse(verified)
        self.assertEqual(values["stop"], 97.0)


if __name__ == "__main__":
    unittest.main()
