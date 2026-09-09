"""
Demonstration of Structural Causal Model DAG Interventional Engine Skill
"""

from client import StructuralCausalModel

def main():
    print("=== Structural Causal Model (SCM) & Do-Calculus Demonstration ===")
    scm = StructuralCausalModel()

    # Model: Seasonality -> MarketingSpend -> WebTraffic -> Revenue
    # Seasonality also directly impacts WebTraffic (confounding)
    scm.add_variable("Seasonality", parents=[], equation=lambda p, u: 1.0 + u)
    scm.add_variable("MarketingSpend", parents=["Seasonality"], equation=lambda p, u: p["Seasonality"] * 1000.0 + u)
    scm.add_variable("WebTraffic", parents=["MarketingSpend", "Seasonality"], equation=lambda p, u: p["MarketingSpend"] * 0.05 + p["Seasonality"] * 50.0 + u)
    scm.add_variable("Revenue", parents=["WebTraffic"], equation=lambda p, u: p["WebTraffic"] * 20.0 + u)

    noise = {"Seasonality": 0.2, "MarketingSpend": 50.0, "WebTraffic": 10.0, "Revenue": 5.0}

    print("Observational Realization:")
    obs = scm.sample(noise)
    for k, v in obs.items():
        print(f"  {k}: {v}")

    # Pearl's do-calculus intervention: do(MarketingSpend = 5000.0)
    print("\nApplying Intervention do(MarketingSpend = 5000.0)...")
    scm.do("MarketingSpend", 5000.0)
    interv = scm.sample(noise)
    for k, v in interv.items():
        print(f"  {k}: {v}")

    assert interv["MarketingSpend"] == 5000.0
    assert interv["WebTraffic"] == 5000.0 * 0.05 + (1.0 + 0.2) * 50.0 + 10.0
    assert interv["Revenue"] == interv["WebTraffic"] * 20.0 + 5.0

    print("\nSCM Interventional Engine Verification PASS!")

if __name__ == "__main__":
    main()
