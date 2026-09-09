"""
MCP Server for Structural Causal Model DAG Interventional Engine Skill
"""

import json
import sys
from client import StructuralCausalModel

def handle_call(name: str, args: dict) -> dict:
    if name == "simulate_scm_intervention":
        scm = StructuralCausalModel()
        scm.add_variable("Seasonality", [], lambda p, u: 1.0 + u)
        scm.add_variable("Spend", ["Seasonality"], lambda p, u: p["Seasonality"] * 1000.0 + u)
        scm.add_variable("Traffic", ["Spend", "Seasonality"], lambda p, u: p["Spend"] * 0.05 + p["Seasonality"] * 50.0 + u)
        scm.add_variable("Revenue", ["Traffic"], lambda p, u: p["Traffic"] * 20.0 + u)
        
        interv_var = args.get("intervention_var")
        interv_val = args.get("intervention_val", 5000.0)
        if interv_var:
            scm.do(interv_var, interv_val)
        noise = args.get("noise", {"Seasonality": 0.0, "Spend": 0.0, "Traffic": 0.0, "Revenue": 0.0})
        res = scm.sample(noise)
        return {"simulated_state": res}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_call(req.get("method"), req.get("params", {}))
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
