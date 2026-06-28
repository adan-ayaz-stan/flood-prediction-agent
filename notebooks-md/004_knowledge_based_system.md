# Step 8: Knowledge-Based System (Forward Chaining Inference Engine)

This notebook implements a Forward Chaining Inference Engine to handle the operational governance of our Earthquake Response System. It translates raw seismic facts into actionable emergency protocols.

### System Synergy & Component Contrast
To achieve a comprehensive disaster response, our pipeline relies on three distinct AI paradigms working in synergy:
* **The Machine Learning Model (Statistical & Probabilistic):** Evaluates historical seismic patterns to predict the mathematical probability of a specific threat level (`Disaster_Urgency`). It tells us *how bad* the event is.
* **The A* Search Agent (Algorithmic & Geographic):** Computes the mathematically optimal graph-based route across a damaged environment to reach the epicenter. It tells us *how to get there*.
* **The Forward Chaining Knowledge Base (Deterministic & Logical):** Applies hardcoded, human-approved expert emergency guidelines to the raw facts of the disaster. It breaks down *exactly what personnel and equipment need to be put on that route*. 

While ML handles the gray areas of prediction, the Knowledge Base ensures strict, deterministic adherence to expert safety protocols.


```python
class ForwardChainingEngine:
    def __init__(self, rules):
        self.rules = rules

    def infer(self, initial_facts):
        """
        Executes the forward chaining algorithm.
        Continuously loops through the rules until no new facts can be inferred.
        """
        inferred_facts = set(initial_facts)
        trace_log = []
        
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                # Check if all 'IF' conditions are present in our known facts
                if all(condition in inferred_facts for condition in rule['if']):
                    # Check if the 'THEN' conclusion is completely new
                    if rule['then'] not in inferred_facts:
                        inferred_facts.add(rule['then'])
                        trace_log.append(f"Fired Rule: {rule['name']} -> Added Fact: '{rule['then']}'")
                        changed = True
                        
        return inferred_facts, trace_log

# Define our 12 Expert Domain Rules for Earthquake Response
earthquake_rules = [
    # Primary Physical Effects
    {"name": "Rule 1", "if": ["high_magnitude", "shallow_depth"], "then": "severe_surface_shaking"},
    {"name": "Rule 2", "if": ["low_magnitude", "low_population_density"], "then": "predicted_threat_level_is_low"},
    {"name": "Rule 3", "if": ["over_water", "severe_surface_shaking"], "then": "tsunami_warning_active"},
    {"name": "Rule 4", "if": ["high_magnitude", "inland"], "then": "severe_infrastructure_damage"},
    
    # Secondary Human Impacts
    {"name": "Rule 5", "if": ["severe_surface_shaking", "high_population_density"], "then": "mass_casualties_expected"},
    {"name": "Rule 6", "if": ["severe_infrastructure_damage", "high_population_density"], "then": "widespread_homelessness"},
    
    # Tactical Resource Deployments
    {"name": "Rule 7", "if": ["predicted_threat_level_is_low"], "then": "deploy_basic_supplies"},
    {"name": "Rule 8", "if": ["mass_casualties_expected"], "then": "deploy_field_hospital"},
    {"name": "Rule 9", "if": ["tsunami_warning_active"], "then": "deploy_navy_rescue_crafts"},
    {"name": "Rule 10", "if": ["tsunami_warning_active"], "then": "initiate_coastal_evacuation_protocols"},
    {"name": "Rule 11", "if": ["severe_infrastructure_damage"], "then": "deploy_heavy_rubble_clearance_vehicles"},
    {"name": "Rule 12", "if": ["widespread_homelessness"], "then": "establish_displaced_persons_camp"},
    {"name": "Rule 13", "if": ["mass_casualties_expected", "severe_infrastructure_damage"], "then": "request_international_medical_aid"}
]

# Initialize the engine
kb_engine = ForwardChainingEngine(earthquake_rules)
print("Forward Chaining Engine Initialized with 13 Expert Rules.")
```

    Forward Chaining Engine Initialized with 13 Expert Rules.



```python
print("=== TEST CASE 1: High-Risk Ocean Quake (Magnitude 9.0) ===")

# Initial facts passed from our ML/Data layer
event_facts_1 = {"high_magnitude", "shallow_depth", "over_water", "high_population_density"}

# Run inference
final_facts_1, execution_trace_1 = kb_engine.infer(event_facts_1)

print("\n--- Execution Trace ---")
for step in execution_trace_1:
    print(step)

print("\n--- Final Generated Action Plan ---")
actionable_deployments = [fact for fact in final_facts_1 if fact.startswith("deploy_") or fact.startswith("request_") or fact.startswith("initiate_") or fact.startswith("establish_")]
for action in actionable_deployments:
    print(f"[ACTION REQUIRED]: {action.replace('_', ' ').title()}")
```

    === TEST CASE 1: High-Risk Ocean Quake (Magnitude 9.0) ===
    
    --- Execution Trace ---
    Fired Rule: Rule 1 -> Added Fact: 'severe_surface_shaking'
    Fired Rule: Rule 3 -> Added Fact: 'tsunami_warning_active'
    Fired Rule: Rule 5 -> Added Fact: 'mass_casualties_expected'
    Fired Rule: Rule 8 -> Added Fact: 'deploy_field_hospital'
    Fired Rule: Rule 9 -> Added Fact: 'deploy_navy_rescue_crafts'
    Fired Rule: Rule 10 -> Added Fact: 'initiate_coastal_evacuation_protocols'
    
    --- Final Generated Action Plan ---
    [ACTION REQUIRED]: Initiate Coastal Evacuation Protocols
    [ACTION REQUIRED]: Deploy Navy Rescue Crafts
    [ACTION REQUIRED]: Deploy Field Hospital



```python
print("=== TEST CASE 2: Low-Risk Land Quake (Magnitude 5.0) ===")

# Initial facts passed from our ML/Data layer
event_facts_2 = {"low_magnitude", "low_population_density", "inland"}

# Run inference
final_facts_2, execution_trace_2 = kb_engine.infer(event_facts_2)

print("\n--- Execution Trace ---")
for step in execution_trace_2:
    print(step)

print("\n--- Final Generated Action Plan ---")
actionable_deployments = [fact for fact in final_facts_2 if fact.startswith("deploy_") or fact.startswith("request_") or fact.startswith("initiate_") or fact.startswith("establish_")]
for action in actionable_deployments:
    print(f"[ACTION REQUIRED]: {action.replace('_', ' ').title()}")
```

    === TEST CASE 2: Low-Risk Land Quake (Magnitude 5.0) ===
    
    --- Execution Trace ---
    Fired Rule: Rule 2 -> Added Fact: 'predicted_threat_level_is_low'
    Fired Rule: Rule 7 -> Added Fact: 'deploy_basic_supplies'
    
    --- Final Generated Action Plan ---
    [ACTION REQUIRED]: Deploy Basic Supplies

