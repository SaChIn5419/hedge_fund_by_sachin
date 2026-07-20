import os
import json
import sys

REGISTRY_PATH = 'data/evidence_registry.json'

def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH, 'r') as f:
        return json.load(f)

def save_registry(data):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def validate_entry(entry):
    required = [
        "experiment_id", "title", "status", "evidence_tier", 
        "competing_hypotheses", "winning_hypothesis", "evidence_evaluation",
        "acceptance_criteria", "discovery_dataset", "validation_dataset", "replication_dataset"
    ]
    for field in required:
        if field not in entry or not entry[field]:
            raise ValueError(f"Missing required field: '{field}' in {entry.get('experiment_id')}")
            
    if not isinstance(entry["competing_hypotheses"], list) or len(entry["competing_hypotheses"]) < 2:
        raise ValueError(f"'competing_hypotheses' must be a list containing at least 2 hypotheses in {entry.get('experiment_id')}")
        
    valid_statuses = ["Exploratory", "Replicated", "Established"]
    if entry["status"] not in valid_statuses:
        raise ValueError(f"Invalid status: '{entry['status']}'. Must be one of {valid_statuses}")
        
    valid_tiers = ["Level 1", "Level 2", "Level 3", "Level 4"]
    if entry["evidence_tier"] not in valid_tiers:
        raise ValueError(f"Invalid evidence tier: '{entry['evidence_tier']}'. Must be one of {valid_tiers}")

def register_experiment(entry):
    validate_entry(entry)
    registry = load_registry()
    
    # Check if exists
    for i, item in enumerate(registry):
        if item["experiment_id"] == entry["experiment_id"]:
            registry[i] = entry
            print(f"Updated existing experiment: {entry['experiment_id']}")
            save_registry(registry)
            return
            
    registry.append(entry)
    print(f"Preregistered new experiment: {entry['experiment_id']}")
    save_registry(registry)

if __name__ == '__main__':
    # Validate the file on execution
    try:
        data = load_registry()
        for item in data:
            validate_entry(item)
        print("Evidence Registry is valid and contains", len(data), "entries with competing hypotheses.")
    except Exception as e:
        print("Registry validation failed:", str(e))
        sys.exit(1)
