import json

def search_resource():
    with open("schema.json", "r") as f:
        schema = json.load(f)
    
    keywords = ["monitoring", "vertex", "ai"]
    
    for provider_name, provider_data in schema.get("provider_schemas", {}).items():
        if "google" not in provider_name:
            continue
        print(f"\nChecking provider: {provider_name}")
        resource_schemas = provider_data.get("resource_schemas", {})
        
        matches = []
        for resource_name in resource_schemas.keys():
            # Check if all keywords are in the resource name, or any of them
            if any(kw in resource_name for kw in keywords):
                matches.append(resource_name)
        
        print(f"Found {len(matches)} potential resource matches:")
        for match in sorted(matches):
            if "monitoring" in match or "vertex" in match:
                print(f"  - {match}")

if __name__ == "__main__":
    search_resource()
