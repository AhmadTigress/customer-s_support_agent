"""
Simple tests for prompt_config.yaml
"""

import yaml

def test_yaml_file():
    """Test YAML file loads"""
    with open("prompt_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    assert config is not None
    print("✓ YAML loaded")

def test_basic_sections():
    """Test main sections exist"""
    with open("prompt_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    sections = ["business", "bot_identity", "system_prompt", "query_types"]
    for section in sections:
        assert section in config
    print("✓ Basic sections exist")

def test_business_info():
    """Test business info"""
    with open("prompt_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    business = config["business"]
    assert business["name"] == "TIGRESS TECH LABS"
    print("✓ Business info")

def test_bot_info():
    """Test bot info"""
    with open("prompt_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    bot = config["bot_identity"]
    assert bot["name"] == "Tigra"
    print("✓ Bot info")

def test_query_types():
    """Test query types"""
    with open("prompt_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    queries = config["query_types"]
    types = ["complaint", "technical", "sales", "general"]
    for qtype in types:
        assert qtype in queries
    print("✓ Query types")

if __name__ == "__main__":
    print("Testing YAML config...")
    
    test_yaml_file()
    test_basic_sections()
    test_business_info()
    test_bot_info()
    test_query_types()
    
    print("✓ All tests passed")