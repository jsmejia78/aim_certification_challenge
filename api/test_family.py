#!/usr/bin/env python3
"""
Simple test for family data functionality
"""

import json
import os

def test_family_data_storage():
    """Test family data storage and retrieval"""
    
    # Sample family data
    sample_family = {
        "mother_name": "Sarah Johnson",
        "father_name": "Michael Johnson",
        "number_of_kids": 2,
        "kids_names": ["Emma", "Lucas"],
        "kids_ages": [8, 5],
        "children": [
            {
                "name": "Emma",
                "age": 8,
                "strengths": ["Creative", "Empathetic", "Good reader"],
                "growth_areas": ["Organization", "Time management", "Focus"]
            },
            {
                "name": "Lucas",
                "age": 5,
                "strengths": ["Energetic", "Social", "Curious"],
                "growth_areas": ["Patience", "Following instructions", "Emotional regulation"]
            }
        ]
    }
    
    # Test file path
    family_file_path = "family_data.json"
    
    print("Testing family data storage...")
    
    # Test saving data
    try:
        with open(family_file_path, "w") as f:
            json.dump(sample_family, f, indent=2)
        print("✅ Family data saved successfully")
    except Exception as e:
        print(f"❌ Failed to save family data: {e}")
        return False
    
    # Test reading data
    try:
        if os.path.exists(family_file_path):
            with open(family_file_path, "r") as f:
                loaded_data = json.load(f)
            
            # Verify data integrity
            if (loaded_data["mother_name"] == sample_family["mother_name"] and
                loaded_data["father_name"] == sample_family["father_name"] and
                loaded_data["number_of_kids"] == sample_family["number_of_kids"]):
                print("✅ Family data loaded and verified successfully")
                print(f"   Mother: {loaded_data['mother_name']}")
                print(f"   Father: {loaded_data['father_name']}")
                print(f"   Children: {loaded_data['number_of_kids']}")
                for i, child in enumerate(loaded_data['children']):
                    print(f"   Child {i+1}: {child['name']} (age {child['age']})")
                    print(f"     Strengths: {', '.join(child['strengths'])}")
                    print(f"     Growth areas: {', '.join(child['growth_areas'])}")
                return True
            else:
                print("❌ Data integrity check failed")
                return False
        else:
            print("❌ Family data file not found after saving")
            return False
    except Exception as e:
        print(f"❌ Failed to load family data: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Family Data Functionality")
    print("=" * 40)
    
    success = test_family_data_storage()
    
    print("=" * 40)
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Some tests failed!")
    
    # Clean up test file
    if os.path.exists("family_data.json"):
        os.remove("family_data.json")
        print("🧹 Test file cleaned up")
