    Skills removed from robot_skills
    {
      "id": "move_roller_to_workplace",
      "name": "Move Roller to Workplace",
      "description": "Pick up the roller from storage and place it at the workplace position",
      "category": "program",
      "api_call": {"method": "POST", "endpoint": "/api/program/execute", "body": {"program": "move_roller_from_storage_to_workplace.prog"}},
      "preconditions": {
        "roller.location": "storage_area"
      },
      "effects": {
        "roller.location": "workplace"
      },
      "trigger_steps": ["consolidate_with_roller"],
      "estimated_duration_sec": 15.0,
      "can_interrupt": false
    },
    {
      "id": "return_roller_to_storage",
      "name": "Return Roller to Storage",
      "description": "Pick up the roller from the workplace and return it to the storage area",
      "category": "program",
      "api_call": {"method": "POST", "endpoint": "/api/program/execute", "body": {"program": "move_roller_from_workplace_to_storage.prog"}},
      "preconditions": {
        "roller.location": "workplace"
      },
      "effects": {
        "roller.location": "storage_area"
      },
      "trigger_after_steps": ["consolidate_with_roller"],
      "estimated_duration_sec": 15.0,
      "can_interrupt": false
    },