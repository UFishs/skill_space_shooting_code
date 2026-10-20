
# stackthree
predefined_instr_dict = {
    "stackthree": {
        '1': 'Stack the red cube on the top of the orange cube, and then stack the yellow cube on the top of the red cube.',
        '2': 'Pick up the red cube.',
        '3': 'Place the red cube on the top of the orange cube.',
        '4': 'Pick up the yellow cube.',
        '5': 'Place the yellow cube on the top of the red cube.',
        's': 'The task is successfully completed.',
        'f': 'The task is failed, please try again.',
        'd': 'The task directly succeed.'
    },
    "your_task": {
        '1': 'Your complete task instruction here.',
        '2': 'Your fisrt subtask instruction here.',
        '3': 'Your second subtask instruction here.',
        # Add more subtasks as needed
        's': 'The task is successfully completed.',
        'f': 'The task is failed, please try again.',
        'd': 'The task directly succeed.'
    }
}

task_instruction_to_name = {
    "Stack the red cube on the top of the orange cube, and then stack the yellow cube on the top of the red cube.": "stackthree",
    "Your complete task instruction here.": "your_task"
}