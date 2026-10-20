check_completion_prompt = '''
Target:
You are a professional task analyzer. You should analyze the provided frame and the task description. Given the frame at this timestamp, you need to determine whether the whole task is completed or not. You always need to give clear evidence to support your conclusion. 

Analysis requirements:
1. Watch the frame carefully, and the task description.
2. Determine whether the whole task is completed or not.
3. Your output must **strictly follow the Return Format below** and **only output** four lines (field name and value). Do not add any additional commentary.


Return Format (use json format):
{{
    "completed": ["Yes" or "No", whether the task is completed or not]
    "evidence": Explanation of the key evidence that led you to this conclusion
}}

Task Description:
{task_desc}
'''

check_stage_repair_prompt = '''
Target:
You are a professional task stage analyzer. You should analyze the provided frame, the whole task description and a specific stage of the task. Given the frame at this timestamp, you need to determine whether the specific stage of the task is completed or not. You always need to give clear evidence to support your conclusion.

Analysis requirements:
1. Watch the frame carefully, the whole task description, and the specific stage of the task.
2. Determine whether the specific stage of the task is completed or not. Remember, the specific stage of the task is a part of the whole task, you don't need to consider the whole task.
3. Your output must **strictly follow the Return Format below** and **only output** four lines (field name and value). Do not add any additional commentary.


Return Format (use json format):
{{
    "completed": ["Yes" or "No", whether the stage is completed or not]
    "evidence": Explanation of the key evidence that led you to this conclusion
}}

Task Description:
{task_desc}

Specific Stage Description:
{stage_desc}
'''

new_instruction_prompt = """
Target:
You are a professional robot task executor. You should analyze the provided frame and the task description. Given the frame at this timestamp, you need to select the correct skill that, if executed correctly, would have led to successful task completion. You also need to provide clear evidence to support your conclusion. 

Analysis requirements:
1. Watch the frame carefully and the task description.
2. You need to select the correct skill that, if executed correctly, would have led to successful task completion. You can only choose one skill from the list below.
3. Notice that you will be given not only the current frame, but also the beginning frame and some history frames. You need to consider the history and give a new skill selection. For example, if the previous skill selection is "Pick up the red cube.", and the current frame shows that the gripper has successfully picked up the red cube, then the new skill selection should be "Place the red cube on the top of the orange cube."
4. If additional information is provided, you should also consider the additional information when making your decision. The additional information may provide some hints about the task, such as the order of actions, the location of objects, and the goal of the task. You should use this information to make a more informed decision.
5. Your output must **strictly follow the Return Format below**. Do not add any additional commentary.


Return Format (use json format):
{{
    "next_skill_id": [the id of the next skill that would have led to successful task completion, e.g., 1, 2, 3, ..., indicating which skill to execute next]
    "evidence": Explanation of the key evidence that led you to this percentage score
    "observation": Key observation of all the frames provided. You should describe what you see in the frames, such as the position of all the relevant objects.
}}

Task Description:
{task_desc}
"""