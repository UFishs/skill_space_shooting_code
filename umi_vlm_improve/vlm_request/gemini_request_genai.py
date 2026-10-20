import os
import base64

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from google.genai.types import HttpOptions
import json
import time

# api_key = os.getenv('GOOGLE_GEMINI_KEY')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
model_name = 'gemini-2.5-pro'
client = genai.Client(
    http_options=HttpOptions(api_version="v1"),
    vertexai=True,
    project="your-project-id",  # Replace with your Google Cloud project ID
    location="global",
)


def pil_to_bytes(image):
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()


def request_check_stage(prompt_template, prompt_content, images, images_desc):
    '''
    prompt_content: dict
    '''
    start_time = time.time()
    
    completed_prompt = prompt_template.format(
        task_desc=prompt_content['task_desc'],
        stage_desc=prompt_content['stage_desc'],
    )
        
    if "additional_information" in prompt_content and prompt_content["additional_information"] is not None:
        print("Additional information is provided.")
        completed_prompt += f"\nAdditional information: {prompt_content['additional_information']}"
        

    contents = []
    contents.append(completed_prompt)

    for id, image in enumerate(images):

        contents.append(
            types.Part.from_bytes(
                data=pil_to_bytes(image),
                mime_type='image/png'
            )
        )
        contents.append(images_desc[id])

    response = client.models.generate_content(
        model=model_name,
        contents=contents
    )
    end_time = time.time()
    time_taken = end_time - start_time
    response_text = response.text
    response_text = response_text.replace('```json', '').replace('```', '')
    json_response = json.loads(response_text)
    json_response['time_taken'] = time_taken
    return json_response

def request_new_instruction(prompt_template, prompt_content, images, images_desc, predefined_instr_dict):
    '''
    prompt_content: dict
    '''
    start_time = time.time()
    
    completed_prompt = prompt_template.format(
        task_desc=prompt_content['task_desc'],
    )

    if "additional_information" in prompt_content and prompt_content["additional_information"] is not None:
        print("Additional information is provided.")
        completed_prompt += f"\nAdditional information: {prompt_content['additional_information']}"

    removed_keys = ['s', 'f', 'd']
    used_instr_dict = {k: v for k, v in predefined_instr_dict.items() if k not in removed_keys}    
    completed_prompt += f"\nPredefined instructions: {json.dumps(used_instr_dict)}"


    contents = []
    contents.append(completed_prompt)

    for id, image in enumerate(images):

        contents.append(
            types.Part.from_bytes(
                data=pil_to_bytes(image),
                mime_type='image/png'
            )
        )
        contents.append(images_desc[id])

    response = client.models.generate_content(
        model=model_name,
        contents=contents
    )
    end_time = time.time()
    time_taken = end_time - start_time
    response_text = response.text
    response_text = response_text.replace('```json', '').replace('```', '')
    json_response = json.loads(response_text)
    json_response['time_taken'] = time_taken
    return json_response

def request_check_completion(prompt_template, prompt_content, images, images_desc):
    '''
    prompt_content: dict
    '''
    start_time = time.time()
    
    completed_prompt = prompt_template.format(
        task_desc=prompt_content['task_desc'],
    )

    if "additional_information" in prompt_content and prompt_content["additional_information"] is not None:
        print("Additional information is provided.")
        completed_prompt += f"\nAdditional information: {prompt_content['additional_information']}"


    contents = []
    contents.append(completed_prompt)

    for id, image in enumerate(images):

        contents.append(
            types.Part.from_bytes(
                data=pil_to_bytes(image),
                mime_type='image/png'
            )
        )
        contents.append(images_desc[id])

    response = client.models.generate_content(
        model=model_name,
        contents=contents
    )
    end_time = time.time()
    time_taken = end_time - start_time
    response_text = response.text
    response_text = response_text.replace('```json', '').replace('```', '')
    json_response = json.loads(response_text)
    json_response['time_taken'] = time_taken
    return json_response