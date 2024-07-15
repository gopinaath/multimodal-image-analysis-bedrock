import json
import logging
import base64
import boto3

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
max_tokens = 2000
prompt_text_identify_objects = "Accurately identify each object, and the object's location in this image. If there are multiple objects of the same type, identify each separately. After identifying, doublecheck if each of the identified object is present the given picture. If it's not present, remove that object from the identified objects. Ignore architectural features like floor, wall, etc. Give response in JSON format.  Use double quotes for constructing json objects. Don't add newline characters. "

prompt_text_identify_objects = prompt_text_identify_objects + "Sample JSON format: {\"objects\": {\"Object-1 Name\": \"Object-1 location\", \"Object-2 name\": \"Object-2 location\"}}.  This is only a sample JSON document with two placeholder items. You will likely have more objects in the image. "

prompt_text_validate_objects = "Answer if the given object can be identified in the image.  If the identified object is present in the image, respond with 'Yes'. If the identified objects are not present in the image, respond with 'No'. If you are unsure, respond with 'Unsure'. Give response in JSON format.  Use double quotes for constructing json objects. Don't add newline characters.  Does the image contain "

def run_multi_modal_prompt(bedrock_runtime, messages, max_tokens):
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
             "messages": messages
        }
    )
    response = bedrock_runtime.invoke_model(
        body=body, modelId=model_id, )
    response_body = json.loads(response.get('body').read())
    return response_body

def get_objects_from_model(input_image, prompt_text):
    try:
        bedrock_runtime = boto3.client(service_name='bedrock-runtime')
        with open(input_image, "rb") as image_file:
            content_image = base64.b64encode(image_file.read()).decode('utf8')

        message = {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": content_image}},
                {"type": "text", "text": prompt_text}
            ]
        }

        messages = [message]
        response = run_multi_modal_prompt(bedrock_runtime, messages, max_tokens)
        logger.debug(response)

        content = response["content"]
        first_item = content[0]
        text = first_item["text"]

        # Extract the JSON string from the text
        start = text.index("{")
        end = text.rindex("}") + 1
        json_string = text[start:end]
        logger.debug(json_string)

        text_json = json.loads(json_string)
        objects = text_json["objects"]
        return objects
    except Exception as e:
        logger.error("Exception occurred: %s", e)   
        return None

def validate_objects(input_image, prompt_text):
    try:
        bedrock_runtime = boto3.client(service_name='bedrock-runtime')

        with open(input_image, "rb") as image_file:
            content_image = base64.b64encode(image_file.read()).decode('utf8')

        message = {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": content_image}},
                {"type": "text", "text": prompt_text}
            ]
        }

        messages = [message]

        response = run_multi_modal_prompt(bedrock_runtime, messages, max_tokens)
        logger.debug(response)

        content = response["content"]
        logger.debug(content)
        return content
    except Exception as e:
        logger.error("Exception occurred: %s", e)   
        return None
    
def main():
    input_image_1 = "./sample-image.jpg"

    # sometimes the JSON output from LLM can be inconsistent. Retry 3 times before giving up
    retry_count = 0
    while retry_count < 3:
        try:
            objects = get_objects_from_model(input_image_1, prompt_text_identify_objects)
            pretty_json_objects_image_1 = json.dumps(objects, indent=4)
            logger.info(pretty_json_objects_image_1)
            break  
        except ValueError as e:
            logger.error(f"Error invoking model: {e}")
            retry_count += 1
            if retry_count == 3:
                logger.error("Maximum retries reached. Exiting.")
                return  
                
    keys = list(objects.keys())
    logger.info(keys)
    key_list_to_be_removed = []

    for key in objects:
        validation_result = validate_objects(input_image_1, prompt_text_validate_objects + str(key) + " ? ")

        first_item = validation_result[0]
        text_field = first_item["text"]
        validation_result_inner_dict = json.loads(text_field)
        # response is sometimes called "response" and sometimes called "answer"
        response_value = validation_result_inner_dict.get("response", validation_result_inner_dict.get("answer", None))
        
        if(not (response_value == "Yes")):
            key_list_to_be_removed.append(key)
    
    logger.info(key_list_to_be_removed)
    logger.info("Objects to be removed :" + json.dumps(objects, indent=4))

    for key in key_list_to_be_removed:
        objects.pop(key)

    logger.info(json.dumps(objects, indent=4))

if __name__ == "__main__":
    main()
