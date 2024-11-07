import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

app = Flask(__name__)

# Gemini API credentials (ensure this is set)
genai.configure(api_key=os.environ["GEMINI_API_KEY"]) # or replace with your key directly, but less secure.


# Clarifai API credentials (replace with your actual credentials)
PAT = '84002ab0dabe47e094afc87e9bcb9c06'  # Or replace with your key directly, but less secure.
USER_ID = 'openai'
APP_ID = 'chat-completion'
MODEL_ID = 'openai-gpt-4-vision'
MODEL_VERSION_ID = '266df29bc09843e0aee9b7bf723c03c2'

# Gemini Model Configuration
generation_config = {
    "temperature": 0.2,  # Adjust as needed
    "max_output_tokens": 8192,  # Adjust as needed
}


@app.route('/')
def index():
    return render_template('index.html')

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-002",  # or the model you prefer
    generation_config=generation_config,
)

# Set up Clarifai gRPC client
channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)
metadata = (('authorization', 'Key ' + PAT),)
userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return render_template('error.html', error_message="No image uploaded")

    image = request.files['image']
    try:
        image_data = image.read()
        image_input = resources_pb2.Input(
            data=resources_pb2.Data(
                image=resources_pb2.Image(base64=image_data)
            )
        )

        post_model_outputs_response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=userDataObject,
                model_id=MODEL_ID,
                version_id=MODEL_VERSION_ID,  # Corrected: Added version_id
                inputs=[image_input]
            ),
            metadata=metadata
        )

        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            raise Exception(
                f"Clarifai API error: {post_model_outputs_response.status.description}"
            )

        clarifai_result = post_model_outputs_response.outputs[0].data.text.raw

        prompt = f"""
    Analyze the following image description from open-ai-gpt4-vision and determine if it pertains to waste materials. If it is related to waste, answer as follows. Otherwise, respond, "I'm not instructed to answer questions unrelated to waste."

    If relevant, provide concise answers for each of these points:
    
    * **Types of waste:** Identify the specific waste materials present in the image.
    * **Degradability:** Briefly state how long each type of waste takes to decompose.
    * **Disposal/Recycling:** Suggest how to responsibly dispose of or recycle each waste type.
    
    Image description:
    ```
    {clarifai_result}
    ```
    """


        chat_session = model.start_chat()
        gemini_response = chat_session.send_message(prompt).text

        return render_template(
            'results.html',
            clarifai_analysis=clarifai_result,
            gemini_response=gemini_response
        )

    except Exception as e:
        return render_template('error.html', error_message=str(e))


if __name__ == '__main__':
    app.run(debug=True)