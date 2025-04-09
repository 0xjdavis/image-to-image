import os
import streamlit as st
import numpy as np
from PIL import Image
import io
import openai
import requests
import json
import datetime
import time
import uuid
import base64
from openai import OpenAI

# Initialize session state for feedback and image analysis
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'last_feedback_id' not in st.session_state:
    st.session_state.last_feedback_id = None
if 'image_analysis' not in st.session_state:
    st.session_state.image_analysis = None

# Setting page config
st.set_page_config(
    page_title="AI Image Generator",
    layout="wide"
)

# Sidebar content
with st.sidebar:
    st.title("AI Image Generator")
    st.markdown("Upload an image, then instruct the model to modify it and create variations.")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display thumbnail in sidebar
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=250)
    
    # Prompt input
    st.subheader("Image Processing")
    prompt = st.text_area("Instructions:", height=100, 
                     placeholder="e.g., 'Transform this into a watercolor painting' or 'Create a variation with more vibrant colors'")
    
    # Model selection and quality controls
    st.subheader("Model & Quality Settings")
    
    model_option = st.selectbox(
        "Model", 
        ["gpt-4o-mini", "gpt-4o", "dall-e-3", "dall-e-2"],
        help="GPT-4o and DALL-E 3 offer higher quality but fewer variation options"
    )
    
    # Show appropriate settings based on model selection
    if model_option in ["dall-e-3", "gpt-4o", "gpt-4o-mini"]:
        image_size = st.selectbox(
            "Image Size", 
            ["1024x1024", "1792x1024", "1024x1792"],
            help="Larger sizes may produce more detailed images"
        )
        
        quality_setting = st.radio(
            "Quality Level",
            ["standard", "hd"],
            help="HD quality produces more detailed images but costs more credits"
        )
        
        style_modifier = st.selectbox(
            "Style",
            [
                "vivid",
                "natural"
            ],
            help="Vivid creates hyper-real images, Natural creates more realistic images"
        )
    else:
        image_size = st.selectbox(
            "Image Size", 
            ["1024x1024", "512x512", "256x256"],
            help="Larger sizes may produce more detailed images"
        )
    
    # Advanced options
    with st.expander("Advanced Options"):
        enable_vision_analysis = st.checkbox("Enable AI Vision Analysis", value=True,
                                        help="Use GPT-4o Vision to analyze the image and improve preservation")
        
        # AI Vision analysis options
        if enable_vision_analysis:
            vision_detail_level = st.radio(
                "Vision Analysis Detail",
                ["basic", "detailed"],
                help="Detailed analysis takes longer but improves preservation"
            )
            
            # Preservation options
            preservation_level = st.slider("Preservation Strictness", min_value=1, max_value=10, value=8,
                                         help="Higher values preserve more of the original image")
    
    # API Key input
    api_key = st.text_input("OpenAI API Key:", type="password")
    
    # Action buttons in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        edit_button = st.button("Edit Image")
    with col2:
        variation_button = st.button("Create Variation")
    with col3:
        enhanced_variation = st.button("Enhanced Variation")

# Main content area
st.title("AI Image Generator & Variation Creator")

# Direct feedback storage function with debugging
def save_feedback_data(feedback_type, prompt, model, quality, style, size):
    """
    Direct feedback storage function that ensures reliable data saving with debugging
    """
    # Generate a unique ID for this feedback
    unique_id = str(uuid.uuid4())
    st.session_state.last_feedback_id = unique_id
    
    # Create a timestamp for this feedback entry
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the feedback data structure
    feedback_entry = {
        "id": f"{feedback_type}_{timestamp}_{unique_id[:8]}",
        "uuid": unique_id,
        "timestamp": timestamp,
        "feedback_type": feedback_type,
        "prompt": prompt,
        "model": model,
        "quality": quality,
        "style": style, 
        "size": size
    }
    
    # Define fixed feedback directory path
    feedback_dir = "/Users/admin/Sites/vibe-coding/Streamlit-Inpaint/feedback_data"
    
    # Debug message - print to console for tracking
    print(f"DEBUG: Saving feedback to {feedback_dir}")
    print(f"DEBUG: Feedback data: {json.dumps(feedback_entry)}")
    
    try:
        # Ensure the directory exists
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Central feedback file path
        central_filepath = os.path.join(feedback_dir, "feedback.json")
        print(f"DEBUG: Central filepath: {central_filepath}")
        
        # Initialize new feedback list
        all_feedback = []
        
        # Load existing feedback if file exists
        if os.path.exists(central_filepath):
            try:
                with open(central_filepath, "r") as f:
                    content = f.read().strip()
                    if content:
                        all_feedback = json.loads(content)
                        print(f"DEBUG: Loaded {len(all_feedback)} existing entries")
                    else:
                        print("DEBUG: Existing file was empty")
                        all_feedback = []
                        
                # Ensure all_feedback is a list
                if not isinstance(all_feedback, list):
                    all_feedback = [all_feedback]
                    
            except Exception as e:
                print(f"DEBUG: Error reading existing file: {str(e)}")
                # Start fresh if there's an issue
                all_feedback = []
        else:
            print("DEBUG: No existing feedback.json file")
        
        # Add the new feedback entry
        all_feedback.append(feedback_entry)
        print(f"DEBUG: New total entries: {len(all_feedback)}")
        
        # Write to the central file
        with open(central_filepath, "w") as f:
            json.dump(all_feedback, f, indent=2)
            print(f"DEBUG: Successfully wrote to central file")
        
        # Also save individual file for redundancy
        individual_filename = f"feedback_{feedback_type}_{timestamp}_{unique_id[:8]}.json"
        individual_filepath = os.path.join(feedback_dir, individual_filename)
        
        # Write to individual file
        with open(individual_filepath, "w") as f:
            json.dump(feedback_entry, f, indent=2)
            print(f"DEBUG: Successfully wrote individual file: {individual_filename}")
        
        # Mark as submitted in session state
        st.session_state.feedback_submitted = True
        
        # Return success
        return True
        
    except Exception as e:
        # Detailed error handling
        error_msg = f"Error saving feedback: {str(e)}"
        print(f"DEBUG ERROR: {error_msg}")
        print(f"DEBUG ERROR: Type: {type(e)}")
        print(f"DEBUG ERROR: Current directory: {os.getcwd()}")
        st.error(error_msg)
        return False


# Streamlined function to read feedback data
def read_feedback_data():
    """
    Simple function to read feedback data from the feedback directory
    """
    feedback_dir = "/Users/admin/Sites/vibe-coding/Streamlit-Inpaint/feedback_data"
    central_filepath = os.path.join(feedback_dir, "feedback.json")
    all_feedback = []
    
    # Check if central feedback file exists
    if os.path.exists(central_filepath):
        try:
            with open(central_filepath, "r") as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
                    if isinstance(data, list):
                        all_feedback.extend(data)
                    else:
                        all_feedback.append(data)
        except Exception as e:
            print(f"Error reading central feedback file: {str(e)}")
    
    # Also check individual feedback files as backup
    if os.path.exists(feedback_dir) and os.path.isdir(feedback_dir):
        json_files = [f for f in os.listdir(feedback_dir) if f.endswith('.json') and f != "feedback.json"]
        
        for file in json_files:
            try:
                with open(os.path.join(feedback_dir, file), 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_feedback.extend(data)
                    else:
                        all_feedback.append(data)
            except Exception as e:
                print(f"Error reading {file}: {str(e)}")
    
    # Remove duplicates based on id
    unique_ids = set()
    unique_feedback = []
    
    for entry in all_feedback:
        entry_id = entry.get("id", None)
        if entry_id and entry_id not in unique_ids:
            unique_ids.add(entry_id)
            unique_feedback.append(entry)
        elif not entry_id:
            # Keep entries without IDs too
            unique_feedback.append(entry)
    
    return unique_feedback


# Get API key from input or environment variable
def get_api_key():
    if api_key:
        return api_key
    elif "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]
    return None


# AI VISION ANALYSIS FUNCTIONS

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string for API calls"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def analyze_image_with_vision(image, detail_level="basic"):
    """
    Analyze the image using GPT-4o Vision to extract key features and details
    
    Args:
        image: PIL Image object
        detail_level: Level of analysis detail (basic, detailed)
    
    Returns:
        dict: Analysis results including objects, colors, composition details
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=get_api_key())
        
        # Encode the image to base64
        base64_image = encode_image_to_base64(image)
        
        # Prepare system message based on detail level
        if detail_level == "detailed":
            system_message = """
            You are an expert computer vision system designed to analyze images in detail.
            Please provide a comprehensive analysis of this image, including:
            1. All objects and their positions (be very specific about locations)
            2. Detailed color palette with specific colors and their locations
            3. Lighting conditions and direction
            4. Spatial relationships between objects
            5. Composition and perspective details
            6. Any unique or distinctive elements that must be preserved
            Format your response as a structured JSON with these categories.
            """
        else:  # basic
            system_message = """
            You are an expert computer vision system designed to analyze images.
            Please provide a basic analysis of this image, including:
            1. Main objects and their general positions
            2. Primary colors used
            3. Basic lighting conditions
            4. Key elements that should be preserved in any variation
            Format your response as a structured JSON with these categories.
            """
        
        # Call the API with the image
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": [
                    {"type": "text", "text": "Please analyze this image thoroughly:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            response_format={"type": "json_object"}
        )
        
        # Extract and parse the JSON response
        analysis_text = response.choices[0].message.content
        analysis = json.loads(analysis_text)
        
        # Log the analysis for debugging
        print(f"Image analysis completed: {len(str(analysis))} characters of data")
        
        return analysis
        
    except Exception as e:
        error_msg = f"Error analyzing image with GPT-4o Vision: {str(e)}"
        print(f"ERROR: {error_msg}")
        st.error(error_msg)
        return None


def generate_enhanced_prompt(original_prompt, image_analysis, preservation_level=8):
    """
    Generate an enhanced prompt that includes the image analysis to better preserve
    key elements while making the requested changes.
    
    Args:
        original_prompt: User's original prompt
        image_analysis: Results from the vision analysis
        preservation_level: How strictly to preserve original elements (1-10)
    
    Returns:
        str: Enhanced prompt for the image generation model
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=get_api_key())
        
        # Create a system message that emphasizes preservation
        preservation_instruction = ""
        if preservation_level >= 8:
            preservation_instruction = "EXTREMELY important to maintain precise fidelity"
        elif preservation_level >= 5:
            preservation_instruction = "important to maintain good fidelity"
        else:
            preservation_instruction = "maintain reasonable fidelity"
        
        system_message = f"""
        You are an expert prompt engineer for image generation models.
        Your task is to enhance the user's prompt by incorporating the detailed image analysis,
        ensuring the original image elements are preserved while making only the specific changes requested.
        It is {preservation_instruction} to the original image in all aspects EXCEPT what the user explicitly
        wants to change. Be extremely specific about what should stay the same.
        """
        
        # Call the API to generate an enhanced prompt
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"""
                Original user prompt: "{original_prompt}"
                
                Image analysis: {json.dumps(image_analysis, indent=2)}
                
                Generate an enhanced prompt that will preserve all the key elements of the original image
                while making ONLY the changes requested in the original prompt. Be extremely specific about
                what should be preserved exactly as is, and what should be changed.
                Focus on accurate placement, composition, and details of the original.
                """}
            ],
            max_tokens=1000
        )
        
        # Extract the enhanced prompt
        enhanced_prompt = response.choices[0].message.content
        
        # Log the enhanced prompt for debugging
        print(f"Enhanced prompt generated ({len(enhanced_prompt)} chars): {enhanced_prompt[:100]}...")
        
        return enhanced_prompt
        
    except Exception as e:
        error_msg = f"Error generating enhanced prompt: {str(e)}"
        print(f"ERROR: {error_msg}")
        return f"Starting with the exact image provided, {original_prompt}. Keep everything else precisely the same."


# Function to generate image with GPT-4o-mini
def generate_image_with_gpt4o_mini(prompt_text, size="1024x1024", quality="standard", style="vivid"):
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=get_api_key())
        
        # Log API key status (not the actual key)
        print(f"API Key present: {get_api_key() is not None}")
        print(f"Using model: gpt-4o-mini")
        print(f"Prompt being sent: {prompt_text}")
        
        # Create a new image based on the prompt using GPT-4o
        print("Sending request to OpenAI API...")
        
        # Use the Chat API with GPT-4o-mini, including a system message for image generation
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates images based on user descriptions."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Generate an image based on this description: {prompt_text}"}
                ]}
            ],
            max_tokens=1000
        )
        
        # Extract the image URL from the response
        print(f"Response received from GPT-4o-mini")
        
        # In a real implementation, we would extract the image URL from the response
        # For now, this is a placeholder until the API details are fully available
        image_url = response.choices[0].message.content
        
        # If the response contains an actual image URL, return it
        # Otherwise, fall back to DALL-E 3 for now
        if "http" in image_url and (".png" in image_url.lower() or ".jpg" in image_url.lower() or ".jpeg" in image_url.lower()):
            return image_url
        else:
            print("GPT-4o-mini did not return a direct image URL. Falling back to DALL-E 3.")
            return generate_image_with_dalle3(prompt_text, size, quality, style)
        
    except Exception as e:
        error_msg = f"Error generating image with GPT-4o-mini: {str(e)}"
        print(f"ERROR: {error_msg}")
        st.error(error_msg)
        
        # Fall back to DALL-E 3 if GPT-4o-mini fails
        print("Falling back to DALL-E 3 after GPT-4o-mini error.")
        try:
            return generate_image_with_dalle3(prompt_text, size, quality, style)
        except Exception as e2:
            error_msg2 = f"Error with fallback to DALL-E 3: {str(e2)}"
            print(f"FALLBACK ERROR: {error_msg2}")
            st.error(error_msg2)
            return None


# Function to generate image with GPT-4o
def generate_image_with_gpt4o(prompt_text, size="1024x1024", quality="standard", style="vivid", image_data=None):
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=get_api_key())
        
        # Log API key status (not the actual key)
        print(f"API Key present: {get_api_key() is not None}")
        print(f"Using model: gpt-4o")
        print(f"Prompt being sent: {prompt_text}")
        
        # Prepare messages depending on whether we have an image
        if image_data:
            # Encode the image to base64
            base64_image = encode_image_to_base64(image_data)
            
            # Use the Chat API with GPT-4o, including the image
            messages = [
                {"role": "system", "content": "You are an expert AI image generator that creates high-quality images based on descriptions while precisely maintaining all elements of reference images except what is explicitly requested to change."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Generate an image based on this description: {prompt_text}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        else:
            # Text-only prompt
            messages = [
                {"role": "system", "content": "You are an expert AI image generator that creates high-quality images based on descriptions."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Generate an image based on this description: {prompt_text}"}
                ]}
            ]
        
        # Create a new image based on the prompt using GPT-4o
        print("Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )
        
        # Extract the image URL from the response
        print(f"Response received from GPT-4o")
        
        # In a real implementation, we would extract the image URL from the response
        # For now, this is a placeholder until the API details are fully available
        image_url = response.choices[0].message.content
        
        # If the response contains an actual image URL, return it
        # Otherwise, fall back to DALL-E 3 for now
        if "http" in image_url and (".png" in image_url.lower() or ".jpg" in image_url.lower() or ".jpeg" in image_url.lower()):
            return image_url
        else:
            print("GPT-4o did not return a direct image URL. Falling back to DALL-E 3.")
            return generate_image_with_dalle3(prompt_text, size, quality, style)
        
    except Exception as e:
        error_msg = f"Error generating image with GPT-4o: {str(e)}"
        print(f"ERROR: {error_msg}")
        st.error(error_msg)
        
        # Fall back to DALL-E 3 if GPT-4o fails
        print("Falling back to DALL-E 3 after GPT-4o error.")
        try:
            return generate_image_with_dalle3(prompt_text, size, quality, style)
        except Exception as e2:
            error_msg2 = f"Error with fallback to DALL-E 3: {str(e2)}"
            print(f"FALLBACK ERROR: {error_msg2}")
            st.error(error_msg2)
            return None


# Function to generate image with DALL-E 3
def generate_image_with_dalle3(prompt_text, size="1024x1024", quality="standard", style="vivid"):
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=get_api_key())
        
        # Log API key status (not the actual key)
        print(f"API Key present: {get_api_key() is not None}")
        print(f"Using model: dall-e-3")
        print(f"Prompt being sent: {prompt_text}")
        
        # Create a new image based on the prompt
        print("Sending request to OpenAI API...")
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt_text,
            n=1,
            size=size,
            quality=quality,
            style=style
        )
        
        # Log response info
        print(f"Response received with status")
        print(f"Generated image URL: {response.data[0].url}")
        
        # Return the URL of the generated image
        return response.data[0].url
    
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        print(f"ERROR: {error_msg}")
        st.error(error_msg)
        return None


# Function to generate image variation with DALL-E 2
def generate_image_variation_dalle2(original_image, size="1024x1024"):
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=get_api_key())
        
        # Log info
        print(f"API Key present: {get_api_key() is not None}")
        print(f"Using model: dall-e-2 for variation")
        
        # Prepare the image for API - resize and convert to PNG
        # DALL-E variation requires square images
        width, height = [int(x) for x in size.split("x")]
        resized_image = original_image.resize((width, height))
        buffered = io.BytesIO()
        resized_image.save(buffered, format="PNG")
        print(f"Image prepared: size {resized_image.size}")
        
        print("Sending variation request to OpenAI API...")
        
        # Create a variation of the image
        response = client.images.create_variation(
            image=io.BytesIO(buffered.getvalue()),
            n=1,
            size=size
        )
        
        # Log response info
        print(f"Response received with status")
        print(f"Generated variation URL: {response.data[0].url}")
        
        # Return the URL of the generated image
        return response.data[0].url
    
    except Exception as e:
        error_msg = f"Error generating variation: {str(e)}"
        print(f"ERROR: {error_msg}")
        st.error(error_msg)
        return None


# Enhanced variation function that uses GPT-4o Vision for analysis
def generate_vision_enhanced_variation(original_image, prompt_text, model="gpt-4o", 
                                      size="1024x1024", quality="standard", style="vivid",
                                      detail_level="basic", preservation_level=8):
    try:
        # Step 1: Analyze the image with GPT-4o Vision
        with st.spinner("Analyzing image with AI Vision..."):
            image_analysis = analyze_image_with_vision(original_image, detail_level)
            if not image_analysis:
                st.warning("Image analysis failed. Proceeding with basic enhancement.")
                image_analysis = {"objects": "unknown", "colors": "unknown", "lighting": "unknown", "key_elements": "all elements"}
            
            # Store the analysis in session state for reuse
            st.session_state.image_analysis = image_analysis
        
        # Step 2: Generate an enhanced prompt using the analysis
        with st.spinner("Crafting enhanced preservation prompt..."):
            enhanced_prompt = generate_enhanced_prompt(prompt_text, image_analysis, preservation_level)
        
        # Step 3: Generate the image with the enhanced prompt
        with st.spinner(f"Generating enhanced image with {model}..."):
            if model == "gpt-4o":
                # Pass both the enhanced prompt and the original image
                return generate_image_with_gpt4o(enhanced_prompt, size, quality, style, image_data=original_image)
            elif model == "gpt-4o-mini":
                return generate_image_with_gpt4o_mini(enhanced_prompt, size, quality, style)
            else:
                # Default to DALL-E 3
                return generate_image_with_dalle3(enhanced_prompt, size, quality, style)
        
    except Exception as e:
        error_msg = f"Error generating vision-enhanced variation: {str(e)}"
        print(f"ERROR: {error_msg}")
        st.error(error_msg)
        return None


# Function to generate image with edits using DALL-E 2
def generate_image_with_dalle2_edit(original_image, prompt_text, size="1024x1024"):
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=get_api_key())
        
        # Log info
        print(f"API Key present: {get_api_key() is not None}")
        print(f"Using model: dall-e-2 for edit")
        print(f"Prompt being sent: {prompt_text}")
        
        # Prepare the image for API - convert to RGBA (with alpha channel)
        rgba_image = original_image.convert("RGBA")
        buffered = io.BytesIO()
        rgba_image.save(buffered, format="PNG")
        print(f"Image prepared: size {rgba_image.size}, format {rgba_image.format}")
        
        # Create a new image based on the input image and prompt
        print("Sending request to OpenAI API...")
        response = client.images.edit(
            image=io.BytesIO(buffered.getvalue()),
            prompt=prompt_text,
            n=1,
            size=size
        )
        
        # Log response info
        print(f"Response received with status")
        print(f"Generated image URL: {response.data[0].url}")
        
        # Return the URL of the generated image
        return response.data[0].url
    
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        print(f"ERROR: {error_msg}")
        st.error(error_msg)
        return None


# Two-step approach (DALL-E 2 variation + model enhancement)
def two_step_variation(original_image, prompt_text, model="dall-e-3", size="1024x1024", quality="standard", style="vivid"):
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=get_api_key())
        
        # Step 1: Create a basic variation with DALL-E 2
        print("Step 1: Creating variation with DALL-E 2")
        
        # Prepare the image for API
        width, height = [int(x) for x in size.split("x")]
        resized_image = original_image.resize((width, height))
        buffered = io.BytesIO()
        resized_image.save(buffered, format="PNG")
        
        variation_response = client.images.create_variation(
            image=io.BytesIO(buffered.getvalue()),
            n=1,
            size=size
        )
        
        # Download the variation
        variation_url = variation_response.data[0].url
        print(f"DALL-E 2 variation created: {variation_url}")
        response = requests.get(variation_url)
        variation_image = Image.open(io.BytesIO(response.content))
        
        # Step 2: Use the selected model to enhance it with the provided prompt
        print(f"Step 2: Enhancing with {model}")
        
        # Use model with precision-focused prompt
        enhanced_prompt = (
            f"Make only the following specific change: {prompt_text}. "
            f"IMPORTANT: Maintain ALL other elements of the original scene exactly as they are - "
            f"including the exact same lighting, colors, perspective, background details, "
            f"desk placement, and all other objects. This is a focused modification task, not a creative reimagining."
        )
        
        print(f"Enhanced prompt being sent: {enhanced_prompt}")
        
        if model == "gpt-4o":
            final_url = generate_image_with_gpt4o(enhanced_prompt, size, quality, style)
        else:
            # Default to DALL-E 3
            final_url = generate_image_with_dalle3(enhanced_prompt, size, quality, style)
        
        print(f"Final enhanced image URL: {final_url}")
        return final_url
    
    except Exception as e:
        error_msg = f"Error in two-step variation: {str(e)}"
        print(f"ERROR: {error_msg}")
        st.error(error_msg)
        return None


# Create columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    if uploaded_file is not None:
        # Display original image full size
        st.subheader("Original Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original", use_container_width=True)

with col2:
    # Check if any button was clicked
    if uploaded_file is not None and (edit_button or variation_button or enhanced_variation):
        # Check for API key
        if not get_api_key():
            st.error("Please provide an OpenAI API key to generate images.")
        elif not prompt and (edit_button or enhanced_variation):
            st.info("Please enter instructions to guide the image generation.")
        else:
            # Extract parameters based on model selection
            selected_size = image_size
            
            if model_option in ["dall-e-3", "gpt-4o", "gpt-4o-mini"]:
                selected_quality = quality_setting
                selected_style = style_modifier
            else:
                selected_quality = "standard"  # Default for DALL-E 2
                selected_style = "vivid"  # Default for DALL-E 2
            
            # Get advanced options if enabled
            use_vision_analysis = False
            vision_detail = "basic"
            preservation_strictness = 8
            
            # Check if advanced options are enabled
            if "enable_vision_analysis" in st.session_state:
                use_vision_analysis = st.session_state.enable_vision_analysis
                
                if use_vision_analysis:
                    vision_detail = st.session_state.vision_detail_level
                    preservation_strictness = st.session_state.preservation_level
            
            # Handle edit button
            if edit_button:
                with st.spinner(f"Generating edited image with {model_option}..."):
                    # Open the image
                    image = Image.open(uploaded_file).convert("RGB")
                    
                    if use_vision_analysis and model_option in ["gpt-4o", "dall-e-3"]:
                        # Use vision-enhanced generation
                        generated_image_url = generate_vision_enhanced_variation(
                            image,
                            prompt,
                            model=model_option,
                            size=selected_size,
                            quality=selected_quality,
                            style=selected_style,
                            detail_level=vision_detail,
                            preservation_level=preservation_strictness
                        )
                    elif model_option == "gpt-4o":
                        # Use GPT-4o with specialized edit prompt
                        edit_prompt = f"Edit this image with the following instructions: {prompt}. Keep everything else EXACTLY the same."
                        generated_image_url = generate_image_with_gpt4o(
                            edit_prompt, 
                            size=selected_size, 
                            quality=selected_quality, 
                            style=selected_style,
                            image_data=image
                        )
                    elif model_option == "dall-e-3":
                        # Use DALL-E 3 with specialized edit prompt
                        edit_prompt = f"Edit this image with the following instructions: {prompt}. Keep everything else EXACTLY the same, including colors, lighting, layout, and objects."
                        generated_image_url = generate_image_with_dalle3(
                            edit_prompt, 
                            size=selected_size, 
                            quality=selected_quality, 
                            style=selected_style
                        )
                    else:
                        # Use DALL-E 2 edit function
                        generated_image_url = generate_image_with_dalle2_edit(
                            image, 
                            prompt, 
                            size=selected_size
                        )
                    
                    if generated_image_url:
                        st.subheader("Edited Image")
                        st.image(generated_image_url, caption=f"Modified by {model_option}", use_container_width=True)
                        
                        # Add download link
                        st.markdown(f"[Download edited image]({generated_image_url})")
                        
                        # Display analysis if it was performed
                        if use_vision_analysis and "image_analysis" in st.session_state and st.session_state.image_analysis:
                            with st.expander("View AI Vision Analysis"):
                                st.json(st.session_state.image_analysis)
                        
                        # Feedback mechanism
                        st.write("### Was this edit accurate to your request?")
                        
                        # Callback functions for feedback buttons
                        def save_positive_edit():
                            success = save_feedback_data(
                                feedback_type="positive_edit", 
                                prompt=prompt, 
                                model=model_option, 
                                quality=selected_quality, 
                                style=selected_style, 
                                size=selected_size
                            )
                            return success
                            
                        def save_negative_edit():
                            success = save_feedback_data(
                                feedback_type="negative_edit", 
                                prompt=prompt, 
                                model=model_option, 
                                quality=selected_quality, 
                                style=selected_style, 
                                size=selected_size
                            )
                            return success
                            
                        # Create unique keys for the buttons
                        feedback_col1, feedback_col2 = st.columns(2)
                        
                        with feedback_col1:
                            # Use on_click parameter for callback function
                            st.button("👍 Yes, good result", 
                                     key="edit_positive_feedback", 
                                     on_click=save_positive_edit)
                            
                            # Show a success message if feedback was submitted
                            if st.session_state.feedback_submitted and st.session_state.last_feedback_id:
                                st.success("✅ Thanks for your feedback! We've saved this for improving our system.")
                                
                        with feedback_col2:
                            # Use on_click parameter for callback function
                            st.button("👎 No, needs improvement", 
                                     key="edit_negative_feedback", 
                                     on_click=save_negative_edit)
                            
                            # Show an error message if feedback was submitted
                            if st.session_state.feedback_submitted and st.session_state.last_feedback_id:
                                st.error("❌ We're sorry the result wasn't what you expected. Your feedback has been saved.")
            
            # Handle variation button
            elif variation_button:
                with st.spinner(f"Generating image variation with {model_option}..."):
                    # Open the image
                    image = Image.open(uploaded_file).convert("RGB")
                    
                    if use_vision_analysis and model_option in ["gpt-4o", "dall-e-3"]:
                        # For variations, use a more creative prompt
                        creative_prompt = f"Create a variation of this image{': ' + prompt if prompt else '.'} Maintain the same general scene and composition but allow for creative interpretation."
                        generated_image_url = generate_vision_enhanced_variation(
                            image,
                            creative_prompt,
                            model=model_option,
                            size=selected_size,
                            quality=selected_quality,
                            style=selected_style,
                            detail_level=vision_detail,
                            preservation_level=5  # Lower preservation for variations
                        )
                    elif model_option in ["gpt-4o", "dall-e-3"]:
                        # Use selected model with variation prompt if prompt is provided
                        if prompt:
                            variation_prompt = f"Create a variation of the uploaded image: {prompt}"
                        else:
                            variation_prompt = "Create a variation of the uploaded image."
                        
                        if model_option == "gpt-4o":
                            generated_image_url = generate_image_with_gpt4o(
                                variation_prompt, 
                                size=selected_size, 
                                quality=selected_quality, 
                                style=selected_style,
                                image_data=image
                            )
                        else:
                            generated_image_url = generate_image_with_dalle3(
                                variation_prompt, 
                                size=selected_size, 
                                quality=selected_quality, 
                                style=selected_style
                            )
                    else:
                        # Use DALL-E 2 variation API (doesn't use prompt)
                        generated_image_url = generate_image_variation_dalle2(
                            image, 
                            size=selected_size
                        )
                    
                    if generated_image_url:
                        st.subheader("Generated Variation")
                        st.image(generated_image_url, caption=f"Variation by {model_option}", use_container_width=True)
                        
                        # Add download link
                        st.markdown(f"[Download variation]({generated_image_url})")
                        
                        # Display analysis if it was performed
                        if use_vision_analysis and "image_analysis" in st.session_state and st.session_state.image_analysis:
                            with st.expander("View AI Vision Analysis"):
                                st.json(st.session_state.image_analysis)
                        
                        # Feedback mechanism
                        st.write("### Was this variation satisfactory?")
                        
                        # Reset feedback state for new feedback
                        st.session_state.feedback_submitted = False
                        st.session_state.last_feedback_id = None
                        
                        # Callback functions for feedback buttons
                        def save_positive_variation():
                            success = save_feedback_data(
                                feedback_type="positive_variation", 
                                prompt=prompt if prompt else "No prompt provided", 
                                model=model_option, 
                                quality=selected_quality, 
                                style=selected_style, 
                                size=selected_size
                            )
                            return success
                            
                        def save_negative_variation():
                            success = save_feedback_data(
                                feedback_type="negative_variation", 
                                prompt=prompt if prompt else "No prompt provided", 
                                model=model_option, 
                                quality=selected_quality, 
                                style=selected_style, 
                                size=selected_size
                            )
                            return success
                        
                        # Create unique keys for the buttons
                        feedback_col1, feedback_col2 = st.columns(2)
                        
                        with feedback_col1:
                            # Use on_click parameter for callback function
                            st.button("👍 Yes, good result", 
                                     key="var_positive_feedback", 
                                     on_click=save_positive_variation)
                                
                            # Show success message if feedback was submitted    
                            if st.session_state.feedback_submitted and st.session_state.last_feedback_id:
                                st.success("✅ Thanks for your feedback! We've saved this for improving our system.")
                        
                        with feedback_col2:
                            # Use on_click parameter for callback function
                            st.button("👎 No, needs improvement", 
                                     key="var_negative_feedback", 
                                     on_click=save_negative_variation)
                                
                            # Show error message if feedback was submitted
                            if st.session_state.feedback_submitted and st.session_state.last_feedback_id:
                                st.error("❌ We're sorry the result wasn't what you expected. Your feedback has been saved.")
            
            # Handle enhanced variation button
            elif enhanced_variation:
                with st.spinner(f"Generating enhanced variation with {model_option} (this may take longer)..."):
                    # Open the image
                    image = Image.open(uploaded_file).convert("RGB")
                    
                    # First analyze the image to help with more specific prompting
                    st.info("Analyzing original image to ensure precise modifications...")
                    
                    # For workspace images specifically
                    if any(keyword in prompt.lower() for keyword in ["workspace", "office", "desk", "chair"]):
                        st.info("Detected workspace/office image - optimizing for precision replacement")
                    
                    if use_vision_analysis and model_option in ["gpt-4o", "dall-e-3"]:
                        # Use vision-enhanced generation with high preservation
                        generated_image_url = generate_vision_enhanced_variation(
                            image,
                            prompt,
                            model=model_option,
                            size=selected_size,
                            quality=selected_quality,
                            style=selected_style,
                            detail_level="detailed",  # Always use detailed for enhanced variation
                            preservation_level=10  # Maximum preservation
                        )
                    elif model_option in ["gpt-4o", "dall-e-3"]:
                        # Use the enhanced approach with the selected model
                        enhanced_prompt = f"""
                        Starting with the EXACT image provided:
                        1. PRECISELY duplicate all elements, including their exact positions, colors, lighting, and proportions
                        2. Then, ONLY make this specific change: {prompt}
                        3. NOTHING else should be changed - preserve ALL other objects, background, composition, perspective, and style EXACTLY
                        
                        This is an EXACT edit task, not a creative variation. The result should look identical to the original except for the ONE specific change requested.
                        """
                        
                        if model_option == "gpt-4o":
                            generated_image_url = generate_image_with_gpt4o(
                                enhanced_prompt,
                                size=selected_size, 
                                quality=selected_quality, 
                                style=selected_style,
                                image_data=image
                            )
                        else:
                            generated_image_url = generate_image_with_dalle3(
                                enhanced_prompt,
                                size=selected_size, 
                                quality=selected_quality, 
                                style=selected_style
                            )
                    else:
                        # Use the two-step approach for DALL-E 2
                        generated_image_url = two_step_variation(
                            image, 
                            prompt,
                            model="dall-e-3",  # Fall back to DALL-E 3 for the enhancement step
                            size=selected_size
                        )
                    
                    if generated_image_url:
                        st.subheader("Enhanced Variation")
                        st.image(generated_image_url, caption=f"Enhanced Variation by {model_option}", use_container_width=True)
                        
                        # Add download link
                        st.markdown(f"[Download enhanced variation]({generated_image_url})")
                        
                        # Display analysis if it was performed
                        if use_vision_analysis and "image_analysis" in st.session_state and st.session_state.image_analysis:
                            with st.expander("View AI Vision Analysis"):
                                st.json(st.session_state.image_analysis)
                        
                        # Feedback mechanism
                        st.write("### Was this enhanced variation accurate to your request?")
                        
                        # Reset feedback state for new feedback
                        st.session_state.feedback_submitted = False
                        st.session_state.last_feedback_id = None
                        
                        # Callback functions for feedback buttons
                        def save_positive_enhanced():
                            success = save_feedback_data(
                                feedback_type="positive_enhanced", 
                                prompt=prompt, 
                                model=model_option, 
                                quality=selected_quality, 
                                style=selected_style, 
                                size=selected_size
                            )
                            return success
                            
                        def save_negative_enhanced():
                            success = save_feedback_data(
                                feedback_type="negative_enhanced", 
                                prompt=prompt, 
                                model=model_option, 
                                quality=selected_quality, 
                                style=selected_style, 
                                size=selected_size
                            )
                            return success
                        
                        # Create unique keys for the buttons
                        feedback_col1, feedback_col2 = st.columns(2)
                        
                        with feedback_col1:
                            # Use on_click parameter for callback function
                            st.button("👍 Yes, good result", 
                                     key="enhanced_positive_feedback", 
                                     on_click=save_positive_enhanced)
                                
                            # Show success message if feedback was submitted    
                            if st.session_state.feedback_submitted and st.session_state.last_feedback_id:
                                st.success("✅ Thanks for your feedback! We've saved this for improving our system.")
                        
                        with feedback_col2:
                            # Use on_click parameter for callback function
                            st.button("👎 No, needs improvement", 
                                     key="enhanced_negative_feedback", 
                                     on_click=save_negative_enhanced)
                                
                            # Show error message if feedback was submitted
                            if st.session_state.feedback_submitted and st.session_state.last_feedback_id:
                                st.error("❌ We're sorry the result wasn't what you expected. Your feedback has been saved to help us improve.")
                                st.info("💡 Tip: Try adjusting your prompt to be more specific about what to change and what to keep the same.")
    
    elif not uploaded_file and (edit_button or variation_button or enhanced_variation):
        st.info("Please upload an image first.")

# Instructions
if not uploaded_file:
    st.write("### Instructions:")
    st.write("""
    1. Upload an image using the sidebar
    2. Enter instructions for the AI model
    3. Select your model and quality settings
    4. Enter your OpenAI API key
    5. Choose one of the actions:
       - **Edit Image**: Directly modify the image based on your instructions
       - **Create Variation**: Generate an alternative version (simple)
       - **Enhanced Variation**: Generate a high-quality variation using advanced prompting
    
    The AI model will process your image according to your selection.
    """)
    
    # Add feedback data viewer
    st.subheader("View Saved Feedback Data")
    if st.button("View Collected Feedback"):
        # Use our improved read_feedback_data function
        all_feedback = read_feedback_data()
        
        if all_feedback:
            st.write(f"Total feedback entries: {len(all_feedback)}")
            st.write("### Feedback Summary")
            
            # Calculate statistics
            positive_count = sum(1 for entry in all_feedback if "positive" in entry.get("feedback_type", ""))
            negative_count = sum(1 for entry in all_feedback if "negative" in entry.get("feedback_type", ""))
            test_count = sum(1 for entry in all_feedback if entry.get("feedback_type") == "test")
            
            # Count by specific feedback types
            edit_positive = sum(1 for entry in all_feedback if entry.get("feedback_type") == "positive_edit")
            variation_positive = sum(1 for entry in all_feedback if entry.get("feedback_type") == "positive_variation")
            enhanced_positive = sum(1 for entry in all_feedback if entry.get("feedback_type") == "positive_enhanced")
            
            # Create a summary tab
            st.write("### Overall Feedback")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positive Feedback", positive_count)
            with col2:
                st.metric("Negative Feedback", negative_count)
            with col3:
                st.metric("Test Entries", test_count)
                
            st.write("### Feedback by Feature")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Edit", edit_positive)
            with col2:
                st.metric("Variation", variation_positive)
            with col3:
                st.metric("Enhanced", enhanced_positive)
            
            # Show recent feedback
            st.write("### Recent Feedback")
            
            # Sort by timestamp (newest first)
            try:
                # Parse timestamps in standard format
                for entry in all_feedback:
                    if "timestamp" in entry and isinstance(entry["timestamp"], str):
                        try:
                            entry["sort_time"] = datetime.datetime.strptime(entry["timestamp"], "%Y%m%d_%H%M%S")
                        except:
                            entry["sort_time"] = datetime.datetime.now()
                
                # Sort by timestamp, newest first
                sorted_feedback = sorted(all_feedback, key=lambda x: x.get("sort_time", datetime.datetime.now()), reverse=True)
            except Exception:
                sorted_feedback = all_feedback
            
            # Display the 5 most recent entries
            for i, entry in enumerate(sorted_feedback[:5]):
                with st.expander(f"Feedback #{i+1} - {entry.get('feedback_type', 'unknown')} - {entry.get('timestamp', 'unknown')}"):
                    st.write(f"**Prompt:** {entry.get('prompt', 'N/A')}")
                    st.write(f"**Model:** {entry.get('model', 'N/A')}")
                    st.write(f"**Quality:** {entry.get('quality', 'N/A')}")
                    st.write(f"**Style:** {entry.get('style', 'N/A')}")
                    st.write(f"**Size:** {entry.get('size', 'N/A')}")
            
            # Option to download all data as JSON
            if st.button("Download All Feedback Data"):
                json_str = json.dumps(all_feedback, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"all_feedback_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("No feedback data found yet.")
    
    # Add a test feedback button
    st.subheader("Test Feedback System")
    
    # Reset feedback state for testing
    if 'test_feedback_submitted' not in st.session_state:
        st.session_state.test_feedback_submitted = False
    
    # Test feedback callback
    def save_test_feedback():
        result = save_feedback_data(
            feedback_type="test",
            prompt="Test prompt",
            model="test-model",
            quality="standard",
            style="vivid",
            size="1024x1024"
        )
        st.session_state.test_feedback_submitted = result
        return result
        
    # Test button with callback
    st.button("Save Test Feedback", 
              key="test_feedback_button",
              on_click=save_test_feedback)
    
    # Show results based on session state
    if st.session_state.test_feedback_submitted:
        st.success("✅ Test feedback successfully saved!")
        
        # Load the feedback data and display the latest entry
        test_feedback = read_feedback_data()
        if test_feedback:
            st.write("Latest test feedback entry:")
            
            # Find the most recent test entry
            test_entries = [entry for entry in test_feedback if entry.get("feedback_type") == "test"]
            if test_entries:
                # Sort by timestamp and get the latest
                sorted_entries = sorted(test_entries, 
                                      key=lambda x: x.get("timestamp", ""), 
                                      reverse=True)
                st.json(sorted_entries[0])
                
                # Show path to the feedback file
                st.info(f"Feedback data is stored in: /Users/admin/Sites/vibe-coding/Streamlit-Inpaint/feedback_data/")
            else:
                st.info("No test entries found yet.")
    
    # Example prompts specifically for AI vision-powered image editing
    st.markdown("### Effective prompts for AI vision-powered precise modifications:")
    
    examples = [
        "Replace the chair with a leather black office chair, but keep everything else exactly the same",
        "Change only the desk color to dark mahogany, keeping all other elements identical",
        "Replace the laptop with a desktop computer, maintaining the exact same environment",
        "Add a coffee mug on the desk near the right corner, keeping everything else unchanged",
        "Change only the view outside the windows to show a snowy landscape",
        "Replace only the floor with hardwood, keeping all furniture and objects exactly as they are",
        "Add a small bookshelf to the left wall, keeping all existing elements unchanged",
        "Change only the lighting to warm sunset lighting, keeping all objects and layout identical"
    ]
    
    for example in examples:
        st.write(f"- {example}")
    
    st.write("### Tips for precise image modifications with AI Vision:")
    st.write("""
    - Be extremely specific about WHAT to change and what to keep the same
    - Use phrases like "keep everything else exactly the same" or "maintain all other elements"
    - For best results with targeted edits, use the **Enhanced Variation** option with AI Vision enabled
    - Use HD quality for more precise modifications with GPT-4o or DALL-E 3
    - Clearly identify the object to modify by its location (e.g., "the chair at the desk")
    - Specify "duplicate this image, but only change..." to emphasize precision
    - If the result changes too much, try making your instruction even more restrictive
    - For office/workspace images, specifically mention keeping desk, windows, plants, and view identical
    - Enable AI Vision in Advanced Options for the most precise results
    """)
    
    # Model Comparison
    st.markdown("## Model Comparison (April 2025)")
    st.markdown("""
    | Model | Vision Support | Strengths | Best For | Limitations |
    |-------|---------------|-----------|----------|-------------|
    | GPT-4o | ✅ Full Vision | Multimodal vision, understands complex context, can generate images directly | Precision edits, complex context-aware modifications | New model, might have inconsistencies |
    | DALL-E 3 | ❌ Text only | High quality, follows detailed prompts well, reliable | Detailed artistic control, consistent results | Cannot directly see images, recreates from description |
    | DALL-E 2 | ❌ Basic edit only | Direct image editing and variations | Simple variations and edits | Lower quality, less control over output |
    """)
    
    # Information about OpenAI's vision capabilities
    st.markdown("## OpenAI Vision for Image Generation (April 2025)")
    st.markdown("""
    ### How AI Vision Analysis Works
    
    When you enable AI Vision Analysis in the Advanced Options, the app:
    
    1. **Analyzes the uploaded image** using GPT-4o's vision capabilities to identify:
       - Objects and their positions
       - Color palette and lighting
       - Composition and spatial relationships
       - Important details that should be preserved
       
    2. **Creates an enhanced prompt** that combines:
       - Your specific modification request
       - Detailed instructions to preserve all other elements
       - Precise descriptions of key components from the original
       
    3. **Generates the new image** with more contextual awareness and precision
    
    This approach significantly improves the accuracy of targeted edits while maintaining
    the integrity of the original image in all unmodified areas.
    """)

    st.markdown("""
    ### AI Vision Technologies
    
    - **GPT-4o Vision**: Can directly perceive and understand image context
    - **DALL-E 3 + Vision-Enhanced Prompting**: Combines GPT-4o's visual understanding with DALL-E 3's generation
    - **Reference-Aware Image Generation**: Uses the original image as a reference to maintain consistency
    
    ### When to Use AI Vision Analysis
    
    - **Use for precision edits** when you want to change just one element
    - **Use for workspace/office scenes** where layout preservation is critical
    - **Use for product photography** to maintain brand consistency
    - **Use when previous attempts** changed too much of the original image
    
    ### Preservation Strictness Setting
    
    The preservation strictness slider in Advanced Options controls how rigidly the system tries to keep elements unchanged:
    
    - **Higher values (8-10)**: Maximum preservation of original elements
    - **Medium values (5-7)**: Balanced approach
    - **Lower values (1-4)**: More creative interpretation while keeping general structure
    """)
