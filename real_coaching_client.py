import os
import sys
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ensure we import the correct schemas
from schema import SkillProtocol, SkillStep, ChecklistItem, FailureCondition
from person_d_coaching import LocalCoachingEngine, Colors

# Try importing OpenCV and PIL for real camera and image processing
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try importing Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

# Load environment variables
load_dotenv()

# The real-time evaluation schema returned by Gemini for a specific frame
class FrameEvaluationResult(BaseModel):
    checkpoint_matches: List[str] = Field(
        default_factory=list, 
        description="List of checklist item IDs that have been SUCCESSFULLY completed and are visible in this frame."
    )
    failure_triggered_id: Optional[str] = Field(
        default=None, 
        description="The ID of the failure condition triggered in this frame, if any. Leave empty if none are triggered."
    )
    rationale: str = Field(
        ..., 
        description="A short, concise sentence explaining what you visually saw in the frame to justify your verdict."
    )

class RealCoachingClient:
    def __init__(self, protocol_path: str):
        # 1. Initialize our dynamic state engine
        self.engine = LocalCoachingEngine(protocol_path)
        
        # 2. Initialize Gemini Client
        if not SDK_AVAILABLE:
            print(f"{Colors.FAIL}{Colors.BOLD}Error: google-genai SDK not installed.{Colors.ENDC}")
            print("Please run: pip install google-genai python-dotenv pillow opencv-python")
            sys.exit(1)
            
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print(f"{Colors.WARNING}{Colors.BOLD}Warning: GEMINI_API_KEY not found in .env file.{Colors.ENDC}")
            print("Please create a .env file with your key: GEMINI_API_KEY=\"your_key\"")
            
        self.client = genai.Client()

    def capture_webcam_frame(self) -> Optional[bytes]:
        """Captures a frame directly from the Mac webcam using OpenCV."""
        if not OPENCV_AVAILABLE:
            print(f"{Colors.WARNING}OpenCV (cv2) not available. Cannot use webcam.{Colors.ENDC}")
            return None

        print(f"\n📸 Opening camera... Please perform the action in front of the lens.")
        cap = cv2.VideoCapture(0) # 0 is standard webcam on mac
        if not cap.isOpened():
            print(f"{Colors.FAIL}Could not open webcam.{Colors.ENDC}")
            return None

        # Give camera a moment to warm up/focus
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"{Colors.FAIL}Failed to grab frame from camera.{Colors.ENDC}")
            return None

        # Convert frame to JPEG bytes
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            return None

        print(f"✅ Frame successfully captured from webcam!")
        return buffer.tobytes()

    def run_live_turn(self, image_path: Optional[str] = None):
        """
        Runs a single visual evaluation turn.
        Takes a real image from image_path, or captures directly from the camera.
        """
        if self.engine.state == "COMPLETED":
            print(f"\n🏆 Training is already completed successfully!")
            return

        # 1. Obtain image bytes
        image_bytes = None
        if image_path:
            if not os.path.exists(image_path):
                print(f"{Colors.FAIL}Local image file not found: {image_path}{Colors.ENDC}")
                return
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        else:
            image_bytes = self.capture_webcam_frame()

        if not image_bytes:
            # Fallback: Let user type a local file path
            print(f"\n{Colors.BOLD}No webcam input available.{Colors.ENDC}")
            user_path = input("Please enter the path to an image file to analyze: ").strip()
            if not user_path:
                return
            self.run_live_turn(image_path=user_path)
            return

        # 2. Get the active step's checklists and failure modes
        active_step: SkillStep = self.engine.protocol.steps[self.engine.current_step_idx - 1]
        
        # Format a clear prompt for the vision model, giving it the specific targets to evaluate
        checklist_str = json.dumps([item.dict() for item in active_step.checklist], indent=2)
        failures_str = json.dumps([fc.dict() for fc in active_step.failure_conditions], indent=2)

        prompt = f"""
You are the computer vision broker for the "Watch Once, Teach Forever" physical skill coaching engine.
Your job is to analyze this image of a user attempting a step in the skill: '{self.engine.protocol.name}'.

Currently Active Step: Step {active_step.index} - "{active_step.name}"
Description: {active_step.description}

Here is the checklist of targets that need to be met for this step to succeed:
{checklist_str}

Here are the specific failure/safety hazards to watch out for:
{failures_str}

Analyze the image carefully. Identify which checklists are met (set high confidence if clearly done, lower if ambiguous or not visible).
Identify if any of the listed failure conditions are triggered (compare fingers, wire overlaps, or knot angles).
Return your output matching the requested schema.
"""

        print(f"📡 Sending real frame to gemini-3.5-flash...")
        
        try:
            response = self.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FrameEvaluationResult,
                    temperature=0.1 # Low temperature for high precision/consistency on evaluation checks
                )
            )
            
            # Parse structured result
            result = FrameEvaluationResult.parse_raw(response.text)
            
            # Print evaluation details from Gemini
            print(f"\n🎯 {Colors.OKCYAN}{Colors.BOLD}[GEMINI VISION VERDICT]{Colors.ENDC}")
            print(f"Rationale: \"{result.rationale}\"")
            print(f"Checklist Matches: {result.checkpoint_matches}")
            print(f"Failure Triggered: {result.failure_triggered_id}")
            print("="*60)

            # 3. Formulate standard event format and route into our coaching engine
            detections = []
            for item_id in result.checkpoint_matches:
                detections.append({"type": "checklist", "id": item_id, "confidence": 0.95})
                
            if result.failure_triggered_id:
                detections.append({"type": "failure", "id": result.failure_triggered_id, "confidence": 0.95})

            synthetic_event = {
                "timestamp": 1.0, # Live stream
                "frame_description": result.rationale,
                "audio_transcript": "",
                "detected_detections": detections
            }

            # Feed the real Gemini evaluation directly into the dynamic coaching engine!
            self.engine.process_frame_event(synthetic_event)

        except Exception as e:
            print(f"{Colors.FAIL}Error calling Gemini API: {e}{Colors.ENDC}")

def discover_protocols() -> Dict[int, tuple[str, str]]:
    """Scans the current directory for valid JSON protocol files conforming to our schema."""
    valid_profiles = {}
    idx = 1
    for filename in sorted(os.listdir('.')):
        if filename.endswith('.json'):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                # Ensure it has the critical elements of our schema
                if "name" in data and "steps" in data and "category" in data:
                    valid_profiles[idx] = (data["name"], filename)
                    idx += 1
            except Exception:
                continue
    return valid_profiles

def main():
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}🚀 WATCH ONCE, TEACH FOREVER — 100% DYNAMIC CLIENT 🚀{Colors.ENDC}")
    print(f"This engine scans your workspace for ANY custom skill profile JSON,")
    print(f"loads it on-the-fly, and dynamically boots the visual coaching loops!")
    print(f"====================================================================\n")

    # Discover profiles in the folder dynamically!
    profiles = discover_protocols()

    if not profiles:
        print(f"{Colors.FAIL}No valid skill protocol JSON files found in this directory.{Colors.ENDC}")
        print("Please drop a valid protocol JSON file (matching schema.py) into this folder.")
        return

    print("Discovered Dynamic Skill Protocols:")
    for num, (name, filename) in profiles.items():
        print(f"  {Colors.BOLD}{num}. {name}{Colors.ENDC} ({Colors.OKBLUE}File: {filename}{Colors.ENDC})")
    
    choice = input("\nSelect skill number, or type a custom JSON file path: ").strip()

    if choice.isdigit() and int(choice) in profiles:
        selected_profile = profiles[int(choice)][1]
    else:
        # Check if they typed a path directly
        if os.path.exists(choice):
            selected_profile = choice
        else:
            print(f"{Colors.FAIL}Invalid selection or file not found: '{choice}'{Colors.ENDC}")
            return

    try:
        client = RealCoachingClient(selected_profile)
    except Exception as e:
        print(f"{Colors.FAIL}Initialization failed: {e}{Colors.ENDC}")
        return

    client.engine.start_coaching()

    # Dynamic loop allowing continuous frame-by-frame analysis
    while client.engine.state != "COMPLETED":
        print(f"\nPress Enter to capture a camera frame, type an image path (e.g. frame.jpg), or 'q' to quit.")
        inp = input("> ").strip()
        if inp.lower() == 'q':
            print("Exiting coaching session.")
            break
        
        path = inp if inp else None
        client.run_live_turn(image_path=path)

if __name__ == "__main__":
    main()
