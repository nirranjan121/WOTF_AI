import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import our unified contract schemas
from schema import SkillProtocol, SkillStep, ChecklistItem, FailureCondition, Severity
from person_d_coaching import Colors

# Check for Google GenAI SDK and PIL
try:
    from google import genai
    from google.genai import types
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

# Load environment variables
load_dotenv()

class Phase1SynthesisPipeline:
    def __init__(self):
        if not SDK_AVAILABLE:
            print(f"{Colors.FAIL}{Colors.BOLD}Error: google-genai SDK not installed.{Colors.ENDC}")
            sys.exit(1)
            
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print(f"{Colors.WARNING}Warning: GEMINI_API_KEY not found in env. Synthesis calls will fail.{Colors.ENDC}")
            
        self.client = genai.Client()

    def analyze_real_video_footage(self, video_path: str, skill_name: Optional[str] = None, category: Optional[str] = None) -> Optional[SkillProtocol]:
        """
        Uploads real video footage of an expert doing a task to the Gemini Files API,
        polls for processing completion, and uses Gemini to synthesize the entire SkillProtocol JSON.
        """
        if not os.path.exists(video_path):
            print(f"{Colors.FAIL}Error: Video file not found at path: {video_path}{Colors.ENDC}")
            return None

        # Handle blank or AUTO inputs for zero-friction rural India deployment
        is_auto_detect = not skill_name or skill_name.upper() == "AUTO" or not category or category.upper() == "AUTO"

        print(f"\n🚀 {Colors.OKBLUE}{Colors.BOLD}[PHASE 1: MULTIMODAL VIDEO UPLOAD]{Colors.ENDC} Uploading '{video_path}' to Gemini Files API...")
        
        try:
            # 1. Upload video using the GenAI Files API
            video_file = self.client.files.upload(file=video_path)
            print(f"✅ File uploaded. Remote Name: {video_file.name}")
            
            # 2. Poll until ACTIVE
            print(f"🌀 {Colors.WARNING}Waiting for Gemini to finish indexing/processing video footage...{Colors.ENDC}")
            start_time = time.time()
            while True:
                # Refresh file status
                video_file = self.client.files.get(name=video_file.name)
                state_name = video_file.state.name if hasattr(video_file.state, 'name') else str(video_file.state)
                
                if state_name == "ACTIVE":
                    print(f"\n⚡️ {Colors.OKGREEN}Video is ACTIVE and ready for visual analysis!{Colors.ENDC} (Took {round(time.time() - start_time, 1)}s)")
                    break
                elif state_name == "FAILED":
                    error_msg = video_file.error.message if hasattr(video_file, 'error') and video_file.error else "Unknown error"
                    print(f"{Colors.FAIL}Error: Video processing failed: {error_msg}{Colors.ENDC}")
                    return None
                else:
                    # Print anim-style loading dots
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    time.sleep(4)

            # 3. Request full-footage multi-agent synthesis
            print(f"\n🧠 {Colors.OKCYAN}{Colors.BOLD}[PERSON B & C: MULTI-AGENT FOOTAGE ANALYSIS]{Colors.ENDC}")
            print("Gemini is watching the video, segmenting steps, and extracting checkpoints & hazards...")

            # Design a high-impact prompt centered on rural vocational trades (spinning, loom weaving, tool repair)
            auto_detect_clause = """
            IDENTIFY SKILL NAME & CATEGORY:
            Since the artisan who uploaded this footage may not speak English or have computer literacy, you must AUTO-DETECT the skill being performed (e.g. 'Pottery Clay Wedging', 'Thread Spinning', 'Hand Sewing Button', 'Tractor Radiator Repair').
            Determine its appropriate industrial/craft category (e.g. 'Crafts', 'Vocational Trades', 'Farming Equipment', 'Tailoring').
            Do NOT leave the skill name or category empty.
            """ if is_auto_detect else f"""
            Skill Name: {skill_name}
            Category: {category}
            """

            prompt = f"""
            You are the expert cognitive synthesis agent designed to assist vocational schools and local artisans in rural India.
            We have uploaded real video footage showing an artisan/expert demonstrating a physical skill.
            
            {auto_detect_clause}
            
            Your task is to analyze the video footage step-by-step:
            1. SEGMENT: Group the continuous video into chronological logical steps.
            2. CHECKLISTS: For each step, define 1-2 exact visual checkpoints required to pass the step.
            3. HAZARDS: For each step, identify 1 common mistake or safety hazard, describe its visual trigger, and write helpful spoken voice guidelines (mitigation_instruction) to correct the learner.
            
            Synthesize all of this information and compile a complete SkillProtocol object.
            Return the output strictly in JSON matching the SkillProtocol schema.
            """

            response = self.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[
                    video_file,
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SkillProtocol,
                    temperature=0.2 # Lower temp for extreme consistency
                )
            )

            # Parse and validate using our strict contract!
            protocol = SkillProtocol.parse_raw(response.text)
            print(f"🎯 {Colors.OKGREEN}{Colors.BOLD}[SYNTHESIS COMPLETED]{Colors.ENDC} Protocol parsed & validated successfully!")
            print(f"Successfully compiled '{protocol.name}' with {len(protocol.steps)} steps and checklists!")
            return protocol

        except Exception as e:
            print(f"{Colors.FAIL}Video analysis failed: {e}{Colors.ENDC}")
            return None

    def segment_demo_stream(self, description_log: List[str]) -> List[Dict[str, Any]]:
        """
        [PERSON B: Segmentation Agent - Text Log Fallback]
        Segments text-based timelines.
        """
        print(f"\n🎬 {Colors.OKBLUE}{Colors.BOLD}[PERSON B: SEGMENTATION AGENT]{Colors.ENDC} Analyzing expert activity timeline...")
        
        prompt = f"""
You are Person B: The Real-Time Segmentation Agent.
Segment these activities into logical steps.

Expert Activity Logs:
{json.dumps(description_log, indent=2)}

Return your response in JSON matching this schema:
List of:
{{
  "index": int,
  "name": "Step Name",
  "description": "What happens in this step",
  "raw_activities": ["activity 1", "activity 2"]
}}
"""
        class StepSegment(BaseModel):
            index: int
            name: str
            description: str
            raw_activities: List[str]

        class SegmentationResult(BaseModel):
            segments: List[StepSegment]

        try:
            response = self.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SegmentationResult,
                    temperature=0.1
                )
            )
            result = SegmentationResult.parse_raw(response.text)
            return [seg.dict() for seg in result.segments]
        except Exception as e:
            print(f"{Colors.FAIL}Segmentation API failed: {e}{Colors.ENDC}")
            return []

    def synthesize_protocol(self, skill_name: str, category: str, segments: List[Dict[str, Any]]) -> Optional[SkillProtocol]:
        """
        [PERSON C: Synthesis Agent - Text Log Fallback]
        Synthesizes text-based segments.
        """
        prompt = f"""
You are Person C: The Skill Synthesis Agent.
Reconcile these step segments into a complete Training Protocol.

Skill: {skill_name}
Category: {category}
Segments: {json.dumps(segments, indent=2)}

Format the output strictly according to the SkillProtocol schema.
"""
        try:
            response = self.client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SkillProtocol,
                    temperature=0.2
                )
            )
            return SkillProtocol.parse_raw(response.text)
        except Exception as e:
            print(f"{Colors.FAIL}Synthesis API failed: {e}{Colors.ENDC}")
            return None

def main():
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}================================================================{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}🎬 WATCH ONCE, TEACH FOREVER — PHASE 1: VIDEO ANALYSIS 🎬{Colors.ENDC}")
    print(f"Upload real video footage of an expert performing a task. Gemini will")
    print(f"analyze the footage, segment it, compile the protocol, and build the JSON.")
    print(f"{Colors.OKBLUE}{Colors.BOLD}================================================================{Colors.ENDC}\n")

    print("Choose Phase 1 Input Method:")
    print("1. 📹 Analyze Real Video Footage (Upload mp4/mov file)")
    print("2. 📝 Simulate with Expert Text Activity Logs")
    mode = input("Your choice (1 or 2): ").strip()

    pipeline = Phase1SynthesisPipeline()
    protocol = None

    if mode == "1":
        video_path = input("Enter path to video file (e.g. expert_demo.mp4): ").strip()
        skill_name = input("Enter Skill Name (Press Enter to AUTO-DETECT from video): ").strip()
        category = input("Enter Category (Press Enter to AUTO-DETECT from video): ").strip()
        
        # Pass empty strings as None so the pipeline triggers 100% autonomous detection
        protocol = pipeline.analyze_real_video_footage(
            video_path, 
            skill_name=skill_name if skill_name else None, 
            category=category if category else None
        )
    else:
        # Fallback to simulated logs
        print("\nChoose simulated logs to synthesize:")
        print("1. Sowing a Button (Crafts)")
        print("2. Upgrading Laptop RAM (Electronics)")
        choice = input("Your choice (1 or 2): ").strip()

        skill_name = "Sewing a Button"
        category = "Crafts"
        logs = []

        if choice == "1":
            logs = [
                "T+0.0s: Expert threads the thread through the eye of a metal needle.",
                "T+3.5s: Expert ties a secure double knot at the long end of the thread.",
                "T+7.0s: Expert places the round 4-hole button on the fabric markings.",
                "T+10.5s: Expert pushes the needle up from the backside of the fabric through bottom-left button hole.",
                "T+14.0s: Expert pulls thread tight, then feeds needle down into the top-right button hole.",
                "T+23.0s: Expert pushes needle to backside and ties off a lock-knot to prevent unraveling."
            ]
        else:
            skill_name = "Upgrading Laptop RAM"
            category = "Electronics"
            logs = [
                "T+0.0s: Expert powers down laptop and places it on anti-static mat.",
                "T+5.0s: Expert uses Phillips screw driver to open case.",
                "T+15.0s: Expert releases RAM latches to pop up RAM.",
                "T+24.0s: Expert slides new RAM stick into empty slot and snaps the clips shut."
            ]

        segments = pipeline.segment_demo_stream(logs)
        if segments:
            protocol = pipeline.synthesize_protocol(skill_name, category, segments)

    if not protocol:
        print(f"{Colors.FAIL}Synthesis failed.{Colors.ENDC}")
        return

    # Export compiled JSON using the actual protocol name resolved by Gemini!
    output_filename = f"{protocol.name.lower().replace(' ', '_')}.json"
    with open(output_filename, 'w') as f:
        json.dump(protocol.dict(), f, indent=2)
        
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 SUCCESS! Skill Protocol Synthesized Live!{Colors.ENDC}")
    print(f"Saved compiled JSON protocol to: {Colors.BOLD}{output_filename}{Colors.ENDC}")
    print(f"You can now run {Colors.BOLD}python3 real_coaching_client.py{Colors.ENDC} or launch")
    print(f"your web dashboard using {Colors.BOLD}python3 app.py{Colors.ENDC} and select the new skill!")
    print(f"====================================================================\n")

if __name__ == "__main__":
    main()
