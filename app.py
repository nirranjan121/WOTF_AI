import os
import json
import time
from typing import List, Optional, Dict
from flask import Flask, render_template, jsonify, request, send_from_directory
from schema import SkillProtocol
from phase1_synthesis import Phase1SynthesisPipeline
from real_coaching_client import RealCoachingClient, FrameEvaluationResult
from person_d_coaching import LocalCoachingEngine
from google.genai import types
from pydantic import BaseModel, Field
from schema import SkillStep, ChecklistItem, FailureCondition

class StreamChunkResult(BaseModel):
    is_new_step: bool = Field(..., description="Whether the artisan has transitioned to or completed a brand new logical milestone step.")
    step_name: Optional[str] = Field(None, description="If is_new_step is true, a beautiful title for the new step.")
    step_description: Optional[str] = Field(None, description="If is_new_step is true, a detailed action-oriented description of the step.")
    checkpoint_description: Optional[str] = Field(None, description="If is_new_step is true, exactly 1 checkpoint to verify this step is correct.")
    failure_trigger: Optional[str] = Field(None, description="If is_new_step is true, exactly 1 safety hazard to watch out for during this step.")
    failure_mitigation: Optional[str] = Field(None, description="If is_new_step is true, verbal guidance to mitigate/fix the safety hazard.")
    voice_guideline: Optional[str] = Field(None, description="If is_new_step is true, a spoken warm guideline to announce when starting this step.")

# Definitive Google Gemini Model Registry
MODEL_REGISTRY = {
    "vision_evaluation": "gemini-3.5-flash",                # Supports Computer Use safety standards
    "live_translation": "gemini-3.5-flash",                  # High-quality translation fallback to avoid preview errors
    "multimodal_flow": "gemini-omni-flash-preview",          # Omni-media flash model
    "voice_coaching": "gemini-3.1-flash-tts-preview",        # Gemini 3.1 Text-to-Speech preview
    "interactive_live": "gemini-3.1-flash-live-preview",     # Gemini Flash Live audio-to-audio preview
    "banana_generator": "gemini-3.1-flash-lite-image"        # Nano Banana 2 Lite for visual assets
}


app = Flask(__name__)

# Track active session states globally
class WebAppState:
    def __init__(self):
        self.active_engine = None
        self.client = None

state = WebAppState()

def discover_protocols_list():
    valid_profiles = []
    for filename in sorted(os.listdir('.')):
        if filename.endswith('.json'):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                if "name" in data and "steps" in data and "category" in data:
                    valid_profiles.append({
                        "name": data["name"],
                        "description": data["description"],
                        "category": data["category"],
                        "steps_count": len(data["steps"]),
                        "filename": filename
                    })
            except Exception:
                continue
    return valid_profiles

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/skills', methods=['GET'])
def get_skills():
    """Returns all available skill profiles."""
    return jsonify(discover_protocols_list())

@app.route('/api/synthesize', methods=['POST'])
def synthesize_skill():
    """Runs Phase 1 synthesis and saves the dynamic profile JSON."""
    # 1. Handle Real Video File Upload
    if 'video' in request.files:
        video_file = request.files['video']
        skill_name = request.form.get('skill_name', '').strip()
        category = request.form.get('category', 'Vocational').strip()

        if not skill_name:
            skill_name = "AUTO"
        if not category:
            category = "AUTO"

        # Save to temporary workspace folder
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, video_file.filename)
        video_file.save(temp_path)

        try:
            pipeline = Phase1SynthesisPipeline()
            protocol = pipeline.analyze_real_video_footage(temp_path, skill_name, category)
        finally:
            # Always clean up temp files to keep the workspace pristine
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        if not protocol:
            return jsonify({"error": "Failed to analyze video footage."}), 500

        output_filename = f"{protocol.name.lower().replace(' ', '_')}.json"
        with open(output_filename, 'w') as f:
            json.dump(protocol.dict(), f, indent=2)

        return jsonify({
            "success": True,
            "filename": output_filename,
            "protocol": protocol.dict()
        })

    # 2. Fallback to simulated logs
    data = request.get_json(silent=True) or {}
    skill_name = data.get('skill_name', 'Unnamed Skill')
    category = data.get('category', 'General')
    logs = data.get('logs', [])
    
    if not logs:
        return jsonify({"error": "No video file or activity logs provided."}), 400
        
    pipeline = Phase1SynthesisPipeline()
    segments = pipeline.segment_demo_stream(logs)
    if not segments:
         return jsonify({"error": "Could not segment demonstration logs."}), 500
         
    protocol = pipeline.synthesize_protocol(skill_name, category, segments)
    if not protocol:
         return jsonify({"error": "Failed to synthesize training protocol."}), 500
         
    output_filename = f"{protocol.name.lower().replace(' ', '_')}.json"
    with open(output_filename, 'w') as f:
        json.dump(protocol.dict(), f, indent=2)
        
    return jsonify({
        "success": True,
        "filename": output_filename,
        "protocol": protocol.dict()
    })

@app.route('/api/start', methods=['POST'])
def start_session():
    """Initializes the coaching state machine for a chosen skill."""
    data = request.get_json(silent=True) or {}
    filename = data.get('filename')
    
    if not filename or not os.path.exists(filename):
        return jsonify({"error": "Invalid profile chosen."}), 400
        
    try:
        # Initialize our dynamic engine
        state.client = RealCoachingClient(filename)
        state.client.engine.start_coaching()
        
        return jsonify({
            "success": True,
            "protocol_name": state.client.engine.protocol.name,
            "steps": [step.dict() for step in state.client.engine.protocol.steps],
            "current_step_idx": state.client.engine.current_step_idx,
            "state": state.client.engine.state
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/capture', methods=['POST'])
def capture_and_verify():
    """Snaps a frame from webcam (or accepts static path), evaluates via Gemini, and updates state."""
    if not state.client:
        return jsonify({"error": "No active training session."}), 400
        
    data = request.get_json(silent=True) or {}
    use_mock = data.get('use_mock', False)
    mock_detections = data.get('mock_detections', [])
    mock_rationale = data.get('mock_rationale', "Action performed correctly.")
    
    # 1. Capture visual frame (real webcam frame or simulated event)
    try:
        if use_mock:
            # Simulated flow for rapid debugging
            synthetic_event = {
                "timestamp": round(time.time() % 100, 1),
                "frame_description": mock_rationale,
                "audio_transcript": "",
                "detected_detections": mock_detections
            }
            state.client.engine.process_frame_event(synthetic_event)
            rationale = mock_rationale
        else:
            # Real camera capture + real Gemini visual evaluation!
            frame_b64 = data.get('frame_b64')
            image_bytes = None
            if frame_b64:
                try:
                    import base64
                    image_bytes = base64.b64decode(frame_b64)
                except Exception as e:
                    print(f"Error decoding browser frame_b64: {e}")
                    
            if not image_bytes:
                image_bytes = state.client.capture_webcam_frame()
                
            if not image_bytes:
                return jsonify({"error": "Camera frame capture failed. Please ensure your browser or system has camera permission."}), 500
                
            active_step = state.client.engine.protocol.steps[state.client.engine.current_step_idx - 1]
            checklist_str = json.dumps([item.dict() for item in active_step.checklist], indent=2)
            failures_str = json.dumps([fc.dict() for fc in active_step.failure_conditions], indent=2)

            prompt = f"""
            You are the visual coach for '{state.client.engine.protocol.name}'.
            Analyze this active frame against:
            Checklists: {checklist_str}
            Safety Hazards: {failures_str}
            """
            
            response = state.client.client.models.generate_content(
                model=MODEL_REGISTRY["vision_evaluation"],
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FrameEvaluationResult,
                    temperature=0.1
                )
            )
            
            result = FrameEvaluationResult.parse_raw(response.text)
            rationale = result.rationale
            
            # Transform to engine input
            detections = []
            for item_id in result.checkpoint_matches:
                detections.append({"type": "checklist", "id": item_id, "confidence": 0.95})
            if result.failure_triggered_id:
                detections.append({"type": "failure", "id": result.failure_triggered_id, "confidence": 0.95})
                
            synthetic_event = {
                "timestamp": round(time.time() % 100, 1),
                "frame_description": result.rationale,
                "audio_transcript": "",
                "detected_detections": detections
            }
            state.client.engine.process_frame_event(synthetic_event)
            
        # Compile response payload
        engine = state.client.engine
        
        # Capture and clear any real-time vocal feedback from the state-machine
        tts_feedback = list(engine.tts_queue)
        engine.tts_queue = []
        
        return jsonify({
            "current_step_idx": engine.current_step_idx,
            "engine_state": engine.state,
            "verified_item_ids": list(engine.verified_item_ids),
            "rationale": rationale,
            "tts_feedback": tts_feedback,
            "is_completed": engine.state == "COMPLETED"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# ==========================================
# 📹 REAL-TIME WEBCAM DIGITIZATION (PHASE 1)
# ==========================================
from schema import SkillStep

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

def capture_webcam_frame_direct() -> Optional[bytes]:
    """Capture a live frame from local webcam using cv2."""
    if not OPENCV_AVAILABLE:
        return None
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    is_success, buffer = cv2.imencode(".jpg", frame)
    if not is_success:
        return None
    return buffer.tobytes()

class LiveDigitizationSession:
    def __init__(self):
        self.filepath = ".live_digitizer_session.json"
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                self.name = data.get("name", "")
                self.category = data.get("category", "Heritage Crafts")
                self.steps = [SkillStep.parse_obj(step) for step in data.get("steps", [])]
                return
            except Exception as e:
                print(f"Error loading persistent digitizer session: {e}")
        
        self.name = ""
        self.category = "Heritage Crafts"
        self.steps = []

    def save(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump({
                    "name": self.name,
                    "category": self.category,
                    "steps": [step.dict() for step in self.steps]
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving persistent digitizer session: {e}")

    def clear(self):
        self.name = ""
        self.category = "Heritage Crafts"
        self.steps = []
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except Exception as e:
                print(f"Error clearing persistent digitizer session file: {e}")

live_digitizer = LiveDigitizationSession()

@app.route('/api/digitize_start', methods=['POST'])
def digitize_start():
    """Initializes or resets a live webcam digitization session."""
    data = request.get_json(silent=True) or {}
    live_digitizer.name = data.get('skill_name', '').strip()
    live_digitizer.category = data.get('category', 'Heritage Crafts').strip()
    live_digitizer.steps = []
    live_digitizer.save()
    
    return jsonify({
        "success": True,
        "name": live_digitizer.name,
        "category": live_digitizer.category,
        "steps_count": 0
    })

@app.route('/api/digitize_snap', methods=['POST'])
def digitize_snap():
    """Grabs a live camera frame, synthesizes 1 step via Gemini, and adds to session."""
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get('frame_b64')
    image_bytes = None
    
    if frame_b64:
        try:
            import base64
            image_bytes = base64.b64decode(frame_b64)
        except Exception as e:
            print(f"Error decoding digitize_snap frame_b64: {e}")
            
    if not image_bytes:
        image_bytes = capture_webcam_frame_direct()
        
    if not image_bytes:
        return jsonify({"error": "Failed to grab frame from camera. Make sure webcam is available and browser/terminal permissions are allowed."}), 500

    craft_name = live_digitizer.name or "Heritage Craft"
    step_idx = len(live_digitizer.steps) + 1

    prompt = f"""
    You are an expert cognitive synthesis agent designed to assist local artisans in rural India.
    An expert is demonstrating the craft "{craft_name}" live in front of their camera, and we have just snapped a live frame showing Step {step_idx} of this craft.

    Analyze the visual image carefully. Extract:
    1. A short, beautiful, high-impact Step Name (e.g. "Mold the Spindle", "Tighten the Warp", "Form the Loop").
    2. A brief, warm, action-oriented description of what the expert's hands are doing in this frame.
    3. Exactly 1 checklist checkpoint that visually verifies this step is correct. Give it a simple unique ID (e.g. "loop_formed") and a clear verification description in 'description'.
    4. Exactly 1 failure/safety hazard to watch out for during this step (e.g. "Thread overlap", "Finger too close to tool"). Give it an ID, trigger_description, severity (WARNING or CRITICAL), and a warm verbal mitigation_instruction.

    Format your output matching the requested schema.
    """

    try:
        from google import genai
        client = genai.Client()
        
        response = client.models.generate_content(
            model=MODEL_REGISTRY["vision_evaluation"],
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SkillStep,
                temperature=0.2
            )
        )
        
        new_step = SkillStep.parse_raw(response.text)
        new_step.index = step_idx  # Ensure sequence is correct
        
        live_digitizer.steps.append(new_step)
        live_digitizer.save()
        
        running_protocol = {
            "name": live_digitizer.name or "Auto-Detected Craft",
            "category": live_digitizer.category,
            "steps": [s.dict() for s in live_digitizer.steps]
        }
        
        return jsonify({
            "success": True,
            "step": new_step.dict(),
            "protocol": running_protocol
        })
        
    except Exception as e:
        return jsonify({"error": f"Synthesis failed: {str(e)}"}), 500

@app.route('/api/digitize_stream_chunk', methods=['POST'])
def digitize_stream_chunk():
    """Analyzes a live stream frame + audio transcript to automatically detect and compile steps on-the-fly."""
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get('frame_b64')
    transcript = data.get('transcript', '').strip()
    image_bytes = None
    
    if frame_b64:
        try:
            import base64
            image_bytes = base64.b64decode(frame_b64)
        except Exception as e:
            print(f"Error decoding streaming frame_b64: {e}")
            
    if not image_bytes:
        image_bytes = capture_webcam_frame_direct()
        
    if not image_bytes:
        return jsonify({"error": "Failed to grab frame from camera. Make sure webcam is available."}), 500

    craft_name = live_digitizer.name or "Heritage Craft"
    step_idx = len(live_digitizer.steps) + 1
    
    existing_steps_str = json.dumps([
        {"index": step.index, "name": step.name, "description": step.description}
        for step in live_digitizer.steps
    ], indent=2)

    prompt = f"""
    You are an active real-time AI synthesis agent observing a live heritage craft demonstration of "{craft_name}" for rural India.
    We have just captured a live visual frame of the expert demonstrating the craft.
    We have also captured the following live audio transcript of the expert's spoken explanation during this segment:
    "{transcript}"

    Here is a list of the logical milestone steps we have already digitized and logged so far in this session:
    {existing_steps_str}

    Your goal is to decide if the expert has transitioned to or completed a brand-new, distinct logical milestone step, OR if they are still performing an already-logged step.
    
    To help you:
    - If the visual or spoken transcript represents a continuation of the last logged step, return `is_new_step: false`.
    - If the visual frame shows a new physical action, or the spoken transcript describes a brand new step transition, return `is_new_step: true` and synthesize the detailed step block.
    """

    try:
        from google import genai
        client = genai.Client()
        
        response = client.models.generate_content(
            model=MODEL_REGISTRY["vision_evaluation"],
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=StreamChunkResult,
                temperature=0.2
            )
        )
        
        chunk_res = StreamChunkResult.parse_raw(response.text)
        
        if chunk_res.is_new_step and chunk_res.step_name and chunk_res.step_description:
            # Construct nested Pydantic SkillStep object
            checklist_items = []
            if chunk_res.checkpoint_description:
                checklist_items.append(ChecklistItem(
                    id=f"step_{step_idx}_check",
                    description=chunk_res.checkpoint_description
                ))
            else:
                checklist_items.append(ChecklistItem(
                    id=f"step_{step_idx}_check",
                    description="Verify visual alignment matches reference."
                ))
                
            failures = []
            if chunk_res.failure_trigger:
                failures.append(FailureCondition(
                    id=f"step_{step_idx}_fail",
                    trigger_description=chunk_res.failure_trigger,
                    severity="WARNING",
                    mitigation_instruction=chunk_res.failure_mitigation or "Carefully realign your hand placement."
                ))
                
            new_step = SkillStep(
                index=step_idx,
                name=chunk_res.step_name,
                description=chunk_res.step_description,
                checklist=checklist_items,
                failure_conditions=failures
            )
            
            live_digitizer.steps.append(new_step)
            live_digitizer.save()
            
            return jsonify({
                "new_step_added": True,
                "step": new_step.dict(),
                "protocol": {
                    "name": live_digitizer.name,
                    "category": live_digitizer.category,
                    "steps": [step.dict() for step in live_digitizer.steps]
                }
            })
            
        return jsonify({
            "new_step_added": False,
            "protocol": {
                "name": live_digitizer.name,
                "category": live_digitizer.category,
                "steps": [step.dict() for step in live_digitizer.steps]
            }
        })
        
    except Exception as e:
        print(f"Error in live streaming analyzer chunk: {e}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/api/digitize_finalize', methods=['POST'])
def digitize_finalize():
    """Saves the live-synthesized steps as a complete launchable SkillProtocol JSON file."""
    try:
        data = request.get_json(silent=True) or {}
        override_name = data.get('skill_name', '').strip()
        override_category = data.get('category', '').strip()
        
        if override_name:
            live_digitizer.name = override_name
        if override_category:
            live_digitizer.category = override_category
            
        live_digitizer.save()
        
        # Force reload session to synchronize states cleanly
        live_digitizer._load()
        
        if not live_digitizer.steps:
            return jsonify({"error": "Cannot finalize an empty craft. Snap or stream at least one step milestone."}), 400
            
        final_name = live_digitizer.name or f"Live_Craft_{int(time.time() % 1000)}"
        final_category = live_digitizer.category or "Heritage Crafts"
        
        protocol = SkillProtocol(
            name=final_name,
            description=f"Interactive coaching course for {final_name} compiled in real-time from webcam footage.",
            category=final_category,
            version="1.0.0",
            steps=live_digitizer.steps
        )
        
        output_filename = f"{protocol.name.lower().replace(' ', '_')}.json"
        with open(output_filename, 'w') as f:
            json.dump(protocol.dict(), f, indent=2)
            
        # Reset persistent digitizer session state
        live_digitizer.clear()
        
        return jsonify({
            "success": True,
            "filename": output_filename,
            "protocol": protocol.dict()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to save protocol: {str(e)}"}), 500

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """Translates coaching instructions or rationales on the fly using Gemini."""
    try:
        data = request.get_json(silent=True) or {}
        text = data.get('text', '').strip()
        target_lang = data.get('target_lang', 'en').strip()
        
        if not text or target_lang == 'en':
            return jsonify({"translated_text": text})
            
        # Supported Regional Languages (antigravity-preview-05-2026 Interactions API)
        lang_names = {
            # Regional Indian Languages
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'mr': 'Marathi',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            # Foreign Languages
            'es': 'Spanish',
            'fr': 'French'
        }
        
        lang_name = lang_names.get(target_lang, target_lang)
        
        from google import genai
        client = genai.Client()
        
        prompt = f"""
        Translate the following coaching guideline, explanation, or correction into natural, conversational, encouraging, simple {lang_name}.
        Keep any technical step labels, index numbers, or chemical/physical measurements intact.
        Return ONLY the raw translated string, without any quotation marks, meta commentary, or extra explanations.
        Text: "{text}"
        """
        
        response = client.models.generate_content(
            model=MODEL_REGISTRY["live_translation"],
            contents=prompt
        )
        
        translated = response.text.strip()
        if translated.startswith('"') and translated.endswith('"'):
            translated = translated[1:-1]
            
        return jsonify({"translated_text": translated})
    except Exception as e:
        print(f"Translation failed: {e}")
        return jsonify({"translated_text": text})


@app.route('/api/speak', methods=['POST'])
def speak_text():
    """Generates high-fidelity spoken audio for the given text using Gemini Audio TTS."""
    try:
        data = request.get_json(silent=True) or {}
        text = data.get('text', '').strip()
        target_lang = data.get('target_lang', 'en').strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        lang_names = {
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'mr': 'Marathi',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'es': 'Spanish',
            'fr': 'French',
            'en': 'English'
        }
        lang_name = lang_names.get(target_lang, 'English')
        
        from google import genai
        from google.genai import types
        import base64
        
        client = genai.Client()
        
        # Spoken instruction prompt requesting native regional speech accent
        prompt = f"Please speak this instruction out loud in a natural, clear, warm, and highly encouraging {lang_name} voice: \"{text}\""
        
        # Try configured audio TTS model, and fall back recursively if necessary
        models_to_try = [
            MODEL_REGISTRY["voice_coaching"],   # gemini-3.1-flash-tts-preview
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            MODEL_REGISTRY["vision_evaluation"] # gemini-3.5-flash
        ]
        
        response = None
        error_msg = ""
        for model in models_to_try:
            try:
                print(f"[TTS API] Requesting speech audio from model {model} in language {lang_name}...")
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
                            )
                        )
                    )
                )
                if response and response.candidates and response.candidates[0].content.parts:
                    break
            except Exception as inner_e:
                print(f"[TTS API] Model {model} failed: {inner_e}")
                error_msg = str(inner_e)
                continue
                
        if not response:
            return jsonify({"error": f"Failed to generate speech with any model: {error_msg}"}), 500
            
        audio_bytes = b""
        mime_type = "audio/wav"
        
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                audio_bytes += part.inline_data.data
                if part.inline_data.mime_type:
                    mime_type = part.inline_data.mime_type
                    
        if not audio_bytes:
            return jsonify({"error": "No audio data returned by Gemini"}), 500
            
        print(f"[TTS API] Generated {len(audio_bytes)} audio bytes of type {mime_type}")
        
        encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
        return jsonify({
            "audio": encoded_audio,
            "mime_type": mime_type
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-step-image', methods=['POST'])
def generate_step_image():
    """Generates a step illustration image dynamically using gemini-3.1-flash-lite-image or Imagen."""
    try:
        data = request.get_json(silent=True) or {}
        prompt = data.get('prompt', '').strip()
        step_name = data.get('step_name', 'Step Illustration').strip()
        
        if not prompt:
            prompt = f"An illustration of {step_name}"
            
        enriched_prompt = f"A high-quality 2D vector graphic blueprint detailing: {prompt}. Beautiful dark technical style background, clean lines, suitable for vocational visual guidance, glowing cyan and yellow accents."
        
        from google import genai
        from google.genai import types
        import base64
        
        client = genai.Client()
        
        image_bytes = None
        error_msg = ""
        
        models_to_try = [
            MODEL_REGISTRY["banana_generator"], # gemini-3.1-flash-lite-image
            "imagen-3.0-generate-002",
            "gemini-2.5-flash-image"
        ]
        
        for model in models_to_try:
            try:
                print(f"[Image Gen] Requesting image from model {model}...")
                response = client.models.generate_images(
                    model=model,
                    prompt=enriched_prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="4:3"
                    )
                )
                if response and response.generated_images:
                    image_bytes = response.generated_images[0].image.image_bytes
                    print(f"[Image Gen] Successfully generated image from {model}!")
                    break
            except Exception as e:
                print(f"[Image Gen] Model {model} failed: {e}")
                error_msg = str(e)
                continue
                
        if image_bytes:
            encoded_img = base64.b64encode(image_bytes).decode('utf-8')
            return jsonify({
                "image": encoded_img,
                "mime_type": "image/jpeg"
            })
            
        # Fallback 2: generate_content response if generate_images is unavailable
        try:
            print("[Image Gen] Trying generate_content fallback with model...")
            response = client.models.generate_content(
                model=MODEL_REGISTRY["banana_generator"],
                contents=[enriched_prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"]
                )
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    image_bytes = part.inline_data.data
                    encoded_img = base64.b64encode(image_bytes).decode('utf-8')
                    return jsonify({
                        "image": encoded_img,
                        "mime_type": part.inline_data.mime_type or "image/jpeg"
                    })
        except Exception as fallback_e:
            print(f"[Image Gen] Content-fallback failed: {fallback_e}")
            
        # Fallback 3: Return blueprint styling signals to let frontend render styled blueprints natively
        print("[Image Gen] Returning stylized technical blueprint fallback.")
        return jsonify({
            "placeholder": True,
            "prompt": prompt,
            "message": "Image generation unavailable under active credentials."
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Create static & template folders if missing
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("\n" + "="*50)
    print("🔥 Web Dashboard Client Running on: http://127.0.0.1:5007 🔥")
    print("="*50 + "\n")
    app.run(host='127.0.0.1', port=5007, debug=True)
