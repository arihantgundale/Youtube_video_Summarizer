from youtube_transcript_api import YouTubeTranscriptApi
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os
from datetime import datetime
import requests
import json

 #ollama -> llama 3.2


def get_ollama_models():
    """Fetch list of available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models
        return []
    except Exception:
        return []


def setup_ollama_summarizer(model_name="llama3"):
    """Set up an Ollama-hosted summarization model"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            raise Exception("Ollama server not responding.")

        available_models = get_ollama_models()
        if not available_models:
            print("No models found in Ollama. Please pull a model (e.g., 'ollama pull llama3').")
            return None

        if model_name not in available_models:
            print(f"Model '{model_name}' not found. Available models: {', '.join(available_models)}")
            print(f"Please pull the model with: 'ollama pull {model_name}'")
            return None

        print(f"Ollama is running. Using model: {model_name}")
        return lambda text, **kwargs: ollama_generate(text, model_name, **kwargs)
    except Exception as e:
        print(f"Error setting up Ollama summarizer: {str(e)}")
        return None


def ollama_generate(prompt, model_name, max_new_tokens=120, min_length=20):
    """Generate text using Ollama API with streaming support"""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.0  # Deterministic output
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        full_response = ""
        for line in response.iter_lines():
            if line:  # Skip empty lines
                json_data = json.loads(line.decode('utf-8'))
                if "response" in json_data:
                    full_response += json_data["response"]
                if json_data.get("done", False):
                    break  # Stop when the response is complete

        if full_response:
            return full_response
        else:
            raise Exception("No response content received from Ollama.")
    except Exception as e:
        raise Exception(f"Ollama API error: {str(e)}")


def get_transcript_text(video_id, lang="en"):
    """Get transcript as a single string"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        full_text = " ".join(entry['text'] for entry in transcript)
        return full_text
    except Exception as e:
        print(f"Error getting transcript: {str(e)}")
        return None


def list_available_languages(video_id):
    """List available transcript languages"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        languages = [t.language_code for t in transcript_list]
        print(f"Available languages: {', '.join(languages)}")
        return languages
    except Exception as e:
        print(f"Error: {str(e)}. No languages found.")
        return []


def clean_transcript(text):
    """Remove noise from transcript"""
    filler_words = {"um", "uh", "like", "you", "know"}
    words = [word for word in text.split() if len(word) > 2 and word.lower() not in filler_words]
    return " ".join(words)


def llm_summarize(transcript_text, summarizer, max_length=120, min_length=20):
    """Summarize the transcript using Ollama"""
    if not transcript_text or not summarizer:
        return "Unable to summarize."

    try:
        max_input_length = 4096  # Adjust if your model supports a different context window
        transcript_text = clean_transcript(transcript_text)
        words = transcript_text.split()

        prompt = (
            "You are an expert summarizer. Provide only the summary content (do not include introductory phrases "
            "like 'Here is a concise, accurate summary of the text:'). Create a concise, accurate summary of the "
            "following text, focusing on main ideas and key details, omitting filler or redundant content:\n\n"
            f"{transcript_text}\n\nSummary:"
        )
        chunk_prompt_template = (
            "You are an expert summarizer. Provide only the summary content (no introductory phrases). "
            "Give a concise summary of the main points of the following text:\n\n"
            "{chunk}\n\nSummary:"
        )

        if len(words) > max_input_length - 100:
            chunks = [" ".join(words[i:i + (max_input_length - 100)])
                      for i in range(0, len(words), max_input_length - 100)]
            intermediate_summaries = []
            for chunk in chunks:
                chunk_prompt = chunk_prompt_template.format(chunk=chunk)
                summary = summarizer(chunk_prompt, max_new_tokens=100, min_length=20)
                intermediate_summaries.append(summary)
            final_input = " ".join(intermediate_summaries)
            final_prompt = (
                "You are an expert summarizer. Provide only the summary content (no introductory phrases). "
                "Combine these summaries into a single, precise summary:\n\n"
                f"{final_input}\n\nSummary:"
            )
            return summarizer(final_prompt, max_new_tokens=max_length, min_length=min_length)
        else:
            return summarizer(prompt, max_new_tokens=max_length, min_length=min_length)
    except Exception as e:
        print(f"Error summarizing: {str(e)}")
        return "Summary generation failed."


def save_summary_to_pdf(summary, video_id, lang):
    """Save the summary to a PDF file"""
    output_dir = "summaries"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/summary_{video_id}_{lang}_{timestamp}.pdf"

    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        style = styles['Normal']
        summary = summary.replace('\n', '<br/>')
        content = [Paragraph(f"Video ID: {video_id} - Language: {lang}", styles['Heading1']),
                   Paragraph(summary, style)]
        doc.build(content)
        print(f"Summary saved to: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving PDF: {str(e)}")
        return None


def main():
    """Main function to run the YouTube transcript summarizer with Ollama"""
    print("Checking Ollama server...")
    available_models = get_ollama_models()
    if not available_models:
        print(
            "No models found. Please install a model (e.g., 'ollama pull llama3') and ensure the server is running ('ollama serve').")
        return

    print(f"Available Ollama models: {', '.join(available_models)}")
    ollama_model = input("Enter Ollama model name (e.g., llama3, mistral) [default: llama3]: ").strip() or "llama3:latest"
    summarizer = setup_ollama_summarizer(ollama_model)
    if not summarizer:
        print("Failed to initialize summarizer. Exiting.")
        return

    video_url = input("Enter a YouTube video URL: ").strip()
    if "youtube.com" in video_url or "youtu.be" in video_url:
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be" in video_url:
            video_id = video_url.split("/")[-1].split("?")[0]
    else:
        video_id = video_url

    available_langs = list_available_languages(video_id)
    if not available_langs:
        print("No languages found.")
        return

    lang = input(f"Pick a language code (like 'en', 'es', 'fr') from {', '.join(available_langs)}: ").strip()
    transcript_text = get_transcript_text(video_id, lang)
    if transcript_text:
        print(f"\nGenerating summary with Ollama ({ollama_model})...")
        summary = llm_summarize(transcript_text, summarizer)
        print("\nSummary:")
        print(summary)
        pdf_filename = save_summary_to_pdf(summary, video_id, lang)

        save = input("\nWould you like to save the full transcript too? (y/n): ").lower().strip()
        if save == 'y':
            output_dir = "transcripts"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/transcript_{video_id}_{lang}_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            print(f"Transcript saved to: {filename}")


if __name__ == "__main__":
    main()