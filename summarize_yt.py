from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os


def setup_llm_summarizer():
    """Set up the summarization model"""
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        print("Summarizer loaded successfully.")
        return summarizer
    except Exception as e:
        print(f"Error setting up summarizer: {str(e)}")
        return None


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


def llm_summarize(transcript_text, summarizer, max_length=150, min_length=30):
    """Summarize the transcript using the LLM"""
    if not transcript_text or not summarizer:
        return "Unable to summarize."

    try:
        max_input_length = 1024  # BART's max input length
        words = transcript_text.split()
        if len(words) > max_input_length:
            chunks = [" ".join(words[i:i + max_input_length])
                      for i in range(0, len(words), max_input_length)]
            summaries = []
            for chunk in chunks:
                summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            return " ".join(summaries)
        else:
            summary = summarizer(transcript_text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing: {str(e)}")
        return "Summary generation failed."


def save_summary_to_pdf(summary, video_id, lang):
    """Save the summary to a PDF file"""
    output_dir = "summaries"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/summary_{video_id}_{lang}_{timestamp}.pdf"

    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        style = styles['Normal']

        # Replace newlines with <br/> for PDF formatting
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
    # Initialize the summarizer
    summarizer = setup_llm_summarizer()
    if not summarizer:
        print("Failed to initialize summarizer. Exiting.")
        return

    # Get video URL and extract ID
    video_url = input("Enter a YouTube video URL: ").strip()
    if "youtube.com" in video_url or "youtu.be" in video_url:
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be" in video_url:
            video_id = video_url.split("/")[-1].split("?")[0]
    else:
        video_id = video_url

    # Check available languages
    available_langs = list_available_languages(video_id)
    if not available_langs:
        print("No languages found.")
        return

    # Get language choice
    lang = input(f"Pick a language code (like 'en', 'es', 'fr') from {', '.join(available_langs)}: ").strip()

    # Get and summarize transcript
    transcript_text = get_transcript_text(video_id, lang)
    if transcript_text:
        print("\nGenerating summary...")
        summary = llm_summarize(transcript_text, summarizer)
        print("\nSummary:")
        print(summary)

        # Save summary to PDF
        pdf_filename = save_summary_to_pdf(summary, video_id, lang)

        # Optional: Save transcript
        save = input("\nWould you like to save the full transcript too? (y/n): ").lower().strip()
        if save == 'y':
            output_dir = "transcripts"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/transcript_{video_id}_{lang}_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            print(f"Transcript saved to: {filename}")


if __name__ == "__main__":
    main()