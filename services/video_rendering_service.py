import tempfile
import whisper
from pytube import YouTube
from pytube.exceptions import VideoUnavailable

def transcribeVideoOrchestrator(youtube_url: str, model_name: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            video = downloadYoutubeVideo(youtube_url, temp_dir)
            if not video:
                return "Video could not be downloaded."
            transcription = transcribe(video, model_name)
            return transcription
        except Exception as e:
            return f"An error occurred: {str(e)}"

def transcribe(video: dict, model_name="medium"):
    try:
        print(f"Transcribing: {video['name']}")
        model = whisper.load_model(model_name)
        result = model.transcribe(video['path'])
        return result["text"]
    except FileNotFoundError:
        return "Transcription file not found."
    except Exception as e:
        return f"An error occurred during transcription: {str(e)}"

def downloadYoutubeVideo(youtube_url: str, directory: str) -> dict:
    try:
        print(f"Processing: {youtube_url}")
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if stream:
            file_path = stream.download(output_path=directory)
            print(f"Download complete: {file_path}")
            return {"name": yt.title, "thumbnail": yt.thumbnail_url, "path": file_path}
        else:
            return None
    except VideoUnavailable:
        print("Video is unavailable.")
        return None
    except Exception as e:
        print(f"An error occurred during video download: {str(e)}")
        return None
