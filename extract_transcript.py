from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

class Transcript:
    def __init__(self, video_id:str):
        self.video_id = video_id
        self.transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        self
    
    def get_transcript(self, language_code:str):
        transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
        
        # you can also directly filter for the language you are looking for, using the transcript list
        try:
            transcript = transcript_list.find_transcript([language_code])  
        except Exception as e:
            print(f'No {language_code} transcript available')
            print(f'We will try to translate the transcript to {language_code}')
            transcript = transcript.translate(language_code)
        
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript.fetch())  # format the transcript in your preferred way
        
        # Assuming 'text' contains the formatted transcript
        with open('transcript.txt', 'w') as file:
            file.write(transcript_text)
        
        return transcript_text
        