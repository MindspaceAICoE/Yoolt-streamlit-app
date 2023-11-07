import openai

MODEL="gpt-3.5-turbo"
def meeting_minutes(transcription) -> dict:
    """The main function which combines all sub-summaries into a python dict/JSON"""
    print(f"Summarizing transcript starting with [{transcription[:50]}...]")
    abstract_summary = abstract_summary_extraction(transcription)
    key_points = key_points_extraction(transcription)
    action_items = action_item_extraction(transcription)
    sentiment = sentiment_analysis(transcription)
    print("Done")
    return {
        'abstract_summary': abstract_summary,
        'key_points': key_points,
        'action_items': action_items,
        'sentiment': sentiment
    }


def abstract_summary_extraction(transcription):
    """Use OpenAI to extract a summary"""
    response = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def key_points_extraction(transcription):
    """Use OpenAI to extract key points"""
    response = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def action_item_extraction(transcription):
    """Use OpenAI to extract action items from the meeting"""
    response = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI expert in analyzing conversations and extracting action items. Please review the text and identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. These could be tasks assigned to specific individuals, or general actions that the group has decided to take. Please list these action items clearly and concisely."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def sentiment_analysis(transcription):
    """Do some sentiment analysis"""
    response = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the following text. Please consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used. Indicate whether the sentiment is generally positive, negative, or neutral, and provide brief explanations for your analysis where possible."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


# async def transcribe_and_summarize_all(parts: List[FilePart]) -> dict:
#     """Here we collect all 25MB chunks (parts) and transcribe them and make one big 
#     meeting minutes summary out of it."""
#     transcription = ""
#     minutes = ""

#     print("Transcribing...")
#     num_parts = len(parts)
#     i = 0
#     for part in parts:
#         print(f"Part {i} of {num_parts}: {part.part}")
#         audio_file = part.part
#         partial_transcription =  transcribe_audio(audio_file)
#         transcription = transcription + partial_transcription
#         print("Done")
#         i += 1
#     minutes = meeting_minutes(transcription)
#     return  minutes