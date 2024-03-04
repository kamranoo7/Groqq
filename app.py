from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    session,
    Response,
)

import weaviate
import openai
from werkzeug.security import generate_password_hash, check_password_hash
import csv
import gspread
from google.oauth2.service_account import Credentials
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import io
import re
import threading
import random
from flask import Flask, request, jsonify
from flask_mail import Mail, Message
import secrets
import uuid
import os
from dotenv import load_dotenv


app = Flask(__name__)
mail = Mail(app)
app.secret_key = "JobBot"

load_dotenv()

with open('mail_config.yaml', 'r') as file:
    mail_config = yaml.safe_load(file)
app.config.update(mail_config)

app.config["MAIL_PASSWORD"] = "zjwj qhxq askj mxbh"

mail.init_app(app)

global audio_speech
audio_speech = None



key= os.getenv('API_KEY')
lm_client = openai.OpenAI(api_key=key)
service_account_file = "cred.json"
credentials = service_account.Credentials.from_service_account_file(
    service_account_file
)
dbclient = bigquery.Client(credentials=credentials, project=credentials.project_id)
credentials_path = "credentials.json"
response = "Done."
app.secret_key = "secret_key"

global intro
intro = """
Welcome to the alpha version of JobBot, created by Capria. JobBot is powered by ChatGPT and can
discuss both opportunities and risks relative to Al automation of your job and implications for your career. Content comes from the writings of authors Ravi Venkatesan and Will Poole as well as select experts. Ask anything about jobs and careers, such as
"What should my daughter study to ensure her job is not replaced by AI?" or "What topics will the authors Ravi's and Will's new book address?" Please use the thumbs-up or down button to give us feedback.
"""


def add_row_to_sheet(data, sheet_id):
    creds = Credentials.from_service_account_file(
        credentials_path, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)

    try:
        sheet = client.open_by_key(sheet_id)
        worksheet = sheet.get_worksheet(0)
        worksheet.append_row(data)
        print("Row added successfully!")
    except Exception as e:
        print("Error: ", e)


# layer_1 = weaviate.Client(
#     url="https://job1-5sc5kfpy.weaviate.network",
#     additional_headers={"X-OpenAI-Api-Key": key}
# )

name_1 = "job1"

# layer_2 = weaviate.Client(
#     url="https://job11-rre6svy1.weaviate.network",
#     additional_headers={"X-OpenAI-Api-Key": key}
# )

name_2 = "job11"



from google.cloud import bigquery

def get_user_by_userID(userID):
    try:

        dataset_name = "my-project-41692-400512.jobbot"
        table_name = "users"

        query = """
            SELECT *
            FROM `{0}.{1}`
            WHERE userID = @userID
        """.format(
            dataset_name, table_name
        )

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("userID", "STRING", userID)]
        )

        query_job = dbclient.query(query, job_config=job_config)  # Make an API request

        results = query_job.result()  # Waits for job to complete

        for row in results:
            return {"name": row.name, "password": row.password, "userID": row.userID , 'isAnswer': row.isAnswer,'role':row.role}

        return None

    except Exception as e:
        print(f"An error occurred while retrieving user: {str(e)}")
        return None

@app.route("/control_panel", methods=["GET", "POST"])
def control_panel():
    global intro
    if request.method == "POST":
        session["language"] = request.form.get("language", "english")
        session["language"] = get_language(session["language"])

        session["intro"] = request.form.get("intro", "")
        intro = session["intro"] if session["intro"] != "" else intro

        session["prompt_level1"] = request.form.get("prompt_level1", "Be nice")
        session["prompt_level2"] = request.form.get("prompt_level2", "Be nice")
        session["prompt_level3"] = request.form.get("prompt_level3", "Be nice")

    role = "admin"
    return render_template(
        "control_panel.html",
        language=session.get("language", "english"),
        intro=intro,
        prompt_level1=session.get("prompt_level1", ""),
        prompt_level2=session.get("prompt_level2", ""),
        prompt_level3=session.get("prompt_level3", ""),
        role = role,
    )


@app.route("/trans")
def trans():
    global intro
    print("/trans-------")
    newList = []
    transwords = [
        "JobBot",
        "User",
        "Enter Your Query",
        "Feedback",
        "Fast Mode",
        intro,
        "Yes",
        "Submit",
        "Close",
        "Slow Mode",
        "Groq Mode",
    ]
    if "language" not in session: 
        session["language"] = "en"
    if session["language"] != "en":
        for item in transwords:
            trans = translate_text(item, session["language"])
            print(trans, "transscripted words ")
            newList.append(trans)

    if not newList:
        newList = transwords

    print(newList, "---------------trans words ")
    return jsonify(newList)


@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    try:
        audio_file = request.files["audioFile"]
        if audio_file:
            audio_bytes = audio_file.read()
            audio_file = FileWithNames(audio_bytes)
            session["greet"], session["language"] = transcribe(audio_file)
            session["language"] = (
                request.form["language"]
                if request.form["language"] != "auto"
                else session["language"]
            )
            session["language"] = get_language(session["language"])
            print("\n\n\n", session["language"], "\n\n\n")

            print("\n\n\n", session["language"], "\n\n\n")
    except Exception as e:
        print(e)
        update_logs(e)
        session["language"] = "english"

    return jsonify({"channel": "chat"})

@app.route("/")
def index():
    if "language" not in session or session["language"] is None:
        session["language"] = "en"
        print("Language set to English")
    print("Redirecting...")
    if "prompt_level1" not in session or session["prompt_level1"] is None:
        session["prompt_level1"] = "Be kind."
    if "prompt_level2" not in session or session["prompt_level2"] is None:
        session["prompt_level2"] = "Be kind."
    if "prompt_level3" not in session or session["prompt_level3"] is None:
        session["prompt_level3"] = "Be kind."
    return render_template("chat.html")


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    print(data)
    try:
        unique_id = data.get("uniqueId")
        thumbs = data.get("type", "Text")
        l2 = data.get("l2ResponseClicked")
        l3 = data.get("l3ResponseClicked")
        feedback_text = data.get("feedback", "Null")
        level = data.get("level", "test")
        print(thumbs, feedback_text, level)
        if not l2 and not l3:
            add_row_to_sheet(
                [
                    session["transcription"],
                    session["level_1_response"],
                    thumbs,
                    feedback_text,
                ],
                "1OvOj468hgwhjrBFqWrHrZtSirrodrUEejBUUr37by_Y",
            )
        if l2 and not l3:
            add_row_to_sheet(
                [
                    session["transcription"],
                    session["level_2_response"],
                    thumbs,
                    feedback_text,
                ],
                "1OvOj468hgwhjrBFqWrHrZtSirrodrUEejBUUr37by_Y",
            )
        if l2 and l3:
            add_row_to_sheet(
                [
                    session["transcription"],
                    session["level_3_response"],
                    thumbs,
                    feedback_text,
                ],
                "1OvOj468hgwhjrBFqWrHrZtSirrodrUEejBUUr37by_Y",
            )
    except Exception as e:
        update_logs(e)

    return jsonify({"status": "success"})


def transcribe(audio_file):
    try:
        response = lm_client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="verbose_json"
        )
        print(response)
        transcription = response.text
        print(transcription,'transceiption.........................................')
        language = response.text + " " + response.language
    except Exception as e:
        print(e)
        update_logs(e)
        transcription = "Error."
        language = "english"

    return transcription, language


class FileWithNames(io.BytesIO):
    name = "audio.wav"


import os
from datetime import datetime


def update_logs(input_string):
    file_exists = os.path.isfile("logs.txt")

    with open("logs.txt", "a" if file_exists else "w") as file:
        if file_exists:
            file.write("\n\n\n\n")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{current_time}\n{input_string}\n")


def process_response(input_string, replacements):
    def replacement(match):
        index = int(match.group(1))
        return (
            f"[{replacements[index]}]" if index < len(replacements) else match.group(0)
        )

    try:
        return re.sub(r"\[(\d+)\]", replacement, input_string)
    except:
        return input_string


import requests


def translate_text(text, target_language):
    print(target_language)
    api_key = "AIzaSyAtfrkxLhTygIJi9Rb-l0duA8fV9LgKZ7M"  # Replace with your API key

    url = "https://translation.googleapis.com/language/translate/v2"
    data = {"q": text, "target": target_language, "format": "text"}
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    response = requests.post(url, headers=headers, params=params, json=data)
    r = response.json()
    print(r)
    return r["data"]["translations"][0]["translatedText"]


def get_language(lang):
    print("getting lang.")
    lang = lang.lower()
    if "arabic" in lang:
        return "ar"
    if "kannada" in lang:
        return "kn"
    if "telugu" in lang:
        return "te"
    if "spanish" in lang:
        return "es"
    if "hebrew" in lang:
        return "he"
    if "japanese" in lang:
        return "ja"
    if "korean" in lang:
        return "ko"
    if "hindi" in lang:
        return "hi"
    if "bengali" in lang:
        return "bn"
    if "tamil" in lang:
        return "ta"
    if "urdu" in lang:
        return "ur"
    if "chinese" in lang:
        return "zh-CN"
    if "french" in lang:
        return "fr"
    if "german" in lang:
        return "de"

    session["language"] = "english"
    return "en"


import requests
import json
import re

def extract_content_words(lines):
    words = []
    for line in lines:
        if not line.strip():
            continue
        try:
            parsed_line = json.loads(line)
            content = parsed_line.get("result", {}).get("content", "").strip()
            if content:
                words.append(content)
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {line}\nError: {e}")

    return words

def groq(question, language, addition = "You are a helpfull assistant"):
    global audio_speech
    api_key = "aeb9ooc4eiTiedootee3dei4aipuub9v"
    url = "https://api.groq.com/v1/request_manager/text_completion"

    user_message = question
    system_message = addition

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = {
        "model_id": "llama2-70b-4096",
        "system_prompt": system_message,
        "user_prompt": user_message,
        "temperature": 0.1,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

    sentence = ""
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                dec = chunk.decode()
                try:
                    sentence += json.loads(dec)['result']['content']
                except:
                    words = extract_content_words(dec.split("\n"))
                    for wrd in words:
                        sentence += " "
                        sentence += wrd
                data = {"response": sentence, "sufficient": False, "endOfStream": False}
                json_data = json.dumps(data)
                yield f"data: {json_data}\n\n"
    else:
        print(f"Request failed with status code: {response.status_code}")

    if language != "en":
        try:
            sentence = translate_text(sentence, language)
        except Exception as e:
            print("\n\n\n\n")
            print(e, language)
            print("\n\n\n\n")
            pass
    
    audio_speech = lm_client.audio.speech.create(model="tts-1", voice="alloy", input=sentence)

    audio_speech.stream_to_file('output.mp3')

    data = {"response": sentence, "sufficient": False, "endOfStream": True}
    json_data = json.dumps(data)
    yield f"data: {json_data}\n\n"


@app.route("/level1", methods=["POST"])
def level1():
    print("level 1....\n\n\n")
    if request.method=="GET":
        return jsonify({"transcription":session["transcription"]})
    
    session["transcription"] = request.form["query"] if "query" in request.form else ""
    session["prompt_level1"] = (
        "" if "prompt_level1" not in session else session["prompt_level1"]
    )

    if request.form["leng"] != "":
        session["language"] = request.form["leng"]

    session["language"] = (
        "english" if session["language"] == "" else session["language"]
    )

    if request.form["fast"] == "true":
        session["toggle"] = "fast"
    if request.form["slow"] == "true":
        session["toggle"] = "slow"
    if request.form["groq"] == "true":
        session["toggle"] = "groq"

    audio_file = request.files["audio"] if "audio" in request.files else None
    
    try:
        if audio_file:
            print(audio_file, request.files["audio"])
            audio_bytes = audio_file.read()
            print(
                "-----------------------------------------------------------------------------"
            )
            audio_file = FileWithNames(audio_bytes)
            print(
                "-----------------------------------------------------------------------------"
            )
            session["transcription"], session["language"] = transcribe(audio_file)
            
            session["language"] = get_language(session["language"])

            print("\n\n\n", session["language"], "\n\n\n")

            if session["language"].lower() != "en":
                session["transcription"] = translate_text(
                    session["transcription"], "en"
                )
                print(
                    "\n\n\n\nEnglish:  ", session["transcription"], session["language"]
                )
    except Exception as e:
        session["language"] = "en"
        print(e)
        update_logs(e)
        session["transcription"] = "Error."
    return jsonify({"message": "Data received, start streaming","transcription":session["transcription"]})


def ask_gpt(question, model, language, addition = "You are a helpfull assistant"):
    global audio_speech
    user_message = question
    system_message = "Answer the question." + addition

    translate = ""
    response = ""

    msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    print("STREAMINGNNNNNNN\\n\n\n\n\n\n")
    stream = lm_client.chat.completions.create(
        model=model,
        messages=msg,
        stream=True,
    )
    for chunk in stream:
        print(chunk,"Chunk\\n\n\n\n\n\n")

        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
            translate = response
            data = {"response": translate, "sufficient": False}
            json_data = json.dumps(data)
            yield f"data: {json_data}\n\n"

    translate = translate_text(translate, language) if language != "en" else translate

    audio_speech = lm_client.audio.speech.create(model="tts-1", voice="nova", input=translate)

    audio_speech.stream_to_file('output.mp3')
    data = {"response": translate, "sufficient": False, "endOfStream": True}
    json_data = json.dumps(data)
    yield f"data: {json_data}\n\n"


@app.route("/level1/stream")
def level1_stream():
    global audio_speech
    try:
        if session["toggle"] == "slow":
            resp = Response(
                ask_gpt(
                    question = session["transcription"],
                    model = "gpt-4",
                    addition = session["prompt_level1"],
                    language = session["language"],
                ),
                content_type="text/event-stream",
            )
            
            return resp

        elif session["toggle"] == "fast":
            resp = Response(
                ask_gpt(
                    question = session["transcription"],
                    model = "gpt-3.5-turbo-16k",
                    addition = session["prompt_level2"],
                    language = session["language"],
                ),
                content_type="text/event-stream",
            )
            return resp

        else:
            resp = Response(
                groq(
                    question = session["transcription"],
                    addition = session["prompt_level3"],
                    language = session["language"],
                ),
                content_type="text/event-stream",
            )
            return resp

    except:
        data = {"response": "Error.", "sufficient": False}
        json_data = json.dumps(data)
        resp = "data: {json_data}\n\n"
        return Response(resp, content_type="text/event-stream")

@app.route('/audioInterval')
def audio_interval():
    def generate_audio():
        global audio_speech
        if audio_speech:
            audio_file_path = './output.mp3'
            audio_speech = None
            with open(audio_file_path, 'rb') as audio_file:
                while True:
                    chunk = audio_file.read(1024)
                    if not chunk:
                        break
                    print('sending audio')
                    yield chunk
        else:
            print('No Audio')
            return {'error':'null'}
    return Response(generate_audio(), mimetype='audio/mpeg')



@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('/'))


if __name__ == "__main__":
    app.run(debug=True)
