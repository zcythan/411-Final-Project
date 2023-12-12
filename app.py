from flask import Flask, render_template, redirect, url_for, request
from lieDetector import lieDetector

aimod = lieDetector()
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    result = ""
    input = {}

    nothing = ""
    if request.method == "POST":
        input['statement'] = request.form['statement']
        input['subject'] = request.form['subject']
        input['speaker'] = request.form['speaker']
        input['jobTitle'] = request.form['jobTitle']
        input['state'] = request.form['state']
        input['party'] = request.form['party']
        input['context'] = request.form['context']

        if input['statement'] != "":
            try:
                request.form['binary_classifier']
                binary_classifier = True
            except:
                binary_classifier = False

            string_input = ""
            for value in input.values():
                string_input += value + " "
            print(string_input)
            result = aimod.predict(string_input, binary_classifier)

    return render_template('index.html', title='Political Deception Detector', states=states, input=input, result=result, nothing=nothing)


states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
    "Washington", "West Virginia", "Wisconsin", "Wyoming"
]
