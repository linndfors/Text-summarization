from flask import Flask, render_template, request
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from main import find_sentence

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        num_selection = int(request.form['num-selection'])
        output_text = process_input(input_text, num_selection)
        return render_template('index.html', output_text=output_text)
    else:
        return render_template('index.html', output_text='')

def process_input(input_text, num_selection):
    # Remove the last letter from the input text
    # call our function
    processed_text = find_sentence(input_text, num_selection, 1)
    result = " ".join(processed_text)
    return result

if __name__ == '__main__':
    app.run(debug=True)
