from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        output_text = process_input(input_text)
        return render_template('index.html', output_text=output_text)
    else:
        return render_template('index.html', output_text='')

def process_input(input_text):
    # Remove the last letter from the input text
    # call our function
    processed_text = input_text
    processed_text += "\nhere will be summarized text!!!!"
    return processed_text

if __name__ == '__main__':
    app.run(debug=True)
