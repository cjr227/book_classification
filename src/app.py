# Import Packages
import base64
import io
import datetime

## Installed Packages
from simpletransformers.classification import ClassificationModel
from pypdf import PdfReader
import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
from flask import Flask
from flask_restful import reqparse, Resource, Api

label_mapping = {'Other': 'Other', 
'Information/Explanation': 'Non-Fiction', 
'News': 'Non-Fiction', 
'Instruction': 'Non-Fiction', 
'Opinion/Argumentation': 'Non-Fiction', 
'Forum': 'Non-Fiction', 
'Prose/Lyrical': 'Fiction', 
'Legal': 'Non-Fiction', 
'Promotion': 'Fiction'}


def extract_text(content):
    try:
        reader = PdfReader(content)
    except:
        print("Error: File cannot be read. Please try again with another file.")
        return None
    file_text = ""
    for page in reader.pages:
        file_text += page.extract_text()
    if file_text.strip() in ['', None]:
        print(
            "Error: This file has no parsable content, and thus a rating cannot be provided. Please try again with "
            "another URL.")
        return None
    #print(file_text)
    return file_text

def get_model_object():
    model_args= {
            "num_train_epochs": 15,
            "learning_rate": 1e-5,
            "max_seq_length": 512,
            "silent": True
            }
    model = ClassificationModel(
    "xlmroberta", 
    "classla/xlm-roberta-base-multilingual-text-genre-classifier", use_cuda=False,
    args=model_args
    )
    return model

def get_model_output(model_obj, model_input):
    prediction, logit_output = model_obj.predict([model_input])
    prediction_enr = label_mapping[model_obj.config.id2label[prediction[0]]]
    return prediction, logit_output, prediction_enr

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if '.pdf' in filename:
            file_text = extract_text(io.BytesIO(decoded))
            model_obj = get_model_object()
            model_pred_raw, model_logit, model_pred = get_model_output(model_obj=model_obj, model_input=file_text)
            tbl = dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in ["filename", "date", "prediction"]],
                    data=[{"filename": filename, 
                    "date": datetime.datetime.fromtimestamp(date), 
                    "prediction": model_pred}],
                    style_cell=dict(textAlign='left'),
                    style_header=dict(backgroundColor='paleturquoise'),
                    style_data=dict(backgroundColor='lavender')
                )
            table_layout = html.Div([dcc.Markdown(
                        "**Prediction**:"),
                        tbl])
            return table_layout
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

# Dash
server = Flask('app')
app = dash.Dash(__name__, server=server, title="Book Genre Classification")

# API
api = Api(server)
parser = reqparse.RequestParser()
parser.add_argument('filepath')

class BookClassifier(Resource):
    def post(self):
        args = parser.parse_args()
        file_text = extract_text(content=args['filepath'])
        if file_text is None:
            return {'Error': 'File cannot be reached or has no parsable content. Please try again with another '
                             'file'}, 400
        else:
            model_obj = get_model_object()
            model_pred_raw, model_logit, model_pred = get_model_output(model_obj=model_obj, model_input=file_text)
            return {'predicted_class_raw': model_obj.config.id2label[model_pred_raw[0]], 
            'logit_score': {model_obj.config.id2label[i]: model_logit[0][i] for i in range(len(model_logit[0]))}, 
            'predicted_class_final': model_pred}, 201

api.add_resource(BookClassifier, '/predict')

# HTML/CSS
## Note that much of this CSS comes from https://codepen.io/chriddyp/pen/bWLwgP.css
logo_image_style = {'textAlign': 'left', "display": "block",
                    "margin-left": "auto", "margin-right": "auto", 'padding-top': 20}
input_style = {'height': '38px', 'padding': '6px 10px',
               'background-color': '#fff', 'border': '1px solid #D1D1D1',
               'border-radius': '4px', 'box-shadow': 'none',
               'box-sizing': 'border-box', 'width': '900px'}
button_style = {'display': 'inline-block',
                'height': '38px',
                'padding': '0 30px',
                'color': '#555',
                'text-align': 'center',
                'font-weight': '600',
                'line-height': '38px',
                'letter-spacing': '.1rem',
                'text-decoration': 'none',
                'white-space': 'nowrap',
                'background-color': 'transparent',
                'border-radius': '4px',
                'border': '1px solid #bbb',
                'cursor': 'pointer',
                'box-sizing': 'border-box'}
div_style = {
    'font-size': '1em',
    'line-height': '0px',
    'margin': '3',
    'margin-bottom': '3rem',
    'position': 'relative',
    'top': '3rem',
    'left': '0',
    'textAlign': 'left',
    'padding-top': '1rem',
    'padding-left': '5rem'}

app.layout = html.Div([
    html.H1("Book Genre Classification"),
    html.Div([
        dcc.Markdown("Upload a PDF document to classify whether its text "
                     "is fiction or non-fiction.")
    ]),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    dcc.Loading(
        id='loading',
        type='circle',
        children=[html.Div(id='output-data-upload')]
    ),
    html.Br(),
    html.Div([
        dcc.Markdown("This work is licensed under a [GNU General Public License, version 3.0]("
                     "https://www.gnu.org/licenses/gpl-3.0.en.html).")
    ])
])

@callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
