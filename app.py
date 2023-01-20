import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
import utils
import io

import plotly.io as pio
pio.templates.default = "plotly"

st.title('Embedding Navigation')

## DROPDOWN
#### client-side text input
#### file input -- csv
#### file input -- xlsx
#### file input -- txt
#### xml??

text_option = st.selectbox(
    'How would you like to upload your text?',
    ('Comma Separated Values (.csv)', 'UTF-8 encoded Text file (.txt)', 'Input text here') # 'Excel sheet (.xlsx)',
)

poss_col_names = ['sentences','sents','sentence','sent','text','texts']
text = ''

if text_option == 'Comma Separated Values (.csv)':
    uploaded_file = st.file_uploader("Choose a file to navigate")
    check_data_type = False
    if uploaded_file is not None:
        org = pd.read_csv(uploaded_file).dropna()
        if all(poss_col_names) not in org.columns:
            object_cols = [org[col] for col in org.columns if org[col].dtype == 'O']
            object_cols = sorted(object_cols, key=lambda x: org[x.name].apply(lambda y: len(y.split())).mean(),reverse=True)
            org = org.rename(columns={object_cols[0].name:'sents'})
    else:
        org = []
# elif text_option == 'Excel sheet (.xlsx)':
#     uploaded_file = st.file_uploader("Choose a file to navigate")
#     check_data_type = False
#     if uploaded_file is not None:
#         org = pd.read_excel(uploaded_file, sheet_name=None)
#     else:
#         org = []
elif text_option == 'UTF-8 encoded Text file (.txt)':
    uploaded_file = st.file_uploader("Choose a file to navigate")
    check_data_type = True
    if uploaded_file is not None:
        text = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    org = None
else:
    check_data_type = True
    uploaded_file = None
    org = None
    text = st.text_area('Text to navigate')

model_option = st.selectbox(
    'What pretrained model do you want to use? (Read more [here](https://www.sbert.net/docs/pretrained_models.html))',
    ('all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'all-mpnet-base-v2', 'all-distilroberta-v1')
)

@st.experimental_memo
def get_data(name, check_data_type, model_name):
    #st.write(f'Reading {name}...')
    if check_data_type and (len(text)>0):
        pp = utils.Preprocesser(text)
        sentences = pp.sentence_tokenize()
        em = utils.Embedder(model_name, sentences)
        return em.embed()
    elif (org is not None) and (not isinstance(org,list)):
        intersect = set(poss_col_names).intersection(set(list(org.columns)))
        col_name = list(intersect)[0]
        sentences = org[col_name].to_list() # must have one of the above
        em = utils.Embedder(model_name, sentences)
        return em.embed()

def display_text(text, **kwargs):
    for _, value in kwargs.items():
        st.write(f"<small style='text-align: right;'><b>{value}</b></small>",unsafe_allow_html=True)
    st.write(text.replace('<br>', ' '))
    st.markdown("<hr style='width: 75%;margin: auto;'>",unsafe_allow_html=True)

if not check_data_type:
    poss_div_names = ['chapter','chapters','book','books','section','sections','paragraph','paragraphs']
    if uploaded_file is not None:
        intersect = set(poss_div_names).intersection(set(list(org.columns)))
        intersect = list(intersect)
else:
    intersect = []

fig_selected = []
if uploaded_file is not None:
    df = get_data(uploaded_file.name, check_data_type, model_option)

    if len(intersect) > 0:
        div = intersect[0]
        df[div] = org[div]
    else:
        div = None

    fig = px.scatter(df, x='x', y='y', size='sent_size', color=div, hover_data=['text'])
    with st.container():
        fig_selected = plotly_events(fig, select_event=True)
else:
    df = get_data('provided text', check_data_type, model_option)
    if df is not None:
        fig = px.scatter(df, x='x', y='y', size='sent_size', hover_data=['text'])
        with st.container():
            fig_selected = plotly_events(fig, select_event=True)

if st.button('Reset'):
    fig_selected = []

subsets = []
if len(fig_selected) > 0:
    st.markdown(f"<p><b>Number of sentences</b>: {len(fig_selected)}</p>",unsafe_allow_html=True)
    st.markdown("<hr>",unsafe_allow_html=True)
    for selected in fig_selected:
        subset = df.loc[(df.x == selected['x']) & (df.y == selected['y'])]
        if check_data_type:
            p = subset.apply(lambda x: display_text(x['text']),axis=1)
            subsets.append(subset)
        else:
            p = subset.apply(lambda x: display_text(x['text'],**{str(i):x[o] for (i,o) in enumerate(intersect)}),axis=1)
            subsets.append(subset)
    data = pd.concat(subsets)
    data.text = data.text.apply(lambda x: x.replace('<br>', ' '))
    st.download_button(
        label = 'Download data as CSV',
        data = data.to_csv().encode('utf-8'),
        file_name = 'embedding_data.csv',
        mime = 'text/csv'
    )
else:
    st.write('Use the select tools in the chart above to select labeled sentences.')

st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)