# %%writefile app.py
# cell to make a streamlit app

from sklearn.datasets import fetch_openml
import plotly.express as px
import pandas as pd
import numpy as np
import umap
import streamlit as st

def call_streamlit():

    TITLE = 'Interactive UMAP demo'

    st.set_page_config(
        page_title=TITLE,
        page_icon='👾',
        layout='centered'
    )

    st.title(TITLE)

    st.markdown('This application is a [Streamlit](https://github.com/streamlit/streamlit) \
        dashboard \
        to analyze [NMIST dataset](https://github.com/zalandoresearch/fashion-mnist) \
        with the help of [UMAP](https://github.com/lmcinnes/umap)')

    return st


@st.cache(persist=True)
def load_data(sample_size=300, seed=0):

    mnist = fetch_openml('mnist_784', version=1)

    # label	to Description dictionary
    label_decoder = {str(k): v for k, v in enumerate((
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'))}

    labels = mnist.target.map(label_decoder)

    # return sample
    np.random.seed(seed)
    idx = np.random.randint(1, mnist.data.shape[0] + 1, sample_size)
    return mnist.data.iloc[idx], labels.iloc[idx]


def get_parameters(st, unique_labels):
    # st.subheader('Choose distance metric')
    umap__metric = st.sidebar.selectbox('This parameter controls how distance is computed in the ambient space of the input data:',
                                        ['euclidean',
                                         'manhattan',
                                         'chebyshev',
                                         'minkowski',
                                         'canberra',
                                         'braycurtis',
                                         # 'haversine',
                                         'mahalanobis',
                                         'wminkowski',
                                         # 'seuclidean'
                                         ])

    umap__n_neighbors = st.sidebar.slider(
        label='This parameter controls how UMAP balances local versus global structure in the data:',
        min_value=2,
        max_value=unique_labels * 10,
        value=5,
        step=1)

    umap__min_dist = st.sidebar.slider('The min_dist parameter controls how tightly UMAP is allowed to pack points together:',
                                       min_value=.1,
                                       max_value=1.0,
                                       value=.5,
                                       step=.1)

    if umap__metric == None:
        umap__metric = 'euclidean'

    return umap__metric, umap__n_neighbors, umap__min_dist


def get_plot(
        data,
        labels,
        umap__metric, umap__n_neighbors, umap__min_dist):

    target = 'Labels:'

    reducer = umap.UMAP(random_state=42,
                        n_components=3,
                        metric=umap__metric,
                        n_neighbors=umap__n_neighbors,
                        min_dist=umap__min_dist)

    embedding = reducer.fit_transform(data)

    df = pd.DataFrame(embedding, columns=['C1', 'C2', 'C3'])

    df[target] = labels.values
    df['marker_size'] = np.full((df.shape[0],), 5, dtype=int)

    fig = px.scatter_3d(
        df,
        x='C1', y='C2', z='C3',
        color=target,
        size='marker_size',
        opacity=0.7)

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    return fig


st = call_streamlit()
data, labels = load_data()
unique_labels = np.unique(labels).size
umap_parameters = get_parameters(st, unique_labels)

fig = get_plot(
    data,
    labels,
    *umap_parameters)

st.write(fig)
