import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, GRU, BatchNormalization, Dropout
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import graphviz
import os
import networkx as nx

def build_model(model_type, units, activation, batch_norm, dropout):
    model = Sequential()
    if model_type == "ANN":
        model.add(Dense(units, activation=activation, input_shape=(100,)))
        if batch_norm:
            model.add(BatchNormalization())
        if dropout:
            model.add(Dropout(0.3))
        model.add(Dense(units // 2, activation=activation))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "CNN":
        model.add(Conv2D(32, (3,3), activation=activation, input_shape=(64, 64, 3)))
        model.add(MaxPooling2D((2,2)))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(units, activation=activation))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "LSTM":
        model.add(LSTM(units, input_shape=(10, 10)))
        model.add(Dense(1, activation='sigmoid'))
    elif model_type == "GRU":
        model.add(GRU(units, input_shape=(10, 10)))
        model.add(Dense(1, activation='sigmoid'))
    return model

def visualize_model(model):
    dot_img_file = "model_architecture.png"
    plot_model(model, to_file=dot_img_file, show_shapes=True, show_layer_names=True)
    return dot_img_file

def generate_graph(model):
    dot = graphviz.Digraph(format='png')
    dot.attr(rankdir='LR')
    for i, layer in enumerate(model.layers):
        dot.node(str(i), layer.name + f'\n{layer.output_shape}')
        if i > 0:
            dot.edge(str(i-1), str(i))
    dot.render("model_graph")
    return "model_graph.png"

def generate_networkx_graph(model):
    G = nx.DiGraph()
    for i, layer in enumerate(model.layers):
        G.add_node(layer.name, shape='rectangle')
        if i > 0:
            G.add_edge(model.layers[i-1].name, layer.name)
    
    plt.figure(figsize=(10, 5))
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10, font_weight='bold')
    plt.title("Neural Network Structure")
    plt.savefig("networkx_graph.png")
    return "networkx_graph.png"

st.sidebar.title("Neural Network Explainer")
model_category = st.sidebar.selectbox("Choose Model Category", ["Custom", "Pretrained"])

if model_category == "Custom":
    model_type = st.sidebar.selectbox("Choose Model Type", ["ANN", "CNN", "LSTM", "GRU"])
    st.title("Neural Network Visualization")
    
    units = st.slider("Number of Units (if applicable)", 32, 512, step=32)
    activation = st.selectbox("Activation Function", ["relu", "sigmoid", "tanh", "softmax"])
    batch_norm = st.checkbox("Include Batch Normalization")
    dropout = st.checkbox("Include Dropout Layer")
    
    if st.button("Build Model"):
        model = build_model(model_type, units, activation, batch_norm, dropout)
        img_file = visualize_model(model)
        graph_file = generate_graph(model)
        networkx_file = generate_networkx_graph(model)
        
        st.image(img_file, caption="Model Architecture")
        st.image(graph_file, caption="Layer Connectivity (Graphviz)")
        st.image(networkx_file, caption="Flowchart-like Model Structure")

elif model_category == "Pretrained":
    st.title("Pretrained Model Visualizations")
    chosen_model = st.selectbox("Select Pretrained Model", ["VGG16", "ResNet50", "InceptionV3", "MobileNet", "DenseNet121", "EfficientNetB0"])
    pretrained_images = {
    "VGG16": "model_images/vgg16.png",
    "ResNet50": "model_images/resnet50.png",
    "InceptionV3": "model_images/inceptionv3.png",
    "MobileNet": "model_images/mobilenet.png",
    "DenseNet121": "model_images/densenet121.png",
    "EfficientNetB0": "model_images/efficientnetb0.png"
}
    
    if chosen_model in pretrained_images:
        st.image(pretrained_images[chosen_model], caption=f"{chosen_model} Structure")
