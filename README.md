# NN-Explainer

A **Streamlit-based Neural Network Explainer** that allows users to build, visualize, and understand different types of neural networks, including **ANNs, CNNs, LSTMs, and GRUs**, along with pretrained models like **VGG16, ResNet50, InceptionV3, MobileNet, DenseNet121, and EfficientNetB0**.

## Features

- **Custom Model Builder**: Define model parameters like number of units, activation functions, batch normalization, and dropout layers.
- **Pretrained Model Visualizations**: Displays architecture diagrams for popular pretrained models.
- **Multiple Visualizations**:
  - **Layer-wise Model Diagram**
  - **Graph-based Model Visualization (Graphviz)**
  - **Flowchart-style Model Representation (NetworkX)**

## Installation

### Prerequisites
Ensure you have Python installed (>=3.8) and install the required dependencies:

```sh
pip install streamlit tensorflow keras matplotlib graphviz networkx pydot pygraphviz
```

### Graphviz Setup (Windows)
If you encounter Graphviz errors, install it manually:
- Download and install [Graphviz](https://graphviz.gitlab.io/download/)
- Add it to the system `PATH` environment variable.

## Running the App

Clone the repository and navigate to the project directory:

```sh
git clone https://github.com/AtharvTrivedi21/NN-Explainer.git
cd NN-Explainer
```

Run the Streamlit app:

```sh
streamlit run app.py
```

## Usage

1. **Choose Model Category**:
   - Select **Custom** to define your own architecture.
   - Select **Pretrained** to visualize existing models.

2. **For Custom Models**:
   - Set the number of units, activation function, and optional layers.
   - Click **Build Model** to generate visualizations.

3. **For Pretrained Models**:
   - Select a pretrained model from the dropdown.
   - The architecture diagram will be displayed.

## File Structure

```
NN-Explainer/
│── app.py                 # Main Streamlit application
│── model_visualizer.py     # Script to generate architecture images
│── requirements.txt        # List of dependencies
│── images/                 # Pretrained model architecture images
│── README.md               # Project documentation
│── .gitignore              # Files to ignore in version control
```

## Example Output

### Custom Model Architecture
![Example](images/custom_model.png)

### Pretrained Model Example (VGG16)
![VGG16](images/vgg16.png)


**Author**: [Atharv Trivedi](https://github.com/AtharvTrivedi21)
