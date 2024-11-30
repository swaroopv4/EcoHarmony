# EcoHarmony-HSD
Unlock the potential of our advanced transformer model with text and voice recognition, designed for detecting hate and offensive speech. This cutting-edge tool analyzes both written and spoken language, effectively identifying harmful content. With its robust capabilities, our model offers a comprehensive solution for promoting safer online environments.

## Setup Instructions

### Prerequisites

- Anaconda or Python (with pip)
- Internet connection for installing dependencies

### Step-by-Step Setup Guide

1. **Clone the Repository**

    ```sh
    git clone https://github.com/your-username/EcoHarmony-SocialPro.git
    cd EcoHarmony-HSD
    ```

2. **Create and Activate the Environment**

    **Using Anaconda:**

    ```sh
    conda create --name social_env python=3.8
    conda activate social_env
    ```

    **Using Python's venv:**

    ```sh
    python -m venv social_env
    source social_env/bin/activate  # On Windows use `social_env\Scripts\activate`
    ```

3. **Navigate to the Project Directory**

    ```sh
    cd EcoHarmony/SocialPro
    ```

4. **Install Requirements**

    ```sh
    pip install -r requirements.txt
    ```

5. **Train the Model**

    ```sh
    python Training.py
    ```

6. **Run the Streamlit App**

    ```sh
    streamlit run app.py
    ```

### Google Colab

For running this project on Google Colab, use the following notebook link:
[Google Colab Notebook](https://colab.research.google.com/drive/1LETo9YZujiTj1pxZvvIg_rGaFZgm-0b_?usp=sharing)

## Project Structure

- `app.py`: The main Streamlit application file.
- `Training.py`: Script to train the machine learning model.
- `requirements.txt`: List of required Python packages.
- `EcoHarmony/SocialPro`: Project directory containing all relevant code and data.

## Features

- **Multi-language Support**: Detects and translates tweets from various languages to English.
- **Real-time Detection**: Analyzes text input in real-time to determine the presence of hate speech.
- **User Feedback**: Allows users to provide feedback on detection results.
- **Advanced Visualization**: Displays sentiment analysis and word clouds for better insights.

## Usage

1. **Enter a Tweet**: Type or paste a tweet into the text area.
2. **Detect Hate Speech**: Click the "Detect" button to analyze the tweet.
3. **View Results**: Check the prediction and sentiment analysis results.

### Contributing

Feel free to contribute to this project by creating pull requests, reporting issues, or suggesting improvements.



