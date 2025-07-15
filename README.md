# 🍽️ StickChef AI - Multimodal Ingredient Analyzer & Recipe Fusion Engine

A Flask-based backend with Streamlit frontend for a food-themed AI tool that analyzes fridge/pantry images and generates personalized fusion recipes.

## 🚀 Features

- **🔍 Computer Vision**: Detects ingredients from fridge/pantry photos using Faster R-CNN
- **🤖 AI Recipe Generation**: Creates original fusion recipes using GPT-2 text generation
- **📊 Flavor Profiles**: Analyzes and visualizes recipe flavor components (6-component analysis)
- **🌟 Wild Card Fusion**: Unexpected cultural blends (Korean-Italian, Mexican-Japanese, etc.)
- **🌱 Sustainability Scoring**: Carbon footprint estimation and eco-friendly tips
- **🎨 Modern UI**: Clean, responsive Streamlit interface
- **📱 One-Click Experience**: Simple upload → detect → generate → enjoy workflow

## 🛠️ Tech Stack

- **Backend**: Flask + Python 3.12+
- **Frontend**: Streamlit
- **Computer Vision**: PyTorch + torchvision (Faster R-CNN)
- **NLP**: Hugging Face Transformers (GPT-2)
- **Visualization**: Matplotlib
- **Image Processing**: PIL/Pillow
- **Voice Support**: SpeechRecognition + PyAudio (optional)

## 🏆 Competitive Advantages

StickChef AI stands out in the 2025 AI recipe landscape:

- **📸 Multimodal Input**: Photo-based ingredient detection vs. text-only competitors
- **🎯 Generative Fusion**: Creates novel recipes vs. database lookups
- **📊 Visual Intelligence**: 6-component flavor profile visualization
- **🌍 Sustainability Focus**: Carbon footprint scoring and eco-tips
- **🎲 Wild Card Mode**: Unexpected cultural fusion combinations
- **⚡ Single-File Architecture**: Easy deployment and maintenance

## 📋 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd StickChefAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python main.py --test --section 1
   python main.py --test --section 2
   python main.py --integration
   ```

## 🎯 Usage

### Quick Start
```bash
streamlit run main.py
```

### Step-by-Step Guide

1. **Upload Image**: Take a photo of your fridge/pantry contents
2. **Analyze**: AI detects ingredients using computer vision
3. **Customize**: Add preferences (cuisine, dietary restrictions, etc.)
4. **Generate**: Get 3 personalized fusion recipe options
5. **Visualize**: View flavor profiles and nutrition info
6. **Export**: Download recipes as text files

### Testing

```bash
# Test individual sections
python main.py --test --section 1  # Image analysis
python main.py --test --section 2  # Recipe generation
python main.py --test --section 3  # Integration tests

# Run all integration tests
python main.py --integration
```

## 🏗️ Architecture

```
StickChef AI
├── Section 1: Image Analysis (Faster R-CNN)
├── Section 2: Recipe Generation (GPT-2)
└── Section 3: Streamlit Frontend
    ├── Flask API (Background Thread)
    ├── File Upload Interface
    ├── Recipe Display Cards
    └── Flavor Profile Charts
```

## 🔧 API Endpoints

- `GET /` - Health check
- `POST /analyze_image` - Ingredient detection from image
- `POST /generate_recipe` - Single recipe generation
- `POST /generate_multiple_recipes` - 3 recipe options

## 🧪 Testing Strategy

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end API testing
- **Error Handling**: Comprehensive exception management
- **Mock Testing**: Isolated component testing

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile
- **Custom Styling**: Professional color scheme and typography
- **Interactive Elements**: Progress indicators and status messages
- **Data Visualization**: Matplotlib charts for flavor profiles
- **Export Options**: Download recipes as text files

## 🔍 Detected Ingredients

The system can detect 20+ food items including:
- Fruits: apple, orange, banana
- Vegetables: broccoli, carrot
- Proteins: chicken, hot dog, sandwich
- Prepared foods: pizza, donut, cake
- Containers: bottle, cup, wine glass

## 🍳 Recipe Generation Features

- **Fusion Cuisine**: Combines multiple culinary traditions
- **Personalization**: Adapts to dietary restrictions and preferences
- **Flavor Analysis**: Generates Sweet/Savory/Spicy/Sour/Bitter/Umami profiles
- **Nutrition Estimates**: Provides basic nutritional information
- **Multiple Options**: 3 different recipe variations per request

## 🚦 Error Handling

- **Image Processing**: Handles invalid formats and empty files
- **API Validation**: Comprehensive input validation
- **Fallback Options**: Manual ingredient input when detection fails
- **User Feedback**: Clear error messages and suggestions

## 📊 Performance

- **Response Time**: ~2-5 seconds for image analysis
- **Recipe Generation**: ~3-8 seconds for 3 recipes
- **Memory Usage**: Optimized for local development
- **Scalability**: Thread-safe Flask backend

## 🔒 Security

- **Input Validation**: All inputs sanitized and validated
- **File Upload**: Restricted to image formats only
- **Error Handling**: No sensitive information in error messages
- **Local Processing**: All AI processing happens locally

## 🎯 Future Enhancements

- [ ] Database integration for recipe storage
- [ ] User authentication and profiles
- [ ] Recipe rating and feedback system
- [ ] Mobile app development
- [ ] Advanced nutrition analysis
- [ ] Ingredient substitution suggestions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **PyTorch**: Computer vision capabilities
- **Hugging Face**: Pre-trained language models
- **Streamlit**: Rapid frontend development
- **COCO Dataset**: Object detection training data

---

**Made with ❤️ by the StickChef AI Team**

For questions or support, please open an issue on GitHub. 