# 🏃‍♂️ Fitness Data Analyzer

An advanced Streamlit application for analyzing fitness data from FIT and TCX files using AI-powered insights.

## ✨ Features

### 📊 **Data Visualization**
- **Heart Rate Analysis**: Interactive charts with training zone overlays
- **Elevation Profiles**: Detailed elevation changes throughout your workout
- **GPS Route Mapping**: Interactive maps showing your workout route
- **Performance Metrics**: Comprehensive workout statistics

### 🤖 **AI-Powered Insights**
- Detailed workout analysis using Google Vertex AI
- Training zone recommendations
- Performance improvement suggestions
- Recovery guidance
- Personalized coaching tips

### 📱 **User Experience**
- Modern, responsive interface
- Tabbed navigation for different data views
- Interactive charts and visualizations
- Chat interface for follow-up questions
- Export capabilities for data and chat history

### 📁 **File Support**
- **FIT Files**: From Garmin, Polar, Suunto devices
- **TCX Files**: Training Center XML format
- Automatic file type detection
- Comprehensive data extraction

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Cloud Account with Vertex AI enabled
- Valid fitness data files (FIT or TCX)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd FIT_Chatbot_Streamlit
   ```

2. **Create virtual environment**
   ```bash
   python -m venv 313
   source 313/bin/activate  # On Windows: 313\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Google Cloud credentials
   ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_CLOUD_REGION=us-central1
MODEL_NAME=gemini-pro
```

### Google Cloud Setup

1. Create a Google Cloud Project
2. Enable Vertex AI API
3. Set up authentication (service account or application default credentials)
4. Configure your project ID and region in the `.env` file

## 📈 Data Analysis Features

### Heart Rate Analysis
- Training zone visualization
- Average, maximum, and minimum heart rate
- Heart rate variability insights
- Zone distribution analysis

### Performance Metrics
- Distance and duration tracking
- Pace and speed analysis
- Calorie burn calculations
- Power data analysis (if available)

### GPS and Route Analysis
- Interactive route mapping
- Elevation profile visualization
- Ascent and descent tracking
- Geographic insights

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Plotly**: Interactive data visualization
- **Folium**: GPS mapping and route visualization
- **FitDecode**: FIT file parsing
- **TCXParser**: TCX file parsing
- **Google Vertex AI**: AI-powered analysis
- **Pandas/NumPy**: Data processing

### File Processing
- Secure temporary file handling
- Automatic cleanup after processing
- Error handling for corrupted files
- Support for large file uploads

## 🔧 Customization

### Adding New Visualizations
1. Create visualization functions in the utility section
2. Add new tabs in the main interface
3. Include data validation and error handling

### Extending AI Analysis
1. Modify the AI prompt templates
2. Add new analysis categories
3. Implement custom metrics calculation

### UI Enhancements
1. Customize the Streamlit theme
2. Add new sidebar options
3. Implement additional export formats

## 🐛 Troubleshooting

### Common Issues

**ImportError: No module named 'X'**
- Ensure all dependencies are installed: `pip install -r requirements.txt`

**Google Cloud Authentication Error**
- Verify your credentials are properly configured
- Check your project ID and region settings

**File Parsing Errors**
- Ensure your files are valid FIT or TCX format
- Try with different files to isolate the issue

**Visualization Not Loading**
- Check browser console for JavaScript errors
- Try refreshing the page or restarting the app

## 📝 Usage Tips

1. **File Upload**: Drag and drop or browse for FIT/TCX files
2. **Navigation**: Use tabs to explore different aspects of your data
3. **AI Chat**: Ask specific questions about your workout performance
4. **Export**: Download analysis reports and chat history
5. **Settings**: Use sidebar options to customize the display

## 🔮 Future Enhancements

- [ ] Multiple file comparison
- [ ] Historical trend analysis
- [ ] Training plan generation
- [ ] Social sharing features
- [ ] Mobile app version
- [ ] Integration with fitness APIs
- [ ] Advanced machine learning insights

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions and support, please open an issue in the GitHub repository.
