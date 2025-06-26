
# 🤖 AI-Powered Text Completion Chat App

This project is a **capstone application** that showcases **Generative AI for modern text completion** using the [Cohere API](https://cohere.ai) and an interactive [Gradio](https://gradio.app) interface. Built for experimentation, visualization, and user-friendly conversations, this app supports **parameter tuning**, **visual analytics**, and **real-time chatting** with memory.

---

## 🚀 What This App Does

- Accepts a user prompt and sends it to Cohere’s large language models (LLMs)
- Displays coherent, creative, and contextual AI completions
- Tracks and displays response time, token usage, and other stats
- Offers **light/dark mode**, conversation memory, model selection, and a polished UI
- Provides a dashboard to run systematic AI experiments and visualize results

---

## 🧠 Features

- ✅ Cohere API integration with error handling and test ping
- 💬 Gradio-based modern chat UI (light/dark mode, emoji avatars, parameter controls)
- 📊 Stats tracking: tokens, response time, word count, chat history
- 📈 Built-in experiments: run prompt tests with different temperature and token settings
- 🔐 API key entry with local memory and onboarding instructions
- 🧪 Visualization: response time, word count, generation speed across runs
- 📚 In-app documentation, sample prompts, tips, and real-world use cases

---

## 🗂️ Repository Structure

```
├── text_completion_app.py   # Main Python app (this file)
├── README.md                # Project documentation (this file)
├── requirements.txt         # Dependencies
├── assets/                  # Optional: screenshots or experiment images
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
python text_completion_app.py
```

OR launch directly in Google Colab (recommended for beginners)

---

## 🔧 Requirements

- Python 3.8+
- Cohere API key (free from [dashboard.cohere.ai](https://dashboard.cohere.ai))
- Internet access

Install dependencies using:

```bash
pip install cohere gradio pandas plotly matplotlib seaborn wordcloud
```

---

## 📊 Experiment Dashboard

Run built-in test cases by visiting the **🧪 Experiments tab** in the app. It will automatically:
- Test different prompts, temperatures, and token counts
- Log results including word count and generation time
- Plot findings for visualization

---

## 📘 Report Location

The full capstone report is in: `AI Text Completion Report.pdf`.

---

## 👩‍💻 Author

Capstone project developed by **Ashlyn Benoy**.

---

## 🔗 Links

- 📚 [Cohere Documentation](https://docs.cohere.ai)
- 🧑‍🔬 [Gradio Documentation](https://gradio.app)
- ✨ [GitHub Repository](https://github.com/ashlynmaria/ai-text-completion-project)

---

## 📝 License

MIT License – open for learning, sharing, and enhancement.
