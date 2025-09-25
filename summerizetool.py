import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from string import punctuation
import heapq
import re

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english') + list(punctuation))
    
    def preprocess_text(self, text):
        """Clean and prepare the text for processing"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[.*?\]', '', text)
        return text.strip()
    
    def summarize(self, text, num_sentences=5):
        """Generate a summary of the text with the specified number of sentences"""
        text = self.preprocess_text(text)
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        words = word_tokenize(text.lower())
        words = [word for word in words if word not in self.stop_words and word.isalnum()]
        
        word_frequencies = FreqDist(words)
        
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    if i not in sentence_scores:
                        sentence_scores[i] = word_frequencies[word]
                    else:
                        sentence_scores[i] += word_frequencies[word]
        
        if not sentence_scores:
            return "Unable to generate summary. Text may be too short or contain mostly stopwords."
            
        top_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        top_sentences.sort()
        
        summary = ' '.join([sentences[i] for i in top_sentences])
        return summary

class SummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Text Summarizer")
        self.root.geometry("900x700")
        self.root.configure(bg="#f5f5f5")
        
        self.summarizer = TextSummarizer()
        
        self.setup_ui()
    
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        title = tk.Label(
            header_frame, 
            text="AI Text Summarizer", 
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title.pack(expand=True)
        
        subtitle = tk.Label(
            header_frame,
            text="Transform long texts into concise 3-5 sentence summaries",
            font=("Arial", 10),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        subtitle.pack(expand=True)
        
        # Main content
        main_frame = tk.Frame(self.root, bg="#f5f5f5", padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)
        
        # Input section
        input_frame = tk.LabelFrame(main_frame, text=" Input Text ", font=("Arial", 12, "bold"), 
                                   bg="#f5f5f5", fg="#2c3e50", padx=10, pady=10)
        input_frame.pack(fill="x", pady=(0, 15))
        
        instructions = tk.Label(
            input_frame,
            text="Paste your article, blog post, or document text in the box below:",
            font=("Arial", 10),
            bg="#f5f5f5",
            fg="#555555",
            justify="left"
        )
        instructions.pack(anchor="w", pady=(0, 10))
        
        self.input_text = scrolledtext.ScrolledText(
            input_frame,
            wrap=tk.WORD,
            width=85,
            height=15,
            font=("Arial", 10),
            bg="white",
            fg="#333333",
            relief=tk.SOLID,
            highlightthickness=1,
            highlightcolor="#3498db",
            highlightbackground="#cccccc"
        )
        self.input_text.pack(fill="x")
        
        # Example text button
        example_btn = tk.Button(
            input_frame,
            text="Load Example Text",
            command=self.load_example,
            font=("Arial", 9),
            bg="#3498db",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2"
        )
        example_btn.pack(anchor="e", pady=(10, 0))
        
        # Controls section
        controls_frame = tk.Frame(main_frame, bg="#f5f5f5")
        controls_frame.pack(fill="x", pady=10)
        
        self.summarize_btn = tk.Button(
            controls_frame,
            text="Generate Summary",
            command=self.generate_summary,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            activebackground="#219653",
            activeforeground="white",
            relief=tk.FLAT,
            padx=25,
            pady=10,
            cursor="hand2"
        )
        self.summarize_btn.pack(side=tk.LEFT)
        
        self.clear_btn = tk.Button(
            controls_frame,
            text="Clear All",
            command=self.clear_text,
            font=("Arial", 11),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            activeforeground="white",
            relief=tk.FLAT,
            padx=25,
            pady=10,
            cursor="hand2"
        )
        self.clear_btn.pack(side=tk.RIGHT)
        
        # Output section
        output_frame = tk.LabelFrame(main_frame, text=" Summary ", font=("Arial", 12, "bold"), 
                                    bg="#f5f5f5", fg="#2c3e50", padx=10, pady=10)
        output_frame.pack(fill="both", expand=True)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            width=85,
            height=8,
            font=("Arial", 10),
            bg="#f8f9fa",
            fg="#333333",
            relief=tk.SOLID,
            highlightthickness=1,
            highlightcolor="#3498db",
            highlightbackground="#cccccc",
            state="disabled"
        )
        self.output_text.pack(fill="both", expand=True)
        
        # Status bar
        self.status = tk.Label(
            self.root, 
            text="Ready", 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            font=("Arial", 9),
            fg="#555555"
        )
        self.status.pack(side=tk.BOTTOM, fill="x")
    
    def load_example(self):
        example_text = """Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Colloquially, the term "artificial intelligence" is often used to describe machines that mimic cognitive functions that humans associate with the human mind, such as learning and problem solving.

As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI, a phenomenon known as the AI effect. For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. Modern machine capabilities generally classified as AI include successfully understanding human speech, competing at the highest level in strategic game systems, autonomously operating cars, intelligent routing in content delivery networks, and military simulations.

Artificial intelligence was founded as an academic discipline in 1955, and in the years since has experienced several waves of optimism, followed by disappointment and the loss of funding (known as an "AI winter"), followed by new approaches, success and renewed funding. After AlphaGo defeated a professional Go player in 2015, artificial intelligence once again attracted widespread global attention. For most of its history, AI research has been divided into sub-fields that often fail to communicate with each other. These sub-fields are based on technical considerations, such as particular goals, the use of particular tools, or deep philosophical differences.

The traditional problems (or goals) of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception and the ability to move and manipulate objects. General intelligence is among the field's long-term goals. Approaches include statistical methods, computational intelligence, and traditional symbolic AI. Many tools are used in AI, including versions of search and mathematical optimization, artificial neural networks, and methods based on statistics, probability and economics. The AI field draws upon computer science, information engineering, mathematics, psychology, linguistics, philosophy, and many other fields.

The field was founded on the assumption that human intelligence can be so precisely described that a machine can be made to simulate it. This raises philosophical arguments about the mind and the ethics of creating artificial beings endowed with human-like intelligence. These issues have been explored by myth, fiction and philosophy since antiquity. Science fiction and futurology have also suggested that, with its enormous potential and power, AI may become an existential risk to humanity."""
        
        self.input_text.delete(1.0, tk.END)
        self.input_text.insert(tk.END, example_text)
        self.status.config(text="Example text loaded. Click 'Generate Summary' to continue.")
    
    def generate_summary(self):
        input_text = self.input_text.get(1.0, tk.END).strip()
        
        if not input_text:
            messagebox.showwarning("Input Error", "Please enter some text to summarize.")
            return
        
        if len(input_text.split()) < 50:
            messagebox.showwarning("Input Error", "Text is too short for summarization. Please provide at least 50 words.")
            return
        
        self.status.config(text="Summarizing...")
        self.root.update()
        
        try:
            summary = self.summarizer.summarize(input_text)
            
            self.output_text.config(state="normal")
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, summary)
            self.output_text.config(state="disabled")
            
            self.status.config(text="Summary generated successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during summarization: {str(e)}")
            self.status.config(text="Error occurred during summarization.")
    
    def clear_text(self):
        self.input_text.delete(1.0, tk.END)
        self.output_text.config(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state="disabled")
        self.status.config(text="Ready")

if __name__ == "__main__":
    root = tk.Tk()
    app = SummarizerApp(root)
    root.mainloop()