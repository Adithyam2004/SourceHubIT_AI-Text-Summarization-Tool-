# AI Quiz Evaluator
# Now fetches questions from a server (example API) and evaluates student answers.

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AIQuizEvaluator:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def evaluate_answer(self, student_answer, correct_answer):
        """
        Compares student answer with correct answer using NLP (TF-IDF + cosine similarity).
        Returns a score (1 for correct, 0 for needs improvement) and similarity value.
        """
        answers = [student_answer, correct_answer]
        vectorizer = TfidfVectorizer().fit_transform(answers)
        vectors = vectorizer.toarray()
        similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        score = 1 if similarity >= self.threshold else 0
        return score, similarity

    def give_feedback(self, score, similarity):
        if score:
            return f"✅ Correct or sufficiently similar! (Similarity: {similarity:.2f})"
        else:
            return f"❌ Needs improvement. (Similarity: {similarity:.2f})\nReview the key points in the correct answer."

def fetch_questions(api_url=None):
    # Return hardcoded questions for local testing
    return [
        {"question": "What does AI stand for?", "answer": "Artificial Intelligence"},
        {"question": "What is NLP?", "answer": "Natural Language Processing"},
        {"question": "Name a benefit of AI in education.", "answer": "Personalized learning"},
        {"question": "What is machine learning?", "answer": "A subset of AI that enables systems to learn from data."},
        {"question": "What is the main advantage of instant feedback?", "answer": "Helps students quickly understand and correct mistakes."}
    ]

if __name__ == "__main__":
    questions = fetch_questions()
    evaluator = AIQuizEvaluator(threshold=0.7)
    total_score = 0
    total_asked = 0
    idx = 0
    print("Type 'cancel' at any time to stop the quiz.\n")
    while idx < len(questions):
        q = questions[idx]
        print(f"\nQ{idx+1}: {q['question']}")
        student = input("Your answer: ")
        if student.strip().lower() == "cancel":
            print("\nQuiz cancelled by student.")
            break
        score, similarity = evaluator.evaluate_answer(student, q['answer'])
        total_score += score
        total_asked += 1
        print(evaluator.give_feedback(score, similarity))
        idx += 1
    print(f"\nTotal Score: {total_score} / {total_asked}")