# # Import necessary libraries and modules
# from django.shortcuts import render
# from django.http import HttpResponse
# from .models import UploadedFile
# import PyPDF2
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.probability import FreqDist
# from heapq import nlargest
# from sklearn.feature_extraction.text import TfidfVectorizer

# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from rest_framework import status


# # Download NLTK data (you can also include this part in your project setup)
# nltk.download("punkt")
# nltk.download("stopwords")

# # Define the PDFSummarizer class
# class PDFSummarizer:
#     def __init__(self, pdf_file):
#         self.pdf_file = pdf_file
#         self.text = self.extract_text_from_pdf()

#     def extract_text_from_pdf(self):
#         text = ""
#         with open(self.pdf_file, "rb") as file:
#             pdf_reader = PyPDF2.PdfReader(file)
#             num_pages = len(pdf_reader.pages)
#             for page_num in range(num_pages):
#                 page = pdf_reader.pages[page_num]
#                 text += page.extract_text()
#         return text

#     def preprocess_text(self):
#         words = word_tokenize(self.text.lower())
#         stop_words = set(stopwords.words("english"))
#         filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
#         return filtered_words

#     def get_keywords(self, filtered_words, num_keywords=5):
#         word_freq = FreqDist(filtered_words)
#         keywords = nlargest(num_keywords, word_freq, key=word_freq.get)
#         return keywords

#     def summarize_text(self, num_sentences=3):
#         vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
#         tfidf_matrix = vectorizer.fit_transform([self.text])
#         feature_names = vectorizer.get_feature_names_out()
#         dense = tfidf_matrix.todense()
#         sentence_scores = {}
#         for i in range(len(self.text.split("."))):
#             score = 0
#             for j in range(len(feature_names)):
#                 score += dense[0, vectorizer.vocabulary_[feature_names[j]]]
#             sentence_scores[i] = score
#         top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
#         summary = ' '.join([self.text.split(".")[j] for j in sorted(top_sentences)])
#         return summary

# # Define the view function
# def home(request):
#     print("home")
#     if request.method == 'POST' and request.FILES.get('pdf_file'):
#         pdf_file = request.FILES['pdf_file']
        
#         # Save the uploaded PDF file to the server
#         uploaded_file = UploadedFile(pdf_file=pdf_file)
#         uploaded_file.save()

#         # Get the path to the saved PDF file
#         pdf_file_path = uploaded_file.pdf_file.path
        
#         # Create a PDFSummarizer instance with the path to the PDF file
#         summarizer = PDFSummarizer(pdf_file_path)
        
#         # Extract text, preprocess, get keywords, and summarize
#         filtered_words = summarizer.preprocess_text()
#         keywords = summarizer.get_keywords(filtered_words)
#         summary = summarizer.summarize_text()
        
#         # Pass the results to the template
#         response_data = {'keywords': keywords, 'summary': summary}
        
#         return Response(response_data, status=status.HTTP_200_OK)
#     else:
#         return Response({'error': 'No PDF file uploaded'}, status=status.HTTP_400_BAD_REQUEST)


from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .models import UploadedFile
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import json
from django.views.decorators.csrf import csrf_exempt

# Download NLTK data (you can also include this part in your project setup)
nltk.download("punkt")
nltk.download("stopwords")

# Define the PDFSummarizer class

from heapq import nlargest

class TextSummarizer:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file

    def preprocess_text(self, text):
        # Removing Square Brackets and Extra Spaces
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        # Removing special characters and digits
        formatted_text = re.sub('[^a-zA-Z]', ' ', text)
        formatted_text = re.sub(r'\s+', ' ', formatted_text)
        return formatted_text

    def summarize_text(self, text, num_sentences=3):
        # Tokenize text into sentences
        sentence_list = nltk.sent_tokenize(text)
        # Preprocess text
        formatted_text = self.preprocess_text(text)
        # Tokenize and remove stopwords
        words = nltk.word_tokenize(formatted_text)
        stopwords = set(nltk.corpus.stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stopwords]
        # Calculate word frequencies
        word_frequencies = {}
        for word in filtered_words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
        # Normalize word frequencies
        maximum_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / maximum_frequency
        # Calculate sentence scores based on word frequencies
        sentence_scores = {}
        for sentence in sentence_list:
            for word in nltk.word_tokenize(sentence.lower()):
                if word in word_frequencies.keys():
                    if len(sentence.split(' ')) < 30:
                        if sentence not in sentence_scores.keys():
                            sentence_scores[sentence] = word_frequencies[word]
                        else:
                            sentence_scores[sentence] += word_frequencies[word]
        # Select top sentences based on scores
        summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)
        return summary

# Define the view function
@csrf_exempt
def summarize_text(request):
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body_data = json.loads(body_unicode)

        text = body_data.get('text', '')

        if(text):
            summarizer = TextSummarizer(text)  
        
            summary = summarizer.summarize_text(text)  # Pass text as argument

        # Prepare response data
            response_data = {'summary': summary}
            return JsonResponse(response_data, status=200)
        
        else:
            return JsonResponse({'error' : 'No Text Provided'}, status = 400)
    else:
        return JsonResponse({'error' : 'Unsupported req method'}, status = 405)

