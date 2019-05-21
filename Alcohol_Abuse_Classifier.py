# Author: Koki Sasagawa
# Department: University of Michigan, Learning Health Sciences & School of Information
# n2c2 challenge 2018

import xml.etree.ElementTree as ET
import nltk
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

file_name = '/Users/koki/Documents/Notes/LHS_712_NLP_Health_Data/n2c2/train/111.xml'

# Words related to drinking. Use as key words.
ALCOHOL = ['drink',
           'alcohol',
           'etoh',
           'ethanol']

# Words for defining met cases:
# Some words retrieved from:
# https://www.merriam-webster.com/thesaurus/alcoholism
# https://www.projectknow.com/research/addiction-glossary-of-terms-and-phrases/
ALCOHOL_MODIFIER = ["abuse",
                    "addiction",
                    "binge",
                    "concern",
                    "dependence",
                    "excessive",
                    "heavy"]

ALCOHOL_MENTAL = ["anxiety",
                  "debauchery",
                  "depression",
                  "dispomania",
                  "dissoluteness",
                  "distraught",
                  "drunkenness",
                  "inebriety",
                  "insobriety",
                  "intemperance",
                  "intoxicate",
                  "bibulousness",]

# Words for negation detection
NEGATION = ["no",
            "without",
            "stop",
            "n't",
            "not",
            "h/o",
            "never",
            "none",
            "nor",
            "non",
            "rare",
            "previous",
            "prior",
            "history",
            "denies",
            "negative"]

# Run the stemmer on feature words
snowball_stemmer = SnowballStemmer("english")
stemmed_alcohol = [snowball_stemmer.stem(word) for word in ALCOHOL]
stemmed_alcohol_modifer = [snowball_stemmer.stem(word) for word in ALCOHOL_MODIFIER]
stemmed_alcohol_mental = [snowball_stemmer.stem(word) for word in ALCOHOL_MENTAL]
stemmed_negation = [snowball_stemmer.stem(word) for word in NEGATION]

# MAIN SCRIPT

# Distance to look for feature in certain direction
left_negation = 5
right_negation = 3
left_modifier = 2
right_modifier = 2

# Read file
def alcohol_abuse_classifier(file_name):
    tree=ET.parse(file_name)
    raw_text = tree.find('.//TEXT').text

    clean_text = re.sub('\\n', ' ', raw_text)
    clean_text = re.sub('\\t','', clean_text)
    clean_text = re.sub('[\s]{2,}', ' ', clean_text)

    sentences = nltk.sent_tokenize(clean_text)

    hotspot_lines = set()

    for i in sentences:
        # filter out anything non-alphabetical characters and a few special characters
        tokenizer = RegexpTokenizer(r'[a-zA-Z\/\']+')
        token = tokenizer.tokenize(i)

        snowball_stemmer = SnowballStemmer("english")
        stemmed_tokens = [snowball_stemmer.stem(word.lower()) for word in token]

        drink_score = 0
        abuse_score = 0
        token_count = len(stemmed_tokens)

        for j in range(token_count):
            if stemmed_tokens[j] in stemmed_alcohol:
                drink_score += 1

                # Negation detection in left direction
                for i in range(1, left_negation+1):
                    if (j >= i) and (stemmed_tokens[j - i] in stemmed_negation):
                        drink_score = 0
                        break

                # Negation detection in right direciton
                for i in range(1, right_negation+1):
                    if (j < token_count - i) and (stemmed_tokens[j + i] in stemmed_negation):
                        drink_score = 0
                        break

                # Modifier detection in left direction
                for i in range(1, left_modifier+1):
                    if (j >= i) and (stemmed_tokens[j - i] in stemmed_alcohol_modifer):
                        abuse_score += 1

                # Modifier detection in right direciton
                for i in range(1, right_modifier+1):
                    if (j < token_count - i) and (stemmed_tokens[j + i] in stemmed_alcohol_modifer):
                        abuse_score += 1

            # Mental Health Detection
            elif stemmed_tokens[j] in stemmed_alcohol_mental:
                abuse_score += 1

        if drink_score >= 1 and abuse_score >= 1:
            hotspot_lines.add(i)

    if hotspot_lines:
        return '<ALCOHOL-ABUSE met="met" />'
    else:
        return '<ALCOHOL-ABUSE met="not met" />'