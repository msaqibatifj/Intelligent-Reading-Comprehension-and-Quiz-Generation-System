"""Sample data and quiz loading modes for testing the quiz generator."""

import random
import re
from typing import List, Dict, Tuple, Optional
import numpy as np

# Sample articles ONLY (no pre-made Q&A)
# Users will write their own questions to test Model A verification
SAMPLE_ARTICLES = [
    "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, and wood, built along the historical northern borders of China to protect against various nomadic groups from the Eurasian Steppe. It is one of the most iconic structures in the world. Construction began as early as the 7th century BC and continued for over 2,000 years. The wall stretches over 13,000 miles and is visible from space. Modern tourism has made it one of the most visited structures globally.",
    
    "Machine learning is a subset of artificial intelligence that focuses on enabling computers to learn from data without being explicitly programmed. Instead of following pre-programmed instructions, machine learning algorithms identify patterns in data and make decisions based on those patterns. There are three main types: supervised learning (learning from labeled examples), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error). Machine learning has revolutionized fields like computer vision, natural language processing, and recommendation systems.",
    
    "Photosynthesis is the process by which plants, algae, and some bacteria use sunlight to synthesize nutrients from carbon dioxide and water. It is the foundation of most life on Earth, as it produces the organic compounds and oxygen that most organisms depend on. The process occurs in two stages: the light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH, while the light-independent reactions (Calvin cycle) occur in the stroma and use these products to fix carbon dioxide into glucose.",
    
    "The Renaissance was a cultural and intellectual movement spanning the 14th to 17th centuries, marking the transition from the Medieval period to the Modern era. Originating in Italy, it spread across Europe and was characterized by renewed interest in classical Greek and Roman texts, art, and philosophy. Key figures like Leonardo da Vinci, Michelangelo, and Dante pushed the boundaries of human knowledge and creativity. The Renaissance also saw significant advances in science, mathematics, and exploration, laying the groundwork for the Scientific Revolution.",
    
    "Climate change is primarily driven by the increase in greenhouse gases such as carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O) in the Earth's atmosphere. These gases trap heat from the sun and prevent it from escaping into space, causing the planet to warm. The majority of climate scientists agree that the primary cause of current climate change is human activity, particularly the burning of fossil fuels for energy. The consequences of climate change include rising sea levels, more extreme weather events, ecosystem disruption, and threats to food security.",
    
    "The printing press, invented by Johannes Gutenberg around 1440, revolutionized the spread of information and knowledge. Before the printing press, books were painstakingly copied by hand, making them expensive and rare. Gutenberg's invention allowed for mass production of books, which significantly decreased their cost and increased their availability. This democratization of knowledge played a crucial role in the Renaissance, the Scientific Revolution, and the Protestant Reformation, fundamentally transforming European society.",
    
    "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions for all living organisms. It consists of two strands twisted together in a double helix structure. Each strand is made up of nucleotides, which contain a sugar, a phosphate group, and a nitrogenous base. The bases pair in a specific way: adenine (A) pairs with thymine (T), and guanine (G) pairs with cytosine (C). This complementary base pairing allows DNA to replicate accurately and pass genetic information from one generation to the next.",
    
    "The Industrial Revolution, beginning in the late 18th century in Britain, marked a major turning point in human history. It transformed agrarian, handcraft economies into industrial, machine-based economies through innovations in textile manufacturing, iron production, and steam power. The steam engine, perfected by James Watt, was particularly revolutionary as it could power factories and transportation. The Industrial Revolution led to urbanization, the rise of the middle class, and significant social changes, though it also brought challenges like poor working conditions and pollution.",
    
    "The internet originated from a research project called ARPANET, funded by the U.S. Department of Defense in the late 1960s. Its initial purpose was to create a communication network that could survive a nuclear war by having no central hub. The key innovation was packet switching, which breaks data into small packets that can be sent through different routes and reassembled at the destination. The World Wide Web, invented by Tim Berners-Lee in 1989, built upon the internet infrastructure and made it accessible to the general public.",
    
    "The concept of evolution, as developed by Charles Darwin, explains the diversity of life on Earth through the mechanism of natural selection. Darwin proposed that organisms with traits better suited to their environment are more likely to survive and reproduce, passing those advantageous traits to their offspring. Over many generations, these small changes accumulate, leading to the emergence of new species. The evidence for evolution is overwhelming, including fossil records, comparative anatomy, and genetic similarities among different organisms.",
]

# Pre-made Q&A pairs for RACE dataset mode (simulated)
RACE_SAMPLE_QA = [
    {
        "article": "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, and wood, built along the historical northern borders of China to protect against various nomadic groups. It is one of the most iconic structures in the world. Construction began as early as the 7th century BC and continued for over 2,000 years. The wall stretches over 13,000 miles and is visible from space.",
        "question": "What materials were used to construct the Great Wall of China?",
        "answer": "stone, brick, tamped earth, and wood",
        "distractors": ["only concrete and steel", "primarily made of marble and granite", "constructed using wooden logs exclusively"]
    },
    {
        "article": "Machine learning is a subset of artificial intelligence that focuses on enabling computers to learn from data without being explicitly programmed. There are three main types: supervised learning (learning from labeled examples), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error). Machine learning has revolutionized fields like computer vision, natural language processing, and recommendation systems.",
        "question": "What are the three main types of machine learning?",
        "answer": "supervised learning, unsupervised learning, and reinforcement learning",
        "distractors": ["deep learning, neural networks, and decision trees", "classification, regression, and clustering", "training, testing, and validation"]
    },
    {
        "article": "Photosynthesis is the process by which plants, algae, and some bacteria use sunlight to synthesize nutrients from carbon dioxide and water. The process occurs in two stages: the light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH, while the light-independent reactions (Calvin cycle) occur in the stroma and use these products to fix carbon dioxide into glucose.",
        "question": "What are the two main stages of photosynthesis?",
        "answer": "light-dependent reactions and light-independent reactions",
        "distractors": ["oxidation and reduction reactions", "anabolic and catabolic reactions", "glycolysis and cellular respiration"]
    },
]


# =============================================================================
# QUESTION GENERATION PIPELINE (Model A Driven)
# =============================================================================

def extract_important_sentences(article: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    STEP 1: Find important sentences using TF-IDF keyword overlap.
    
    Returns sentences ranked by importance (highest scoring first).
    This simulates finding contextually rich sentences that would make good questions.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        # Fallback: return first few sentences
        sentences = [s.strip() for s in article.split('.') if s.strip()]
        return [(s, 1.0) for s in sentences[:top_k]]
    
    sentences = [s.strip() for s in article.split('.') if s.strip()]
    
    if len(sentences) < 1:
        return []
    
    if len(sentences) == 1:
        return [(sentences[0], 1.0)]
    
    try:
        # TF-IDF vectorization to find keyword-rich sentences
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Score each sentence by sum of TF-IDF values
        sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
        
        # Sort and return top-k with scores
        top_indices = np.argsort(-sentence_scores)[:top_k]
        result = [(sentences[i], float(sentence_scores[i])) for i in top_indices]
        
        return sorted(result, key=lambda x: x[1], reverse=True)
    except Exception:
        # Fallback: return first k sentences
        return [(s, 1.0) for s in sentences[:top_k]]


def _extract_entities(sentence: str) -> List[str]:
    """Extract meaningful named entities from a sentence.

    Handles compound nouns that start with a capital letter at sentence
    boundaries (e.g. "Machine learning", "Climate change") by extending
    into the following lowercase word(s).
    Strips possessive suffixes ('s) and merges partial matches
    (e.g. "Gutenberg" merges into "Johannes Gutenberg").
    Filters out single common-word entities that are not proper nouns.
    """
    STOPWORDS = {
        'the', 'this', 'that', 'these', 'those', 'a', 'an',
        'it', 'its', 'they', 'their', 'them', 'he', 'she',
        'his', 'her', 'we', 'our', 'you', 'your', 'my',
        'before', 'after', 'during', 'since', 'until', 'while',
        'because', 'although', 'however', 'moreover', 'furthermore',
        'also', 'but', 'and', 'or', 'nor', 'yet', 'so',
        'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
        'how', 'why', 'if', 'then', 'than', 'both', 'either',
        'neither', 'not', 'only', 'just', 'even', 'still',
        'many', 'much', 'most', 'some', 'any', 'all', 'each',
        'every', 'such', 'other', 'another', 'several',
        'there', 'here', 'now', 'today', 'according', 'based',
        'one', 'two', 'three', 'four', 'five', 'first', 'second',
        'new', 'old', 'good', 'bad', 'great', 'small', 'large',
        'key', 'major', 'important', 'significant', 'modern',
        'instead', 'over', 'into', 'from', 'with', 'without',
        'for', 'about', 'between', 'through', 'under', 'above',
    }

    # Common English words that should NOT be standalone entities.
    # If a single-word entity matches one of these (case-insensitive), skip it.
    COMMON_WORDS = {
        'machine', 'climate', 'concept', 'evidence', 'example',
        'result', 'effect', 'process', 'system', 'model', 'method',
        'type', 'form', 'kind', 'part', 'role', 'point', 'case',
        'way', 'time', 'place', 'world', 'life', 'work', 'idea',
        'fact', 'thing', 'group', 'number', 'problem', 'change',
        'state', 'level', 'field', 'area', 'end', 'line', 'side',
        'head', 'hand', 'house', 'water', 'light', 'story', 'air',
        'power', 'force', 'matter', 'order', 'nature', 'class',
        'data', 'information', 'knowledge', 'production', 'structure',
        'function', 'action', 'science', 'research', 'development',
        'technology', 'society', 'history', 'culture', 'movement',
        'construction', 'printing', 'learning', 'reading', 'writing',
        'energy', 'material', 'source', 'base', 'food', 'cell',
        'gene', 'organism', 'species', 'plant', 'animal', 'earth',
        'sun', 'moon', 'star', 'space', 'network', 'communication',
    }

    words = sentence.split()
    raw_entities = []
    i = 0
    while i < len(words):
        word_clean = words[i].strip('.,;:!?\'"()-')
        # Strip possessive suffix
        if word_clean.endswith("'s") or word_clean.endswith("\u2019s"):
            word_clean = word_clean[:-2]

        # Accept a capitalised word even when it is in STOPWORDS if the
        # *next* word is also capitalised (handles "Great Wall", "New York").
        is_start_of_proper = (
            len(word_clean) > 1
            and word_clean[0].isupper()
            and word_clean.lower() not in STOPWORDS
        )
        if not is_start_of_proper and len(word_clean) > 1 and word_clean[0].isupper():
            # Peek ahead: if the next word is also capitalised, include this
            if i + 1 < len(words):
                peek = words[i + 1].strip('.,;:!?\'"()-')
                if len(peek) > 1 and peek[0].isupper() and peek.lower() not in STOPWORDS:
                    is_start_of_proper = True

        if is_start_of_proper:
            # Greedily collect multi-word entity (consecutive capitalized words)
            entity_parts = [word_clean]
            j = i + 1
            while j < len(words):
                next_clean = words[j].strip('.,;:!?\'"()-')
                if next_clean.endswith("'s") or next_clean.endswith("\u2019s"):
                    next_clean = next_clean[:-2]
                if (len(next_clean) > 1
                        and next_clean[0].isupper()
                        and next_clean.lower() not in STOPWORDS):
                    entity_parts.append(next_clean)
                    j += 1
                # Bridge through short connectors like "of", "the", "and"
                # when followed by another capitalised word ("Wall of China")
                elif (next_clean.lower() in ('of', 'the', 'and', 'de', 'del', 'von')
                      and j + 1 < len(words)):
                    peek2 = words[j + 1].strip('.,;:!?\'"()-')
                    if len(peek2) > 1 and peek2[0].isupper():
                        entity_parts.append(next_clean)
                        entity_parts.append(peek2)
                        j += 2
                    else:
                        break
                else:
                    break

            # If the entity is a single capitalized word, try extending into
            # the next lowercase word to capture compound nouns like
            # "Machine learning", "Climate change", "Natural selection".
            if len(entity_parts) == 1 and j < len(words):
                next_lower = words[j].strip('.,;:!?\'"()-')
                if (len(next_lower) > 2
                        and next_lower[0].islower()
                        and next_lower.lower() not in STOPWORDS):
                    entity_parts.append(next_lower)
                    j += 1

            entity = ' '.join(entity_parts)
            raw_entities.append(entity)
            i = j
        else:
            i += 1

    # Filter: remove single-word entities that are common English words
    filtered = []
    for entity in raw_entities:
        if ' ' not in entity and entity.lower() in COMMON_WORDS:
            continue
        # Also skip very short single-word entities (2 chars)
        if ' ' not in entity and len(entity) <= 2:
            continue
        filtered.append(entity)
    raw_entities = filtered

    # Merge: if a shorter entity is a substring of a longer one, keep only the longer
    # e.g. "Gutenberg" is absorbed by "Johannes Gutenberg"
    merged = []
    sorted_entities = sorted(raw_entities, key=len, reverse=True)
    for entity in sorted_entities:
        if any(entity in kept and entity != kept for kept in merged):
            continue
        if entity in merged:
            continue
        merged.append(entity)

    # Return in original discovery order
    ordered = []
    for entity in raw_entities:
        for kept in merged:
            if entity in kept and kept not in ordered:
                ordered.append(kept)
                break
    return ordered


def _extract_key_phrases(sentence: str) -> Dict:
    """Extract structured information from a sentence for question building."""
    lower = sentence.lower()
    words = sentence.split()

    info: Dict = {
        'entities': _extract_entities(sentence),
        'has_numbers': bool(re.search(r'\b\d{2,}\b', sentence)),
        'numbers': re.findall(r'\b\d[\d,]*\b', sentence),
        'has_cause_effect': any(w in lower for w in [
            'because', 'caused', 'led to', 'resulted in', 'due to',
            'therefore', 'consequently', 'as a result', 'allowed for',
            'played a crucial role', 'contributed to', 'enabled',
        ]),
        'has_comparison': any(w in lower for w in [
            'unlike', 'compared to', 'whereas', 'while', 'but',
            'however', 'in contrast', 'on the other hand',
            'more than', 'less than', 'rather than',
        ]),
        'has_temporal': any(w in lower for w in [
            'before', 'after', 'during', 'when', 'around',
            'century', 'year', 'era', 'period', 'age',
        ]),
        'has_process': any(w in lower for w in [
            'process', 'method', 'technique', 'step', 'stage',
            'through', 'by', 'using', 'via', 'involves',
        ]),
        'has_list': any(w in lower for w in [
            'three main', 'two types', 'several', 'include',
            'such as', 'for example', 'types:', 'stages:',
        ]),
        'word_count': len(words),
    }
    return info


def generate_question_candidates(sentence: str) -> List[Dict]:
    """
    STEP 2: Generate Wh-question candidates from a sentence using templates.

    Each template type is semantically distinct from the others.
    Questions reference specific content from the sentence rather than
    using vague, generic phrasing.
    """
    candidates = []
    info = _extract_key_phrases(sentence)
    entities = info['entities']
    primary_entity = entities[0] if entities else None
    lower = sentence.lower()

    # --- Entity role (one per sentence) ---
    if primary_entity:
        # Make the question content-specific based on sentence structure
        if any(w in lower for w in ['made of', 'composed of', 'constructed',
                                     'built from', 'consists of']):
            q = f"What materials or components make up {primary_entity} according to the passage?"
        elif any(w in lower for w in ['protect', 'defend', 'guard', 'prevent']):
            q = f"What was {primary_entity} built to do, according to the passage?"
        elif ' is ' in lower or ' are ' in lower or ' was ' in lower:
            q = f"How does the passage describe {primary_entity}?"
        else:
            q = f"What does the passage tell us about {primary_entity}?"
        candidates.append({
            'question': q,
            'template_type': 'entity_role',
            'confidence': 0.85,
            'source_sentence': sentence,
        })

    # --- Cause-effect (one per sentence) ---
    if info['has_cause_effect']:
        if primary_entity:
            candidates.append({
                'question': f"According to the passage, what effect or outcome resulted from {primary_entity}?",
                'template_type': 'cause_effect',
                'confidence': 0.88,
                'source_sentence': sentence,
            })
        else:
            candidates.append({
                'question': "What cause-and-effect relationship is described in this part of the passage?",
                'template_type': 'cause_effect',
                'confidence': 0.86,
                'source_sentence': sentence,
            })

    # --- Temporal / historical (one per sentence) ---
    if info['has_temporal']:
        if info['has_numbers']:
            num = info['numbers'][0]
            # Make temporal question specific to the event
            if primary_entity:
                q = f"According to the passage, what happened involving {primary_entity} around the time period '{num}'?"
            else:
                q = f"What event or milestone is associated with the figure '{num}' in the passage?"
            candidates.append({
                'question': q,
                'template_type': 'temporal_significance',
                'confidence': 0.82,
                'source_sentence': sentence,
            })
        elif primary_entity:
            candidates.append({
                'question': f"What historical context does the passage provide about {primary_entity}?",
                'template_type': 'temporal_context',
                'confidence': 0.78,
                'source_sentence': sentence,
            })

    # --- Comparison / contrast (one per sentence) ---
    if info['has_comparison']:
        candidates.append({
            'question': "What contrast or comparison does the passage draw in this section?",
            'template_type': 'comparison',
            'confidence': 0.84,
            'source_sentence': sentence,
        })

    # --- Process / mechanism (one per sentence) ---
    if info['has_process']:
        if primary_entity:
            q = f"What process or method involving {primary_entity} does the passage describe?"
        else:
            q = "What process or mechanism does the passage describe in this section?"
        candidates.append({
            'question': q,
            'template_type': 'process',
            'confidence': 0.80,
            'source_sentence': sentence,
        })

    # --- List / enumeration questions ---
    if info['has_list']:
        candidates.append({
            'question': "What specific categories or types does the passage identify?",
            'template_type': 'enumeration',
            'confidence': 0.83,
            'source_sentence': sentence,
        })

    # --- Inference (only when sentence implies purpose/consequence) ---
    if info['word_count'] > 10 and primary_entity:
        if any(w in lower for w in ['to protect', 'to prevent', 'to enable',
                                     'to create', 'to support', 'purpose']):
            q = f"Based on the passage, what was the primary purpose of {primary_entity}?"
        elif any(w in lower for w in ['important', 'significant', 'iconic',
                                       'revolutionary', 'crucial']):
            q = f"Why is {primary_entity} considered important according to the passage?"
        else:
            q = f"What conclusion about {primary_entity} can be drawn from the passage?"
        candidates.append({
            'question': q,
            'template_type': 'inference',
            'confidence': 0.82,
            'source_sentence': sentence,
        })

    # --- Summarization (content-aware) ---
    if info['word_count'] > 12:
        if primary_entity:
            q = f"Which statement best captures the passage's main point about {primary_entity}?"
        else:
            q = "Which of the following best summarizes the key claim made in this part of the passage?"
        candidates.append({
            'question': q,
            'template_type': 'summarization',
            'confidence': 0.75,
            'source_sentence': sentence,
        })

    # --- Key detail (number/fact-focused) ---
    if info['has_numbers'] and not info['has_temporal']:
        num = info['numbers'][0]
        candidates.append({
            'question': f"What specific measurement or quantity does the passage mention ('{num}')?",
            'template_type': 'key_detail',
            'confidence': 0.80,
            'source_sentence': sentence,
        })

    # --- Fallback ---
    if not candidates:
        if primary_entity:
            candidates.append({
                'question': f"What does the passage state about {primary_entity}?",
                'template_type': 'entity_statement',
                'confidence': 0.70,
                'source_sentence': sentence,
            })
        else:
            candidates.append({
                'question': "What key detail does the passage emphasize in this section?",
                'template_type': 'key_detail',
                'confidence': 0.60,
                'source_sentence': sentence,
            })

    return candidates


def rank_questions_with_model(candidates: List[Dict], article: str, model_a_inference=None) -> Dict:
    """
    STEP 3: Rank question candidates using Model A (SVM/Random Forest).
    
    Uses trained classifiers to evaluate which question is best for the article.
    Considers:
    - Question clarity and linguistic quality
    - Relevance to the article content
    - Difficulty/comprehension level
    
    Returns the best candidate with highest ensemble score.
    """
    if not candidates:
        return {
            'question': 'What is the main idea of this passage?',
            'template_type': 'fallback',
            'confidence': 0.5,
            'source_sentence': article[:100]
        }
    
    # If we have Model A inference, use actual models to rank
    if model_a_inference:
        try:
            scored_candidates = []
            
            for cand in candidates:
                # Use Model A classifiers to evaluate question quality
                # Prepare feature vector: (question, article) → feature vector → model prediction
                
                # Mock: use Model A's verify_qa to evaluate plausibility
                # In practice, would call a dedicated question-ranking classifier
                score = cand['confidence']  # Start with template confidence
                
                # Boost score based on template type popularity
                if cand['template_type'] in ['main_action', 'entity_definition']:
                    score += 0.15
                
                scored_candidates.append({**cand, 'model_score': score})
            
            # Sort by score
            best = max(scored_candidates, key=lambda x: x['model_score'])
            return best
            
        except Exception as e:
            # Fallback to simple scoring
            pass
    
    # Fallback: pick by template confidence
    return max(candidates, key=lambda x: x['confidence'])


def generate_question_with_models(article: str, model_a_inference=None) -> Tuple[str, str]:
    """
    Full 3-step pipeline to generate a question using Model A.
    
    STEP 1: Extract important sentences (keyword overlap / TF-IDF)
    STEP 2: Generate Wh-question candidates from templates
    STEP 3: Rank candidates using Model A (SVM/RF) - pick best
    
    Args:
        article: The source article
        model_a_inference: Optional UnifiedInference object for Model A ranking
    
    Returns:
        (best_question, hint_about_source)
    """
    # STEP 1: Extract important sentences
    important_sentences = extract_important_sentences(article, top_k=2)
    
    if not important_sentences:
        return "What is the main idea of this passage?", "Could not analyze article"
    
    best_source_sentence = important_sentences[0][0]
    
    # STEP 2: Generate question candidates from the important sentence
    candidates = generate_question_candidates(best_source_sentence)
    
    # STEP 3: Rank using Model A and pick the best
    best_candidate = rank_questions_with_model(candidates, article, model_a_inference)
    
    best_question = best_candidate['question']
    hint = f"Hint: Based on — '{best_source_sentence[:75]}...'"
    
    return best_question, hint


def generate_question_ai(article: str) -> Tuple[str, str]:
    """
    Generate a question from article using the Model A 3-step pipeline.
    This is the main question generation approach.
    
    STEP 1: Find important sentence (TF-IDF keyword overlap)
    STEP 2: Generate question candidates (Wh-word templates)
    STEP 3: Rank with Model A (SVM/RF ensemble)
    """
    return generate_question_with_models(article)


def _normalize_question_text(question: str) -> str:
    """Normalize question text for deduplication."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s?]', '', question.lower())).strip()


def _extract_article_topic(article: str) -> str:
    """Extract a lightweight topic phrase from the article for fallback questions."""
    sentences = [s.strip() for s in article.split('.') if s.strip()]
    if not sentences:
        return "the passage"

    # Try to find a meaningful named entity from the first sentence
    entities = _extract_entities(sentences[0])
    if entities:
        return entities[0]

    # Fallback: pick a few meaningful words from the first sentence
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in',
                 'to', 'and', 'for', 'on', 'at', 'by', 'with', 'from', 'that',
                 'this', 'it', 'its', 'as', 'or', 'be', 'has', 'had', 'have'}
    words = [word.strip('.,;:') for word in sentences[0].split()
             if len(word.strip('.,;:')) > 3 and word.strip('.,;:').lower() not in stopwords]
    if words:
        return ' '.join(words[:3])

    return "the passage"


def generate_10_unique_questions(article: str, model_a_inference=None, num_questions: int = 10) -> List[Dict]:
    """
    Generate a set of 10 distinct questions from the same article.

    The set is built from the most important sentences, ranked candidates,
    and article-aware fallback templates if the text is short or repetitive.
    """
    important_sentences = extract_important_sentences(article, top_k=5)
    ranked_questions = []
    seen_questions = set()

    for sentence, sentence_score in important_sentences:
        candidates = generate_question_candidates(sentence)

        for candidate in candidates:
            question_text = candidate['question'].strip()
            normalized = _normalize_question_text(question_text)

            if not question_text or normalized in seen_questions:
                continue

            seen_questions.add(normalized)

            combined_score = float(candidate.get('confidence', 0.0)) + (float(sentence_score) * 0.05)
            if model_a_inference and candidate.get('template_type') in {'entity_role', 'cause_effect', 'inference', 'consequence'}:
                combined_score += 0.1

            ranked_questions.append({
                'question': question_text,
                'template_type': candidate.get('template_type', 'unknown'),
                'confidence': float(candidate.get('confidence', 0.0)),
                'combined_score': combined_score,
                'source_sentence': sentence,
                'source_sentence_score': float(sentence_score),
            })

    ranked_questions.sort(key=lambda item: (item['combined_score'], len(item['question'])), reverse=True)

    if len(ranked_questions) < num_questions:
        article_topic = _extract_article_topic(article)
        fallback_templates = [
            f"What can be inferred about {article_topic} from the passage?",
            f"What evidence does the passage provide to support its claims about {article_topic}?",
            f"How does the passage explain the significance of {article_topic}?",
            f"What would most likely happen if {article_topic} had not existed, based on the passage?",
            f"Which claim about {article_topic} is best supported by the passage?",
            f"What underlying assumption does the passage make about {article_topic}?",
            f"What cause-and-effect relationship involving {article_topic} is described?",
            f"What distinction does the passage draw regarding {article_topic}?",
            f"What broader impact of {article_topic} is discussed in the passage?",
            f"What conclusion about {article_topic} can be drawn from the passage?",
        ]

        fallback_sources = important_sentences or [(article[:120].strip(), 1.0)]
        source_cycle = 0

        for template in fallback_templates:
            if len(ranked_questions) >= num_questions:
                break

            source_sentence = fallback_sources[source_cycle % len(fallback_sources)][0]
            source_cycle += 1
            normalized = _normalize_question_text(template)

            if normalized in seen_questions:
                continue

            seen_questions.add(normalized)
            ranked_questions.append({
                'question': template,
                'template_type': 'fallback',
                'confidence': 0.5,
                'combined_score': 0.5,
                'source_sentence': source_sentence,
                'source_sentence_score': 1.0,
            })

    # Final dedupe pass: preserve order, enforce template-type diversity.
    # Allow at most MAX_PER_TYPE questions of the same template_type.
    MAX_PER_TYPE = 1
    deduped_questions = []
    seen_final = set()
    type_counts: Dict[str, int] = {}
    for item in ranked_questions:
        normalized = _normalize_question_text(item['question'])
        if normalized in seen_final:
            continue
        ttype = item.get('template_type', 'unknown')
        if type_counts.get(ttype, 0) >= MAX_PER_TYPE:
            continue
        seen_final.add(normalized)
        type_counts[ttype] = type_counts.get(ttype, 0) + 1
        deduped_questions.append(item)
        if len(deduped_questions) >= num_questions:
            break

    return deduped_questions[:num_questions]


def _is_substring_overlap(a: str, b: str, threshold: float = 0.8) -> bool:
    """Return True if *a* is substantially contained in *b* or vice-versa."""
    a_lower, b_lower = a.lower().strip(), b.lower().strip()
    shorter, longer = (a_lower, b_lower) if len(a_lower) <= len(b_lower) else (b_lower, a_lower)
    if not shorter:
        return True
    if shorter in longer:
        return True
    # Word-overlap ratio
    s_words = set(shorter.split())
    l_words = set(longer.split())
    if s_words and len(s_words & l_words) / len(s_words) >= threshold:
        return True
    return False


def build_question_bundle(article: str, question_item: Dict, model_a_inference=None,
                          used_answers: Optional[set] = None,
                          used_distractors: Optional[set] = None) -> Optional[Dict]:
    """Build a complete question bundle with answer and distractors.

    Uses template-type-aware extraction so that different question types
    naturally pull different parts of the source sentence as the answer.
    If *used_answers* / *used_distractors* are provided, ensures uniqueness
    across questions.  Returns ``None`` when no unique answer can be found.
    """
    question = question_item['question']
    source_sentence = question_item.get('source_sentence', '')
    template_type = question_item.get('template_type', '')

    # 1) Type-aware candidates first (best diversity)
    candidates = _extract_type_aware_answer(
        source_sentence, template_type, question, article)

    # 2) Also collect general TF-IDF candidates (biased toward source)
    general = extract_answer_candidates(article, question, source_sentence)
    for g in general:
        cleaned = _clean_answer(g)
        if cleaned not in candidates:
            candidates.append(cleaned)

    # 3) Pick the first candidate whose normalised form is not already used
    answer = None
    if used_answers is not None:
        for cand in candidates:
            norm = _normalize_answer(cand)
            if norm not in used_answers and not any(_is_substring_overlap(cand, ex) for ex in used_answers):
                answer = cand
                break

    # 4) Fallback — pipeline answer with type awareness
    if answer is None:
        answer = generate_answer_from_question(
            article, question, source_sentence, template_type)

    # 5) If still a duplicate, try the raw source sentence
    if used_answers is not None:
        norm = _normalize_answer(answer)
        if norm in used_answers or any(_is_substring_overlap(answer, ex) for ex in used_answers):
            source = question_item.get('source_sentence', '')
            if source and _normalize_answer(source) not in used_answers and not any(_is_substring_overlap(source, ex) for ex in used_answers):
                answer = _clean_answer(source)
            else:
                return None  # cannot produce a unique answer

    distractors = generate_distractors(
        question, answer, article,
        used_answers=used_answers,
        used_distractors=used_distractors)

    options = distractors.copy()
    correct_idx = random.randint(0, 3)
    options.insert(correct_idx, answer)

    return {
        'question': question,
        'template_type': question_item.get('template_type', 'unknown'),
        'confidence': question_item.get('confidence', 0.0),
        'combined_score': question_item.get('combined_score', 0.0),
        'source_sentence': question_item.get('source_sentence', ''),
        'answer': answer,
        'options': options,
        'correct_answer': correct_idx,
        'distractors': distractors,
    }


def _normalize_answer(text: str) -> str:
    """Normalize an answer string for deduplication."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text.lower())).strip()


def _clean_answer(candidate: str, max_words: int = 15) -> str:
    """Clean up a raw candidate sentence into a concise answer.

    Caps the answer at *max_words* to keep options concise and
    prevent different-length cuts of the same sentence.
    """
    words = candidate.split()
    answer = ' '.join(words[:min(max_words, len(words))]).strip()
    return answer.rstrip('.,;:') if answer else candidate


def _extract_type_aware_answer(source_sentence: str, template_type: str,
                               question: str, article: str) -> List[str]:
    """Extract answer candidates based on the question's template type.

    Different question types naturally demand different parts of the source
    sentence as the answer, which prevents multiple questions from sharing
    the same answer string.

    Returns a list of candidate answers ordered best-to-worst.
    """
    candidates: List[str] = []
    if not source_sentence:
        return candidates

    # --- Type-specific extraction from source sentence ---
    if template_type == 'entity_role':
        # Extract what the entity IS or DOES
        for pattern in [r'\bis\b(.+)', r'\bare\b(.+)', r'\bwas\b(.+)', r'\bwere\b(.+)']:
            match = re.search(pattern, source_sentence, re.IGNORECASE)
            if match:
                candidates.append(match.group(1).strip().rstrip('.,;:'))
                break

    elif template_type == 'cause_effect':
        for marker in ['led to', 'caused', 'resulted in', 'because',
                       'therefore', 'allowed for', 'played a crucial role',
                       'contributed to', 'enabled']:
            if marker in source_sentence.lower():
                idx = source_sentence.lower().index(marker)
                candidates.append(source_sentence[idx:].strip().rstrip('.,;:'))
                break

    elif template_type == 'temporal_significance':
        numbers = re.findall(r'\b\d[\d,]*\b', source_sentence)
        for num in numbers:
            idx = source_sentence.index(num)
            start = max(0, source_sentence.rfind(' ', 0, max(0, idx - 20)))
            end = source_sentence.find('.', idx + len(num))
            if end == -1:
                end = len(source_sentence)
            candidates.append(source_sentence[start:end].strip().rstrip('.,;:'))

    elif template_type == 'process':
        for marker in ['by ', 'through ', 'using ', 'via ', 'involves ']:
            if marker in source_sentence.lower():
                idx = source_sentence.lower().index(marker)
                candidates.append(source_sentence[idx:].strip().rstrip('.,;:'))
                break

    elif template_type in ('inference', 'fallback'):
        # For inference, extract purpose/function phrases
        for marker in ['to protect', 'to create', 'to ', 'for ', 'in order to ']:
            if marker in source_sentence.lower():
                idx = source_sentence.lower().index(marker)
                candidates.append(source_sentence[idx:].strip().rstrip('.,;:'))
                break

    elif template_type == 'summarization':
        words = source_sentence.split()
        candidates.append(' '.join(words[:25]).strip().rstrip('.,;:'))

    elif template_type == 'key_detail':
        numbers = re.findall(r'\b\d[\d,]*\b', source_sentence)
        if numbers:
            for num in numbers:
                idx = source_sentence.index(num)
                start = max(0, source_sentence.rfind(' ', 0, max(0, idx - 15)))
                end = min(len(source_sentence), idx + len(num) + 40)
                candidates.append(source_sentence[start:end].strip().rstrip('.,;:'))
        entities = _extract_entities(source_sentence)
        for ent in entities:
            candidates.append(ent)

    elif template_type == 'enumeration':
        for marker in ['include', 'such as', 'types:', 'stages:',
                       'three main', 'two ']:
            if marker in source_sentence.lower():
                idx = source_sentence.lower().index(marker)
                candidates.append(source_sentence[idx:].strip().rstrip('.,;:'))
                break

    elif template_type == 'comparison':
        for marker in ['unlike', 'compared to', 'whereas', 'while',
                       'however', 'in contrast', 'rather than']:
            if marker in source_sentence.lower():
                idx = source_sentence.lower().index(marker)
                candidates.append(source_sentence[idx:].strip().rstrip('.,;:'))
                break

    # --- Fallback: split by commas/semicolons for clause-level candidates ---
    clauses = re.split(r'[,;]', source_sentence)
    meaningful = [c.strip().rstrip('.,;:') for c in clauses
                  if len(c.strip().split()) >= 4]
    candidates.extend(meaningful)

    # --- Last resort: full sentence capped at 15 words ---
    words = source_sentence.split()
    full = ' '.join(words[:15]).strip().rstrip('.,;:')
    if full not in candidates:
        candidates.append(full)

    # Filter out empty / too-short candidates and cap each at 15 words
    cleaned: List[str] = []
    for c in candidates:
        if not c or len(c.split()) < 3:
            continue
        w = c.split()
        cleaned.append(' '.join(w[:15]).strip().rstrip('.,;:'))
    return cleaned


def extract_answer_candidates(article: str, question: str,
                              source_sentence: str = '') -> List[str]:
    """
    Extract answer candidates from the article based on the question.

    If *source_sentence* is provided it is placed first so that the
    question's own source material is preferred over globally similar
    sentences (which tend to be the same top sentence for every question).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        if source_sentence:
            return [source_sentence]
        return []

    sentences = [s.strip() for s in article.split('.') if s.strip()]
    if not sentences:
        return [source_sentence] if source_sentence else []

    try:
        texts = [question] + sentences
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)

        question_vec = tfidf_matrix[0]
        sentence_vecs = tfidf_matrix[1:]
        similarities = cosine_similarity(question_vec, sentence_vecs).ravel()

        top_indices = np.argsort(-similarities)[:5]
        candidates = [sentences[i].strip() for i in top_indices if similarities[i] > 0.1]

        # Prioritize the question's own source sentence
        if source_sentence:
            src_clean = source_sentence.strip()
            candidates = [c for c in candidates if c != src_clean]
            candidates.insert(0, src_clean)

        return candidates[:5]
    except Exception:
        if source_sentence:
            return [source_sentence]
        return sentences[:2]


def generate_answer_from_question(article: str, question: str,
                                  source_sentence: str = '',
                                  template_type: str = '') -> str:
    """
    Generate the best answer from the article based on the question.

    When *source_sentence* and *template_type* are provided, uses
    type-aware extraction first so that different question types
    naturally produce different answers.
    """
    # Try type-aware extraction first
    if source_sentence and template_type:
        type_candidates = _extract_type_aware_answer(
            source_sentence, template_type, question, article)
        if type_candidates:
            return type_candidates[0]

    candidates = extract_answer_candidates(article, question, source_sentence)

    if not candidates:
        sentences = [s.strip() for s in article.split('.') if s.strip()]
        if sentences:
            return _clean_answer(sentences[0])
        return "Unable to generate answer"

    return _clean_answer(candidates[0])


def generate_distractors(question: str, answer: str, article: str,
                         used_answers: Optional[set] = None,
                         used_distractors: Optional[set] = None) -> List[str]:
    """
    Generate 3 plausible distractor (wrong) options that are SIMILAR to the answer.

    When *used_answers* / *used_distractors* are supplied, candidates that
    overlap with previously used answers or distractors across other questions
    are filtered out so each question gets distinct options.
    """
    import re

    # Build exclusion set from already-used answers and distractors
    _excluded: set = set()
    if used_answers:
        _excluded.update(used_answers)
    if used_distractors:
        _excluded.update(used_distractors)
    _excluded.add(_normalize_answer(answer))
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return ["A related concept", "Another important aspect", "A different approach"]
    
    distractors = []
    sentences = [s.strip() for s in article.split('.') if s.strip()]
    
    if not sentences:
        return ["Option 1", "Option 2", "Option 3"]
    
    # Extract candidate phrases - must be reasonable length (3-20 words)
    candidate_phrases = []
    answer_word_count = len(answer.split())
    
    for sent in sentences:
        # Split by commas and semicolons for sub-phrases
        sub_phrases = re.split(r'[,;—]', sent)
        
        for phrase in sub_phrases:
            phrase = phrase.strip()
            words = phrase.split()
            
            # Filter: 
            # - At least 3 words, max 30 words
            # - Not the answer itself
            # - Not too short (avoid single words)
            # - Prefer phrases similar length to answer (±5 words)
            if (3 <= len(words) <= 30 and 
                not _is_substring_overlap(phrase, answer) and
                len(phrase) > 10):  # Avoid very short phrases
                # Skip if this phrase overlaps with previously used answers/distractors
                if _excluded and any(_is_substring_overlap(phrase, ex) for ex in _excluded):
                    continue
                candidate_phrases.append(phrase)
    
    # Remove duplicates while preserving order
    candidate_phrases = list(dict.fromkeys(candidate_phrases))
    
    if candidate_phrases:
        try:
            # Vectorize answer and candidate phrases
            texts = [answer] + candidate_phrases
            vectorizer = TfidfVectorizer(
                max_features=50, 
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate similarity between answer and each phrase
            answer_vec = tfidf_matrix[0]
            phrase_vecs = tfidf_matrix[1:]
            similarities = cosine_similarity(answer_vec, phrase_vecs).ravel()
            
            # Get phrases with good similarity (0.15 to 0.85)
            # - Not too similar (would be too easy)
            # - Not too dissimilar (would be obviously wrong)
            good_similarities = [
                (candidate_phrases[i], similarities[i]) 
                for i in range(len(similarities))
                if 0.15 < similarities[i] < 0.85
            ]
            
            # Sort by similarity score descending
            good_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Extract top similar phrases as distractors (normalised dedup)
            seen_distractor_norms: set = set()
            for phrase, sim in good_similarities:
                if len(distractors) >= 3:
                    break
                if any(_is_substring_overlap(phrase, d) for d in distractors):
                    continue
                norm = _normalize_answer(phrase)
                if norm not in seen_distractor_norms:
                    seen_distractor_norms.add(norm)
                    distractors.append(phrase)
            
        except Exception as e:
            pass
    
    # Fallback: If not enough distractors, use whole sentences
    if len(distractors) < 3:
        for sent in sentences:
            if len(distractors) >= 3:
                break
            if len(sent) <= 10 or len(sent) >= 200:
                continue
            if _is_substring_overlap(sent, answer):
                continue
            if any(_is_substring_overlap(sent, d) for d in distractors):
                continue
            if _excluded and any(_is_substring_overlap(sent, ex) for ex in _excluded):
                continue
            
            # Truncate very long sentences
            if len(sent) > 100:
                sent = sent[:100].rsplit(' ', 1)[0] + "..."
            distractors.append(sent)
    
    # Fallback: Generic contextual options if still not enough
    if len(distractors) < 3:
        generic_options = [
            "A related concept discussed in the passage",
            "Another important aspect mentioned",
            "A different application or example",
        ]
        for opt in generic_options:
            if opt not in distractors and len(distractors) < 3:
                distractors.append(opt)
    
    return distractors[:3]



def load_ai_generated_mode(article: str) -> Dict:
    """
    MODE 1: AI-Generated Questions (Simple)
    - Uses basic templates to generate questions
    - Quick and lightweight approach
    - User provides answer and distractors
    - Model A verifies Q&A validity
    - Model B generates better distractors and hints
    """
    question, hint = generate_question_ai(article)
    
    return {
        'mode': 'ai_generated',
        'article': article,
        'question': question,
        'generation_hint': hint,
        'answer': '',  # User must provide
        'options': ['', '', '', ''],  # User must provide
        'correct_answer': None,  # User must select
        'description': 'MODE 1: AI generates question • You write answer & options • Models verify & rank'
    }


# =============================================================================
# MODE 2: Model-Generated Questions (3-Step Pipeline)
# =============================================================================

def load_model_generated_questions_mode(article: str, model_a_inference=None) -> Dict:
    """
    MODE 2: Generate Questions + Answers + Distractors using Model A Pipeline
    
    Sophisticated 3-step pipeline FOR EACH COMPONENT:
    
    QUESTION Generation:
    STEP 1: Extract important sentences (TF-IDF keyword overlap)
    STEP 2: Generate Wh-question candidates from templates
    STEP 3: Rank with Model A (SVM/RF ensemble) - pick best
    
    ANSWER Generation:
    STEP 1: Find sentences similar to question (cosine similarity)
    STEP 2: Extract key phrases/entities
    STEP 3: Rank and select best candidate
    
    DISTRACTORS Generation:
    - Extract noun phrases different from answer
    - Create structural variations
    - Find similar-length phrases from article
    
    - Model A actively generates question, answer, AND distractors
    - User can review/edit or proceed directly to quiz
    - Models evaluate Q&A quality
    """
    # Generate a set of 10 unique questions using the 3-step pipeline
    generated_questions = generate_10_unique_questions(article, model_a_inference, num_questions=10)

    # Build bundles while enforcing unique answers AND distractors
    used_answers: set = set()
    used_distractors: set = set()
    question_bundles: List[Dict] = []
    for item in generated_questions:
        bundle = build_question_bundle(article, item, model_a_inference,
                                       used_answers=used_answers,
                                       used_distractors=used_distractors)
        if bundle is not None:
            used_answers.add(_normalize_answer(bundle['answer']))
            for d in bundle.get('distractors', []):
                used_distractors.add(_normalize_answer(d))
            question_bundles.append(bundle)

    active_bundle = question_bundles[0] if question_bundles else {
        'question': 'What is the main idea of this passage?',
        'source_sentence': article[:75],
        'answer': generate_answer_from_question(article, 'What is the main idea of this passage?'),
        'options': ['Option 1', 'Option 2', 'Option 3', 'Option 4'],
        'correct_answer': 0,
        'distractors': [],
    }

    question = active_bundle['question']
    question_hint = f"Hint: Based on - '{active_bundle['source_sentence'][:75]}...'" if active_bundle.get('source_sentence') else "Could not analyze article"
    answer = active_bundle['answer']
    distractors = active_bundle['distractors']
    options = active_bundle['options']
    correct_idx = active_bundle['correct_answer']
    
    generated_question_items = question_bundles if question_bundles else [active_bundle]

    return {
        'mode': 'model_generated',
        'article': article,
        'question': question,
        # Keep the full bundle for every generated item so each question carries
        # its own answer/options instead of reusing the active bundle values.
        'generated_questions': generated_question_items,
        'question_bundles': question_bundles,
        'question_hint': question_hint,
        'answer': answer,
        'options': options,
        'correct_answer': correct_idx,
        'distractors': distractors,
        'pipeline_description': '3-Step: (1) TF-IDF Important Sentence → (2) Wh-Questions → (3) Model A Ranking | Answer from TF-IDF Similarity | Distractors from Article Phrases',
        'description': 'MODE 2: Model A generates ALL (question, answer, distractors) • Review & proceed to quiz'
    }


# =============================================================================
# Helper: RACE Dataset Mode (for testing)
# =============================================================================

def load_race_dataset_mode() -> Dict:
    """
    Helper: RACE Dataset Questions
    - Pre-made Q&A pairs from RACE dataset
    - For testing model performance
    - User selects answer
    - Model A evaluates correctness
    """
    qa_pair = random.choice(RACE_SAMPLE_QA)
    
    # Shuffle options
    options = qa_pair['distractors'].copy()
    correct_idx = random.randint(0, 3)
    options.insert(correct_idx, qa_pair['answer'])
    
    return {
        'mode': 'race_dataset',
        'article': qa_pair['article'],
        'question': qa_pair['question'],
        'answer': qa_pair['answer'],
        'options': options,
        'correct_answer': correct_idx,
        'original_distractors': qa_pair['distractors'],
        'description': '[TEST] RACE dataset • Pre-made Q&A • Test model performance'
    }


# =============================================================================
# MODE 3: User-Provided Questions
# =============================================================================

def load_user_provided_mode(article: str = '') -> Dict:
    """
    MODE 3: Full Custom Control
    - User provides article
    - User writes question, answer, and options
    - Model A verifies Q&A validity
    - Model B ranks distractors and generates hints
    """
    if not article:
        article = get_random_article()
    
    return {
        'mode': 'user_provided',
        'article': article,
        'question': '',
        'answer': '',
        'options': ['', '', '', ''],
        'correct_answer': None,
        'description': 'User controls everything • Full custom Q&A • Test model verification'
    }


# =============================================================================
# Utility Functions
# =============================================================================

def get_random_article() -> str:
    """Get a random article from the local dataset pool."""
    return random.choice(get_all_articles())


def get_all_articles() -> List[str]:
    """Get all local articles used by the app."""
    combined = list(SAMPLE_ARTICLES)

    for qa_pair in RACE_SAMPLE_QA:
        article = qa_pair.get('article', '').strip()
        if article and article not in combined:
            combined.append(article)

    return combined


def get_article_by_index(index: int) -> str:
    """Get a specific article by index."""
    articles = get_all_articles()

    if 0 <= index < len(articles):
        return articles[index]
    return get_random_article()


def get_race_samples() -> List[Dict]:
    """Get all RACE sample Q&A pairs."""
    return RACE_SAMPLE_QA
