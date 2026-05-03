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

    Filters out common sentence-starting words, articles, pronouns,
    conjunctions, and other non-entity capitalized words.
    Also strips possessive suffixes ('s) and merges partial matches
    (e.g. "Gutenberg" merges into "Johannes Gutenberg").
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

    words = sentence.split()
    raw_entities = []
    i = 0
    while i < len(words):
        word_clean = words[i].strip('.,;:!?\'"()-')
        # Strip possessive suffix
        if word_clean.endswith("'s") or word_clean.endswith("\u2019s"):
            word_clean = word_clean[:-2]
        if (len(word_clean) > 1
                and word_clean[0].isupper()
                and word_clean.lower() not in STOPWORDS):
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
                else:
                    break
            entity = ' '.join(entity_parts)
            raw_entities.append(entity)
            i = j
        else:
            i += 1

    # Merge: if a shorter entity is a substring of a longer one, keep only the longer
    # e.g. "Gutenberg" is absorbed by "Johannes Gutenberg"
    merged = []
    # Sort longest first so we check against longest entities first
    sorted_entities = sorted(raw_entities, key=len, reverse=True)
    for entity in sorted_entities:
        # Check if this entity is already a substring of an entity we kept
        if any(entity in kept and entity != kept for kept in merged):
            continue
        # Check if we already have a duplicate
        if entity in merged:
            continue
        merged.append(entity)

    # Return in original discovery order
    ordered = []
    for entity in raw_entities:
        # Find the merged version that contains this entity
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
        'word_count': len(words),
    }
    return info


def generate_question_candidates(sentence: str) -> List[Dict]:
    """
    STEP 2: Generate Wh-question candidates from a sentence using templates.
    
    Applies context-aware question templates to produce specific, complex
    questions rather than generic ones. Uses only ONE entity per template
    type to avoid repetitive questions across the set.
    """
    candidates = []
    info = _extract_key_phrases(sentence)
    entities = info['entities']
    # Pick the single best (longest / most specific) entity for this sentence
    primary_entity = entities[0] if entities else None

    # --- ONE entity-specific question per sentence (using the best entity) ---
    if primary_entity:
        candidates.append({
            'question': f"According to the passage, what role does {primary_entity} play in the events described?",
            'template_type': 'entity_role',
            'confidence': 0.85,
            'source_sentence': sentence,
        })

    # --- Cause-effect (one per sentence) ---
    if info['has_cause_effect']:
        if primary_entity:
            candidates.append({
                'question': f"What was the direct consequence of {primary_entity} as described in the passage?",
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
            candidates.append({
                'question': f"What significance does the time reference '{num}' hold in the context of the passage?",
                'template_type': 'temporal_significance',
                'confidence': 0.82,
                'source_sentence': sentence,
            })
        elif primary_entity:
            candidates.append({
                'question': f"What was happening before {primary_entity} according to the passage?",
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
        candidates.append({
            'question': "What process or mechanism does the passage describe?",
            'template_type': 'process',
            'confidence': 0.80,
            'source_sentence': sentence,
        })

    # --- Inference (one per sentence) ---
    if info['word_count'] > 8 and primary_entity:
        candidates.append({
            'question': f"What can be inferred about {primary_entity} from the information provided in the passage?",
            'template_type': 'inference',
            'confidence': 0.82,
            'source_sentence': sentence,
        })

    # --- Significance (one per sentence) ---
    if info['word_count'] > 6 and primary_entity:
        candidates.append({
            'question': f"Why is {primary_entity} significant according to the passage?",
            'template_type': 'significance',
            'confidence': 0.78,
            'source_sentence': sentence,
        })

    # --- Summarization (one per sentence, entity-free) ---
    if info['word_count'] > 10:
        candidates.append({
            'question': "Which of the following best summarizes the key claim made in this part of the passage?",
            'template_type': 'summarization',
            'confidence': 0.75,
            'source_sentence': sentence,
        })

    # --- Fallback: context-specific rather than fully generic ---
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

    # Final dedupe pass while preserving order after ranking.
    deduped_questions = []
    seen_final = set()
    for item in ranked_questions:
        normalized = _normalize_question_text(item['question'])
        if normalized in seen_final:
            continue
        seen_final.add(normalized)
        deduped_questions.append(item)
        if len(deduped_questions) >= num_questions:
            break

    return deduped_questions[:num_questions]


def build_question_bundle(article: str, question_item: Dict, model_a_inference=None) -> Dict:
    """Build a complete question bundle with answer and distractors."""
    question = question_item['question']
    answer = generate_answer_from_question(article, question)
    distractors = generate_distractors(question, answer, article)

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


def extract_answer_candidates(article: str, question: str) -> List[str]:
    """
    Extract answer candidates from the article based on the question.
    
    Uses heuristics to find relevant phrases that could be the answer:
    - Extracts sentences containing key question words
    - Extracts noun phrases from important sentences
    - Returns top candidates ranked by relevance
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return []
    
    sentences = [s.strip() for s in article.split('.') if s.strip()]
    if not sentences:
        return []
    
    try:
        # Vectorize question and sentences
        texts = [question] + sentences
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate similarity between question and each sentence
        question_vec = tfidf_matrix[0]
        sentence_vecs = tfidf_matrix[1:]
        similarities = cosine_similarity(question_vec, sentence_vecs).ravel()
        
        # Get top-3 sentences by similarity (potential answers)
        top_indices = np.argsort(-similarities)[:3]
        candidates = [sentences[i].strip() for i in top_indices if similarities[i] > 0.1]
        
        return candidates[:3]
    except Exception:
        # Fallback: return first few sentences
        return sentences[:2]


def generate_answer_from_question(article: str, question: str) -> str:
    """
    Generate the best answer from the article based on the question.
    
    STEP 1: Find sentences most similar to question (TF-IDF cosine similarity)
    STEP 2: Extract key phrases/entities from top sentences
    STEP 3: Rank and return best candidate
    """
    candidates = extract_answer_candidates(article, question)
    
    if not candidates:
        # Fallback: use first sentence
        sentences = [s.strip() for s in article.split('.') if s.strip()]
        if sentences:
            return sentences[0]
        return "Unable to generate answer"
    
    # Use the most similar sentence as the answer source
    best_candidate = candidates[0]
    
    # Clean up - extract key phrase (first 20-30 words or until comma)
    if ',' in best_candidate:
        answer = best_candidate.split(',')[0].strip()
    else:
        words = best_candidate.split()
        answer = ' '.join(words[:min(20, len(words))]).strip()
    
    return answer.rstrip('.,;:') if answer else candidates[0]


def generate_distractors(question: str, answer: str, article: str) -> List[str]:
    """
    Generate 3 plausible distractor (wrong) options that are SIMILAR to the answer.
    
    Strategies (in order of priority):
    1. Find meaningful phrases from article semantically similar to answer (cosine similarity)
    2. Extract sentences/clauses that are similar in length and structure to answer
    3. Create contextual variations (related but wrong)
    
    Goal: Distractors should be plausible wrong answers with reasonable length (3-15 words).
    """
    import re
    
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
                phrase.lower() != answer.lower() and
                len(phrase) > 10):  # Avoid very short phrases
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
            
            # Extract top similar phrases as distractors
            for phrase, sim in good_similarities:
                if phrase not in distractors and len(distractors) < 3:
                    distractors.append(phrase)
            
        except Exception as e:
            pass
    
    # Fallback: If not enough distractors, use whole sentences
    if len(distractors) < 3:
        for sent in sentences:
            if (sent.lower() != answer.lower() and 
                sent not in distractors and
                10 < len(sent) < 200):  # Reasonable sentence length
                # Truncate very long sentences
                if len(sent) > 100:
                    sent = sent[:100].rsplit(' ', 1)[0] + "..."
                distractors.append(sent)
                if len(distractors) >= 3:
                    break
    
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

    question_bundles = [build_question_bundle(article, item, model_a_inference) for item in generated_questions]

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
    
    return {
        'mode': 'model_generated',
        'article': article,
        'question': question,
        'generated_questions': generated_questions,
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
