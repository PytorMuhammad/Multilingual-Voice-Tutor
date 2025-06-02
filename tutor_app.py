def clean_and_fix_response(user_input, response_text):
    """Clean response and ensure it makes conversational sense"""
    
    # CRITICAL FIX: Check user's language preference first
    response_language = st.session_state.response_language
    
    # If user selected single language mode, enforce it strictly
    if response_language == "cs":
        # Force Czech only response
        clean_text = re.sub(r'\[de\].*?(?=\[cs\]|$)', '', response_text, flags=re.DOTALL)
        clean_text = re.sub(r'\[cs\]\s*', '', clean_text).strip()
        if not clean_text:
            if 'guten tag' in user_input.lower() or 'wie geht' in user_input.lower():
                return "[cs] Dobrý den! Mám se dobře, děkuji."
            elif 'jak se máte' in user_input.lower():
                return "[cs] Mám se dobře, děkuji! A vy?"
            else:
                return "[cs] Děkuji za vaši zprávu."
        return f"[cs] {clean_text}"
    
    elif response_language == "de":
        # Force German only response  
        clean_text = re.sub(r'\[cs\].*?(?=\[de\]|$)', '', response_text, flags=re.DOTALL)
        clean_text = re.sub(r'\[de\]\s*', '', clean_text).strip()
        if not clean_text:
            if 'dobrý den' in user_input.lower() or 'jak se máte' in user_input.lower():
                return "[de] Guten Tag! Mir geht es gut, danke."
            elif 'guten tag' in user_input.lower() or 'wie geht' in user_input.lower():
                return "[de] Mir geht es gut, danke! Und Ihnen?"
            else:
                return "[de] Vielen Dank für Ihre Nachricht."
        return f"[de] {clean_text}"
    
    # FIXED AUTO MODE: Detect user's primary language and respond accordingly
    elif response_language == "auto":
        detected_lang = detect_user_primary_language_advanced(user_input)
        
        if detected_lang == "cs":
            # User spoke primarily Czech - respond in Czech only
            clean_text = re.sub(r'\[de\].*?(?=\[cs\]|$)', '', response_text, flags=re.DOTALL)
            clean_text = re.sub(r'\[cs\]\s*', '', clean_text).strip()
            if not clean_text:
                return "[cs] Ahoj! Mám se dobře, děkuji. Jsem AI asistent. O čem si chceš povídat?"
            return f"[cs] {clean_text}"
            
        elif detected_lang == "de":
            # User spoke primarily German - respond in German only  
            clean_text = re.sub(r'\[cs\].*?(?=\[de\]|$)', '', response_text, flags=re.DOTALL)
            clean_text = re.sub(r'\[de\]\s*', '', clean_text).strip()
            if not clean_text:
                return "[de] Hallo! Mir geht es gut, danke. Ich bin ein AI-Assistent. Worüber möchtest du sprechen?"
            return f"[de] {clean_text}"
            
        # If both languages detected, use existing bilingual logic
        # Fall through to existing logic below
    
    # For "both" mode, use existing logic
    # Remove any English explanations that shouldn't be there
    lines = response_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip English explanations (lines without language markers that contain English)
        if not re.search(r'\[([a-z]{2})\]', line):
            # Check if it's likely English
            english_indicators = ['you', 'are', 'the', 'and', 'or', 'is', 'this', 'that', 'would', 'like', 'know', 'practice', 'specific', 'anything']
            if any(word in line.lower() for word in english_indicators):
                continue  # Skip English lines
        
        cleaned_lines.append(line)
    
    # Rebuild response
    cleaned_response = ' '.join(cleaned_lines)
    
    # If response is empty after cleaning, provide a default
    if not cleaned_response.strip():
        # Detect user's language and respond appropriately
        if 'guten tag' in user_input.lower() or 'wie geht' in user_input.lower():
            cleaned_response = "[de] Guten Tag! Mir geht es gut, danke."
        elif 'dobrý den' in user_input.lower() or 'jak se máte' in user_input.lower():
            cleaned_response = "[cs] Dobrý den! Mám se dobře, děkuji."
        else:
            cleaned_response = "[cs] Děkuji za vaši zprávu. [de] Vielen Dank für Ihre Nachricht."
    
    # Ensure language markers are present
    if not re.search(r'\[([a-z]{2})\]', cleaned_response):
        # Add appropriate language markers based on content
        cleaned_response = add_appropriate_language_markers(cleaned_response, user_input)
    
    return cleaned_response.strip()

def detect_user_primary_language_advanced(user_input):
    """Advanced detection for Auto mode - determines if user spoke primarily one language"""
    
    # Remove any existing language markers for clean detection
    clean_text = re.sub(r'\[[a-z]{2}\]', '', user_input).strip()
    
    # Split into words for analysis
    words = re.findall(r'\b\w+\b', clean_text.lower())
    
    if not words:
        return "mixed"  # Default if no words
    
    # Czech indicators (more comprehensive)
    czech_chars = set("áčďéěíňóřšťúůýž")
    czech_words = {
        "ahoj", "jak", "se", "máš", "máte", "můžeš", "můžeme", "jsem", "jsi", "je", "jsou", 
        "prosím", "děkuji", "dobrý", "dobře", "ano", "ne", "já", "ty", "on", "ona", 
        "my", "vy", "oni", "den", "noc", "chci", "dnes", "zítra", "včera", "tady", "tam", 
        "proč", "kde", "kdy", "co", "kdo", "to", "ten", "ta", "mít", "jít", "dělat", 
        "vidět", "slyšet", "vědět", "říct", "sobě", "mi", "ti", "si"
    }
    
    # German indicators (more comprehensive)
    german_chars = set("äöüß")
    german_words = {
        "hallo", "wie", "geht", "es", "ihnen", "ich", "du", "er", "sie", "wir", "ihr", 
        "sind", "ist", "bin", "habe", "haben", "bitte", "danke", "gut", "ja", "nein",
        "der", "die", "das", "ein", "eine", "zu", "von", "mit", "für", "auf",
        "wenn", "aber", "oder", "und", "nicht", "auch", "so", "was", "wo",
        "wann", "wer", "warum", "möchte", "kann", "muss", "soll", "darf", "will", "mir"
    }
    
    # Count evidence
    czech_evidence = 0
    german_evidence = 0
    
    # Character-based evidence (strong indicator)
    for char in clean_text.lower():
        if char in czech_chars:
            czech_evidence += 3  # Strong weight for special chars
        elif char in german_chars:
            german_evidence += 3
    
    # Word-based evidence (very strong indicator)
    for word in words:
        if word in czech_words:
            czech_evidence += 5  # Very strong weight for specific words
        elif word in german_words:
            german_evidence += 5
    
    # Calculate percentages
    total_evidence = czech_evidence + german_evidence
    
    if total_evidence == 0:
        return "mixed"  # No clear language detected
    
    cs_percent = (czech_evidence / total_evidence) * 100
    de_percent = (german_evidence / total_evidence) * 100
    
    # Determine primary language (need >70% to be considered "primary")
    if cs_percent > 70:
        return "cs"
    elif de_percent > 70:
        return "de"
    else:
        return "mixed"  # Both languages present
