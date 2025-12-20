# ניתוח בעיות איכות בפלטים

## דוגמאות פלטים בעייתיים

### דוגמה 1: "Large language models"
**פלט:** "large language models are known for producing fabricated content is also shaped by what sequence lengths, vocabulary size. however, data filtering and repetition can cause the model to decline tool use rather than only model compute."

**בעיות:**
- משפט לא שלם: "are known for producing fabricated content is also shaped"
- קפיצת נושא: עובר מ-"fabricated content" ל-"sequence lengths" ללא מעבר חלק
- חוסר קוהרנטיות: "decline tool use rather than only model compute" לא ברור

### דוגמה 2: "because the system"
**פלט:** "because the system can reject tool calls more reliably, but it does not fully predict the next tokens into an actual output string. the contradiction here is that while multimodal models exhibit impressive pattern recognition, they are not the same"

**בעיות:**
- משפט לא שלם: "they are not the same" - חסר השלמה
- קפיצת נושא: עובר מ-"tool calls" ל-"multimodal models" ללא הקשר
- חוסר קוהרנטיות: "contradiction here" לא ברור מה הסתירה

### דוגמה 3: "Transformer architectures"
**פלט:** "transformer architectures underpin most modern llms use relative- position methods, and monitoring for memorization behaviors. data parallelism replicates the model' s ability to discard irrelevant information, while later training adds harder technical material."

**בעיות:**
- משפט לא שלם: "underpin most modern llms use relative-position methods" - שני נושאים מחוברים ללא קשר
- קפיצת נושא: עובר מ-"relative-position methods" ל-"monitoring for memorization" ללא מעבר
- חוסר קוהרנטיות: "data parallelism replicates the model's ability to discard irrelevant information" לא ברור

### דוגמה 4: "Attention mechanisms"
**פלט:** "attention mechanisms focus on state changes what is that should be beneficial. if the model to reduce sequence lengths, and a pretraining on instruction contracts ready text about the model can be trained, and how smooth was filtered."

**בעיות:**
- משפטים לא שלמים: "what is that should be beneficial" - לא ברור
- חוסר דקדוק: "if the model to reduce" - חסר פועל עזר
- חוסר קוהרנטיות: "how smooth was filtered" - לא ברור מה זה אומר

## בעיות עיקריות שזוהו

### 1. חוסר קוהרנטיות סמנטית
**תיאור:** הטקסט קופץ בין נושאים ללא מעבר חלק או קשר לוגי.

**סיבות אפשריות:**
- Context window קטן מדי (30 טוקנים) - לא מספיק הקשר לשמירה על נושא
- Topic drift detection חלש מדי - העונש על סטייה מהנושא (0.5) לא מספיק חזק
- אין מנגנון חזק לשמירה על נושא ברמת הפסקה/משפט

**מיקום בקוד:**
- `reasoner/generate.py:322` - topic_drift_penalty חלש מדי
- `reasoner/generate.py:259` - context_window = 30 קטן מדי

### 2. משפטים לא שלמים
**תיאור:** משפטים נקטעים באמצע או לא מתחברים היטב.

**סיבות אפשריות:**
- Sentence closure mechanism לא מספיק חזק
- אין מנגנון לזיהוי משפטים לא שלמים לפני סיום
- Top-k/top-p sampling עלול לבחור טוקנים שלא מתאימים לסיום משפט

**מיקום בקוד:**
- `reasoner/generate.py:527-537` - sentence closure logic
- `reasoner/generate.py:539-548` - sentence completeness check חלש מדי

### 3. קפיצות נושאים
**תיאור:** הטקסט עובר מנושא אחד לאחר ללא מעבר חלק או הסבר.

**סיבות אפשריות:**
- Anchor mechanism משתמש רק ב-24 טוקנים הראשונים - לא מתעדכן עם הזמן
- אין מנגנון לשמירה על נושא דינמי (topic tracker) ב-generate רגיל
- Semantic similarity לא מספיק חזקה לשמירה על קוהרנטיות נושא

**מיקום בקוד:**
- `reasoner/generate.py:303` - anchor_tokens = ids[:k0] - רק טוקנים ראשונים
- `reasoner/generate.py:257-270` - context vector לא כולל topic tracking

### 4. חוסר דקדוק ותחביר
**תיאור:** משפטים עם שגיאות דקדוק או תחביר לא תקין.

**סיבות אפשריות:**
- Bigram model לא מספיק חזק לשמירה על דקדוק תקין
- אין מנגנון לזיהוי שגיאות דקדוק
- Selector (trigram/bigram) לא מספיק חזק ב-generate רגיל

**מיקום בקוד:**
- `reasoner/generate.py:272` - רק bigram_logp, אין trigram
- אין מנגנון לזיהוי שגיאות דקדוק

### 5. שילוב חלקי משפטים
**תיאור:** נראה שהמערכת מחברת חלקי משפטים שלא מתחברים.

**סיבות אפשריות:**
- Semantic similarity עלולה למצוא קשרים בין טוקנים שלא מתחברים תחבירית
- אין מנגנון לזיהוי משפטים שלמים לפני בחירת טוקן חדש
- Context window קטן מדי לא רואה את המבנה המלא של המשפט

**מיקום בקוד:**
- `reasoner/generate.py:286-287` - semantic similarity לא מספיק חזקה
- `reasoner/generate.py:259` - context_window קטן מדי

## השוואה בין generate רגיל ל-generate-dual

### generate רגיל:
- משתמש רק ב-bigram + semantic similarity
- אין topic tracker דינמי
- אין sequence-level scoring
- פחות קוהרנטי

### generate-dual:
- משתמש ב-selector (trigram/bigram) + reasoner
- יש topic tracker דינמי
- יש sequence-level scoring
- יותר קוהרנטי אבל עדיין בעייתי

## המלצות לשיפור

1. **הגדלת context window** - מ-30 ל-50-60 טוקנים
2. **חיזוק topic drift detection** - העלאת העונש מ-0.5 ל-1.0-1.5
3. **הוספת topic tracker דינמי** גם ב-generate רגיל
4. **חיזוק sentence closure mechanism** - זיהוי טוב יותר של סיום משפט
5. **הוספת מנגנון לזיהוי משפטים לא שלמים** - בדיקה לפני בחירת טוקן
6. **שיפור coherence scoring** - בונוס חזק יותר לטוקנים קוהרנטיים
7. **שימוש ב-trigram גם ב-generate רגיל** - לא רק bigram
