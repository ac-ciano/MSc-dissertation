from datasets import Dataset, ClassLabel
import pandas as pd
import re
import os


CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]

def load_data(type='validation'):
    """Load and prepare the same data as used in fine-tuning"""
    # Load csv with pandas (same as in fine-tuning script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if type == 'validation':
        csv_path = os.path.join(base_dir, './data/posts_with_labels.csv')
    elif type == 'test':
        csv_path = os.path.join(base_dir, './data/test_set.csv')
    elif type == 'unlabeled':
        csv_path = os.path.join(base_dir, './data/posts_without_labels.csv')
    else:
        raise ValueError("Dataset type must be either 'validation' or 'test'")
    dataset_pd = pd.read_csv(csv_path)

    return dataset_pd

def extract_label_from_response(response_text, thinking_enabled=False):
    response_text = response_text.strip()
    
    if thinking_enabled:
        # Look for "Final Classification:" pattern first
        final_classification_match = re.search(r'final classification:\s*([a-zA-Z]+)', response_text, re.IGNORECASE)
        if final_classification_match:
            extracted_label = final_classification_match.group(1).lower().strip()
            if extracted_label in CLASS_LABELS:
                return extracted_label
        
        # If no "Final Classification:" found, look after </thinking> tag
        thinking_end = response_text.find('</thinking>')
        if thinking_end != -1:
            after_thinking = response_text[thinking_end + len('</thinking>'):].strip()
            # Look for exact label matches in the text after thinking
            for label in CLASS_LABELS:
                if re.search(rf'\b{re.escape(label.lower())}\b', after_thinking.lower()):
                    return label
    
    # Fallback to original logic for non-thinking or if thinking extraction fails
    response_text_lower = response_text.lower()
    
    # Remove common prefixes that might appear
    response_text_lower = re.sub(r'^(the|this|post|is|classified|as|category|:|\s)+', '', response_text_lower)
    
    # Look for exact label matches
    for label in CLASS_LABELS:
        # word boundary matching
        if re.search(rf'\b{re.escape(label.lower())}\b', response_text_lower):
            return label
        if response_text_lower.strip().startswith(label.lower()):
            return label
    
    # If no exact match, try to find partial matches
    if 'indicator' in response_text_lower or 'no explicit' in response_text_lower:
        return 'indicator'
    elif 'ideation' in response_text_lower or 'suicidal thought' in response_text_lower:
        return 'ideation'
    elif 'behavior' in response_text_lower or 'self-harm' in response_text_lower or 'plan' in response_text_lower:
        return 'behavior'
    elif 'attempt' in response_text_lower or 'suicide attempt' in response_text_lower:
        return 'attempt'
    
    return "LABEL_NOT_FOUND"

def get_inference_prompt(style='calculators', thinking_enabled=False):
    if style == 'base':
        if thinking_enabled:
            inference_prompt_style = f"""You are tasked with classifying social media posts into suicide risk categories. Think through this step-by-step before providing your final answer.

First, analyze the post using this thinking process:

<thinking>
1. CONTENT ANALYSIS: What are the key elements in this post?
   - Identify any mentions of death, dying, suicide, self-harm
   - Note emotional indicators (hopelessness, despair, etc.)
   - Look for specific plans, methods, or preparations
   - Check for references to past attempts

2. EXPLICIT SUICIDE MENTIONS: 
   - Is suicide explicitly mentioned? (Yes/No)
   - Are there direct expressions of wanting to die or not be alive? (Yes/No)

3. PLAN OR METHOD IDENTIFICATION:
   - Are specific methods mentioned? (tools, locations, substances)
   - Are there signs of preparation? (notes, acquiring means)

4. TEMPORAL CONTEXT:
   - Does this refer to past events or current/future intentions?
   - Are past attempts specifically described?

5. CATEGORY REASONING: Based on the above analysis, which category fits best?
</thinking>

Now classify the post into one of these categories:

---
Description of risk categories:
- {CLASS_LABELS[0]}: The post content has no explicit suicidal expression or has explicit expression of resolved suicidal issues.
- {CLASS_LABELS[1]}: The post content has explicit suicidal expression but there is no plan or tendency to commit suicide.
- {CLASS_LABELS[2]}: The post content has explicit suicidal expression and there is plan or tendency to act out self-harm or suicide, or mentioned historical experience of self-harm behaviour.
- {CLASS_LABELS[3]}: The post content has explicit expression concerning recent suicide attempt, or mentioned historical experience of suicide attempt.

---
IMPORTANT RULES:
- The suicide risk only corresponds to the person writing the post, not other people potentially being mentioned
- Only answer with one word: 'indicator', 'ideation', 'behavior', or 'attempt'
- Never answer with anything different than one of the four options
- Never answer with NaN or empty answer
- Show your thinking process, then provide your final classification

---
User Post: {{}}

Your response format:
<thinking>
[Your step-by-step analysis here]
</thinking>

Final Classification: [your answer]"""
        else:
            inference_prompt_style = f"""Based on the social media post provided, return the category that the post belongs to based on the following four categories and their definitions.
---
Description of risk categories:
- {CLASS_LABELS[0]}: The post content has no explicit suicidal expression or has explicit expression of resolved suicidal issues.
- {CLASS_LABELS[1]}: The post content has explicit suicidal expression but there is no plan or tendency to commit suicide.
- {CLASS_LABELS[2]}: The post content has explicit suicidal expression and there is plan or tendency to act out self-harm or suicide, or mentioned historical experience of self-harm behaviour.
- {CLASS_LABELS[3]}: The post content has explicit expression concerning recent suicide attempt, or mentioned historical experience of suicide attempt.
---
Note that the suicide risk only corresponds to the person writing the post not of other people potentially being mentioned.
Only answer with one word. It should be always one of the following  'indicator', 'ideation', 'behavior', 'attempt'   
Never answer with something different than one of the four options. Never answer with NaN or empty answer
---
User post:
{{}}"""
    elif style == 'calculators':
        if  thinking_enabled:
            inference_prompt_style=f"""You are tasked with classifying social media posts into suicide risk categories. Think through this step-by-step before providing your final answer.

First, analyze the post using this thinking process:
                
<thinking>
1. CONTENT ANALYSIS: What are the key elements in this post?
   - Identify any mentions of death, dying, suicide, self-harm
   - Note emotional indicators (hopelessness, despair, etc.)
   - Look for specific plans, methods, or preparations
   - Check for references to past attempts
2. EXPLICIT SUICIDE MENTIONS: 
   - Is suicide explicitly mentioned? (Yes/No)
   - Are there direct expressions of wanting to die or not be alive? (Yes/No)
   - If unclear, lean toward 'indicator' over 'ideation'
3. PLAN OR METHOD IDENTIFICATION:
   - Are specific methods mentioned? (tools, locations, substances)
   - Are there signs of preparation? (notes, acquiring means)
   - If uncertain about explicit planning, choose 'ideation' over 'behavior'
4. TEMPORAL CONTEXT:
   - Does this refer to past events or current/future intentions?
   - Are past attempts specifically described?
5. CATEGORY REASONING: Based on the above analysis, which category fits best?
                
</thinking>
Now classify the post into one of these categories:
**Indicator:**
Definition: The post or content lacks any explicit expression concerning suicide. There may be general expressions of distress or sadness, but no direct mention or indication of suicidal thoughts or feelings. May include also a vague feeling about dying or death but no explicit mentioning of suicide. Note that the mentioning of suicide need to be very explicit in order to not be 'indicator'. Very large general dissatisfaction about life and hopelessness is still 'indicator' when suicide/killing oneself is not mentioned. When in doubt whether the suicide was explicitly mentioned or not choose 'indicator' over 'ideation'.

**Ideation:**
Definition: The content includes explicit expressions of suicidal thoughts or feelings, i.e. a desire to not be alive any longer, but without any specific plan to commit suicide. This can range from vague thoughts about not wanting to live to stronger, more explicit desires to end ones life (albeit without a specific plan how to do so). If no desire to die or to commit suicide is expressed, consider it 'indicator'. If the post contains specific ideas of how to commit suicide (ways/locations/means/methods) consider it 'behavior'.
                
**Behavior:**
Definition: The post includes explicit expressions of suicidal thoughts and also mentions (even if only briefly) a plan to commit suicide or exhibits self-harming behavior. It must include some form of explicit planning like a specific method or preparations taken (e.g. suicide note, lethal medication/drugs, tools/weapons (e.g. knives/guns/ropes) suitable to end ones life, suitable locations (e.g. bridges/cliffs/buildings to jump off from, train lines to get run over by). If you are unsure whether there is an explicit plan or not choose 'ideation' over 'behavior'.
                
**Attempt:**
Definition: The content describes past attempts at suicide. This category is focused on historic actions rather than current plans. A concrete action need to have happened in the past (e.g. overdose). When someone merely thought of an attempt in the past this classifies not as an 'attempt'. Note that when a post refers to past attempts but also mentions current plans then label it as 'attempt'.
                
---
IMPORTANT RULES:
- The suicide risk only corresponds to the person writing the post, not other people potentially being mentioned
- Only answer with one word: 'indicator', 'ideation', 'behavior', or 'attempt'
- Never answer with anything different than one of the four options
- Never answer with NaN or empty answer
                
---
User Post: {{}}
                
Your response format:
<thinking>
[Your step-by-step analysis here]
</thinking>
                
Final Classification: [your answer]"""
        else:
            inference_prompt_style=f"""Based on the social media post provided, return the category that the post belongs to based on the following four categories and their definitions.
---
 Indicator:
Definition: The post or content lacks any explicit expression concerning suicide. There may be general expressions of distress or sadness, but no direct mention or indication of suicidal thoughts or feelings. May include also a vague feeling about dying or death but no explicit mentioning of suicide. Note that the mentioning of suicide need to be very explicit in order to not be 'indicator'. Very large general dissatisfaction about life and hopefullness is still 'indicator' when suicide/killing oneself is not mentioned. When in doubt whether the suicide was explicitly mentioned or not choose 'indicator' over 'ideation'
Ideation:
Definition: The content includes explicit expressions of suicidal thoughts or feelings, i.e. a desire to not be alive any longer, but without any specific plan to commit suicide. This can range from vague thoughts about not wanting to live to stronger, more explicit desires to end ones life (albeit without a specific plan how to do so). If no desire to die or to commit suicide is expressed, consider it 'indicator'. If the post contains specific ideas of how to commit suicide (ways/locations/means/methods) consider it 'behavior'. 
Behavior:
Definition: The post includes explicit expressions of suicidal thoughts and also mentions (even if only briefly) a plan to commit suicide or exhibits self-harming behavior. It must include some form of explicit planning like a specific method or preperations taken (e.g. suicide note, lethal medication/drugs, tools/weapons (e.g. knifes/guns/ropes) suitable to end ones life, suitable locations (e.g. bridges/cliffs/buildings to jump off from, train lines to get run over by). If you are unsure whether there is a explicit plan or not choose 'ideation' over 'behavior'
Attempt:
Definition: The content describes past attempts at suicide. This category is focused on historic actions rather than current plans. A concrete action need to have happened in the past (e.g. overdose). When someone merely thought of an attempt in the past this classifies not as an 'attempt'
Note that when a post refers to past attempts but also mentions current plans then label it as 'attempt'
---
Note that the suicide risk only corresponds to the person writing the post not of other people potentially being mentioned.
Only answer with one word. It should be always one of the following  'indicator', 'ideation', 'behavior', 'attempt'   
Never answer with something different than one of the four options. Never answer with NaN or empty answer
---
User Post:
{{}}"""
            
    elif style == 'calculators_ev(errex)_gemini_1':
        if thinking_enabled:
            inference_prompt_style = f'''You are an expert AI assistant tasked with classifying the suicide risk level of social media posts. Your goal is to accurately categorize the post into one of four risk levels, following a clear hierarchy of severity.

First, carefully analyze the post using the revised thinking process. Then, provide the final classification.

Risk Hierarchy (Most to Least Severe):
Attempt > Behavior > Ideation > Indicator

If a post contains elements of multiple categories (e.g., a past attempt and current plans), you must classify it as the highest risk category present.

Step 1: Thinking Process
Check for a Plan (Behavior?): If not an attempt, does the post express suicidal thoughts and mention any form of plan? A "plan" is not just a method. Look for:

Method: Mentioning tools, substances, weapons (e.g., pills, buying a rope).

Timing: Specifying a time for the act (e.g., "I'll do it tonight," "on my 21st birthday," "when I get home").

Location: Mentioning a specific place (e.g., a bridge, a train station).

Preparation: Describing preparatory actions (e.g., writing a note, saying goodbye).
If suicidal thoughts and any element of a plan are present, the category is Behavior.

Check for Active Suicidal Desire (Ideation?): If not an attempt or behavior, does the post express a desire to not be alive, to die, or to end one's life? This goes beyond general sadness.

Look for phrases like "I want to die," "I want to end it all," "I hate living," "I'm going to die soon," "I wish I were dead."

The specific word "suicide" is not required. Focus on the expressed intent.

Check for Negated Thoughts (Indicator?): Before finalizing on Ideation, check if the user is expressing suicidal thoughts that they do not agree with or want. For example, "My brain tells me to kill myself, but I don't actually want to." These are intrusive thoughts and should be classified as Indicator.

Default to Indicator: If none of the above categories fit, the post expresses general distress, hopelessness, or mental health struggles without a stated desire to die. This is an Indicator.


Step 2: Category Definitions
Indicator:

Definition: The post shows general distress, hopelessness, or discusses suicide thematically but lacks a personal, active desire to die.

Crucial Rule: This category includes posts where the user mentions suicidal thoughts but clearly rejects them or states they do not want to act on them (e.g., intrusive or unwanted thoughts).

Ideation:

Definition: The post explicitly expresses a desire to no longer be alive or to die, but does not include any concrete plan (method, timing, location, or preparation).

Examples: "I hate living," "I wish I were dead," "I'm going to die soon." The desire is clear, even if passive.

Behavior:

Definition: The post includes both suicidal ideation and a component of planning. A plan can be a method, a specific timeframe, a location, or a preparatory act. The presence of any of these, even if vague, elevates the risk from Ideation to Behavior.

Examples: "I want to be gone tonight," "I plan to kill myself at 21."

Attempt:

Definition: The post describes a past act of self-harm or a suicide attempt. This includes aborted attempts (e.g., "I put the belt around my neck but then stopped") and non-suicidal self-injury (NSSI) like cutting. The focus is on the past action.

Hierarchy Rule: If a post mentions a past attempt AND current plans/ideation, the category is Attempt.

Step 3: Examples to Guide Your Reasoning
Here are some examples of how to classify difficult posts:

Post: "I want to be gone tonight. I'm in so much mental pain, I want this pain to turn into physical. I want it to be over tonight, I don't wanna survive it."

Correct Classification: behavior

Reasoning: The user expresses a clear desire to die and provides a specific timeframe ("tonight"), which constitutes a plan.

Post: "i plan to kill myself at 21, whether i want to or not, i'm 18 now."

Correct Classification: behavior

Reasoning: The user states an explicit intent to die and provides a future timeframe ("at 21"). This is a form of planning.

Post: "I'm inconsolable. I'm going to die soon"

Correct Classification: ideation

Reasoning: "I'm going to die soon" is a direct expression of a desire/expectation to not be alive, which is stronger than a general indicator. It lacks a plan, so it is Ideation.

Post: "my brain keeps telling me to kill myself, but I don't actually want to. It's so tiring to try to love yourself when your own mind is against you."

Correct Classification: indicator

Reasoning: The user explicitly mentions suicidal thoughts but negates or rejects them ("but I don't actually want to"). This is a key feature of the Indicator category.

Post: "I put a belt around my neck, and sat down against the doorknob in the closet tonight. Right before I passed out, I stopped, pulled it off, and started crying. I'm so close to just doing it."

Correct Classification: attempt

Reasoning: The user describes a specific past action ("I put a belt around my neck...tonight...I stopped"). This is an aborted attempt and is the highest risk factor in the post, even with the current ideation.

Step 4: Provide Your Final Answer
First, write out your step-by-step thinking process in a <thinking> block.

The suicide risk only corresponds to the person writing the post.

Only answer with one word: 'indicator', 'ideation', 'behavior', or 'attempt'.

User Post: {{}}

Your response format:

[Your new, improved step-by-step analysis here]


Final Classification: [your answer]'''

    elif style == 'calculators_ev(noex)_gemini_1':
        if not thinking_enabled:
            inference_prompt_style = f'''Your task is to classify the provided social media post into one of four categories related to suicide risk: Indicator, Ideation, Behavior, or Attempt. Analyze the post content meticulously based on the definitions and decision guidance below.

Category Definitions:

Indicator:

Core Criterion: The post expresses distress, sadness, hopelessness, worthlessness, or general negative sentiment (e.g., "life is unbearable," "I feel so empty," "I can't take this anymore") BUT lacks any explicit, direct mention of suicide, killing oneself, or a clearly stated desire to die.

Vague expressions about wishing for an end to suffering, wanting to disappear, or general statements about death (e.g., "I wish it would all stop," "I wonder what death is like") fall into 'Indicator' UNLESS suicide or killing oneself is explicitly stated as the method or desire.

Extreme general dissatisfaction with life or hopelessness is 'Indicator' if suicide or a direct wish to end one's life is not explicitly and unambiguously stated.

Crucial Rule: If there is any doubt that suicide or a desire to die is explicitly and directly mentioned, or if such an idea is only implied or could be interpreted in multiple ways, you must classify the post as Indicator. This is the default category if specific suicidal intent is not unequivocally clear.

Ideation:

Core Criterion: The post contains explicit and direct expressions of suicidal thoughts, a clearly stated desire to die, to no longer be alive, or to kill oneself, but without mentioning a specific, personal plan, method the user intends to use, or active preparations for suicide.

Examples: "I want to kill myself," "I wish I were dead," "I've been seriously thinking about ending my life."

Abstractly mentioning methods (e.g., "pills could do it," "jumping seems like a way out," "sometimes I think about how a rope could end it") is still 'Ideation' IF it's not presented as a concrete personal plan or if there's no stated intent to use that method.

Crucial Rule: If no explicit desire to die or commit suicide is directly stated (refer to 'Indicator' rule), it is 'Indicator'. If a specific, personal plan or preparation is mentioned, consider 'Behavior'. Do NOT classify as 'Ideation' if suicidal intent is only hinted at or ambiguous.

Behavior:

Core Criterion: The post includes explicit suicidal thoughts/desire to die (as defined in 'Ideation') AND also mentions a concrete, personal plan, a specific method the user indicates intent to use, or active preparations being made to commit suicide.

This requires more than just thinking about suicide; it involves some level of personalized planning or action towards it.

Examples of concrete plan/preparation elements linked to personal intent:

Mentioning a specific method with clear intent to use it personally (e.g., "I'm going to overdose on these pills tonight," "I have a rope and I'm planning to use it in the garage").

Stating acquisition or possession of specific means for suicide (e.g., "I just bought a gun for this purpose," "I've been saving up my medication to take it all at once").

Referring to a specific time or location for a planned attempt (e.g., "Friday is the day I end it," "I'm going to the bridge after everyone's asleep").

Mentioning preparatory actions like writing a suicide note, saying goodbyes with suicidal context, or putting affairs in order due to a planned suicide.

Crucial Rule: If you are unsure whether the plan is concrete, personal, and indicates active preparation or intent, classify as 'Ideation'. The plan must be more than an abstract thought or general mention of methods.

Attempt:

Core Criterion: The content clearly and unambiguously describes a specific suicidal action that the user has already taken in the past. A concrete harmful action with suicidal intent must have occurred.

Examples: "I overdosed last year and was hospitalized," "When I tried to hang myself a few months ago, the rope broke," "My past attempt with pills didn't work."

Simply thinking about an attempt in the past, or regretting not attempting, is NOT an 'Attempt'; it would likely be 'Ideation' or 'Indicator' based on current expression.

Special Rule for Current Plans with Past Attempt: If a post clearly describes a past attempt (as defined above) AND also mentions current, specific plans or preparations for a new attempt (i.e., meeting the full criteria for Behavior for the current plans), it should be classified as Attempt. If the current expressions are only Ideation-level or Indicator-level alongside a past attempt, it remains 'Attempt' based on the past action, but this special rule emphasizes the severity when Behavior-level current plans are also present.

Decision Guidance - Follow these questions in strict order:

Past Attempt Check: Does the post clearly describe a specific, concrete suicidal action the poster took in the past?

If YES:

Does the post also describe current, specific, and concrete plans or preparations for a new attempt (i.e., current statements meet the full criteria for Behavior)? If YES, classify as Attempt.

If NO current Behavior-level plans, but a clear past attempt is described, classify as Attempt.

If NO clear past suicidal action (or only thoughts of past actions), proceed to Q2.

Explicit Suicidal Content Check: Does the post contain an explicit and direct mention of suicide, killing oneself, or a clearly stated desire to die/not be alive?

If NO (i.e., it's vague, ambiguous, implied, or general distress without direct suicidal statements): Classify as Indicator. (This is a critical filter).

If YES, proceed to Q3.

Concrete Plan/Preparation Check: Does the post include a concrete, personal plan, a specific method the user indicates intent to use, or mention of active preparations for suicide? (Refer to 'Behavior' examples).

If YES: Classify as Behavior.

If NO (i.e., explicit suicidal thoughts/desire to die are present, but no such concrete, personal plan/preparation): Classify as Ideation.

Output Format:

You must only answer with a single word.

Your answer must be one of the following four options: 'Indicator', 'Ideation', 'Behavior', 'Attempt'.

Never answer with anything different. Do not provide explanations, additional text, NaN, or an empty answer.

User Post: {{}}
'''

    elif style == 'calculators_ev_gemini_1':
        if thinking_enabled:
            inference_prompt_style = f'''You are tasked with classifying social media posts into suicide risk categories. Think through this step-by-step before providing your final answer.

First, analyze the post using this thinking process:
        
<thinking>
1. CONTENT ANALYSIS: What are the key elements in this post?
    - Identify any mentions of death, dying, suicide, self-harm (direct and indirect language).
    - Note emotional indicators (hopelessness, despair, worthlessness, anhedonia, etc.).
    - Look for specific plans: methods, means, locations, **specific timeframes (e.g., "tonight", "next week", a specific date/age like "at 21")**, or preparations (e.g., writing notes, acquiring means, saying goodbyes).
    - Check for clear descriptions of past **concrete actions** taken to attempt suicide (e.g., "I took pills," "I tried to cut my wrists").

2. EXPLICIT SUICIDE MENTIONS / CLEAR DESIRE TO DIE:
    - Does the post contain explicit mentions of "suicide," "kill myself," "end my life," or very similar direct phrases? (Yes/No)
    - OR, are there other direct and unambiguous expressions of wanting to die, not be alive, or cease existing (e.g., "I want to die," "I hate living," "I wish I was dead," "I don't want to be here anymore," "I want it all to end," "I'm going to die soon" when in a context of severe distress)? (Yes/No)
    - If 'No' to both of the above (i.e., no explicit suicide terms AND no other clear desire to die expressed), then the post is likely 'Indicator'. Otherwise, proceed to analyze for 'Ideation', 'Behavior', or 'Attempt'.

3. PLAN OR METHOD IDENTIFICATION (for Behavior):
    - Building on step 2 (suicidal intent/desire to die is present), does the post also mention:
        - A specific **method** for suicide (e.g., "gonna overdose on pills," "jumping off the bridge")?
        - Specific **means** being acquired or possessed with intent (e.g., "I have the pills ready," "bought a gun")?
        - A specific **location** being considered for suicide (e.g., "thinking about that tall building")?
        - A specific **timeframe** for carrying out suicide (e.g., "I'm doing it tonight," "I plan to end it on my 21st birthday")?
        - Clear signs of **preparation** (e.g., "writing my suicide note," "giving away my things," "saying final goodbyes")?
    - If YES to any of these elements of planning, it strongly suggests 'Behavior'.
    - Guideline: If suicidal intent/desire to die is present, but you are genuinely uncertain about the presence of an explicit plan (as defined by method, means, location, specific timeframe, or preparation), choose 'Ideation' over 'Behavior'.

4. PAST ATTEMPT IDENTIFICATION:
    - Does the post describe a **specific past concrete action** the poster took to try to end their life (e.g., "I overdosed on pills last year but survived," "when I tried to hang myself it didn't work," "my failed attempt with bleach")?
    - Vague references (e.g., "I've tried before," "my past attempts," "I've been suicidal for years and almost did it") without describing the actual action are NOT sufficient for 'Attempt'. It must be a description of what they *did*.
    - If yes, this indicates 'Attempt'.

5. CATEGORY REASONING: Based on the above analysis, and strictly following the definitions below, which category fits best?
    - Start by determining if it's more than 'Indicator' based on Step 2.
    - If it is, then differentiate between 'Ideation' (desire/thoughts to die without a plan), 'Behavior' (desire/thoughts to die WITH a plan, as defined in Step 3), and 'Attempt' (a described past concrete action).
    - Adhere to the specific nuances in the definitions provided.
        
</thinking>
Now classify the post into one of these categories:

**Indicator:**
Definition: The post or content lacks any explicit expression concerning suicide AND lacks a clear desire to die or end one's life. There may be general expressions of distress, sadness, hopelessness, or dissatisfaction with life (e.g., "life sucks," "I'm so unhappy," "I feel worthless," "I'm struggling"). May include vague feelings about death or dying in a passive sense, but no indication of self-inflicted intent or a wish to cease living. If the post expresses a clear desire not to be alive (e.g., "I hate living," "I want to disappear," "I wish I wouldn't wake up"), it should be 'Ideation', not 'Indicator'.

**Ideation:**
Definition: The content includes explicit expressions of suicidal thoughts or feelings (e.g., using terms like "suicide," "kill myself") OR a clear desire to not be alive any longer (e.g., "I want to die," "I hate living," "I wish I could disappear," "I want it to be over," "I'm going to die soon" when in a context of distress), but WITHOUT any specific plan to commit suicide. A plan is defined by mentioning a specific method, means, location, timeframe, or preparations. This category can range from vague thoughts about not wanting to live to stronger, more explicit desires to end one's life. If no such desire to die or commit suicide is expressed, consider it 'Indicator'. If the post contains specific elements of a plan (as detailed under 'Behavior'), consider it 'Behavior'.

**Behavior:**
Definition: The post includes explicit expressions of suicidal thoughts/a desire to die (as defined in 'Ideation') AND ALSO mentions (even if only briefly) a plan to commit suicide or exhibits self-harming behavior intended to lead to suicide. A plan MUST include one or more of the following:
    * a specific **method** (e.g., "I'm going to take all these pills," "thinking of jumping from the bridge"),
    * specific **means** being acquired or possessed with suicidal intent (e.g., "I've written a suicide note," "I just bought a rope," "getting my affairs in order"),
    * a specific **timeframe** for the act (e.g., "I'm ending it tonight," "I plan to do it on my birthday," "I'll kill myself at 21"),
    * mention of specific **tools/weapons** (e.g., knives/guns/ropes) or **substances** (e.g., lethal medication/drugs) with clear intent for suicide,
    * or specific **locations** being considered for suicide (e.g., bridges/cliffs/buildings to jump off from, train lines).
    If you are unsure whether there is an explicit plan (containing at least one of the elements above), choose 'Ideation' over 'Behavior'.

**Attempt:**
Definition: The content describes one or more **specific past concrete actions** the poster took to try to end their life (e.g., "I overdosed last year," "when I tried to cut my wrists," "my failed attempt with pills five months ago"). This category is focused on historic actions rather than current plans or general past ideation. A concrete action (what the person *did*) must be described. Merely thinking of an attempt in the past, stating "I've been suicidal before," or making vague references like "my previous crisis" or "after what I tried" without detailing the action, does not classify as 'Attempt'. Note that when a post refers to past attempts (describing a concrete action) and also mentions current plans for suicide, label it as 'Attempt'.
        
---
IMPORTANT RULES:
- The suicide risk only corresponds to the person writing the post, not other people potentially being mentioned.
- Only answer with one word: 'indicator', 'ideation', 'behavior', or 'attempt'.
- Never answer with anything different than one of the four options.
- Never answer with NaN or empty answer.
        
---
User Post: {{}}
        
Your response format:
<thinking>
[Your step-by-step analysis here]
</thinking>
        
Final Classification: [your answer]'''

            
    elif style == 'calculators_ev_gemini_1.1':
        if thinking_enabled:
            inference_prompt_style = f'''You are tasked with classifying social media posts into suicide risk categories. Think through this step-by-step before providing your final answer.

First, analyze the post using this thinking process:
        
<thinking>
1. CONTENT ANALYSIS: What are the key elements in this post?
    - Identify any mentions of death, dying, suicide, self-harm, or specific methods/means.
    - Note emotional indicators (hopelessness, despair, anhedonia, being a burden, etc.).
    - Look for specific plans, methods (tools, locations, substances), or preparations (notes, acquiring means).
    - Check for references to past actions, including those initiated but not completed.
    - Note the imminence of any stated intentions (e.g., "tonight," "soon").
    - Assess the writer's stated desire: Is there a genuine wish to die, or are they expressing distress about *unwanted* suicidal thoughts?

2. EXPLICIT SUICIDE/DEATH INTENT:
    - Does the post explicitly mention "suicide," "kill myself," or similar direct phrases? (Yes/No)
    - Does the post express a clear desire to die or not be alive (e.g., "I want to be gone," "I want it to be over," "I'm going to die soon" in a context of despair)? (Yes/No)
    - If suicidal thoughts are mentioned, does the writer explicitly state they *do not* want these thoughts or *do not* want to act on them (e.g., "my brain says X, but I don't want to")? This might lean towards 'Indicator' if the primary sentiment is distress about the thoughts themselves, rather than a desire to die.
    - If suicide/death is not explicitly mentioned or the desire is clearly negated or unwanted, lean towards 'Indicator'.

3. PLAN, METHOD, OR PREPARATION IDENTIFICATION:
    - Is a specific method, tool, substance, or location mentioned or strongly implied in relation to a suicidal act? (e.g., pills, gun, bridge, "overdose," "jump").
    - Are there signs of preparation (e.g., "wrote a note," "got the pills," "saying goodbye")?
    - If there's a desire to die but no clear indication of plan/method/preparation, it's likely 'Ideation'. If there is *any* mention of a method or preparation, it leans towards 'Behavior'.

4. TEMPORAL CONTEXT & ACTION ASSESSMENT:
    - Does the post describe a past event, a current/future intention, or both?
    - Crucially, does it describe a *specific, potentially lethal action* that was initiated in the past or very recently, even if it was stopped, interrupted, or failed? (e.g., "I took pills but woke up," "I tried to [method] but stopped myself/was found"). This is key for 'Attempt'.
    - If a past action is described, was it a concrete suicidal act or just thinking about one?

5. CATEGORY REASONING: Based on the above analysis, which category fits best?
    - Distinguish 'Indicator' (general distress, or unwanted thoughts about suicide without desire) from 'Ideation' (desire to die, explicit suicidal thoughts).
    - Distinguish 'Ideation' (desire/thoughts, no plan) from 'Behavior' (desire/thoughts *with* a plan, method, or preparation).
    - Distinguish 'Behavior' (current/future plan) from 'Attempt' (a past or very recent *initiated action* that was potentially lethal).
    - If a past attempt is mentioned alongside current ideation/behavior, 'Attempt' takes precedence if the past action was concrete and potentially lethal.
        
</thinking>
Now classify the post into one of these categories:

**Indicator:**
Definition: The post expresses general distress, sadness, hopelessness, or discusses suicide abstractly, but **lacks a clear, current, personal expression of desire to die or suicidal intent from the author.** It may include mentions of suicidal thoughts if these are explicitly stated as unwanted and not representing a current personal desire to die (e.g., "my brain tells me to kill myself, but I don't actually want to die"). Vague feelings about dying or death, or very general dissatisfaction with life, fall here if suicidal intent isn't explicit. **If in doubt between Indicator and Ideation due to ambiguity about genuine suicidal desire, choose Indicator.**

**Ideation:**
Definition: The content includes **explicit personal expressions of suicidal thoughts or feelings, or a desire to not be alive any longer,** but without any specific plan, method, or preparations mentioned for carrying it out. This can range from phrases like "I want to die," "I wish I wasn't here," "I'm going to end it all," or "I'm going to die soon" (when stated in a context of despair), to more direct thoughts of suicide. **If no desire to die or commit suicide is personally expressed by the author for themselves, consider 'Indicator'. If a plan/method is present, consider 'Behavior'.**

**Behavior:**
Definition: The post includes explicit personal expressions of suicidal thoughts/desire to die **AND also mentions (even if only briefly) a plan to commit suicide, a specific method, or preparations being taken.** This must include some form of explicit planning element like a specific method (e.g., "thinking of overdosing on X pills," "want to jump from Y"), tools/weapons (e.g., "have a gun," "getting a rope"), suitable locations (e.g., "going to the bridge"), or preparations (e.g., "writing my suicide note," "giving things away"). **If you are unsure whether there is an explicit plan/method, choose 'Ideation' over 'Behavior'.**

**Attempt:**
Definition: The content describes **past or very recent, specific, self-initiated actions by the author that were potentially lethal, even if interrupted by self or others, or ultimately failed to cause death** (e.g., "I took all my pills last night but my friend found me," "I tried to hang myself today but I stopped right before passing out," "when I overdosed two years ago"). A concrete, potentially life-threatening action by the author must have been *initiated*. This category is focused on these historic or very recent actions. If a post refers to such past attempts and also mentions current plans/ideation, it should still be labeled 'Attempt' if the past action is clearly described. Merely *thinking* of an attempt in the past is 'Ideation' or 'Behavior' (if plan was involved then).

---
IMPORTANT RULES:
- The suicide risk only corresponds to the person writing the post, not other people potentially being mentioned.
- Only answer with one word: 'indicator', 'ideation', 'behavior', or 'attempt'.
- Never answer with anything different than one of the four options.
- Never answer with NaN or empty answer.
    
---
User Post: {{}}
    
Your response format:
<thinking>
[Your step-by-step analysis here]
</thinking>
    
Final Classification: [your answer]'''
            
    elif style == 'calculators_ev_gemini_2.0':
        if thinking_enabled:
            inference_prompt_style = f'''You are tasked with classifying social media posts into suicide risk categories. Think through this step-by-step before providing your final answer.

First, analyze the post using this thinking process:
        
<thinking>
1. CONTENT ANALYSIS: What are the key elements in this post?
    - Identify any mentions of death, dying, suicide, self-harm, or specific methods/means (even if vague).
    - Note emotional indicators (hopelessness, despair, anhedonia, being a burden, hating life, etc.).
    - Look for specific plans, methods (tools, locations, substances), or preparations (notes, acquiring means).
    - Check for references to past actions, including those initiated but not completed. Note the timeframe if mentioned (e.g., "yesterday," "last year").
    - Note the imminence of any stated intentions (e.g., "tonight," "soon," "planning to").
    - Assess the writer's stated desire: Is there a genuine wish to die or for life to end, or are they expressing distress about *unwanted* suicidal thoughts?

2. EXPLICIT SUICIDE/DEATH INTENT:
    - Does the post explicitly mention "suicide," "kill myself," or similar direct phrases? (Yes/No)
    - Does the post express a clear desire to die, not be alive, or for life to end (e.g., "I want to be gone," "I want it to be over," "I hate living," "life isn't worth it," "I'm going to die soon" in a context of despair)? (Yes/No)
    - If suicidal thoughts are mentioned, does the writer explicitly state they *do not* want these thoughts or *do not* want to act on them (e.g., "my brain says X, but I don't want to")? This strongly leans towards 'Indicator' if the primary sentiment is distress about the thoughts themselves, rather than a desire to die.
    - If suicide/death is not explicitly mentioned OR the desire is clearly negated/unwanted, lean towards 'Indicator'. However, strong expressions like "I hate living" generally imply ideation unless explicitly negated.

3. PLAN, METHOD, OR PREPARATION IDENTIFICATION:
    - Is a specific method, tool, substance, or location mentioned or strongly implied in relation to a suicidal act? (e.g., pills, gun, bridge, "overdose," "jump").
    - Are there signs of preparation (e.g., "wrote a note," "got the pills," "saying goodbye")?
    - **Crucially, if explicit desire to die (from step 2) is present and has high imminence (e.g., "tonight," "now"), consider if even a vague reference to action (e.g., "make this pain physical," "do something") implies a method or plan in context.** This is important for Behavior.
    - If there's a desire to die but no indication of plan/method/preparation (even considering the point above), it's likely 'Ideation'. If there is *any* mention of a method, preparation, or intended action (even if vague but coupled with strong intent/imminence), it leans towards 'Behavior'.

4. TEMPORAL CONTEXT & ACTION ASSESSMENT:
    - Does the post describe a past event, a current/future intention, or both?
    - Does it describe a *specific, potentially lethal action* that was initiated by the author in the past (e.g., "I took pills but woke up," "I tried to [method] but stopped," "when I overdosed X years ago," "attempted suicide before")?
        - Note: "tried to" or "attempted" implies an initiated action.
        - This is key for 'Attempt'. The time elapsed since the attempt (e.g., "yesterday" vs. "many years ago") does not disqualify it as an 'Attempt' if it is clearly stated as a past self-initiated suicidal action by the author and mentioned in the context of their current feelings or thoughts.

5. CATEGORY REASONING: Based on the above analysis, which category fits best?
    - Distinguish 'Indicator' (general distress, or explicitly unwanted thoughts about suicide without genuine personal desire to die) from 'Ideation' (expressions of desire to die/life to end, explicit suicidal thoughts). Phrases like "I hate living" should be considered Ideation unless desire is negated.
    - Distinguish 'Ideation' (desire/thoughts, no plan/method/action) from 'Behavior' (desire/thoughts *with* a plan, method, preparation, or intended action, even if method is vague but intent/imminence is high).
    - Distinguish 'Behavior' (current/future plan/intended action) from 'Attempt' (a past or very recent *initiated action* that was potentially lethal, regardless of when it happened, if mentioned by the author).
    - **'Attempt' takes precedence if a past initiated action is clearly described by the author**, even if current ideation/behavior is also present. The historical nature (e.g., "many years ago") does not negate this if the author links it to their current state or brings it up.
    - If in doubt between Indicator and Ideation, *only* choose Indicator if there's genuine ambiguity about personal suicidal desire *after* considering phrases like "I hate living" as likely Ideation. Explicitly unwanted thoughts are Indicator.
    - If in doubt between Ideation and Behavior due to vagueness of plan/method, lean towards Behavior if imminence and intent are very high. Otherwise, if still truly ambiguous about a plan/method, Ideation.
        
</thinking>
Now classify the post into one of these categories:

**Indicator:**
Definition: The post expresses general distress, sadness, hopelessness, or discusses suicide abstractly, but **lacks a clear, current, personal expression of desire to die or suicidal intent from the author.** It may include mentions of suicidal thoughts **only if these are explicitly stated as unwanted and not representing a current personal desire to die** (e.g., "my brain tells me to kill myself, but I don't actually want to die"). Vague feelings about dying or death, or general dissatisfaction with life, fall here **only if suicidal intent isn't personally and explicitly expressed or strongly implied (e.g., through phrases like "I hate living" which would be Ideation).** If suicidal desire is ambiguous *after careful consideration* and not explicitly unwanted, *and* there's no plan/action, this may be Indicator.

**Ideation:**
Definition: The content includes **explicit personal expressions of suicidal thoughts or feelings, a desire to not be alive any longer, or profound expressions of wanting life to end (e.g., "I hate living," "life isn't worth it," "I wish I could disappear").** This can range from phrases like "I want to die," "I wish I wasn't here," "I'm going to end it all," or "I'm going to die soon" (when stated in a context of despair), to more direct thoughts of suicide. This category applies if these expressions are present **but without any specific plan, method, intended action, or preparations mentioned for carrying it out.** If thoughts are explicitly unwanted, it's 'Indicator'. If a plan/method/intended action is present, consider 'Behavior'.

**Behavior:**
Definition: The post includes explicit personal expressions of suicidal thoughts/desire to die (as in 'Ideation') **AND also mentions (even if only briefly) a plan to commit suicide, a specific method, an intended action (even if the method is vague, like "want to make the pain physical," especially if coupled with high imminence), or preparations being taken.** This must include some form of explicit planning or intended action element like a specific method (e.g., "thinking of overdosing on X pills," "want to jump from Y"), tools/weapons (e.g., "have a gun," "getting a rope"), suitable locations (e.g., "going to the bridge"), preparations (e.g., "writing my suicide note," "giving things away"), or a stated intention to act with some indication of *how*, however vague, if imminence is high. **If there is genuine uncertainty about any plan, method, or intended action even after considering context, imminence, and intent, choose 'Ideation'.**

**Attempt:**
Definition: The content describes **past or very recent, specific, self-initiated actions by the author that were potentially lethal, even if interrupted by self or others, or ultimately failed to cause death** (e.g., "I took all my pills last night but my friend found me," "I tried to hang myself today but I stopped right before passing out," "when I overdosed two years ago," "I attempted suicide years ago and am feeling that way again"). A concrete, potentially life-threatening action by the author must have been *initiated*. This category is focused on these historic or very recent actions. **This includes attempts described as historical (e.g., "many years ago") if the author brings them up in the context of their current feelings, thoughts, or struggles.** If a post refers to such past attempts and also mentions current plans/ideation, it should still be labeled 'Attempt' if the past action is clearly described. Merely *thinking* of an attempt in the past is 'Ideation' or 'Behavior' (if a plan was involved then).

---
IMPORTANT RULES:
- The suicide risk only corresponds to the person writing the post, not other people potentially being mentioned.
- Only answer with one word: 'indicator', 'ideation', 'behavior', or 'attempt'.
- Never answer with anything different than one of the four options.
- Never answer with NaN or empty answer.
    
---
User Post: {{}}
    
Your response format:
<thinking>
[Your step-by-step analysis here]
</thinking>
    
Final Classification: [your answer]'''

    elif style == 'calculators_ev_claude_1':
        if thinking_enabled:
            inference_prompt_style = f'''Improved Suicide Risk Classification Prompt
You are tasked with classifying social media posts into suicide risk categories.
Think through this step-by-step before providing your final answer.
First, analyze the post using this thinking process:
<thinking>
1. CONTENT ANALYSIS: What are the key elements in this post?
- Identify any mentions of death, dying, suicide, self-harm, or desire to "not be
alive"
- Note emotional indicators (hopelessness, despair, etc.)
- Look for specific plans, methods, preparations, or timeframes
- Check for references to past attempts or self-harm actions
2. SUICIDAL INTENT IDENTIFICATION:
- Does the post express a desire to die, end life, or not be alive? (Look for
both explicit and strongly implied intent)
- Consider phrases like: "want to die," "end it all," "be gone," "don't want to
survive," "get it over with," "go through with it"
- Intrusive thoughts about suicide ("brain tells me to kill myself") still count
as suicidal content
- If there's clear intent to not be alive, classify as ideation or higher, NOT
indicator
3. PLAN OR METHOD ASSESSMENT:
- Are specific methods mentioned? (tools, substances, locations, actions)
- Are there temporal plans? (specific timeframes like "at 21," "tonight," "soon")
- Are there signs of preparation or research? (acquiring means, seeking
information)
- Any mention of specific methods OR timeframes qualifies as behavior-level
planning
4. PAST ACTION EVALUATION:
- Are past suicide attempts described? (concrete actions taken with intent to
die)
- Are past self-harm actions described? (may be attempts even if not explicitly
suicidal)
- Past actions take precedence - if mentioned, classify as attempt
5. CATEGORY REASONING: Based on the above analysis, which category fits best?
</thinking>
Now classify the post into one of these categories:
**Indicator:**
Definition: The post lacks explicit or strongly implied expressions of wanting to
die or not be alive. May include general distress, sadness, or vague mentions of
death/dying without clear suicidal intent. Expressions like "I hate living" or "no
point in existing" without additional context suggesting desire to die should be
considered indicator. However, if combined with other elements suggesting suicidal
intent, classify higher.
**Ideation:**
Definition: The content includes expressions of suicidal thoughts, feelings, or
desires to not be alive, but WITHOUT specific methods, timeframes, or preparations.
This includes:
- Direct statements: "want to die," "kill myself," "end my life," "don't want to
survive"
- Implied intent: "want to be gone," "get it over with," "go through with it"
- Intrusive thoughts: "brain tells me to kill myself" (even if person says they
don't want to)
- Questions about suicide methods without stating specific plans
- If unclear between indicator and ideation, and there's suggestion of not wanting
to be alive, choose ideation
**Behavior:**
Definition: The post includes suicidal thoughts AND mentions specific methods,
timeframes, preparations, or planning. This includes:
- Specific methods: tools, weapons, substances, locations (bridges, cliffs), actions
(hanging, overdose)
- Temporal plans: "kill myself at 21," "tonight," "soon" with context of suicide
- Preparations: acquiring means, writing notes, researching methods
- Recent self-harm with continuing suicidal intent
- If uncertain between ideation and behavior, and there's ANY specific planning
element, choose behavior
**Attempt:**
Definition: The post describes past concrete actions taken with intent to die or
cause serious self-harm, regardless of whether explicitly labeled as "suicide
attempts." This includes:
- Clear past suicide attempts: "I tried to overdose," "put a belt around my neck"
- Past self-harm actions that could be suicide attempts: "tried to cut myself,"
"took pills"
- When both past attempts and current plans are mentioned, classify as attempt
- Past actions take precedence over current ideation/planning
---
IMPORTANT RULES:
- The suicide risk only corresponds to the person writing the post, not other people
potentially being mentioned
- Only answer with one word: 'indicator', 'ideation', 'behavior', or 'attempt'
- Never answer with anything different than one of the four options
- Never answer with NaN or empty answer
- When in doubt between adjacent categories, consider the most serious elements
present
- Past actions (attempt) > Current specific plans (behavior) > General suicidal
thoughts (ideation) > Distress without suicidal intent (indicator)
---
User Post: {{}}
Your response format:
<thinking>
[Your step-by-step analysis here]
</thinking>
Final Classification: [your answer]'''
            
                
    elif style == 'calculators_ev_claude_1.1':
        if thinking_enabled:
            inference_prompt_style = f'''# Improved Suicide Risk Classification Prompt

You are tasked with classifying social media posts into suicide risk categories. Think through this step-by-step before providing your final answer.

First, analyze the post using this thinking process:

## Analysis Framework

<thinking>
1. **SUICIDE CONTENT IDENTIFICATION:**
   - Look for ANY mention of: suicide, killing self, ending life, wanting to die, being dead, not wanting to be alive
   - Include phrases like: "going to die", "want to be gone", "don't want to survive", "want it to be over"
   - Note: Even if the person says they don't want to act on thoughts, explicit mention of suicide/death still counts

2. **URGENCY AND TIMING INDICATORS:**
   - Current/immediate: "tonight", "today", "now", "going to", "want to"
   - Past events: "I tried", "last time", "when I", "I did"
   - General/ongoing: "sometimes", "always", "keeps telling me"

3. **PLAN/METHOD ASSESSMENT:**
   - Specific methods: pills, jumping, hanging, weapons, etc.
   - Specific locations: bridges, buildings, train tracks, etc.
   - Preparations: notes, acquiring means, specific setups
   - Research/seeking: asking about methods, looking for ways

4. **ACTION HISTORY:**
   - Past attempts: "I tried to", "I overdosed", "I put the belt around my neck and..."
   - Interrupted attempts: actions that were started but stopped
   - Self-harm behaviors: cutting, taking pills, etc.

5. **DECISION LOGIC:**
   - If NO explicit suicide/death wishes mentioned → INDICATOR
   - If explicit suicide thoughts but NO plan/method → IDEATION  
   - If explicit suicide thoughts AND plan/method/preparation → BEHAVIOR
   - If describes past suicide attempt (completed action) → ATTEMPT
</thinking>

## Classification Categories

**INDICATOR:**
Posts expressing distress, hopelessness, or general dissatisfaction with life WITHOUT any explicit mention of suicide, death wishes, or wanting to die. This includes severe depression, loneliness, and life problems, but the person does not explicitly express wanting to end their life.

**IDEATION:** 
Posts with explicit expressions of wanting to die, commit suicide, or end their life, but WITHOUT specific plans, methods, or preparations. This includes:
- "I want to kill myself" (without method)
- "I don't want to be alive"
- "I wish I was dead"
- Asking "why shouldn't I kill myself" (without method)

**BEHAVIOR:**
Posts with explicit suicidal thoughts AND specific plans, methods, preparations, or research about suicide methods. This includes:
- Mentioning specific methods (pills, jumping, hanging, etc.)
- Identifying specific locations (bridges, buildings, etc.)
- Describing preparations (acquiring means, writing notes)
- Seeking information about suicide methods
- Interrupted attempts where action was initiated

**ATTEMPT:**
Posts describing past completed suicide attempts - actual actions that were taken with intent to die, regardless of outcome. Must involve a concrete action that occurred in the past (overdose, jumping, hanging attempt, etc.). If past attempts are mentioned alongside current plans, still classify as ATTEMPT.

---

## Critical Classification Rules

1. **Explicit vs. Implicit:** Phrases like "I'm going to die soon" or "I want to be gone" ARE explicit expressions of death wishes when in context of distress
2. **Intrusive Thoughts:** Even if someone says "my brain tells me to kill myself but I don't want to," this is still IDEATION because suicide is explicitly mentioned
3. **Interrupted Actions:** If someone describes putting a noose around their neck, taking pills, etc., but stopped - this is BEHAVIOR (plan + action), not ATTEMPT
4. **Past vs. Present:** Focus on what happened vs. what is planned/desired now
5. **When in Doubt:** 
   - Between INDICATOR/IDEATION: Choose IDEATION if ANY explicit death/suicide mention
   - Between IDEATION/BEHAVIOR: Choose BEHAVIOR if ANY method/plan mentioned
   - Between BEHAVIOR/ATTEMPT: Choose ATTEMPT only for completed past actions

---

**IMPORTANT RULES:**
- The suicide risk only corresponds to the person writing the post, not other people mentioned
- Only answer with one word: 'indicator', 'ideation', 'behavior', or 'attempt'
- Never answer with anything different than one of the four options
- Never answer with NaN or empty answer

---

**User Post:** {{}}

**Your response format:**
<thinking>
[Your step-by-step analysis here]
</thinking>

**Final Classification:** [your answer]'''

    elif style == 'calculators_ev_deepseek_1':
        if thinking_enabled:
            inference_prompt_style = f'''You are tasked with classifying social media posts into suicide risk categories. Analyze posts systematically using this framework:

<thinking>
1. CONTENT DECONSTRUCTION:
   - Identify explicit suicide references: "suicide", "kill myself", "end my life", "not want to live"
   - Note indirect death references: "not wake up", "be gone", "painless way to die"
   - Flag emotional markers: hopelessness, worthlessness, unbearable pain
   - Detect specific methods: pills, hanging, jumping, firearms
   - Identify preparations: acquiring means, writing notes, scouting locations
   - Recognize past actions: "tried to", "attempted", "overdosed"

2. INTENT ANALYSIS (CRITICAL):
   - Active endorsement: Does writer claim ownership of thoughts? ("I want to die")
   - Passive experience: Are thoughts presented as intrusive? ("my brain tells me")
   - Rejection noted? Explicit disavowal? ("but I won't do it")

3. TEMPORAL FRAMEWORK:
   - Past focus: Historical descriptions (e.g., "last week I tried")
   - Present intent: Current plans (e.g., "doing it tonight")
   - Future ideation: Non-specific timing (e.g., "someday")

4. CATEGORY DECISION TREE:
   Start here → 
   a) Any past suicide attempt described? → ATTEMPT
   b) Explicit suicide intent + (specific method/preparation OR immediate timeline) → BEHAVIOR
   c) Explicit suicide desire without plan/immediacy → IDEATION
   d) ONLY distress/metaphors/rejected thoughts → INDICATOR
</thinking>

Category Definitions with Key Clarifications:

**Indicator** (Distress without actionable suicide intent)
- General unhappiness, hopelessness, or despair
- Metaphorical death references ("want to disappear")
- Rejected/intrusive thoughts ("brain tells me to kill myself but I don't want to")
- NO expressions of active suicidal desire
- *Clarification*: "I'm going to die soon" without explicit suicide context → indicator

**Ideation** (Active suicidal desire without actionable plan)
- Explicit wish to die: "want to die", "don't deserve to live"
- Non-actionable methods: "painless way" (without specification)
- No preparation or immediate timeline
- *Key boundary*: Becomes BEHAVIOR if paired with "tonight" or "now"

**Behavior** (Intent + Capability)
- Specific method stated: pills, cutting, jumping location
- Preparation evidence: bought pills, wrote note
- Imminent timeline: "tonight", "right now", "this weekend"
- *Critical expansion*: Expressions of immediate intent ("end it tonight") qualify even without method details

**Attempt** (Past concrete action)
- Description of actual self-harm with suicidal intent
- Includes interrupted/aborted attempts ("put belt around neck but stopped")
- Overrules concurrent ideation/behavior when mentioned
- *Clarification*: "tried to" or "attempted" always → attempt

---
IRREDUCIBLE RULES:
1. Writer-focused: Ignore mentions of others' actions
2. Temporal hierarchy: Attempt > Behavior > Ideation
3. Immediacy trumps: "tonight" beats non-specific ideation
4. Rejection downgrade: "I don't actually want to" → indicator
5. Output exactly one: 'indicator', 'ideation', 'behavior', or 'attempt'

---
User Post: {{}}

Response Format:
<thinking>
[Apply framework steps 1-4]
</thinking>
Final Classification: [single category]'''
            
              
    elif style == 'calculators+grok3_from_grok3_1-0':
        if thinking_enabled:
            inference_prompt_style = f'''You are tasked with classifying social media posts into suicide risk categories: Indicator, Ideation, Behavior, or Attempt. Analyze the post step-by-step using the thinking process below before providing your final answer.

#### Thinking Process
<thinking>
1. *Content Analysis*: Identify the key elements in the post:
   - Look for mentions of death, dying, suicide, self-harm, or specific methods/means (e.g., pills, guns, jumping).
   - Note emotional indicators (e.g., hopelessness, despair, loneliness, worthlessness).
   - Identify any specific plans (e.g., timing like "tonight"), methods (e.g., tools, substances, locations), or preparations (e.g., writing a note, acquiring means).
   - Check for references to past actions (e.g., "I tried," "I took pills"), including those initiated but not completed.
   - Assess the imminence of intentions (e.g., "tonight," "soon").
   - Evaluate the writer’s stated desire: Is it a genuine wish to die, or distress about unwanted suicidal thoughts?
   - *Key Check*: Look for any indication of past attempts or ongoing actions (e.g., "I’ve taken something").

2. *Explicit Suicide/Death Intent*:
   - Does the post explicitly mention "suicide," "kill myself," or similar phrases? (Yes/No)
   - Does it express a clear, personal desire to die or not be alive (e.g., "I want to die," "I want to be gone," "I’m going to die soon" in a despairing context)? (Yes/No)
   - *Clarification*: Even if "suicide" isn’t mentioned, phrases like "I want to die" or "I don’t want to be alive" count as explicit intent unless clearly negated (e.g., "I don’t mean it").
   - If suicidal thoughts are present but the writer explicitly states they do not want to act on them (e.g., "my brain says kill myself, but I don’t want to"), lean toward *Indicator*.
   - If intent is vague or absent, lean toward *Indicator*.

3. *Plan, Method, or Preparation Identification*:
   - Is a specific method (e.g., "overdose," "jump off a bridge"), tool, substance, or location mentioned or implied?
   - Are there signs of preparation (e.g., "I got the pills," "wrote a note")?
   - Is there a clear intention to act imminently (e.g., "tonight," "soon"), even without a method?
   - *Rule: If there’s a desire to die but no plan, method, or imminent intent, it’s **Ideation. If any of these elements are present, it’s **Behavior*.

4. *Temporal Context & Action Assessment*:
   - Does the post describe a past event, current/future intent, or both?
   - Does it detail a specific, potentially lethal action initiated by the author in the past or very recently (e.g., "I took pills last night," "I tried to hang myself")?
   - *Key Check: If it implies an action is *currently happening or just happened (e.g., "I haven’t taken enough"), classify as *Attempt*.
   - If past actions are mentioned, were they concrete (e.g., "I overdosed") or just thoughts/plans?

5. *Category Reasoning*:
   - *Indicator* vs. *Ideation: Use **Indicator* for general distress or unwanted thoughts without clear intent; use *Ideation* for explicit desire to die.
   - *Ideation* vs. *Behavior: Use **Ideation* if no plan/imminence; use *Behavior* if plan, method, or imminent intent is present.
   - *Behavior* vs. *Attempt"": Use **Behavior* for plans without action; use *Attempt* for past or ongoing lethal actions.
   - If a past attempt is mentioned with current thoughts, prioritize *Attempt* if the past action was concrete.
</thinking>

#### Categories
*Indicator*:  
- *Definition: The post expresses general distress, sadness, hopelessness, or mentions suicide abstractly, but **lacks a clear, current, personal desire to die or suicidal intent from the author*. Includes suicidal thoughts if explicitly unwanted (e.g., "my brain says kill myself, but I don’t want to"). Vague statements (e.g., "I hate living") stay here unless they clearly express a desire to die.  
- *Key: If the post says "I want to die" or similar, it’s **Ideation* unless negated.

*Ideation*:  
- *Definition: The post contains **explicit personal expressions of suicidal thoughts or a desire to not be alive* (e.g., "I want to die," "I wish I was gone," "I’m going to die soon" in despair), but *lacks specific plans, methods, preparations, or imminent intent*. Includes vague intent to die without "suicide" explicitly stated, unless negated.  
- *Key: Moves to **Behavior* if a plan or imminent intent appears.

*Behavior*:  
- *Definition: The post includes **explicit suicidal thoughts/desire to die AND mentions a plan, method, preparation, or imminent intent* (e.g., "I’ll jump off a bridge," "I’m doing it tonight"). Includes specific methods (e.g., "overdose on pills"), tools (e.g., "got a rope"), or imminent timing (e.g., "tonight") even without a method if context implies action soon.  
- *Key: Moves to **Attempt* if an action has been initiated.

*Attempt*:  
- *Definition: The post describes **past or very recent, specific, self-initiated actions by the author that were potentially lethal* (e.g., "I took pills last night," "I tried to overdose"). Includes ongoing actions (e.g., "I haven’t taken enough"). Past attempts take precedence if concrete, even with current thoughts.  
- *Key: Mere thoughts of past attempts are **Ideation* or *Behavior*.

#### Important Rules
- Classify based only on the writer’s risk, not others mentioned.
- Answer with one word: *indicator, **ideation, **behavior, or **attempt*.
- Do not use anything other than these four options (no NaN, empty responses).

#### Response Format
<thinking>
[Step-by-step analysis]
</thinking>
Final Classification: [answer]'''

    elif style == 'grok3_1.0':
        if thinking_enabled:
            inference_prompt_style = f'''You are tasked with classifying social media posts into suicide risk categories: Indicator, Ideation, Behavior, and Attempt. Follow this step-by-step thinking process before providing your final answer.

Thinking Process

1. **CONTENT ANALYSIS**: What are the key elements in this post?
   - Identify mentions of death, dying, suicide, self-harm, or strong implications of wanting to end one’s life (e.g., "I want to be gone," "I’m going to die soon").
   - Note emotional indicators (e.g., hopelessness, despair, shame, loneliness).
   - Look for specific plans, methods, or preparations (e.g., tools, locations, actions taken).
   - Check for references to past attempts, including incomplete or interrupted ones.


EXPLICIT SUICIDE MENTIONS:

Is suicide explicitly mentioned (e.g., "suicide," "kill myself") or strongly implied (e.g., "I want it to be over," "I’m going to die soon")? (Yes/No)
Are there direct or implied expressions of wanting to die or not be alive? (Yes/No)
If unclear, consider emotional context: severe distress with implied intent leans toward 'ideation' over 'indicator.'


PLAN OR METHOD IDENTIFICATION:

Are specific methods mentioned? (e.g., "pills," "jump off a bridge")
Are there signs of preparation or actions taken? (e.g., "I wrote a note," "I put a belt around my neck")
Does the post suggest imminent intent (e.g., "tonight") even without a method? If so, lean toward 'behavior' over 'ideation.'


TEMPORAL CONTEXT:

Does this refer to past events (e.g., "I tried last week") or current/future intentions (e.g., "I want to do it now")?
Are past attempts described, including incomplete ones (e.g., "I started but stopped")?


CATEGORY REASONING: Based on the analysis, which category fits best? Use the definitions below.



Categories

Indicator:

Definition: The post lacks explicit or strongly implied expressions of suicidal intent. It may show general distress, sadness, or vague references to death (e.g., "life isn’t worth living") without a clear desire to die. Severe hopelessness alone isn’t enough—suicidal intent must be explicit or strongly implied to move beyond this category.
Example: "I’m so tired of everything, I don’t know how to keep going."


Ideation:

Definition: The post includes explicit or strongly implied suicidal thoughts or feelings (e.g., "I want to die," "I’m going to die soon") without specific plans or actions. This ranges from passive thoughts (e.g., "I wish I weren’t here") to stronger desires to end life, but stops short of concrete steps.
Example: "I’m inconsolable. I’m going to die soon."


Behavior:

Definition: The post includes explicit or strongly implied suicidal thoughts plus either: (1) a specific plan or method (e.g., "I’ll jump off a bridge"), (2) preparations or actions taken (e.g., "I got the pills"), or (3) strong intent with imminent timing (e.g., "I want to be gone tonight") even without a method. This category bridges thoughts and actions.
Example: "I want to be gone tonight. I’m in so much pain, I need it to stop."


Attempt:

Definition: The post describes a past suicide attempt, including incomplete or self-interrupted actions (e.g., "I tried to hang myself but stopped"). A concrete action must have been initiated with suicidal intent. If the post also mentions current plans or ideation, still classify as 'attempt' if a past attempt is clear.
Example: "I put a belt around my neck tonight but stopped before I passed out."




Important Rules

The suicide risk applies only to the person writing the post, not others mentioned.
Only answer with one word: 'indicator', 'ideation', 'behavior', or 'attempt'.
Never use anything other than these four options or leave it blank.


User Post
{{}}

Response Format

[Your step-by-step analysis here]


Final Classification: [your answer]
'''

    return inference_prompt_style