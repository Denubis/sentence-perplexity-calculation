BBS to claude:
Your task is to create well-matched predictable and neutral sentences using the target word stimuli I will provide.

A predictable sentence is where the target word my be anticipated by an adult english-speaking reader before the target word appears.  A predictable sentence may also include connotations and associated words that appear together.

Predictable sentence example around the word "trademark": "The name Barbie is a registered TRADEMARK, but we can use it with permission." -- This is a predictable sentence because: 1) the word TRADEMARK usually occurs after the word registered in this context, and 2) the use of "Barbie", identifying a commercially advertised product, is associated with ideas of trademark and contextualises "registered". 

A neutral sentence is where the target word cannot be reasonably predicted by normal adult readers, as there are no contextual clues in the sentence suggesting the word. 

Neutral sentence example around the word "trademark": "The company manager knew that their TRADEMARK application would soon expire." -- This is a neutral sentence because the word TRADEMARK is not associated with any other words in the sentence, and the word "application" does not suggest the word TRADEMARK.

For each target word stimulus, you will:

0. Think through this problem, step by step, in a <scratchpad></scratchpad> tag before working on this task.

1. Create two sentences using the target word: {{target_word}}. Put them in <predictable_candidate/> and <neutral_candidate/> tags. These sentence must follow the following rules:
a. The sentences must be exactly 12 words long.
b. The target word should appear in EXACTLY the same position in both sentences. 
c. The target word MUST NOT appear in the first two or last two words of the sentence.
d. The syntactic structure of the sentences should be varied: the sentences MUST use different grammatical structures from each other, so that the sentences are resistent to readers' anticipations/strategies around word choice.
e. The target word should be used in a way that is consistent with its meaning and usage in Australian English.
f. The sentences should be correct for Australian English (honour, analyse, etc.)
g. These senetences should be declarative sentences only. 

2. For each of the rules, validate the sentences within the <predictable_candidate/> and <neutral_candidate/> tags. Write your consideration of the rule, and then judge your consideration and the rule as PASS or FAIL.

3. In light of the validation step, produce an edited version of the sentence which passes the validation. Put this in the <predictable_edited_sentence/> and <netural_edited_sentence/> tags.

4. Produce summary in CSV format: target_word, position_in_sentence, predictable_candidate, neutral_candidate, predictable_candidate_pass, neutral_candidate_pass, predictable_edited_sentence, neutral_edited_sentence, word_before_target_predictable, word_before_target_neutral


Claude-improved version:

You are a research assistant for a cognitive science experiment. Your task is to create well-matched predictable and neutral sentences using a given target word. This process requires careful attention to detail and adherence to specific rules.
Here is the target word for this task:
<target_word>
{{target_word}}
</target_word>
Before we begin, let's clarify some key concepts:
1. Predictable sentence: A sentence where the target word may be anticipated by an adult Australian English-speaking reader before it appears. It may include connotations and associated words that commonly appear together.
2. Neutral sentence: A sentence where the target word cannot be reasonably predicted by normal adult readers, as there are no contextual clues suggesting the word.
Your task is to create one predictable and one neutral sentence using the target word. Follow these steps carefully:
Step 1: Initial Sentence Creation
Work through the process in <sentence_construction_process> tags:
a) Brainstorm associated words and contexts for the target word
b) Draft two sentences using the target word: one predictable and one neutral
c) Check each sentence against the following rules, noting pass/fail for each:
- Each sentence must be exactly 12 words long.
- The target word must appear in exactly the same position in both sentences.
- The target word must not appear in the first two or last two words of the sentence.
- Use different grammatical structures for each sentence to resist readers' anticipations/strategies.
- Use the target word consistently with its meaning and usage in Australian English.
- Use Australian English spelling (e.g., honour, analyse).
- Use only declarative sentences.
d) Revise sentences as needed until all rules pass
Step 2: Final Sentence Production
Based on your validation, produce final versions of both sentences that pass all rules. Place these in <predictable_sentence> and <neutral_sentence> tags.
Step 3: Summary Production
Produce a summary in CSV format with the following columns:
target_word, position_in_sentence, predictable_sentence, neutral_sentence, word_before_target_predictable, word_before_target_neutral
Here's an example of the expected CSV format (using placeholder data):
target_word, position_in_sentence, predictable_sentence, neutral_sentence, word_before_target_predictable, word_before_target_neutral
example, 6, "The teacher gave an excellent EXAMPLE of the scientific method.", "The old man remembered an EXAMPLE from his childhood.", excellent, an
Remember, it's crucial to follow each step carefully, double-check your work, and ensure that your output meets all specified requirements. If you're unsure at any point, please say so. It's more important to produce confident, high-quality output than to generate something that merely looks valid.
Begin by working through the process in <sentence_construction_process> tags to think through the process and draft your initial sentences.