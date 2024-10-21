from __future__ import annotations
__all__ = ['BAD_WORDS_LIST', 'BEAM_WIDTH', 'CUM_LOG_PROBS', 'DRAFT_INPUT_IDS', 'DRAFT_LOGITS', 'EARLY_STOPPING', 'EMBEDDING_BIAS', 'END_ID', 'FREQUENCY_PENALTY', 'INPUT_IDS', 'LENGTH_PENALTY', 'MAX_NEW_TOKENS', 'MIN_LENGTH', 'OUTPUT_IDS', 'OUTPUT_LOG_PROBS', 'PAD_ID', 'PRESENCE_PENALTY', 'PROMPT_EMBEDDING_TABLE', 'PROMPT_VOCAB_SIZE', 'RANDOM_SEED', 'REPETITION_PENALTY', 'RETURN_CONTEXT_LOGITS', 'RETURN_GENERATION_LOGITS', 'RETURN_LOG_PROBS', 'RUNTIME_TOP_K', 'RUNTIME_TOP_P', 'SEQUENCE_LENGTH', 'STOP_WORDS_LIST', 'TEMPERATURE']
BAD_WORDS_LIST: str = 'bad_words_list'
BEAM_WIDTH: str = 'beam_width'
CUM_LOG_PROBS: str = 'cum_log_probs'
DRAFT_INPUT_IDS: str = 'draft_input_ids'
DRAFT_LOGITS: str = 'draft_logits'
EARLY_STOPPING: str = 'early_stopping'
EMBEDDING_BIAS: str = 'embedding_bias'
END_ID: str = 'end_id'
FREQUENCY_PENALTY: str = 'frequency_penalty'
INPUT_IDS: str = 'input_ids'
LENGTH_PENALTY: str = 'len_penalty'
MAX_NEW_TOKENS: str = 'request_output_len'
MIN_LENGTH: str = 'min_length'
OUTPUT_IDS: str = 'output_ids'
OUTPUT_LOG_PROBS: str = 'output_log_probs'
PAD_ID: str = 'pad_id'
PRESENCE_PENALTY: str = 'presence_penalty'
PROMPT_EMBEDDING_TABLE: str = 'prompt_embedding_table'
PROMPT_VOCAB_SIZE: str = 'prompt_vocab_size'
RANDOM_SEED: str = 'random_seed'
REPETITION_PENALTY: str = 'repetition_penalty'
RETURN_CONTEXT_LOGITS: str = 'return_context_logits'
RETURN_GENERATION_LOGITS: str = 'return_generation_logits'
RETURN_LOG_PROBS: str = 'return_log_probs'
RUNTIME_TOP_K: str = 'runtime_top_k'
RUNTIME_TOP_P: str = 'runtime_top_p'
SEQUENCE_LENGTH: str = 'sequence_length'
STOP_WORDS_LIST: str = 'stop_words_list'
TEMPERATURE: str = 'temperature'