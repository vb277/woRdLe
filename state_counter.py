from wordle_environment import make
import numpy as np



def count_states(custom_settings = {}, include_incomplete_words = True):
    custom_settings['action_mode'] = 4
    custom_settings['state_representation'] = 'int'
    env = make(custom_settings)
    unique_rows = 0
    i = 0
    for vocab_word in env.vocab_tuples:
        states = []
        vocab_set = set(vocab_word)
        for word, _, __, word_set, ____ in env.hidden_words:
            if states == [] or not vocab_set.isdisjoint(word_set):
                env.reset(word)
                new_state = env.step(vocab_word)[0][:,0]
                state_is_new = True
                for state in states:
                    if np.array_equal(state, new_state):
                        state_is_new = False
                        break
                if state_is_new:
                    states.append(new_state)
                    unique_rows += 1
            i += 1
            if i % 10000 == 0:
                print(f'{i} input & hidden word combinations checked')
    unique_nonfinal_rows = unique_rows - len(env.hidden_words)
    print(f'{unique_rows} unique input & hidden word combinations found')

    incomplete_word_options = 1
    for i in range(1, env.word_length + 1):
        incomplete_word_options += len(env.alphabet) ** i

    unique_states = 1
    for i in range(env.max_guesses):
        unique_states += unique_nonfinal_rows ** i * (unique_rows - unique_nonfinal_rows)
        if i != env.max_guesses - 1 and include_incomplete_words:
            unique_states += unique_nonfinal_rows ** (i + 1) * incomplete_word_options
        else:
            unique_states += unique_nonfinal_rows ** (i + 1)

    print(f'{unique_states} unique states found')
    return unique_states



custom_settings = {
    'word_length':3,
    'vocab_file':'word_lists/three_letter_abcdefgh.txt',
    'hidden_words_file':'word_lists/three_letter_abcdefgh.txt'
}
count_states(custom_settings)