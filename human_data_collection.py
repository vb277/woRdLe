from wordle_environment import make
import json



custom_settings = {
    'word_length': 3,
    'truncation_limit': 1000,
    'correct_guess_reward': 10,
    'early_guess_reward': 0.2,
    'colour_rewards': (0, 0.05, 0.1),
    'valid_word_reward': 0,
    'invalid_word_reward': 0,
    'step_reward': -0.0001,
    'repeated_guess_reward': 0,
    'alphabet': 'abcdefgh',
    'vocab_file': 'word_lists/three_letter_abcdefgh.txt',
    'hidden_words_file': 'word_lists/three_letter_abcdefgh.txt',
#    'max_hidden_word_options': 8,
#    'hidden_word_subset_seed': 1,
#    'state_representation': 'one_hot_grid'
}
custom_render_settings = {
    'render_mode':'gui',
    'animation_duration':0.4
}
env = make(custom_settings, custom_render_settings)

try:
    with open('human_data.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    data = {
        'settings':custom_settings,
        'episodes':[]
    }

for i in range(20):
    env.play(save_dict = data)
    with open('human_data.json', 'w') as f:
        json.dump(data, f)
    print(data)