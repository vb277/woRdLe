import numpy as np
import random

import gymnasium

import pygame

from collections.abc import Iterable
from numbers import Number

import time
from math import sin, cos



def override_settings(default_settings, custom_settings):
    for key in custom_settings:
        if key not in default_settings:
            raise KeyError(f'{key} is not a setting')
        if custom_settings[key] != None:
            default_settings[key] = custom_settings[key]



def read_word_file(filename):
    with open(filename, 'r') as file:
        word_list = [line.strip() for line in file]
    return word_list



def encode_word_list(words, alphabet, word_length):
    words_as_tuples = []
    words_as_bits = []
    for word in words:
        if len(word) != word_length:
            raise ValueError(f'word #{words.index(word)} - {word} does not match word length {word_length}')
        for character in word:
            if character not in alphabet:
                i = words.index(word)
                raise ValueError(f'word #{i} - {word} contains characters not in alphabet {alphabet}')
        word_tuple = tuple(alphabet.index(character) for character in word)
        words_as_tuples.append(word_tuple)
        word_bits = 0
        for character_int in word_tuple:
            word_bits <<= 5
            word_bits |= character_int
        words_as_bits.append(word_bits)
    return tuple(words), tuple(words_as_tuples), tuple(words_as_bits)



def bits_to_word(bits, alphabet):
    word = []
    while bits != 0:
        character_bits = bits & 31
        word.insert(0, alphabet[character_bits])
        bits >>= 5
    return ''.join(word)



def get_letter_counts(word):
    word_as_set = frozenset(word)
    hidden_word_counts = {letter: 0 for letter in word_as_set}
    for letter in word:
        hidden_word_counts[letter] += 1
    return word_as_set, hidden_word_counts



class Wordle_Environment(gymnasium.Env):

    # 1 = Full control over all keys. Backspace: -1, enter: -2
    # 2 = Controls letter keys and backspace, but automatically presses enter after filling row.
    # 3 = Controls letter keys only. Automatically enters after five letters and clears row if invalid.
    # 4 = Inputs one row at a time.
    # 5 = Chooses word from vocab list.

    def __encode_state(self):
        if self.state_representation == 2:
            return self.state
        
        letter_k = self.blank_letter_number + 1 - self.state_representation
        colour_k = 4 - self.state_representation
        total_k = letter_k + colour_k
        
        encoded_state = np.full(
            self.state.shape[1:] + (total_k,),
            0,
            'uint8'
        )

        for g in range(self.max_guesses):
            for p in range(self.word_length):
                letter = self.state[0, g, p]
                if self.state_representation == 0 or letter != self.blank_letter_number:
                    encoded_state[g, p, letter] = 1
                colour = self.state[1, g, p]
                if self.state_representation == 0 or colour != 3:
                    encoded_state[g, p, letter_k + colour] = 1

        return encoded_state.flatten()


    def __init__(self, custom_settings = {}):

        settings = {
            'word_length' : 5,
            'alphabet' : 'qwertyuiopasdfghjklzxcvbnm',
            'vocab_file' : None,
            'hidden_words_file' : None,

            #If specified, chooses a random subset of the hidden word list to use for this environment instance
            #Gradually increase the subset size to make it easier for the agent at the start
            #Make sure to use the same seed each time, or it will choose a new random subset of words
            'max_hidden_word_options' : None,
            'hidden_word_subset_seed' : 2024,

            #1 = full control over all keys
            #2 = controls letter keys and backspace, but automatically presses enter after filling row
            #3 = controls letter keys only. Automatically enters after five letters and clears row if invalid
            #4 = inputs one row at a time
            #5 = chooses word from vocab list
            'action_mode' : 3,

            'max_guesses' : 6,

            'correct_guess_reward' : 2,
            'early_guess_reward' : 1,
            'colour_rewards' : (-0.02, 0.1, 0.2), #(grey, yellow, green)
            'reward_last_attempt_only' : True,
            'invalid_word_reward' : -1,
            'valid_word_reward' : 0,
            'backspace_reward' : 0,
            'step_reward' : 0, #Reward applied at every step in addition to state-specific rewards
            'repeated_guess_reward' : -1, #Reward for re-entering a guess which has already been evaluated

            'truncation_limit' : None,

            'state_representation':'one_hot'
        }

        override_settings(settings, custom_settings)

        default_word_files = {
            2 : ('word_lists/vocab_two_letter.txt', 'word_lists/hidden_words_two_letter.txt'),
            3 : ('word_lists/vocab_three_letter.txt', 'word_lists/hidden_words_three_letter.txt'),
            4 : ('word_lists/vocab_four_letter.txt', 'word_lists/hidden_words_four_letter.txt'),
            5 : ('word_lists/vocab_five_letter.txt', 'word_lists/hidden_words_five_letter.txt'),
            10 : ('word_lists/vocab_ten_letter.txt', 'word_lists/hidden_words_ten_letter.txt'),
            20 : ('word_lists/vocab_twenty_letter.txt', 'word_lists/hidden_words_twenty_letter.txt')
        }

        #Check all fields that must be positive nonzero integers or iterables of positive nonzero integers
        for f in (
            'word_length',
            'hidden_word_subset_seed',
            'max_hidden_word_options',
            'max_guesses',
            'truncation_limit'
        ):
            value = settings[f]
            if value != None and not (isinstance(value, int) and value > 0):
                raise ValueError(f'{f} must be a positive nonzero integer')

        word_length = settings['word_length']
        self.word_length = word_length
        self.bitmasks = tuple(31 << (i * 5) for i in range(word_length))

        alphabet = settings['alphabet']
        if not isinstance(alphabet, str):
            raise ValueError('alphabet must be a string')
        if len(set(alphabet)) != len(alphabet):
            raise ValueError('alphabet must not contain duplicate characters')
        if ''.join(alphabet.split()) != alphabet:
            raise ValueError('alphabet must not contain whitespace')
        if len(alphabet) > 26:
            raise ValueError('alphabet must not contain more than 26 characters')
        self.alphabet = alphabet
        self.blank_letter_number = len(alphabet)
        
        if settings['vocab_file'] == None:
            if word_length not in default_word_files:
                error_message = 'there are no default vocab files for ' + str(word_length)
                error_message += '-letter words; you must provide your own'
                raise KeyError(error_message)
            else:
                vocab_file = default_word_files[word_length][0]
        else:
            if not isinstance(settings['vocab_file'], str):
                raise ValueError('vocab_file must be a string')
            vocab_file = settings['vocab_file']
        _, self.vocab_tuples, vocab_bits = encode_word_list(read_word_file(vocab_file), alphabet, word_length)
        self.vocab_bits = frozenset(vocab_bits)

        if settings['hidden_words_file'] == None:
            if word_length not in default_word_files:
                error_message = 'there are no default hidden word files for ' + str(word_length)
                error_message += '-letter words; you must provide your own'
                raise KeyError(error_message)
            else:
                hidden_words_file = default_word_files[word_length][1]
        else:
            if not isinstance(settings['hidden_words_file'], str):
                raise ValueError('hidden_words_file must be a string')
            hidden_words_file = settings['hidden_words_file']
        hidden_words = read_word_file(hidden_words_file)
        
        max_hidden_word_options = settings['max_hidden_word_options']
        if max_hidden_word_options != None:
            if max_hidden_word_options > len(hidden_words):
                raise ValueError('max_hidden_word_options is larger than the hidden word list')
            seed = settings['hidden_word_subset_seed']
            if not isinstance(seed, (int, float, str, bytes, bytearray)):
                raise ValueError('hidden_word_subset_seed must be int, float, str, bytes, or bytearray')
            random.Random(seed).shuffle(hidden_words)
            hidden_words = hidden_words[:max_hidden_word_options]

        hidden_words_zipped = zip(*encode_word_list(hidden_words, alphabet, word_length))
        self.hidden_words = tuple(t + get_letter_counts(t[1]) for t in hidden_words_zipped)
        self.hidden_words_len = len(hidden_words)

        if settings['action_mode'] not in (1, 2, 3, 4, 5):
            raise ValueError('action_mode must be 1, 2, 3, 4 or 5')
        self.action_mode = settings['action_mode']

        self.max_guesses = settings['max_guesses']

        for key in ('colour_rewards',):
            value = settings[key]
            if not (isinstance(value, Iterable) and all(isinstance(e, Number) for e in value)):
                raise ValueError(f'{key} must be an iterable containing numbers')
            
        for key in (
            'invalid_word_reward',
            'valid_word_reward',
            'backspace_reward',
            'step_reward',
            'correct_guess_reward',
            'early_guess_reward',
            'repeated_guess_reward'
        ):
            if not isinstance(settings[key], Number):
                raise ValueError(f'{key} must be a number')
            
        if not isinstance(settings['reward_last_attempt_only'], bool):
            raise TypeError('reward_last_attempt_only must be True or False')
        
        correct_guess_rewards = []
        for i in range(self.max_guesses - 1, -1, -1):
            correct_guess_rewards.append(settings['correct_guess_reward'] + settings['early_guess_reward'] * i)
        self.correct_guess_rewards = tuple(correct_guess_rewards)

        self.grey_reward, self.yellow_reward, self.green_reward = settings['colour_rewards']
        self.reward_last_attempt_only = settings['reward_last_attempt_only']

        self.invalid_word_reward = settings['invalid_word_reward']
        self.valid_word_reward = settings['valid_word_reward']
        self.backspace_reward = settings['backspace_reward']
        self.step_reward = settings['step_reward']
        self.repeated_guess_reward = settings['repeated_guess_reward']
            
        if len(self.correct_guess_rewards) != self.max_guesses:
            raise ValueError('length of correct_guess_rewards does not match max_guesses')
        
        self.truncation_limit = settings['truncation_limit']

        match self.action_mode:
            case 1:
                self.action_space = gymnasium.spaces.Discrete(len(alphabet) + 2, start = -2)
            case 2:
                self.action_space = gymnasium.spaces.Discrete(len(alphabet) + 1, start = -1)
            case 3:
                self.action_space = gymnasium.spaces.Discrete(len(alphabet))
            case 4:
                self.action_space = gymnasium.spaces.MultiDiscrete([len(alphabet)] * word_length, 'uint8')
            case 5:
                self.action_space = gymnasium.spaces.Discrete(len(self.vocab_tuples))

        state_representation_options = ('one_hot','one_hot_small','int')
        if settings['state_representation'] not in state_representation_options:
            raise ValueError('state_representation must be int, one_hot or one_hot_small')
        self.state_representation = state_representation_options.index(settings['state_representation'])

        subarray_shape = (self.max_guesses, word_length)
        self.starting_state = np.stack((
                np.full(subarray_shape, self.blank_letter_number, dtype='uint8'),
                np.full(subarray_shape, 3, dtype='uint8')
            ))
        self.starting_row = np.full((word_length,), self.blank_letter_number, dtype='uint8')

        if self.state_representation == 2:
            self.observation_space = gymnasium.spaces.Box(0, self.starting_state, dtype='uint8')
        else:
            if self.state_representation == 0:
                extra_options = 5
            elif self.state_representation == 1:
                extra_options = 3
            state_len = self.max_guesses * word_length * (len(alphabet) + extra_options)
            self.observation_space = gymnasium.spaces.Box(
                0,
                1,
                (self.max_guesses * word_length * (len(alphabet) + extra_options),),
                'uint8'
                )
            self.observation_space = gymnasium.spaces.MultiBinary(state_len)

        self.metadata['render_modes'] = ['command_line', 'gui']


    def reset(self, word = None, seed = None):

        super().reset(seed=seed)

        self.info = {
            'step': 0,

            'total_reward':0,

            'correct_guess':False,
            'invalid_word': False,
            'incomplete_word': False,
            'invalid_words_entered':0,
            'incomplete_words_entered':0
        }

        if word == None:
            i = self.np_random.integers(self.hidden_words_len)
            (
                self.info['hidden_word'],
                self.hidden_word_tuple,
                self.hidden_word_bits,
                self.hidden_word_set,
                self.hidden_word_counts
            ) = self.hidden_words[i]
        else:
            hidden_word_tuple = []
            hidden_word_bits = 0
            for character in word:
                letter = self.alphabet.index(character)
                hidden_word_bits = hidden_word_bits << 5 | letter
                hidden_word_tuple.append(letter)
            self.info['hidden_word'] = word
            self.hidden_word_tuple = hidden_word_tuple
            self.hidden_word_bits = hidden_word_bits
            self.hidden_word_set, self.hidden_word_counts = get_letter_counts(hidden_word_tuple)

        self.state = np.copy(self.starting_state)

        self.position = 0
        self.guess_num = 0
        #self.current_row_code = 0
        #self.current_row = np.copy(self.starting_row)

        return (self.__encode_state(), self.info)
    

    def step(self, action):
        self.info['step'] += 1
        self.info['invalid_word'] = False
        self.info['incomplete_word'] = False
        self.info['correct_guess'] = False

        reward = self.step_reward
        terminal = False
        truncated = self.info['step'] == self.truncation_limit

        enter_word = False

        if self.action_mode <= 3:

            if action == -2:
                enter_word = True

            elif action == -1:
                if self.position != 0:
                    self.position -= 1
                    self.state[0, self.guess_num, self.position] = self.blank_letter_number

            else:
                if self.position != self.word_length:
                    self.state[0, self.guess_num, self.position] = action
                    self.position += 1
                    if self.position == self.word_length and self.action_mode != 1:
                        enter_word = True

        elif self.action_mode == 4:
            self.state[0, self.guess_num] = action
            enter_word = True

        elif self.action_mode == 5:
            self.state[0, self.guess_num] = self.vocab_tuples[action]
            enter_word = True

        if enter_word:
                
            if self.blank_letter_number not in self.state[0, self.guess_num]:
                if all(self.state[0, self.guess_num] == self.hidden_word_tuple):
                    terminal = True
                    reward += self.correct_guess_rewards[self.guess_num] + self.valid_word_reward
                    self.state[1, self.guess_num] = 2
                    self.info['correct_guess'] = True

                elif tuple(self.state[0, self.guess_num]) in self.vocab_tuples:
                    if self.state[0, self.guess_num].tolist() in self.state[0, :self.guess_num].tolist():
                        reward += self.repeated_guess_reward

                    current_row_counts = {letter: 0 for letter in self.state[0, self.guess_num]}
                    row_reward = 0
                    for i, letter in enumerate(self.state[0, self.guess_num]):
                        if letter == self.hidden_word_tuple[i]:
                            current_row_counts[letter] += 1
                    for i, letter in enumerate(self.state[0, self.guess_num]):
                        if letter == self.hidden_word_tuple[i]:
                            self.state[1, self.guess_num, i] = 2
                            row_reward += self.green_reward
                        elif letter in self.hidden_word_set:
                            current_row_counts[letter] += 1
                            if current_row_counts[letter] <= self.hidden_word_counts[letter]:
                                self.state[1, self.guess_num, i] = 1
                                row_reward += self.yellow_reward
                            else:
                                self.state[1, self.guess_num, i] = 0
                                row_reward += self.grey_reward

                        else:
                            self.state[1, self.guess_num, i] = 0
                            row_reward += self.grey_reward

                    if self.guess_num == self.max_guesses - 1:
                        terminal = True
                        reward += row_reward
                    else:
                        self.guess_num += 1
                        self.position = 0
                        if not self.reward_last_attempt_only:
                            reward += row_reward
                    reward += self.valid_word_reward

                else:
                    reward += self.invalid_word_reward
                    self.info['invalid_word'] = True
                    self.info['invalid_words_entered'] += 1
                    if self.action_mode > 2:
                        self.state[0, self.guess_num] = self.blank_letter_number
                        self.position = 0

            #Word is incomplete
            else:
                reward += self.invalid_word_reward
                self.info['incomplete_word'] = True
                self.info['incomplete_words_entered'] += 1

        self.info['total_reward'] += reward

        return self.__encode_state(), reward, terminal, truncated, self.info


    def render(self):
        display_string = '\n'
        for row in range(self.max_guesses):
            for position in range(self.word_length):
                color = ['0', '33', '32', '0'][self.state[1, row, position]]
                letter = (self.alphabet + '.')[self.state[0, row, position]]
                display_string += f'\033[{color}m{letter}'
            display_string += '\n'
        display_string += '\033[0m'
        print(display_string)


    def play(self, hidden_word = None, save_dict = None):
        action_history = []
        step_rewards = []

        self.reset(hidden_word)
        while True:
            self.render()

            user_input = input()

            if self.action_mode <= 3:
                actions = []
                for c in user_input:
                    if c in ('1', '2'):
                        actions.append(- int(c))
                    else:
                        actions.append(self.alphabet.index(c))

            elif self.action_mode == 4:
                actions = (tuple(self.alphabet.index(character) for character in user_input),)

            elif self.action_mode == 5:
                actions = (int(user_input),)

            rewards = []
            for a in actions:
                state, reward, terminal, truncated, info = self.step(a)
                action_history.append(a)
                step_rewards.append(reward)
                rewards.append(str(reward))
                if terminal or truncated:
                    break
            print(' '.join(rewards))
            if terminal or truncated:
                self.render()
                if save_dict:
                    save_dict['episodes'].append({
                        'info':info,
                        'actions':action_history,
                        'rewards':step_rewards,
                        'total_reward':sum(step_rewards)
                    })
                break



def key_coords(index, scale, screen_width, screen_height):
    if index <= 9:
        x, y = 109.5 + 74 * index, 681.5
    elif index <= 18:
        x, y = 142.5 + 74 * (index - 10), 780.5
    else:
        x, y = 216.5 + 74 * (index - 19), 879.5
    return (scale * ((screen_width - 880) / 2 + x), scale * (screen_height - 1000 + y))
    


def get_letter_colours(state, max_guesses, word_length, alphabet):
    letter_colours = [3] * len(alphabet)
    for x in range(max_guesses):
        for y in range(word_length):
            letter_code, colour_code = state[:, x, y]
            if colour_code == 3:
                return letter_colours
            letter_colours[letter_code] = max(colour_code, letter_colours[letter_code])
    return letter_colours



def draw_square(
        screen, font, scale, center_coords,
        letter = None, colour_code = 3, x_scale = 1, y_scale = 1, erase = False
        ):
    x_scale_factor = scale * x_scale
    y_scale_factor = scale * y_scale
    rect = pygame.Rect(
        center_coords[0] - 39 * x_scale_factor,
        center_coords[1] - 39 * y_scale_factor,
        78 * x_scale_factor,
        78 * y_scale_factor
        )
    if erase:
        pygame.draw.rect(screen, (255, 255, 255), rect)
    elif letter == None:
        pygame.draw.rect(screen, (255, 255, 255), rect)
        pygame.draw.rect(screen, (211, 214, 218), rect, round(4.5 * scale))
    elif colour_code == 3:
        pygame.draw.rect(screen, (135, 138, 140), rect, round(4.5 * scale))
    else:
        colour_value = ((120, 124, 126), (201, 180, 88), (106, 170, 100))[colour_code]
        pygame.draw.rect(screen, colour_value, rect)

    if letter != None:
        if colour_code == 3:
            letter_colour = (0, 0, 0)
        else:
            letter_colour = (255, 255, 255)
        img = font.render(letter.upper(), True, letter_colour)
        img = pygame.transform.scale_by(img, (x_scale, y_scale))
        rect = img.get_rect()
        rect.center = (center_coords)
        screen.blit(img, rect)



def draw_message(screen, scale, font, message = None):

    screen_width = screen.get_size()[0]

    if message == None:
        #Erase previous message
        rect = pygame.Rect(0, 0, screen_width, 90 * scale)
        pygame.draw.rect(screen, (255, 255, 255), rect)

    else:
        img = font.render(message, True, (255, 255, 255))

        coords = (screen_width / 2, 57 * scale)

        #Draw box
        rect = pygame.Rect(0, 0, img.get_size()[0] + 40 * scale, 63 * scale)
        rect.center = coords
        r = round(4 * scale)
        pygame.draw.rect(screen, (0, 0, 0), rect, 0, r, r, r, r)

        #Display message
        rect = img.get_rect()
        rect.center = (coords)
        screen.blit(img, rect)



class Wordle_GUI_Wrapper(gymnasium.Wrapper):

    def __initalise_pygame(self):
        screen_width = max(880, 85.5 * self.env.word_length + 15) #Normally 880
        screen_height = 86 * self.env.max_guesses + 484 #Normally 1000
        self.screen_width, self.screen_height = screen_width, screen_height

        pygame.init()
        screen = pygame.display.set_mode((self.scale * screen_width, self.scale * screen_height))
        self.screen = screen
        pygame.display.set_caption('Wordle Environment')
        screen.fill((255, 255, 255))
        pygame.display.update()


    def __init__(self, env = None, custom_render_settings = {}):

        self.env = env
        env.reset()
    
        render_settings = {
            'render_mode':'gui', #'command_line' or 'gui',
            'scale': 2/3,
            'animation_duration': 1.0 #1 is normal speed, 0 is instant
        }

        override_settings(render_settings, custom_render_settings)

        if (not isinstance(render_settings['scale'], Number)) or render_settings['scale'] <= 0:
                raise ValueError('Scale must be a number greater than zero')
        self.scale = render_settings['scale']

        value = render_settings['animation_duration']
        if (not isinstance(value, Number)) or value < 0:
            raise ValueError('animation_duration must be a non-negative number')

        
        self.animation_duration = render_settings['animation_duration']

        self.success_messages = ('Genius', 'Magnificent', 'Impressive', 'Splendid')[:env.max_guesses]
        self.success_messages += ('Great',) * (env.max_guesses - 5)
        if env.max_guesses >= 6:
            self.success_messages += ('Phew',)

        self.render()

    
    def reset(self, word = None):
        result = self.env.reset(word)
        if self.currently_rendered:
            self.render()
        return result


    def step(self, action):
        if not self.currently_rendered:
            return self.env.step(action)
        
        scale = self.scale
        env = self.env
        screen = self.screen

        old_state = np.copy(env.state)
        guess_num, position = env.guess_num, env.position

        result = env.step(action)
        _, reward, terminal, truncated, info = result
        state = env.state

        #Action is a single letter
        if self.env.action_mode <= 3 and action >= 0 and action <= self.env.blank_letter_number:
            added_letters = (env.alphabet[action],)
        #Action is a sequence of letters
        elif self.env.action_mode == 4:
            added_letters = (env.alphabet[l] for l in action)
        #Action is the index of a word in the vocab list
        elif self.env.action_mode == 5:
            added_letters = (env.alphabet[l] for l in env.vocab_tuples[action])
        #Action is backspace or enter
        else:
            added_letters = None
        
        #Add letters
        if added_letters != None and position != env.word_length:
            row_coords = self.board_coords[guess_num]
            for letter in added_letters:
                coords = row_coords[position]
                scale_up = 1
                anim_time = 0.11333333333 * self.animation_duration
                finish_time = time.time() + anim_time
                while True:
                    draw_square(
                        screen, None, scale, coords, x_scale = scale_up, y_scale = scale_up, erase = True
                        )
                    if anim_time == 0:
                        completion = 0
                    else:
                        completion = (finish_time - time.time()) / anim_time
                    scale_up = max(sin(completion * 3.14), 0) * 0.12 + 1
                    draw_square(
                        screen, self.board_font, scale, coords, letter,
                        x_scale = scale_up, y_scale = scale_up
                        )
                    pygame.display.update()
                    if time.time() >= finish_time:
                        break

                old_state[0, guess_num, position] = env.alphabet.index(letter)
                position += 1

        clear_keypress_buffer = False

        #Invalid word animation
        if info['invalid_word'] or info['incomplete_word']:
            clear_keypress_buffer = True

            if info['invalid_word']:
                message = 'Not in word list'
            else:
                message = 'Not enough letters'
            draw_message(screen, scale, self.message_font, message)

            square_colours = old_state[1, guess_num]
            letters = [None] * env.word_length
            for i, l in enumerate(old_state[0, guess_num]):
                if l != self.env.blank_letter_number:
                    letters[i] = env.alphabet[l]

            anim_time = 0.62 * self.animation_duration
            max_offset = 7
            shakes = 10

            background_rect = pygame.Rect(
                scale * (self.x_origin - 78 / 2 - max_offset),
                scale * (105 + 86 * guess_num),
                scale * (env.word_length * 85.5 + 78 + max_offset * 2),
                scale * 94
                )
            row_coords = self.board_coords[guess_num]

            start_time = time.time()
            finish_time = start_time + anim_time
            while True:
                if anim_time == 0:
                    completion = 0
                else:
                    completion = min(1, (time.time() - start_time) / anim_time)
                radians = completion * 3.14
                offset = sin(radians * shakes) * sin(radians) * max_offset

                pygame.draw.rect(screen, (255, 255, 255), background_rect)
                for i, center_coords in enumerate(row_coords):
                    new_coords = (center_coords[0] + scale * offset, center_coords[1])
                    draw_square(screen, self.board_font, scale, new_coords, letters[i], square_colours[i])

                pygame.display.update()

                if time.time() >= finish_time:
                    break

            draw_message(screen, scale, self.message_font)

        correct_answer = all(state[1, guess_num] == 2)

        #Flip letters, revealing colours
        if state[1, guess_num, 0] != old_state[1, guess_num, 0]:
            clear_keypress_buffer = True

            if guess_num + 1 == env.max_guesses and not correct_answer:
                draw_message(screen, scale, self.message_font, info['hidden_word'].upper())
                
            changed_key_colours = []

            row_coords = self.board_coords[guess_num]
            for i in range(env.word_length):
                l = state[0, guess_num, i]
                letter = env.alphabet[l]
                colour = state[1, guess_num, i]
                coords = row_coords[i]

                anim_time = self.animation_duration / 3
                y_scale = 1
                finish_time = time.time() + anim_time
                while True:
                    draw_square(screen, None, scale, coords,
                                x_scale = 1.1, y_scale = y_scale * 1.1, erase = True)
                    if anim_time == 0:
                        completion = 0
                    else:
                        completion = max((finish_time - time.time()) / anim_time, 0)
                    if completion > 0.5:
                        colour_code = 3
                    else:
                        colour_code = colour
                    y_scale = abs(cos(completion * 3.14))
                    draw_square(screen, self.board_font, scale, coords, letter, colour_code, y_scale = y_scale)
                    pygame.display.update()
                    if time.time() >= finish_time:
                        draw_square(screen, self.board_font, scale, coords, letter, colour, y_scale = 1)
                        pygame.display.update()
                        break

                if (colour + 1) % 4 > (self.letter_colours[l] + 1) % 4:
                    self.letter_colours[l] = colour
                    changed_key_colours.append(l)

            #Change keyboard colours
            for l in changed_key_colours:
                letter = env.alphabet[l]
                c = self.letter_colours[l]
                coords = self.key_coords[l]

                #Erase previous key colour
                slightly_bigger_scale = scale * 1.1
                bg_rect = (
                    coords[0] - 32.5 * slightly_bigger_scale,
                    coords[1] - 43.5 * slightly_bigger_scale,
                    65 * slightly_bigger_scale,
                    87 * slightly_bigger_scale
                )
                pygame.draw.rect(screen, (255, 255, 255), bg_rect)

                #Draw background rectangle
                rectangle_colour = ((120, 124, 126), (201, 180, 88), (106, 170, 100), (211, 214, 218))[c]
                bg_rect = (
                    coords[0] - 32.5 * scale,
                    coords[1] - 43.5 * scale,
                    65 * scale,
                    87 * scale
                )
                r = round(4 * scale)
                pygame.draw.rect(screen, rectangle_colour, bg_rect, 0, r, r, r, r)

                #Draw letter
                letter_colour = ((255, 255, 255), (255, 255, 255), (255, 255, 255), 0)[c]
                img = self.keyboard_font.render(letter.upper(), True, letter_colour)
                rect = img.get_rect()
                rect.center = (coords)
                screen.blit(img, rect)

            draw_message(screen, scale, self.message_font)

        #Move letters up and down to celebrate if the player gets the right answer
        if correct_answer and self.animation_duration != 0:

            row_coords = self.board_coords[guess_num]
            letters = [env.alphabet[l] for l in state[0, guess_num]]

            row_above_exists = guess_num != 0
            if row_above_exists:
                row_above_coords = self.board_coords[guess_num - 1]
                row_above_letters = [env.alphabet[l] for l in state[0, guess_num - 1]]
                row_above_colours = state[1, guess_num - 1]

            anim_time = (5/6 + 0.1 * (env.word_length - 1)) * self.animation_duration
            offsets = [(0.1 * i * self.animation_duration) / anim_time for i in range(env.word_length)]
            finish_time = time.time() + anim_time
            while True:
                completion = 1 - max((finish_time - time.time()) / anim_time, 0)
                for i in range(env.word_length):

                    #Erase previous image
                    rect = pygame.Rect(
                    row_coords[i][0] - 39 * scale,
                    row_coords[i][1] - 121 * scale,
                    78 * scale,
                    165 * scale
                    )
                    pygame.draw.rect(screen, (255, 255, 255), rect)

                    if row_above_exists:
                        draw_square(
                            screen, self.board_font, scale, row_above_coords[i], row_above_letters[i],
                            row_above_colours[i]
                        )

                    cell_completion = max(completion * (1 + offsets[i]) - offsets[i], 0)
                    sin_value = sin((cell_completion * 6 - 0.5) * 3.14) + 1
                    coords = (
                        row_coords[i][0],
                        row_coords[i][1] - (sin_value * 37 * (1 - cell_completion) ** 2) * scale
                    )
                    draw_square(screen, self.board_font, scale, coords, letters[i], 2)
                draw_message(screen, scale, self.message_font, self.success_messages[guess_num])
                pygame.display.update()
                if completion == 1:
                    break
                
            draw_message(screen, scale, self.message_font)
        
        #Remove deleted letters
        for i in range(env.word_length):
            if state[0, guess_num, i] != old_state[0, guess_num, i]:
                draw_square(screen, None, scale, self.board_coords[guess_num][i])
        pygame.display.update()

        self.keypress_buffer = pygame.event.get()
        for event in self.keypress_buffer:
            if event.type == pygame.QUIT:
                self.stop_render()
        if clear_keypress_buffer:
            self.keypress_buffer = []

        return result


    def render(self):
        scale = self.scale
        env = self.env
        alphabet = env.alphabet

        self.currently_rendered = True
        self.old_state = env.state
        self.keypress_buffer = []

        self.__initalise_pygame()
        screen_width, screen_height = self.screen_width, self.screen_height
        screen = self.screen

        self.backspace_icon = pygame.image.load('gui_images/backspace_icon.png').convert()
        self.backspace_icon = pygame.transform.smoothscale_by(self.backspace_icon, self.scale)

        enter_icon_font = pygame.font.SysFont(None, round(self.scale * 26))
        self.enter_icon = enter_icon_font.render('ENTER', True, (0, 0, 0))

        board_font = pygame.font.SysFont(None, round(scale * 68))
        self.board_font = board_font
        x_origin = (screen_width - 85.5 * (env.word_length - 1)) / 2
        self.x_origin = x_origin
        board_coords = []
        for r_pos in range(env.max_guesses):
            row_coords = []
            for l_pos in range(env.word_length):
                center_coords = (scale * (x_origin + l_pos * 85.5), scale * (152 + r_pos * 86))
                row_coords.append(center_coords)
                letter_code = env.state[0, r_pos, l_pos]
                try:
                    letter = alphabet[letter_code]
                except IndexError:
                    draw_square(screen, board_font, scale, center_coords)
                else:
                    colour_code = env.state[1, r_pos, l_pos]
                    draw_square(screen, board_font, scale, center_coords, letter, colour_code)
            board_coords.append(row_coords)
        self.board_coords = board_coords

        self.keyboard_font = pygame.font.SysFont(None, round(scale * 42))
        self.letter_colours = get_letter_colours (
            env.state, env.max_guesses, env.word_length, alphabet
            )
        self.key_coords = [key_coords(i, scale, screen_width, screen_height) for i in range(len(alphabet))]
        for i, colour in enumerate(self.letter_colours):
            center_coords = self.key_coords[i]

            rectangle_colour = ((120, 124, 126), (201, 180, 88), (106, 170, 100), (211, 214, 218))[colour]
            bg_rect = (
                    center_coords[0] - 32.5 * scale,
                    center_coords[1] - 43.5 * scale,
                    65 * scale,
                    87 * scale
                    )
            r = round(4 * scale)
            pygame.draw.rect(screen, rectangle_colour, bg_rect, 0, r, r, r, r)
            
            letter_colour = ((255, 255, 255), (255, 255, 255), (255, 255, 255), 0)[colour]
            img = self.keyboard_font.render(alphabet[i].upper(), True, letter_colour)
            rect = img.get_rect()
            rect.center = (center_coords)
            screen.blit(img, rect)

        #Backspace key
        if env.action_mode <= 2:
            box_rect = pygame.Rect(
                    (screen_width / 2 + 261) * scale,
                    (screen_height - 164) * scale,
                    98 * scale,
                    87 * scale
                    )
            r = round(4 * scale)
            pygame.draw.rect(screen, (211, 214, 218), box_rect, 0, r, r, r, r)

            img_rect = self.backspace_icon.get_rect()
            img_rect.center = box_rect.center
            screen.blit(self.backspace_icon, img_rect)

        #Enter key
        if env.action_mode == 1:
            box_rect = pygame.Rect(
                    (screen_width / 2 - 367) * scale,
                    (screen_height - 164) * scale,
                    98 * scale,
                    87 * scale
                    )
            r = round(4 * scale)
            pygame.draw.rect(screen, (211, 214, 218), box_rect, 0, r, r, r, r)

            img_rect = self.enter_icon.get_rect()
            img_rect.center = box_rect.center
            screen.blit(self.enter_icon, img_rect)

        pygame.display.update()

        self.message_font = pygame.font.SysFont('franklingothicdemi', round(scale * 22))


    def stop_render(self):
        self.currently_rendered = False
        pygame.quit()


    def play(self, hidden_word = None, keybindings = None, save_dict = None):

        if keybindings == None:
            if self.env.action_mode <= 4:
                keybindings = {}
                for i, l in enumerate('abcdefghijklmnopqrstuvwxyz'):
                    if l in self.env.alphabet:
                        keybindings[i + 97] = self.env.alphabet.index(l)
                if self.env.action_mode <= 2:
                    keybindings[8] = -1
                    if self.env.action_mode == 1:
                        keybindings[13] = -2
            if self.env.action_mode == 5:
                keybindings = {i+48:i for i in range(10)}
                keybindings[13] = -2
            
        self.reset(hidden_word)
        self.render()
        terminal, truncated = False, False

        action_buffer = []
        action_counter = 0

        action_history = []
        step_rewards = []

        while True:
            events = [e for e in pygame.event.get()] + self.keypress_buffer
            self.keypress_buffer = []
            for event in events:
                if event.type == pygame.KEYDOWN:
                    try:
                        action = keybindings[event.key]
                    except KeyError:
                        pass
                    else:

                        if self.env.action_mode <= 3:
                            state, reward, terminal, truncated, info = self.step(action)
                            action_history.append(action)
                            step_rewards.append(reward)

                        elif self.env.action_mode == 4:
                            action_buffer.append(action)
                            if len(action_buffer) == self.env.word_length:
                                state, reward, terminal, truncated, info = self.step(action_buffer)
                                action_history.append(action_buffer)
                                step_rewards.append(reward)
                                action_buffer = []

                        elif self.env.action_mode == 5:
                            if action == -2:
                                state, reward, terminal, truncated, info = self.step(action_counter)
                                action_history.append(action_counter)
                                step_rewards.append(reward)
                                action_counter = 0
                            else:
                                action_counter = action_counter * 10 + action

                elif event.type == pygame.QUIT:
                    self.stop_render()
                    return
                
            if terminal or truncated or not self.currently_rendered:
                if save_dict:
                    save_dict['episodes'].append({
                        'info':info,
                        'actions':action_history,
                        'rewards':step_rewards,
                        'total_reward':sum(step_rewards)
                    })
                break


    def close(self):
        self.stop_render()



def make(custom_settings = {}, custom_render_settings = {}):
    """
Returns a Wordle environment instance.

Accepts a custom settings dictionary with any of the following fields:

- 'word_length' - 5 by default. If not 2-5, you must specify a 'vocab_file' and 'hidden_words_file'.
- 'alphabet' - Valid characters the player can input. 'abcdefghijklmnopqrstuvwxyz' by default.
- 'vocab_file' - Path to a text file containing the valid word list.
- 'hidden_words_file' - Path to a text file containing all possible hidden/answer words.

- 'max_hidden_word_options' - If specified, plays with a pseudorandom subset of the hidden word list.
Gradually increase the subset size to make it easier for the agent at the start.
- 'hidden_word_subset_seed' - Determines the order in which words are chosen for the subset.
Make sure to use the same seed each time, or it will choose a new random subset of words.

- 'action_mode' - How much control over input is given to the agent. 3 by default.
    - 1 = Full control over all keys.
    - 2 = Controls letter keys and backspace, but automatically presses enter after filling row.
    - 3 = Controls letter keys only. Automatically enters after five letters and clears row if invalid.
    - 4 = Inputs one row at a time.
    - 5 = Chooses word from vocab list.

- 'max_guesses' - The number of rows on the Wordle board. 6 by default.

- 'correct_guess_reward' - Base reward given for winning the game by guessing the hidden word. 2 by default.
- 'early_guess_reward' - Bonus reward given for each remaining unused guess when the game is won. 1 by default.
- 'colour_rewards' - Iterable containing rewards for grey, yellow and green letters in an incorrect guess.
(-0.02, 0.1, 0.2) by default.
- 'reward_last_attempt_only' - True if colour rewards should only be applied to the final guess, otherwise
False. True by default.
- 'invalid_word_reward' - Reward given when an invalid word is entered. -1 by default.
- 'valid_word_reward' - Reward given when a valid word is entered. 0 by default.
- 'backspace_reward' - Reward given when backspace is inputted. 0 by default.
- 'step_reward' - Reward applied at every step in addition to state-specific rewards. 0 by default.
- 'repeated_guess_reward' - Reward given when the player re-enters a valid word they have already
entered this episode. -1 by default.

- 'truncation_limit' - If specified, will truncate each episode after this many steps.

- 'state_representation' - Customise the encoding of the observation space
    - 'one_hot' (default) - Letters and colours are treated as categories and converted to bool matrices
using one-hot encoding, then the array is flattened
    - 'one_hot_small' - Similar to one_hot, but instead of treating empty letters/colours as separate
categories, all the bools for that letter/colour slot are set to False
    - 'int' - The board is represented as a 3D array where the first dimension is (0: letters, 1: colours),
letters are represented as their alphabet index (or one higher than the max index for an empty letter), and
colours are (0 = grey, 1 = yellow, 2 = green, 3 = empty)

And a custom render settings dictionary with any of the following fields:

- 'render_mode' - Either 'command_line' or 'gui'.
- 'scale' - Factor by which to scale the window. Default is 2/3
- 'animation_duration' - Factor by which animation times are multiplied. 1 is normal speed, 0 is instant.
    """
    
    env = Wordle_Environment(custom_settings)

    if 'render_mode' in custom_render_settings:
        render_mode = custom_render_settings['render_mode']
    else:
        render_mode = 'command_line'

    env.render_mode = render_mode

    if render_mode == 'gui':
        return Wordle_GUI_Wrapper(env, custom_render_settings)
    elif render_mode == 'command_line':
        return env
    else:
        raise ValueError(f'there is no such render_mode as {render_mode}')



if __name__ == '__main__':
    env = make(custom_render_settings = {'render_mode':'gui'})
    env.play()