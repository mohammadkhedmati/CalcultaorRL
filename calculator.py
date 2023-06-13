import numpy as np
import pandas as pd

ACTIONS = ['push', '+', '-', '*', '/', '**']
STACK = []
OPERANDS = []
# OPERAND = [2,3,4,5]
number_state = 3
EPSILON = .9
TARGET_NUMBER = 10


def get_operands():
    numbers = int(input('How many operands do you need? \n '))
    while numbers != 0:
        number = input('Enter your number : \n ')
        OPERANDS. append(number)
        numbers -= 1
    TARGET_NUMBER = int(input('Enter your target number : '))


def create_qTable():
    table = pd.DataFrame(np.zeros(number_state, len(ACTIONS)))
    return table


def choose_action(state, qTable):
    state_actions = qTable.iloc[state, :]
    if np.random.uniform > EPSILON or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idmax()

    return action_name


def get_env_feedback(state, action):
    if action == ACTIONS[0]:
        if state == 0 or state == 1 and len(OPERANDS) != 0:
            state_ += 1
            reward = 0
        if state == 1 and len(OPERANDS) == 0:
            state_ = 1
            reward = -1
        else:
            state_ = 2
            reward = 0
    else:
        if state == 0:
            reward = -1
            state_ = 0
        elif state == 1:
            reward = 0
            state_ = 0
        elif state == 2:
            if len(STACK) == 1 and len(OPERANDS) == 0 and STACK[0] == TARGET_NUMBER:
                state_ = 'TERMINAL'
                reward = 1
            elif len(STACK) == 1 and len(OPERANDS) != 0:
                state_ = 0
                reward = 0
            else:
                state_ = 1
                reward = 0
    return state_, reward


def update_env(state, episods, stepCounter):
    if state == 'TERMINAL':
        print(STACK)
    elif state == 0:
        STACK.append(OPERANDS[0])
        OPERANDS.remove(OPERANDS[0])
    elif state == 1:
        STACK.append(OPERANDS[0])
        OPERANDS.remove(OPERANDS[0])
    elif state == 2:
        STACK.append(OPERANDS[0])
        OPERANDS.remove(OPERANDS[0])
