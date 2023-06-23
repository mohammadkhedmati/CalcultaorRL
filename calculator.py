import numpy as np
import pandas as pd

ACTIONS = ['push', '+', '-', '*', '/', '**']
STACK = []
OPERANDS = []
OPERANDS_2 = []
# OPERAND = [2,3,4,5]
number_state = 3
EPSILON = .9
MAX_EPISODES = 150
GAMMA = .9
ALPHA = .1
TARGET_VALUE = None
SELECTED_ACTIONS = []


def get_operands():
    numbers = int(input('How many operands do you need? \n '))
    while numbers != 0:
        number = int(input('Enter your number : \n '))
        OPERANDS.append(number)
        numbers -= 1
    global OPERANDS_2
    global TARGET_VALUE
    TARGET_VALUE = int(input('Enter your target number : '))
    OPERANDS_2 = OPERANDS[:]


def create_qTable():
    table = pd.DataFrame(
        np.zeros(shape=(number_state, len(ACTIONS))), columns=ACTIONS)
    return table


def operation(action_name):
    if action_name == ACTIONS[0]:
        STACK.append(OPERANDS[0])
        OPERANDS.remove(OPERANDS[0])
    elif action_name == ACTIONS[1]:
        num1 = STACK[-1]
        num2 = STACK[-2]
        STACK.pop()
        STACK.pop()
        STACK.append(num1 + num2)
    elif action_name == ACTIONS[2]:
        num1 = STACK[-1]
        num2 = STACK[-2]
        STACK.pop()
        STACK.pop()
        STACK.append(num1 - num2)
    elif action_name == ACTIONS[3]:
        num1 = STACK[-1]
        num2 = STACK[-2]
        STACK.pop()
        STACK.pop()
        STACK.append(num1 * num2)
    elif action_name == ACTIONS[4]:
        num1 = STACK[-1]
        num2 = STACK[-2]
        STACK.pop()
        STACK.pop()
        STACK.append(num1 / num2)
    elif action_name == ACTIONS[5]:
        num1 = STACK[-1]
        num2 = STACK[-2]
        STACK.pop()
        STACK.pop()
        STACK.append(num1 ** num2)


def choose_action(state, qTable):
    state_actions = qTable.iloc[state, :]
    if np.random.uniform() > EPSILON or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()

    try:
        operation(action_name=action_name)
        return action_name
    except:
        return action_name


def reset_env():
    global OPERANDS
    global SELECTED_ACTIONS
    OPERANDS = OPERANDS_2[:]
    SELECTED_ACTIONS = []
    if len(STACK) != 0:
        for i in STACK:
            STACK.pop()


def get_env_feedback(state, action):
    if action == ACTIONS[0]:
        if state == 0 or state == 1 and len(OPERANDS) != 0:
            S_ = state + 1
            reward = 0
        if state == 1 and len(OPERANDS) == 0:
            S_ = 1
            reward = -1
            reset_env()
        else:
            S_ = 2
            reward = 0
    else:
        if state == 0:
            reward = -1
            S_ = 0
            reset_env()
        elif state == 1:
            reward = 0
            S_ = 0
        elif state == 2:
            if len(STACK) == 1 and len(OPERANDS) == 0 and STACK[0] == TARGET_VALUE:
                S_ = 'TERMINAL'
                reward = 1
            elif len(STACK) == 1 and len(OPERANDS) != 0:
                S_ = 0
                reward = 0
            elif len(STACK) == 1 and len(OPERANDS) == 0:
                S_ = 0
                reward = -1
                reset_env()
            else:
                S_ = 1
                reward = 0
    return S_, reward


def update_env(state, episode, stepCounter, action, total_Reward):
    if state == 'TERMINAL':
        print(
            f'terminal reached !!!! \n stack is : ---- {STACK} ----- \n operands is : ---- {OPERANDS} -----')
    else:
        print(
            f'episode : {episode} \n action in this step : {action} \n total actions : {SELECTED_ACTIONS} \n total reward is : {total_Reward} \n state is {state} !!!! \n stack is : ---- {STACK} ----- \n operands is : ---- {OPERANDS} -----')
        # print(
        #     f' \n episode : {episode} \n action in this step : {action} \n total reward is : {total_Reward} \n state is {state} !!!! \n stack is : ---- {STACK} ----- \n operands is : ---- {OPERANDS} ----- \n')


def rl():
    global SELECTED_ACTIONS
    operands = get_operands()
    q_table = create_qTable()
    print(q_table)
    total_Reward = 0
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminate = False
        update_env(state=S, episode=episode,
                   stepCounter=step_counter, action=None, total_Reward=total_Reward)
        while not is_terminate:
            A = choose_action(state=S, qTable=q_table)

            S_, R = get_env_feedback(
                state=S, action=A)

            q_predict = q_table.loc[S, A]
            total_Reward += R
            SELECTED_ACTIONS.append(A)
            if S_ != 'TERMINAL':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminate = True

            q_table.loc[S, A] += ALPHA * (q_target-q_predict)
            S = S_
            update_env(state=S, episode=episode,
                       stepCounter=step_counter, action=A, total_Reward=total_Reward)
            step_counter += 1
        print('\n', q_table, '\n')
    return q_table


if __name__ == "__main__":
    q_table = rl()
    # print()
