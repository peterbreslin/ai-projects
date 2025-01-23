import pickle
import random
from pathlib import Path

# Get the directory of the current file
directory = Path(__file__).parent.resolve()

"""
Building an agent to play Tic-Tac-Toe using Q-Learning.

This basic tabular reinforcement learning approach trains the agent with the Q-Learning algorithm, which uses the Bellman Equation to determine and update the Q-values from interacting with a Tic-Tac-Toe environment that is built from scratch. 

The QTable is saved and the agent's performance is then evaluated post-training by playing games against the computer (the computer chooses random moves every time). Finally, the agent plays us, the human, in a command-line game of Tic-Tac-Toe.
"""


class TTTQLearning:
    def __init__(self, N=int(1e7), alpha=0.2, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        self.Qtable = {}
        self.N = N
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay


    # ------------ ------------ ------------ ------------ ------------ --
    # Helper functions for building and interacting with the environment
    # ------------ ------------ ------------ ------------ ------------ --
    def possible_actions(self, state):
        """
        Returns a list of available actions (the empty spots on the Tic-Tac-Toe board).
        NOTE: 0 is an available action.
        List Comprehension: return [i for i, x in enumerate(state) if x == 0]
        """
        actions = []
        for action, x in enumerate(state):
            if x == 0:
                actions.append(action)
        return actions


    def check_winner(self, state, player):
        """
        Checks whether the given player has won the game by examining all possible winning combinations on the board. For each winning combination, check whether all the cells in that combination are occupied by the given player.
        """
        winning_rows = [(0,1,2), (3,4,5), (6,7,8)]
        winning_cols = [(0,3,6), (1,4,7), (2,5,8)]
        winning_diag = [(0,4,8), (2,4,6)]
        winning_states = winning_rows + winning_cols + winning_diag

        # Check if all cells in each winning combo are occupied by the given player
        return any(all(state[cell] == player for cell in combo) for combo in winning_states)
    

    def update_state(self, state, action, player):
        """
        Creates a new board state after a player makes a move, marking the cell at the given action index with the player's number.
        """
        new_state = state[:] # shallow copy of list so original state remains unchanged
        new_state[action] = player # represents the player's move on the board
        return new_state


    def next_state_and_reward(self, state, action):
        """
        Simulates the game mechanics for one move by Player 1 and a subsequent response by Player 2.
        Returns the new state and the reward.
        """
        # Update state with Player 1's move
        new_state = self.update_state(state, action, 1)

        # Check if Player 1 wins
        if self.check_winner(new_state, 1):
            return (new_state, 1)  # Reward for winning

        # Check if there are no empty spaces (0) left on the board (then game is a draw)
        elif 0 not in new_state:
            return (new_state, 0.1)  # Small reward given for draw

        # Update state with a random move from Player 2
        else:
            actions = self.possible_actions(new_state)
            random_action = random.choice(actions)
            new_state = self.update_state(new_state, random_action, 2)

            # Check if Player 2 wins
            if self.check_winner(new_state, 2):
                return (new_state, -1)  # Player 1 gets a penalty for losing
            else:
                return (new_state, 0)  # No consequence for Player 1
            

    # ------------ ------------ ----------
    # Training the agent using Q-Learning
    # ------------ ------------ ----------
    def train(self):
        """
        Trains the agent using Q-learning and saves the QTable as a pickle dict.
        - Each key in the dictionary is a board state (as a string).
        - The value for each key is a list of 9 Q-values.
        - When the algorithm encounters a new state, it initializes its Q-values to 0.
        - Ensures the QTable has an entry for any new state encountered during training.
        """
        for episode in range(self.N):
            if (episode + 1) % int(1e6) == 0:
                print(f'Episode: {episode + 1}')

            state = [0] * 9  # starting state = empty 3x3 board
            current_player = 1 if random.random() < 0.5 else 2

            # If Player 2 starts, make a random move
            if current_player == 2:
                actions = self.possible_actions(state)
                random_action = random.choice(actions)
                state = self.update_state(state, random_action, 2)
                current_player = 1  # Switch to Player 1

            while True:
                # Qtable uses string representations of states as keys for easy lookup
                state_str = str(state)

                # Add current state to Qtable if not there yet
                if state_str not in self.Qtable:
                    self.Qtable[state_str] = [0] * 9

                actions = self.possible_actions(state)
                if not actions: # No more actions possible
                    break

                # Take action: Exploration vs Exploitation with epsilon-greedy
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(actions)
                else:
                    action = max(actions, key=lambda x: self.Qtable[state_str][x])

                # Observe result (new state and reward)
                new_state, reward = self.next_state_and_reward(state, action)

                new_state_str = str(new_state)
                if new_state_str not in self.Qtable:
                    self.Qtable[new_state_str] = [0] * 9

                # Update Q-Value using Bellman Equation
                self.Qtable[state_str][action] += self.alpha * (
                            reward + self.gamma * max(self.Qtable[new_state_str]) - self.Qtable[state_str][action])

                # Move to the next state
                state = new_state

                if reward != 0:  # Game ended
                    break

            # Decay epsilon
            epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

        # Save Qtable as a pickle file
        with open('TTT_Qtable.p', 'wb') as file:
            pickle.dump(self.Qtable, file)

        
    # ------------ ------------ ------------ ------------ ---
    # Helper functions for playing an Agent vs Computer game
    # ------------ ------------ ------------ ------------ ---
    def agent_vs_computer(self, use_qtable=True):
        """
        The agent will be Player 1; Player 2 is the opponent
        """
        state = [0] * 9
        game_log = []

        # Determine who starts
        current_player = 1 if random.random() < 0.5 else 2

        while True:
            game_log.append(state[:])  # Log current state
            state_str = str(state)
            actions = self.possible_actions(state)

            if not actions:
                return 0, game_log  # Draw

            # Agent's turn
            if current_player == 1 and use_qtable:
                if state_str in self.Qtable:
                    action = max(actions, key=lambda x: self.Qtable[state_str][x])
                else:
                    action = random.choice(actions)  # Random if state not in Qtable

            # Opponent's turn
            else:
                action = random.choice(actions)

            state = self.update_state(state, action, current_player)
            if self.check_winner(state, current_player):
                return current_player, game_log

            # Switch player
            current_player = 1 if current_player == 2 else 2

        
    def evaluate(self, n_games = 100):

        with open(f'{directory}/TTT_Qtable.p', 'rb') as file:
            self.Qtable = pickle.load(file)

        wins = 0
        draws = 0
        losses = 0
        for _ in range(n_games):
            result, game_log = self.agent_vs_computer(use_qtable=True)

            # The agent (Player 1) wins
            if result == 1:
                wins += 1

            # Draw
            elif result == 0:
                draws += 11

            # The opponent (Player 2) wins
            else:
                losses += 1

        print("Results over 100 games:")
        print(f"The agent won {wins}")
        print(f"The agent lost {losses}")
        print(f"Draws: {draws}")
        print(f"Winning percentage: {(wins / n_games) * 100}%")


    # ------------ ------------ ------------ ------------ 
    # Helper functions for playing an Agent vs Human game
    # ------------ ------------ ------------ ------------ 
    def transform_state(self, state):
        """
        Takes in current state of the board and swaps the roles of Player 1 (X) and Player 2 (O)
        """
        return [2 if x == 1 else 1 if x == 2 else 0 for x in state]


    def print_board(self, state):
        """
        Prints the current state of the board to the console in a user-friendly format.
        chars dictionary maps the board values (0, 1, 2) to the characters ' ', 'X', 'O'.
        Top Row: state[0], state[1], state[2]
        Mid Row: state[3], state[4], state[5]
        Bot Row: state[6], state[7], state[8]
        """
        chars = {0: ' ', 1: 'X', 2: 'O'}
        for i in range(3):
            print(f"{chars[state[3 * i]]} | {chars[state[3 * i + 1]]} | {chars[state[3 * i + 2]]}")

            # Separator rows after each row (except last one)
            if i < 2:
                print("---------")


    def agent_move(self, state):
        """
        Determines agent's move based on the current board state. Move chosen using a Q-table if available or randomly if the state is unknown to the agent.
        """
        transformed_state = self.transform_state(state)
        state_str = str(transformed_state)

        if state_str not in self.Qtable:
            return random.choice(self.possible_actions(state))

        return max(self.possible_actions(state), key=lambda x: self.Qtable[state_str][x])


    def agent_vs_human(self):
        """
        Player 1 is the human, Player 2 is the agent
        """
        state = [0] * 9
        current_player = 1 if random.random() < 0.5 else 2

        print("Board positions are indexed as follows:\n")
        print("0 | 1 | 2")
        print("---------")
        print("3 | 4 | 5")
        print("---------")
        print("6 | 7 | 8")
    
        print("\nWhen prompted to make a move, enter the number corresponding to the position you want to play. For example, entering '0' will place your move in the top-left corner.\n")

        while True:
            self.print_board(state)

            if current_player == 1:
                actions = self.possible_actions(state)
                move = int(input(f"\nEnter your move (0-8): "))

                while move not in actions:
                    print("\nInvalid move, try again.")
                    move = int(input(f"Enter your move (0-8): "))
                state = self.update_state(state, move, 1)

            else:
                print("\nAgent's turn:")
                move = self.agent_move(state)
                state = self.update_state(state, move, 2)

            if self.check_winner(state, current_player):
                self.print_board(state)
                if current_player == 1:
                    print("\nYou win!")
                else:
                    print("\nAgent wins!")
                break
            elif 0 not in state:
                self.print_board(state)
                print("\nIt's a draw!")
                break

            current_player = 1 if current_player == 2 else 2


if __name__ == '__main__':

    # Initialize the Q-Learning agent
    agent = TTTQLearning()

    # Check if a Q-table has already been saved
    try:
        with open(f'{directory}/TTT_Qtable.p', 'rb') as file:
            Qtable = pickle.load(file)
    except FileNotFoundError:
        print('No Q-table found. Training a new agent...')
        agent.train()
        print('Training complete. Q-table saved as TTT_Qtable.p')

    print("Evaluating the agent's performance")
    agent.evaluate()

    print("Game Time!")
    agent.agent_vs_human()