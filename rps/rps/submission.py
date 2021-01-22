import random


class Agent:
    __agent = None

    def __init__(self, configuration):
        self.configuration = configuration
        self.my_history = []
        self.opponents_history = []

    def run(self, observation):
        if observation.step > 0:
            self.opponents_history.append(observation.lastOpponentAction)
        return self.execute(observation)

    def execute(self, observation):
        pass


class RandomAgent(Agent):
    def __init__(self, configuration):
        super().__init__(configuration)

    def execute(self, observation):
        if observation.step > 0:
            return random.randint(0, self.configuration.signs - 1)
        else:
            return 0


class AgentFactory:
    __agent_types = ['Random']
    __agent = None

    @staticmethod
    def get_agent(agent_type, configuration):
        if agent_type not in AgentFactory.__agent_types:
            raise Exception("Unknown agent type")
        if agent_type == 'Random':
            if AgentFactory.__agent is None:
                AgentFactory.__agent = RandomAgent(configuration)
            return AgentFactory.__agent


def my_agent(observation, configuration):
    return AgentFactory.get_agent('Random', configuration).run(observation)
