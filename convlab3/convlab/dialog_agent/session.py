"""Dialog controller classes."""
from abc import ABC, abstractmethod
import random
from convlab.dialog_agent.agent import Agent


class Session(ABC):

    """Base dialog session controller, which manages the agents to conduct a complete dialog session.
    """

    @abstractmethod
    def next_agent(self):
        """Decide the next agent to generate a response.

        In this base class, this function returns the index randomly.

        Returns:
            next_agent (Agent): The index of the next agent.
        """
        pass

    @abstractmethod
    def next_response(self, observation):
        """Generated the next response.

        Args:
            observation (str or dict): The agent observation of next agent.
        Returns:
            response (str or dict): The agent's response.
        """
        pass

    @abstractmethod
    def init_session(self):
        """Init the agent variables for a new session."""
        pass


class BiSession(Session):
    """The dialog controller which aggregates several agents to conduct a complete dialog session.

    Attributes:
        sys_agent (Agent):
            system dialog agent.

        user_agent (Agent):
            user dialog agent.

        kb_query (KBquery):
            knowledge base query tool.

        dialog_history (list):
            The dialog history, formatted as [[user_uttr1, sys_uttr1], [user_uttr2, sys_uttr2], ...]
    """

    def __init__(self, sys_agent: Agent, user_agent: Agent, kb_query=None, evaluator=None):
        """
        Args:
            sys_agent (Agent):
                An instance of system agent.

            user_agent (Agent):
                An instance of user agent.

            kb_query (KBquery):
                An instance of database query tool.

            evaluator (Evaluator):
                An instance of evaluator.
        """
        self.sys_agent = sys_agent
        self.user_agent = user_agent
        self.kb_query = kb_query
        self.evaluator = evaluator

        self.dialog_history = []
        self.__turn_indicator = 0

        self.init_session()

    def next_agent(self):
        """The user and system agent response in turn."""
        if self.__turn_indicator % 2 == 0:
            next_agent = self.user_agent
            agent = "user"
        else:
            next_agent = self.sys_agent
            agent = "sys"
        self.__turn_indicator += 1
        # print(agent + " " + str(self.__turn_indicator))
        return next_agent

    def next_response(self, observation):
        next_agent = self.next_agent()
        response = next_agent.response(observation)
        # print(response)
        return response

    def next_turn(self, last_observation):
        """Conduct a new turn of dialog, which consists of the system response and user response.

        The variable type of responses can be either 1) str or 2) dialog act, depends on the dialog mode settings of the
        two agents which are supposed to be the same.

        Args:
            last_observation:
                Last agent response.
        Returns:
            sys_response:
                The response of system.

            user_response:
                The response of user simulator.

            session_over (boolean):
                True if session ends, else session continues.

            reward (float):
                The reward given by the user.
        """
        user_response = self.next_response(last_observation)
        if self.evaluator:
            self.evaluator.add_sys_da(self.user_agent.get_in_da_eval(), self.sys_agent.dst.state['belief_state'])
            self.evaluator.add_usr_da(self.user_agent.get_out_da())

        session_over = self.user_agent.is_terminated()
        if hasattr(self.sys_agent, 'dst'):
            self.sys_agent.dst.state['terminated'] = session_over
        # if session_over and self.evaluator:
            # prec, rec, f1 = self.evaluator.inform_F1()
            # print('inform prec. {} rec. {} F1 {}'.format(prec, rec, f1))
            # print('book rate {}'.format(self.evaluator.book_rate()))
            # print('task success {}'.format(self.evaluator.task_success()))
        reward = self.user_agent.get_reward() if self.evaluator is None else self.evaluator.get_reward(session_over)
        sys_response = self.next_response(user_response)
        self.dialog_history.append([self.user_agent.name, user_response])
        self.dialog_history.append([self.sys_agent.name, sys_response])
        return sys_response, user_response, session_over, reward

    def train_policy(self):
        """
        Train the parameters of system agent policy.
        """
        self.sys_agent.policy.train()

    def init_session(self, **kwargs):
        self.sys_agent.init_session()
        self.user_agent.init_session(**kwargs)
        if self.evaluator:
            self.evaluator.add_goal(self.user_agent.policy.get_goal())
        self.dialog_history = []
        self.__turn_indicator = 0

