import abc
import typing

BaseAction = typing.TypeVar("BaseAction")

class BaseState(typing.Generic[BaseAction], metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def reset(self, *args, **kwargs):
        """
        Reset this state into a new initial state
        """
        pass

    @abc.abstractmethod
    def get_next_state(self, action: BaseAction) -> tuple["BaseState", float]:
        """
        Based on the given action, transform this state into a next state and outputs the reward corresponding to the aaction.
        """
        pass

    @abc.abstractmethod
    def evluate(self) -> float:
        """
        Evaluate this state
        """
        pass

    @abc.abstractmethod
    @property
    def is_terminated(self) -> bool:
        """
        Indicate whether the state is terminated
        """
        pass

