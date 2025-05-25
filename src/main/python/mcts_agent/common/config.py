import typing
import mcts.config

class ConfigSchema(typing.TypedDict):
    mcts_config: mcts.config.MCTSConfig