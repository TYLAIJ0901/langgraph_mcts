import typing
import common.base_types

# Define BaseAction for convenience, referring to the adjusted import
BaseAction = common.base_types.BaseAction

class GraphState(typing.TypedDict):
    game_state: common.base_types.BaseState
    action_records: list[BaseAction]