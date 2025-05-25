import dataclasses

@dataclasses.dataclass
class MCTSConfig:
    exploration_constant: float = 1.414
    num_simulations: int = 1000
    uct_C: float = 1.414 # C_p in the UCT formula UCT = Q(s,a) + C_p * sqrt(ln(N(s))/N(s,a))
