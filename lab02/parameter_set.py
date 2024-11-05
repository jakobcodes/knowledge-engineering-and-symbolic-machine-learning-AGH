from dataclasses import dataclass

@dataclass
class ParameterSet:
    """Represents a set of parameters for Q-learning experiment."""
    learning_rate: float
    discount_factor: float
    epsilon_start: float
    epsilon_decay: float
    epsilon_min: float
    n_bins: int
    name: str
    
    def to_dict(self) -> dict:
        return {
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_start': self.epsilon_start,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'n_bins': self.n_bins,
            'name': self.name
        }