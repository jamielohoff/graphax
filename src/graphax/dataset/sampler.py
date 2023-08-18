from typing import Sequence
from chex import Array, PRNGKey


class ComputationalGraphSampler:
    """
    TODO add documentation
    """
    storage_shape: Sequence[int]
    min_num_intermediates: int
    
    def __init__(self, 
                min_num_intermediates: int = 50,
                storage_shape: Sequence[int] = [20, 105, 20]) -> None:
        """initializes a fixed repository of possible vertex games

        Args:
            num_games (int): _description_
            info (chex.Array): _description_
            key (chex.PRNGKey, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.storage_shape = storage_shape
        self.min_num_intermediates = min_num_intermediates
            
    def sample(self, 
                num_samples: int = 1, 
                key: PRNGKey = None,
                **kwargs) -> Sequence[tuple[str, Array]]:
        """Samples from the repository of possible games

        Args:
            x (_type_): _description_

        Returns:
            Any: _description_
        """
        pass
    
