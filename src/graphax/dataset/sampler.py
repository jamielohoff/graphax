from typing import Sequence
import chex
from ..core import GraphInfo, make_graph_info


class ComputationalGraphSampler:
    """
    TODO add documentation
    """
    max_info: GraphInfo
    min_num_intermediates: int
    
    def __init__(self, 
                min_num_intermediates: int = 12,
                max_info: GraphInfo = make_graph_info([10, 30, 5])) -> None:
        """initializes a fixed repository of possible vertex games

        Args:
            num_games (int): _description_
            info (chex.Array): _description_
            key (chex.PRNGKey, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.max_info = max_info
        self.min_num_intermediates = min_num_intermediates
            
    def sample(self, 
                num_samples: int = 1, 
                key: chex.PRNGKey = None,
                **kwargs) -> Sequence[tuple[str, chex.Array, GraphInfo]]:
        """Samples from the repository of possible games

        Args:
            x (_type_): _description_

        Returns:
            Any: _description_
        """
        pass
    
