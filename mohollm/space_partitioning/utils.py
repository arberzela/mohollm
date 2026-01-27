import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class Region:
    """
    A class used to represent a Region in space partitioning.

    Attributes
    ----------
    volume : float
        The unnormalized volume of the region.
    normalized_volume : float
        The normalized volume of the region.
    center: Dict
        The center configuration the region got build around.
    center_fval: Dict
        The function value at the center configuration.
    boundaries : List[Dict[str, List]]
        A list of dictionaries where each dictionary represents the boundaries of the region.
        The keys of the dictionary are the names of the dimensions (e.g., 'x', 'y', 'z'), and
        the values are tuples representing the minimum and maximum values of the boundaries
        in that dimension.
    points : List[Dict]
        A list of dictionaries where each dictionary represents a point in the region.
    points_fvals : List[Dict]
        A list of dictionaries where each dictionary represents the function values
        at the points in the region.
    points_indices : List[int]
        A list of indices corresponding to the location in the original statistics.observed_fvals
    range_parameter_keys : List[str]
        A list of keys representing the range parameters for the region.
    """

    volume: float
    normalized_volume: float
    center: Dict
    center_fval: Dict
    boundaries: Dict[str, List]
    points: List[Dict]
    points_fvals: List[Dict]
    points_indices: List[int]
    range_parameter_keys: List[str]
    integer_parameter_keys: List[str]
    float_parameter_keys: List[str]

    def __str__(self) -> str:
        """
        Returns a string representation of the region boundaries.
        Returns
        -------
        str
            A string representing the region's boundaries.
        """
        boundaries_str_lines = ["{"]
        for dim, val in self.boundaries.items():
            if dim in self.range_parameter_keys:
                if dim in self.integer_parameter_keys:
                    boundaries_str_lines.append(
                        f"  {dim}: range(int({[int(v) for v in val]})),"
                    )
                elif dim in self.float_parameter_keys:
                    #boundaries_str_lines.append(f"  {dim}: range(float([{math.ceil(val[0], 3)}, {math.floor(val[1], 3)}])),")
                    formatted_values = [float(f"{v:.3f}") for v in val] 
                    boundaries_str_lines.append(f"  {dim}: range(float({formatted_values})),")
            else:
                boundaries_str_lines.append(f"  {dim}: choice({val}),")

                # # Create a list of values for the 'choice' parameter
                # choice_values = []
                # for v in val:
                #     print("is float: ", v, isinstance(v, (float, int)))
                #     print("is bool: ", v, isinstance(v, bool))

                #     if isinstance(v, (float, int)) and not isinstance(v, bool):
                #         choice_values.append(v)
                #     elif isinstance(v, bool):
                #         choice_values.append(str(v).lower())
                #     else:
                #         choice_values.append(f"{str(v)}")
                # print(choice_values)
                # boundaries_str_lines.append(f"  {dim}: choice({choice_values}),")

        boundaries_str_lines[-1] = boundaries_str_lines[-1].rstrip(",")
        boundaries_str_lines.append("}")
        boundaries_str = "\n".join(boundaries_str_lines)
        return boundaries_str

    def to_dict(self) -> Dict:
        """
        Converts the Region instance into a dictionary.

        Returns
        -------
        Dict
            A dictionary representation of the Region instance.
        """
        return {
            "volume": self.volume,
            "normalized_volume": self.normalized_volume,
            "center": self.center,
            "center_fval": self.center_fval,
            "boundaries": self.boundaries,
            "points": self.points,
            "points_fvals": self.points_fvals,
            "points_indices": self.points_indices,
            "range_parameter_keys": self.range_parameter_keys,
            "integer_parameter_keys": self.integer_parameter_keys,
            "float_parameter_keys": self.float_parameter_keys,
        }


@dataclass
class BoundingBox:
    """
    A class used to represent a Bounding Box in space partitioning.

    Attributes
    ----------
    volume : float
        The unnormalized volume of the bounding box.
    boundaries : List[Dict[str, Tuple[float, float]]]
        A list of dictionaries where each dictionary represents the boundaries
        of the bounding box in a specific dimension. The keys are the dimension
        names and the values are tuples representing the minimum and maximum
        values in that dimension.
    range_parameter_keys : List[str]
        A list of keys representing the range parameters for the bounding box.
    """

    volume: float
    boundaries: Dict[str, Tuple[float, float]]
    range_parameter_keys: List[str]

    def get_dimension_names(self) -> List[str]:
        """
        Retrieves the dimension names in the order they appear in the bounding
        box boundaries.

        Returns
        -------
        List[str]
            A list of dimension names in the order they are first encountered.
        """
        return list(self.boundaries.keys())

    def calculate_volume(self) -> float:
        """
        Calculates the volume of the bounding box based on its boundaries.
        We assume that the bounding box has the shape of an n-dimensional
        hyperrectangle. The volume is just the product of each side length
        in each dimension.

        Returns
        -------
        float
            The calculated volume of the bounding box.
        """
        _volume = 1.0
        for dim, bound in self.boundaries.items():
            # For range parameters, use the length of the range
            if dim in self.range_parameter_keys:
                min_val = bound[0]
                max_val = bound[1]
                _volume *= max_val - min_val
            elif isinstance(bound, list):
                # For categorical dimensions, use the number of choices
                _volume *= len(bound)
        self.volume = _volume
