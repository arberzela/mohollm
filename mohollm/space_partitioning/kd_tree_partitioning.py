import logging
import numpy as np

from mohollm.space_partitioning.space_partitioning_strategy import (
    SPACE_PARTITIONING_STRATEGY,
)
from mohollm.space_partitioning.utils import Region, BoundingBox
from typing import List, Dict, Tuple
from scipy.spatial import KDTree


logger = logging.getLogger("KDTreePartitioning")


class KDNode:
    """
    A node in the KD-tree that represents a region in the space.

    Attributes:
        split_dim: The dimension used for splitting this node
        split_value: The value at which the dimension is split
        left: The left child node (points with value <= split_value in split_dim)
        right: The right child node (points with value > split_value in split_dim)
        points: List of points in this node (only for leaf nodes)
        fvals: List of function values for points in this node (only for leaf nodes)
        boundaries: Dictionary of dimension bounds for this node's region
    """

    def __init__(self, split_dim=None, split_value=None):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = None
        self.right = None
        self.points = []
        self.fvals = []
        self.boundaries = {}

        # Keeps track of which features are a range and not a choice
        self.range_parameter_keys = []
        self.integer_parameter_keys = []
        self.float_parameter_keys = []


class KDTreePartitioning(SPACE_PARTITIONING_STRATEGY):
    """
    A class that performs KD-tree partitioning of a space.

    The KD-tree recursively partitions the space by splitting along dimensions.
    Each leaf node of the tree represents a region containing points.
    """

    def __init__(self):
        self.bounding_box = None
        self.root = None
        self.leaf_nodes = []
        self._iter = 0

    def adaptive_leafsize(self, t, d):
        """
        Calculate adaptive leaf size for KD-tree based on iteration.
        m0: Initial leaf size * d
        lam: Growth rate parameter

        Args:
            t: Current iteration
            d: Dimension of the search space

        Returns:
            Leaf size
        """
        adaptive_leaf_settings = self.space_partitioning_settings.get(
            "adaptive_leaf_settings", {}
        )
        if adaptive_leaf_settings.get("use_dimension_scaling", True):
            m0 = adaptive_leaf_settings.get("m0", None) * d
        else:
            m0 = adaptive_leaf_settings.get("m0", None)
        lam = adaptive_leaf_settings.get("lam", None)
        return int(m0) + int(np.ceil(lam * np.log1p(t)))

    def prefilter_points(
        self, points: List[Dict], bounding_box=None
    ) -> Tuple[np.ndarray, List[str], Dict[str, List], Dict[str, List[float]]]:
        """
        Convert data points to numerical values for KD-tree processing.
        For numerical values, use the actual values.
        For categorical values, map to numerical indices.
        Also determines the numerical min/max bounds for each dimension.

        Args:
            points: List of point dictionaries
            bounding_box: Optional bounding box containing the choices for each dimension

        Returns:
            Tuple containing:
            - numpy array with numerical values for points
            - list of dimension names
            - dictionary mapping dimension names to their original choice lists
            - dictionary mapping dimension names to their [min, max] numerical bounds
        """
        if not points:
            return np.array([]), [], {}, {}

        # Get all dimension names from the first point
        all_dimension_names = list(points[0].keys())
        dimension_choices = {}
        is_categorical = {}  # Track which dimensions are categorical

        # Get the choices for each dimension
        if bounding_box and hasattr(bounding_box, "boundaries"):
            for dim in all_dimension_names:
                if dim in bounding_box.boundaries:
                    dimension_choices[dim] = bounding_box.boundaries[dim]
                else:
                    # If dimension not in bounding_box, collect unique values from points
                    unique_values = sorted(
                        list(set(point[dim] for point in points if dim in point))
                    )
                    dimension_choices[dim] = unique_values
        else:
            # If no bounding_box, collect unique values for each dimension from points
            for dim in all_dimension_names:
                unique_values = sorted(
                    list(set(point[dim] for point in points if dim in point))
                )
                dimension_choices[dim] = unique_values

        # Determine which dimensions are categorical vs. numerical
        for dim in all_dimension_names:
            choices = dimension_choices.get(dim, [])
            if not choices:  # If a dimension ended up with no choices (e.g. missing in all points and no bbox)
                is_categorical[dim] = (
                    True  # Treat as categorical by default, though it's an edge case
                )
                logger.warning(
                    f"Dimension {dim} has no defined choices. Treating as categorical."
                )
                continue
            try:
                # Attempt to interpret all choices as numeric
                all_numerical = all(
                    isinstance(val, (int, float))
                    or (
                        isinstance(val, str)
                        and val.replace(".", "", 1).lstrip("-+").isdigit()
                    )  # handles negative and float strings
                    for val in choices
                )
                is_categorical[dim] = not all_numerical
            except (
                ValueError,
                TypeError,
            ):  # Should not be reached if previous check is robust
                is_categorical[dim] = True

        # Determine numerical bounds for KD-tree
        numerical_bounds_dict = {}
        for dim in all_dimension_names:
            choices = dimension_choices.get(dim, [])
            if not choices:
                logger.warning(
                    f"Dimension {dim} has no choices. Assigning bounds [0.0, 0.0]."
                )
                numerical_bounds_dict[dim] = [0.0, 0.0]
                continue

            if is_categorical[dim]:
                # For categorical, numerical values are indices
                min_val = 0.0
                max_val = float(len(choices) - 1) if choices else 0.0
            else:
                # For numerical, numerical values are the actual float values
                try:
                    numerical_values = [float(c) for c in choices]
                    min_val = min(numerical_values)
                    max_val = max(numerical_values)
                except (ValueError, TypeError):
                    # Fallback if conversion fails unexpectedly (should be caught by is_categorical)
                    logger.warning(
                        f"Conversion failed for numerical dim {dim} choices during bounds calculation. Treating as categorical for bounds."
                    )
                    min_val = 0.0
                    max_val = float(len(choices) - 1)
            numerical_bounds_dict[dim] = [min_val, max_val]

        # Convert data points to numerical values for KD-tree
        numerical_data = []
        for point in points:
            point_numerical = []
            for dim in all_dimension_names:
                choices_for_dim = dimension_choices.get(dim, [])
                if (
                    not choices_for_dim
                ):  # Should a point have a dim not in dimension_choices
                    point_numerical.append(0.0)  # Default value
                    logger.warning(
                        f"Dimension {dim} not found in dimension_choices for point {point}. Using 0.0."
                    )
                    continue

                if is_categorical[dim]:
                    # Categorical: convert to index in choices list
                    try:
                        val_to_find = point[dim]
                        if val_to_find in choices_for_dim:
                            idx = choices_for_dim.index(val_to_find)
                        else:
                            # Try converting to same type as first element in choices if direct match fails
                            # This handles cases like int vs str representation if choices are e.g. [1,2,3] and point[dim] is "1"
                            try:
                                converted_val = type(choices_for_dim[0])(val_to_find)
                                if converted_val in choices_for_dim:
                                    idx = choices_for_dim.index(converted_val)
                                else:  # Fallback if still not found
                                    logger.warning(
                                        f"Value {val_to_find} (or typed {converted_val}) not found in choices for {dim}. Using index 0."
                                    )
                                    idx = 0
                            except (ValueError, TypeError, IndexError):
                                logger.warning(
                                    f"Value {val_to_find} not found in choices for {dim} and type conversion failed. Using index 0."
                                )
                                idx = 0
                        point_numerical.append(float(idx))
                    except (
                        ValueError,
                        TypeError,
                        KeyError,
                    ):  # Added KeyError for point[dim]
                        logger.warning(
                            f"Value {point.get(dim)} not found or error processing for categorical dim {dim}. Using index 0."
                        )
                        point_numerical.append(0.0)
                else:
                    # Numerical: use actual value as float
                    try:
                        point_numerical.append(float(point[dim]))
                    except (ValueError, TypeError, KeyError):
                        logger.warning(
                            f"Could not convert {point.get(dim)} to float for numerical dim {dim}. Attempting index fallback."
                        )
                        # Fall back to index if direct float conversion fails for a supposedly numerical dim
                        try:
                            if point[dim] in choices_for_dim:
                                idx = choices_for_dim.index(point[dim])
                                point_numerical.append(float(idx))
                            else:  # If value not in choices, use 0.0
                                point_numerical.append(0.0)
                        except (ValueError, TypeError):
                            point_numerical.append(0.0)  # Final fallback

            numerical_data.append(point_numerical)

        return (
            np.array(numerical_data, dtype=np.float64),
            all_dimension_names,
            dimension_choices,
            numerical_bounds_dict,
        )

    def partition(
        self,
        points: List[Dict],
        fvals: List[Dict],
        max_depth: int = 15,
        min_points: int = 1,
    ) -> List[Region]:
        """
        Partition the space using a KD-tree based on indexed choices.

        Args:
            points: List of point dictionaries
            fvals: List of function value dictionaries
            max_depth: Maximum depth of the KD-tree
            min_points: Minimum number of points in a leaf node

        Returns:
            List of Region objects representing the partitioned space
        """
        logger.debug("Starting KD-tree partitioning")

        if not points:
            logger.warning("Input points list is empty. Cannot perform partitioning.")
            return []

        self._iter = len(self.statistics.observed_configs)
        # Convert points to indices based on choices
        points_array, dimension_names, dimension_choices, converted_dimension_ranges = (
            self.prefilter_points(points, self.bounding_box)
        )

        dimensions = len(dimension_names)
        logger.debug(f"Using {dimensions} dimensions for KD-tree")

        if self.bounding_box is None:
            raise ValueError(
                "Bounding box must be set before partitioning. Please initialize the bounding box. Add an entry to the config file with the 'parameter_constraints' key."
            )

        # Build the KD-tree using index-based data
        logger.debug(
            f"Building KD-tree with indexed points array shape: {points_array.shape}"
        )
        num_leafs = self.adaptive_leafsize(self._iter, dimensions)
        logger.debug(f"Adaptive leaf size: {num_leafs}")
        self.kdtree = KDTree(points_array, leafsize=num_leafs, balanced_tree=False)

        kdtree_input_bounds = list(converted_dimension_ranges.values())
        leaf_cells = self.get_kd_leaf_cells(self.kdtree, kdtree_input_bounds)

        logger.debug("KD-tree built successfully")

        # Convert leaf cells to Region objects, mapping indices back to original values
        regions = self._convert_to_regions_from_cells(
            leaf_cells, points, fvals, dimension_names, dimension_choices
        )

        logger.debug(f"Created {len(regions)} regions from KD-tree")
        return regions

    def get_kd_leaf_cells(self, kdtree, bounds):
        """
        Extract exact hyperrectangles (leaf cells) from a KD-tree.

        Args:
            kdtree: scipy.spatial.KDTree object
            bounds: Bounds for numerical dimensions [[min_x1, max_x1], ..., [min_xd, max_xd]]

        Returns:
            List of tuples (cell_bounds, cell_indices) where:
            - cell_bounds is a tuple (mins, maxs) defining the cell boundaries
            - cell_indices is a list of point indices belonging to this cell
        """

        def recursive_extract_cells(node, cell_bounds, depth=0):
            """
            Recursively traverse the KD-tree and extract leaf cell boundaries.

            Args:
                node: Current node (can be a leafnode or innernode)
                cell_bounds: Current cell boundaries [[min_x1, max_x1], ..., [min_xd, max_xd]]
                depth: Current depth in the tree (used to determine splitting dimension)

            Returns:
                List of (cell_bounds, cell_indices) for all leaf nodes under this node
            """

            # Check if this is a leaf node
            if hasattr(node, "idx"):  # Leaf node
                # Convert cell_bounds to min/max format
                mins = np.array([b[0] for b in cell_bounds])
                maxs = np.array([b[1] for b in cell_bounds])

                # Return the leaf cell and its point indices
                return [((mins, maxs), node.idx)]

            # Otherwise, it's an inner node
            split_dim = node.split_dim
            split_val = node.split

            # Create bounds for left and right children
            left_bounds = [b.copy() for b in cell_bounds]
            right_bounds = [b.copy() for b in cell_bounds]

            left_bounds[split_dim][1] = (
                split_val  # Left child: upper bound becomes split value
            )
            right_bounds[split_dim][0] = (
                split_val  # Right child: lower bound becomes split value
            )

            # Recursively process left and right children
            cells = []
            if node.less is not None:
                cells.extend(recursive_extract_cells(node.less, left_bounds, depth + 1))
            if node.greater is not None:
                cells.extend(
                    recursive_extract_cells(node.greater, right_bounds, depth + 1)
                )

            return cells

        # Initialize cell bounds to cover the entire search space
        initial_bounds = [[bounds[i][0], bounds[i][1]] for i in range(len(bounds))]

        # Extract all leaf cells from the tree
        leaf_cells = recursive_extract_cells(kdtree.tree, initial_bounds)

        return leaf_cells

    def _convert_to_regions_from_cells(
        self,
        leaf_cells,
        original_points,
        original_fvals,
        dimension_names,
        dimension_choices,
    ) -> List[Region]:
        """
        Convert KD-tree leaf cells to Region objects, mapping values back to original choices.

        Args:
            leaf_cells: List of (cell_bounds, cell_indices) tuples
            original_points: Original list of point dictionaries
            original_fvals: Original list of function value dictionaries
            dimension_names: List of dimension names
            dimension_choices: Dictionary mapping dimension names to their choice lists

        Returns:
            List of Region objects
        """
        regions = []

        if not self.bounding_box:
            logger.error("Bounding box not initialized correctly")
            return []

        for (mins, maxs), point_indices in leaf_cells:
            # Create boundaries dictionary, mapping numerical bounds to original choices
            boundaries = {}
            for i, dim_name in enumerate(dimension_names):
                choices = dimension_choices[dim_name]
                if dim_name in self.range_parameter_keys:
                    boundaries[dim_name] = [mins[i], maxs[i]]
                    continue

                # Check if values in this dimension are numerical or categorical
                if all(
                    isinstance(c, (int, float))
                    or (isinstance(c, str) and c.replace(".", "", 1).isdigit())
                    for c in choices
                ):
                    # For numerical dimensions, try to use the actual values
                    try:
                        # Convert choices to float for comparison
                        numerical_choices = [float(c) for c in choices]
                        # Find all choices that fall within the cell bounds
                        selected_choices = [
                            choices[j]
                            for j, val in enumerate(numerical_choices)
                            if mins[i] <= val <= maxs[i]
                        ]

                        # If no values found within bounds, take the closest ones
                        if not selected_choices:
                            lower_idx = min(
                                range(len(numerical_choices)),
                                key=lambda i: abs(numerical_choices[i] - mins[i]),
                            )
                            upper_idx = min(
                                range(len(numerical_choices)),
                                key=lambda i: abs(numerical_choices[i] - maxs[i]),
                            )
                            selected_choices = choices[lower_idx : upper_idx + 1]
                    except (ValueError, TypeError):
                        # Fall back to index-based approach if numerical conversion fails
                        min_idx = max(0, int(round(mins[i])))
                        max_idx = min(len(choices) - 1, int(round(maxs[i])))
                        selected_choices = choices[min_idx : max_idx + 1]
                else:
                    # For categorical dimensions, use index-based approach
                    min_idx = max(0, int(round(mins[i])))
                    max_idx = min(len(choices) - 1, int(round(maxs[i])))
                    selected_choices = choices[min_idx : max_idx + 1]

                # Store the selected choices
                if len(selected_choices) == 1:
                    boundaries[dim_name] = selected_choices
                else:
                    boundaries[dim_name] = selected_choices

            # Get the points and function values for this cell
            cell_points = [original_points[i] for i in point_indices]
            cell_fvals = [original_fvals[i] for i in point_indices]
            cell_indices = point_indices

            # For volume calculation (approximate based on number of choices)
            volume = 1.0
            for dim_name in dimension_names:
                if dim_name in self.range_parameter_keys:
                    # For range parameters, use the length of the range
                    min_val = boundaries[dim_name][0]
                    max_val = boundaries[dim_name][1]
                    volume *= max_val - min_val
                elif isinstance(boundaries[dim_name], list):
                    # For categorical dimensions, use the number of choices
                    volume *= len(boundaries[dim_name])
                else:
                    volume *= 1  # Single choice

            normalized_volume = volume / self.bounding_box.volume

            # Determine the center of the region using the original points if available
            if cell_points:
                center = cell_points[0]
                center_fval = cell_fvals[0]
            else:
                center = {}
                for dim_name in dimension_names:
                    choices = boundaries[dim_name]
                    if isinstance(choices, list):
                        center[dim_name] = choices[len(choices) // 2]
                    else:
                        center[dim_name] = choices
                center_fval = {}
            regions.append(
                Region(
                    volume=volume,
                    normalized_volume=normalized_volume,
                    center=center,
                    center_fval=center_fval,
                    boundaries=boundaries,
                    points=cell_points,
                    points_fvals=cell_fvals,
                    points_indices=cell_indices,
                    range_parameter_keys=self.range_parameter_keys,
                    integer_parameter_keys=self.integer_parameter_keys,
                    float_parameter_keys=self.float_parameter_keys,
                )
            )
            logger.debug(f"Region boundaries: {boundaries}")

        return regions
