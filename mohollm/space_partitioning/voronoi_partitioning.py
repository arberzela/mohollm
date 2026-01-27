import numpy as np
import logging

from mohollm.space_partitioning.space_partitioning_strategy import (
    SPACE_PARTITIONING_STRATEGY,
)
from mohollm.space_partitioning.utils import Region
from typing import List, Dict

logger = logging.getLogger("VoronoiPartitioning")


class VoronoiPartitioning(SPACE_PARTITIONING_STRATEGY):
    def partition(self, points: List[Dict], fvals: List[Dict]) -> List[Region]:
        logger.debug("Starting Voronoi partitioning")
        boundaries = self._compute_region_line_boundaries(points)
        logger.debug("Computed region line boundaries")
        volumes = self.calculate_voronoi_cell_volumes(points)
        logger.debug("Calculated Voronoi cell volumes")
        converted_names = self._convert_region_line_boundaries(
            points, fvals, boundaries, volumes
        )
        logger.debug("Converted region line boundaries to Region objects")
        return converted_names

    def _compute_region_line_boundaries(self, candidates, tol=1e-7):
        """
        For each candidate point in n dimensions, compute the two boundary points
        along each coordinate axis that passes through the candidate.
        """
        logger.debug("Computing region line boundaries")
        tmp = []
        for candidate in candidates:
            tmp.append(list(candidate.values()))

        candidates = np.asarray(tmp)
        N, n_dims = candidates.shape

        if self.bounding_box is None:
            # Create a default bounding box based on the range of the data
            mins = np.min(candidates, axis=0) - 1.0
            maxs = np.max(candidates, axis=0) + 1.0
            # TODO: If no ranges/boundary is provided we use the points in the convex hull
            # TODO: and create a bounding box instance from this. Maybe in the constructor?
            self.bounding_box = [(mins[i], maxs[i]) for i in range(n_dims)]

        all_points = candidates

        all_boundaries = []

        dimension_names = self.bounding_box.get_dimension_names()

        for idx, p in enumerate(candidates):
            boundaries_for_p = np.empty((n_dims, 2, n_dims))

            for dim, dim_name in enumerate(dimension_names):
                # For each dimension, find where the Voronoi cell boundaries intersect
                # the line through p parallel to this axis

                # Initialize the bounds to the bounding box
                lower_bound = self.bounding_box.boundaries[dim_name][0]
                upper_bound = self.bounding_box.boundaries[dim_name][1]

                for j, q in enumerate(all_points):
                    if np.array_equal(p, q):  # Skip the same point
                        continue

                    # For the boundary between p and q, we need to find where
                    # the perpendicular bisector intersects the axis through p

                    # Calculate midpoint between p and q
                    midpoint = (p + q) / 2.0

                    # Calculate normal vector to the bisector plane
                    normal = q - p
                    # If the normal vector has no component in this dimension, skip
                    if abs(normal[dim]) < tol:
                        continue

                    # Find the intersection of the bisector with the line through p
                    # parallel to dimension dim

                    # The line is parameterized as: p + t * unit_vector(dim)
                    # where unit_vector(dim) is a unit vector along dimension dim

                    # For a point on this line to be on the bisector, it must satisfy:
                    # (x - midpoint) · normal = 0

                    # Substituting: (p + t * unit_vector(dim) - midpoint) · normal = 0
                    # Solving for t: t = (midpoint - p) · normal / (unit_vector(dim) · normal)

                    # Since unit_vector(dim) is just a vector with 1 in position dim and 0 elsewhere,
                    # unit_vector(dim) · normal = normal[dim]

                    t = np.dot(midpoint - p, normal) / normal[dim]

                    # The intersection point is then:
                    intersection = p.copy()
                    intersection[dim] = p[dim] + t

                    # Check if this creates a bound
                    if t > 0:  # Upper bound
                        upper_bound = min(upper_bound, intersection[dim])
                    else:  # Lower bound
                        lower_bound = max(lower_bound, intersection[dim])

                # Create the boundary points
                lower_point = p.copy()
                lower_point[dim] = lower_bound

                upper_point = p.copy()
                upper_point[dim] = upper_bound

                boundaries_for_p[dim, 0, :] = lower_point
                boundaries_for_p[dim, 1, :] = upper_point

            all_boundaries.append(boundaries_for_p)

        return all_boundaries

    def monte_carlo_volume(
        self, p_i, points_array, domain_min, domain_max, n_samples=1000000
    ):
        """
        Approximate the volume of the Voronoi cell for point p_i using Monte Carlo integration.

        Parameters:
            p_i (np.ndarray): The central point whose cell volume is approximated.
            points_array (np.ndarray): Array containing all points.
            domain_min (np.ndarray): Array of minima for each dimension.
            domain_max (np.ndarray): Array of maxima for each dimension.
            n_samples (int): Number of random samples to use.

        Returns:
            float: Approximated volume of the Voronoi cell for p_i.
        """
        n_dims = len(domain_min)

        # Compute the volume of the bounding box
        box_volume = self.bounding_box.volume

        # Precompute terms for the Voronoi inequalities of p_i.
        # For each p_j, the inequality is: (p_j - p_i)^T x - 0.5*(||p_j||^2 - ||p_i||^2) <= 0.
        A_list = []
        b_list = []
        for p_j in points_array:
            if np.all(p_j == p_i):
                continue
            A_list.append(p_j - p_i)
            b_list.append(0.5 * (np.dot(p_j, p_j) - np.dot(p_i, p_i)))
        A_list = np.array(A_list)
        b_list = np.array(b_list)

        # Generate random samples uniformly in the bounding box
        samples = np.random.uniform(
            low=domain_min, high=domain_max, size=(n_samples, n_dims)
        )

        # I. e. there is only one point and it will span the full space
        if A_list.size == 0:
            return 1.0

        # Check halfspace constraints: (p_j - p_i)^T x - b <= 0 for all p_j
        # A sample is inside the Voronoi cell if it satisfies all inequalities.
        satisfies = np.all(np.dot(samples, A_list.T) <= b_list, axis=1)
        fraction_inside = np.sum(satisfies) / n_samples

        # The approximated volume is the fraction times the bounding box volume.
        return fraction_inside * box_volume

    # Example function to compute volumes for all cells using Monte Carlo integration.
    def calculate_voronoi_cell_volumes(self, points):
        """
        Calculate an approximate volume for each bounded Voronoi cell using Monte Carlo integration.
        This method avoids computing the convex hull at high dimensions.

        Parameters:
            points (list or np.ndarray): List-of-dict or array of point coordinates.
            n_samples (int): Number of random samples to use for each cell.

        Returns:
            list: Approximated volumes for each Voronoi cell.
        """
        # Convert list-of-dict points to numpy array if needed.
        tmp = []
        for point in points:
            # Assume point is a dict; adjust as necessary.
            tmp.append(list(point.values()))
        points_array = np.array(tmp)
        n_dims = points_array.shape[1]

        # Get bounding box information
        dimensions = self.bounding_box.get_dimension_names()
        domain_min = np.array(
            [self.bounding_box.boundaries[dim][0] for dim in dimensions]
        )
        domain_max = np.array(
            [self.bounding_box.boundaries[dim][1] for dim in dimensions]
        )

        volumes = []
        for p_i in points_array:
            vol = self.monte_carlo_volume(p_i, points_array, domain_min, domain_max)
            volumes.append(vol)

        return volumes

    def _convert_region_line_boundaries(self, points, fvals, boundaries, volumes):
        """
        Converts the output of compute_region_line_boundaries into a list of Region objects.

        For each candidate region (each element is an array of shape (n_dims, 2, n_dims)),
        we create a Region object with boundaries for each dimension. For each dimension d,
        the lower bound is taken as boundaries[i][d, 0, d] and the upper bound as boundaries[i][d, 1, d].

        Infinite bounds are replaced by the corresponding self.bounding_box limits if provided.

        Parameters:
        boundaries (list): List of arrays (one per candidate) with shape (n_dims, 2, n_dims).

        Returns:
            list of Region: Each Region object represents a candidate region's bounds.
        """
        regions = []
        if not boundaries:
            return regions

        n_dims = boundaries[0].shape[0]
        dimension_names = self.bounding_box.get_dimension_names()
        for point, fval, region, volume in zip(points, fvals, boundaries, volumes):
            region_dict = {}
            for d in range(n_dims):
                lower_bound = region[d, 0, d]
                upper_bound = region[d, 1, d]
                if self.bounding_box is not None:
                    if lower_bound == -np.inf:
                        lower_bound = self.bounding_box[d][0]
                    if upper_bound == np.inf:
                        upper_bound = self.bounding_box[d][1]
                # TODO: The conversion to int() is only temporary for NB201. This has to go!
                # region_dict[dimension_names[d]] = (int(lower_bound), int(upper_bound))
                # TODO: Temporary for ZDT
                region_dict[dimension_names[d]] = (
                    round(lower_bound, 2),
                    round(upper_bound, 2),
                )

            normalized_volume = volume / self.bounding_box.volume
            regions.append(
                Region(
                    volume=volume,
                    normalized_volume=normalized_volume,
                    center=point,
                    center_fval={key: round(value, 2) for key, value in fval.items()},
                    boundaries=region_dict,
                )
            )
        return regions
