import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
POSITIONS = ("tl", "tr", "bl", "br")
INPUT_COLORS = (1, 2, 3, 4)
MAX_ATTEMPTS = 500
FORBIDDEN_PRIORITY_ORDERS = {
    ("tl", "tr", "bl", "br"),
    ("bl", "br", "tl", "tr"),
    ("br", "bl", "tr", "tl"),
}


def _apply_position_transform(patch: np.ndarray, position: str) -> np.ndarray:
    if position == "tl":
        return np.fliplr(patch)
    if position == "tr":
        return np.flipud(patch)
    if position == "bl":
        return np.rot90(patch, 2)
    if position == "br":
        return patch.copy()
    raise ValueError(f"unknown position: {position}")


def _foreground_count(patch: np.ndarray) -> int:
    return int(np.count_nonzero((patch != BLACK) & (patch != GREY)))


def _find_separator_index(lines: np.ndarray) -> int:
    hits = [idx for idx in range(lines.shape[0]) if np.all(lines[idx] == GREY)]
    assert len(hits) == 1, "expected exactly one full grey separator line"
    return int(hits[0])


def _extract_quadrants(grid: np.ndarray) -> dict[str, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    sep_r = _find_separator_index(g)
    sep_c = _find_separator_index(g.T)
    top = sep_r
    bottom = g.shape[0] - sep_r - 1
    left = sep_c
    right = g.shape[1] - sep_c - 1
    assert top == bottom and left == right and top == left, "quadrants must be equal-sized"
    q = top
    assert q >= 2, "quadrants are too small"

    return {
        "tl": g[:q, :q],
        "tr": g[:q, sep_c + 1 :],
        "bl": g[sep_r + 1 :, :q],
        "br": g[sep_r + 1 :, sep_c + 1 :],
    }


def _overlay_by_counts(transformed: dict[str, np.ndarray], counts: dict[str, int]) -> np.ndarray:
    order = sorted(POSITIONS, key=lambda pos: counts[pos])
    ordered_counts = [counts[pos] for pos in order]
    assert len(set(ordered_counts)) == 4, "foreground counts must be pairwise distinct"

    q = transformed["tl"].shape[0]
    out = np.zeros((q, q), dtype=int)
    filled = np.zeros((q, q), dtype=bool)
    for pos in order:
        layer = transformed[pos]
        mask = (layer != BLACK) & (~filled)
        out[mask] = layer[mask]
        filled |= mask
    return out


def _overlay_in_order(transformed: dict[str, np.ndarray], order: tuple[str, ...]) -> np.ndarray:
    q = transformed["tl"].shape[0]
    out = np.zeros((q, q), dtype=int)
    filled = np.zeros((q, q), dtype=bool)
    for pos in order:
        layer = transformed[pos]
        mask = (layer != BLACK) & (~filled)
        out[mask] = layer[mask]
        filled |= mask
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the four quadrant patterns after position-coded mirroring, with sparser original quadrants taking visibility priority.

    Input structure:
    - The input is a square grid split by one full grey row and one full grey column into four equal quadrants.
    - Each quadrant contains a sparse monochrome pattern on black background; the separator cross is grey.
    - The four quadrants use distinct foreground colors, and their foreground pixel counts are all different.

    Rule:
    - Remove the grey separator cross and extract the four quadrants.
    - Apply the position-coded transform to each quadrant: top-left = horizontal flip, top-right = vertical flip, bottom-left = 180-degree rotation, bottom-right = unchanged.
    - Count each quadrant's original foreground pixels before transforming.
    - Transparently overlay the four transformed quadrants into one quadrant-sized output: black is transparent, and any overlap is resolved by choosing the quadrant with the smallest original foreground count.

    Key insight:
    - Two independent hidden signals must be combined: quadrant position determines the mirror type, while original pixel density determines layer priority.
    """
    quadrants = _extract_quadrants(np.asarray(grid, dtype=int))
    transformed = {pos: _apply_position_transform(quadrants[pos], pos) for pos in POSITIONS}
    counts = {pos: _foreground_count(quadrants[pos]) for pos in POSITIONS}
    return _overlay_by_counts(transformed, counts)


def _grow_cluster(q: int, size: int, occupied: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    free = [(r, c) for r in range(q) for c in range(q) if (r, c) not in occupied]
    if len(free) < size:
        return None
    for _ in range(80):
        start = random.choice(free)
        cluster = {start}
        while len(cluster) < size:
            frontier = []
            for r, c in cluster:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < q and 0 <= nc < q and (nr, nc) not in occupied and (nr, nc) not in cluster:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cluster.add(random.choice(frontier))
        if len(cluster) == size:
            return cluster
    return None


def _sample_group_sizes(q: int) -> dict[str, int] | None:
    for _ in range(120):
        sizes = {
            "r0_only": random.randint(1, 2),
            "r0_conflict": random.randint(1, 2),
            "r0_triple": random.randint(1, 2),
            "r1_only": random.randint(1, 2),
            "r1_conflict": random.randint(1, 3),
            "r2_only": random.randint(1, 2),
            "r2_conflict": random.randint(1, 3),
            "r3_only": random.randint(1, 3),
        }
        total = sum(sizes.values())
        if total > q * q - 4:
            continue

        counts = {
            0: sizes["r0_only"] + sizes["r0_conflict"] + sizes["r0_triple"],
            1: sizes["r1_only"] + sizes["r1_conflict"] + sizes["r0_triple"],
            2: sizes["r2_only"] + sizes["r2_conflict"] + sizes["r0_conflict"] + sizes["r1_conflict"],
            3: sizes["r3_only"] + sizes["r2_conflict"] + sizes["r1_conflict"] + sizes["r0_triple"],
        }
        ordered = [counts[i] for i in range(4)]
        if ordered[0] < ordered[1] < ordered[2] < ordered[3] and ordered[3] <= int(q * q * 0.6):
            return sizes
    return None


def _build_abstract_masks(q: int) -> tuple[list[np.ndarray], dict[str, set[tuple[int, int]]]] | None:
    sizes = _sample_group_sizes(q)
    if sizes is None:
        return None

    group_order = sorted(sizes, key=lambda name: (-sizes[name], name))
    placed: dict[str, set[tuple[int, int]]] = {}
    occupied: set[tuple[int, int]] = set()
    for name in group_order:
        cluster = _grow_cluster(q, sizes[name], occupied)
        if cluster is None:
            return None
        placed[name] = cluster
        occupied |= cluster

    members = {
        "r0_only": (0,),
        "r0_conflict": (0, 2),
        "r0_triple": (0, 1, 3),
        "r1_only": (1,),
        "r1_conflict": (1, 2, 3),
        "r2_only": (2,),
        "r2_conflict": (2, 3),
        "r3_only": (3,),
    }

    masks = [np.zeros((q, q), dtype=bool) for _ in range(4)]
    for name, coords in placed.items():
        for rank in members[name]:
            for r, c in coords:
                masks[rank][r, c] = True
    return masks, placed


def _assemble_input(original_quadrants: dict[str, np.ndarray]) -> np.ndarray:
    q = original_quadrants["tl"].shape[0]
    grid = np.full((2 * q + 1, 2 * q + 1), BLACK, dtype=int)
    grid[q, :] = GREY
    grid[:, q] = GREY
    grid[:q, :q] = original_quadrants["tl"]
    grid[:q, q + 1 :] = original_quadrants["tr"]
    grid[q + 1 :, :q] = original_quadrants["bl"]
    grid[q + 1 :, q + 1 :] = original_quadrants["br"]
    return grid


def _colored_layers_from_order(
    masks_by_rank: list[np.ndarray], order_positions: tuple[str, ...], colors_by_position: dict[str, int]
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    transformed: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for rank, pos in enumerate(order_positions):
        color = colors_by_position[pos]
        layer = np.where(masks_by_rank[rank], color, BLACK)
        transformed[pos] = layer.astype(int)
        counts[pos] = int(np.count_nonzero(masks_by_rank[rank]))
    return transformed, counts


def _extract_originals_from_transformed(transformed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {pos: _apply_position_transform(transformed[pos], pos) for pos in POSITIONS}


def _visible_colors(output: np.ndarray) -> set[int]:
    return set(map(int, np.unique(output))) - {BLACK}


def _conflict_cell_count(transformed: dict[str, np.ndarray]) -> int:
    occupancy = np.zeros(transformed["tl"].shape, dtype=int)
    for pos in POSITIONS:
        occupancy += transformed[pos] != BLACK
    return int(np.count_nonzero(occupancy >= 2))


def _semantic_checks(
    grid: np.ndarray,
    transformed: dict[str, np.ndarray],
    originals: dict[str, np.ndarray],
    counts: dict[str, int],
) -> bool:
    out = main(grid)
    expected = _overlay_by_counts(transformed, counts)
    if not np.array_equal(out, expected):
        return False
    if len(_visible_colors(out)) != 4:
        return False
    if _conflict_cell_count(transformed) < 3:
        return False

    for pos in ("tl", "tr", "bl"):
        if np.array_equal(originals[pos], transformed[pos]):
            return False
        alt_layers = dict(transformed)
        alt_layers[pos] = originals[pos]
        if np.array_equal(_overlay_by_counts(alt_layers, counts), out):
            return False

    tested_orders = [
        ("tl", "tr", "bl", "br"),
        ("bl", "br", "tl", "tr"),
        ("br", "bl", "tr", "tl"),
    ]
    for order in tested_orders:
        if np.array_equal(_overlay_in_order(transformed, order), out):
            return False

    for pos in POSITIONS:
        color = int(np.max(transformed[pos]))
        if color == BLACK or int(np.count_nonzero(out == color)) == 0:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        q = random.randint(5, 7)
        built = _build_abstract_masks(q)
        if built is None:
            continue
        masks_by_rank, _groups = built

        possible_orders = [
            order for order in itertools.permutations(POSITIONS) if order not in FORBIDDEN_PRIORITY_ORDERS
        ]
        order_positions = random.choice(possible_orders)
        colors = random.sample(list(INPUT_COLORS), len(POSITIONS))
        colors_by_position = {pos: color for pos, color in zip(POSITIONS, colors)}

        transformed, counts = _colored_layers_from_order(masks_by_rank, order_positions, colors_by_position)
        originals = _extract_originals_from_transformed(transformed)
        grid = _assemble_input(originals)

        if _semantic_checks(grid, transformed, originals, counts):
            return grid

    raise RuntimeError("failed to generate a valid density-priority mirror-overlay instance")
