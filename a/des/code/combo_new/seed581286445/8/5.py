import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
COLORS = np.array([RED, GREEN, BLUE], dtype=int)

# Fixed task-specific food chain: red beats green, green beats blue, blue beats red.
BEATS = {RED: GREEN, GREEN: BLUE, BLUE: RED}
PREDATOR_OF = {prey: predator for predator, prey in BEATS.items()}

ROLE_OTHER = 0
ROLE_PREDATOR = 1
ROLE_PREY = 2

CHANGE_ADJACENT = np.array(
    [
        [ROLE_OTHER, ROLE_PREDATOR, ROLE_OTHER],
        [ROLE_PREDATOR, ROLE_PREY, ROLE_PREY],
        [ROLE_OTHER, ROLE_PREY, ROLE_PREDATOR],
    ],
    dtype=int,
)
CHANGE_OPPOSITE = np.array(
    [
        [ROLE_OTHER, ROLE_PREDATOR, ROLE_OTHER],
        [ROLE_PREY, ROLE_PREY, ROLE_PREY],
        [ROLE_OTHER, ROLE_PREDATOR, ROLE_PREDATOR],
    ],
    dtype=int,
)
CHANGE_THREE = np.array(
    [
        [ROLE_OTHER, ROLE_PREDATOR, ROLE_OTHER],
        [ROLE_PREDATOR, ROLE_PREY, ROLE_PREDATOR],
        [ROLE_OTHER, ROLE_PREY, ROLE_PREY],
    ],
    dtype=int,
)
STAY_ONE = np.array(
    [
        [ROLE_OTHER, ROLE_PREDATOR, ROLE_OTHER],
        [ROLE_PREY, ROLE_PREY, ROLE_OTHER],
        [ROLE_PREDATOR, ROLE_PREY, ROLE_PREDATOR],
    ],
    dtype=int,
)

CHANGE_PATCHES = (
    (CHANGE_ADJACENT, 2),
    (CHANGE_OPPOSITE, 2),
    (CHANGE_THREE, 3),
)
STAY_PATCHES = ((STAY_ONE, 1),)

GRID_SIZE_RANGE = (8, 12)
GEN_ATTEMPTS = 300
_GEN_CALL_COUNT = 0


def _predator_neighbor_counts(grid: np.ndarray, predator: int) -> np.ndarray:
    predator_mask = grid == predator
    counts = np.zeros(grid.shape, dtype=int)
    counts[1:, :] += predator_mask[:-1, :]
    counts[:-1, :] += predator_mask[1:, :]
    counts[:, 1:] += predator_mask[:, :-1]
    counts[:, :-1] += predator_mask[:, 1:]
    return counts


def _center_predator_count(grid: np.ndarray, r: int, c: int, predator: int) -> int:
    count = 0
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and int(grid[rr, cc]) == predator:
            count += 1
    return count


def _other_color(prey: int, predator: int) -> int:
    for color in COLORS:
        value = int(color)
        if value not in (prey, predator):
            return value
    raise AssertionError("expected exactly three task colors")


def _materialize_patch(role_patch: np.ndarray, prey: int) -> np.ndarray:
    predator = PREDATOR_OF[prey]
    other = _other_color(prey, predator)
    actual = np.empty_like(role_patch)
    actual[role_patch == ROLE_OTHER] = other
    actual[role_patch == ROLE_PREDATOR] = predator
    actual[role_patch == ROLE_PREY] = prey
    return actual


def _random_variant(role_patch: np.ndarray) -> np.ndarray:
    variant = role_patch.copy()
    if random.random() < 0.5:
        variant = np.fliplr(variant)
    return np.rot90(variant, random.randrange(4)).copy()


def _apply_symmetry(
    grid: np.ndarray, witnesses: list[dict[str, int]], rot_k: int, flip_lr: bool
) -> tuple[np.ndarray, list[dict[str, int]]]:
    out = np.rot90(grid, rot_k).copy()
    if flip_lr:
        out = np.fliplr(out).copy()

    n = grid.shape[0]
    moved: list[dict[str, int]] = []
    for witness in witnesses:
        r = witness["row"]
        c = witness["col"]
        for _ in range(rot_k):
            r, c = n - 1 - c, r
        if flip_lr:
            c = n - 1 - c
        moved.append(
            {
                "row": int(r),
                "col": int(c),
                "prey": int(witness["prey"]),
                "wanted_count": int(witness["wanted_count"]),
            }
        )
    return out, moved


def _build_base_grid(n: int) -> np.ndarray:
    rows, cols = np.indices((n, n))
    phase = random.randrange(3)
    family = random.choice(("cell_affine", "block_affine", "banded"))

    if family == "cell_affine":
        a, b = random.choice(((1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)))
        color_idx = (a * rows + b * cols + phase) % 3
    elif family == "block_affine":
        block_h = random.randint(1, 3)
        block_w = random.randint(1, 3)
        a, b = random.choice(((1, 1), (1, 2), (2, 1), (2, 2)))
        color_idx = (a * (rows // block_h) + b * (cols // block_w) + phase) % 3
    else:
        band = random.randint(1, 3)
        color_idx = ((rows // band) + 2 * (cols // band) + phase) % 3

    return COLORS[color_idx].copy()


def _stamp_patch(
    grid: np.ndarray,
    center: tuple[int, int],
    prey: int,
    role_patch: np.ndarray,
    wanted_count: int,
) -> dict[str, int] | None:
    patch = _materialize_patch(role_patch, prey)
    r, c = center
    for pr in range(3):
        for pc in range(3):
            rr = r + pr - 1
            cc = c + pc - 1
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                grid[rr, cc] = int(patch[pr, pc])

    predator = PREDATOR_OF[prey]
    actual_count = _center_predator_count(grid, r, c, predator)
    if actual_count != wanted_count:
        return None
    if int(grid[r, c]) != prey:
        return None

    return {
        "row": int(r),
        "col": int(c),
        "prey": int(prey),
        "wanted_count": int(wanted_count),
    }


def _interior_slots(n: int) -> list[tuple[int, int]]:
    inner = 2
    outer = n - 3
    return [(inner, inner), (inner, outer), (outer, inner), (outer, outer)]


def _mode_specs(mode: int, n: int) -> list[dict[str, object]]:
    shuffled_colors = random.sample([int(color) for color in COLORS], 3)

    if mode == 0:
        slots = _interior_slots(n)
        specs = []
        for prey in shuffled_colors:
            role_patch, wanted = random.choice(CHANGE_PATCHES)
            specs.append(
                {
                    "center": slots[len(specs)],
                    "prey": prey,
                    "role_patch": _random_variant(role_patch),
                    "wanted_count": wanted,
                }
            )
        extra_kind = random.choice(("change", "stay"))
        role_patch, wanted = random.choice(CHANGE_PATCHES if extra_kind == "change" else STAY_PATCHES)
        specs.append(
            {
                "center": slots[-1],
                "prey": random.choice(shuffled_colors),
                "role_patch": _random_variant(role_patch),
                "wanted_count": wanted,
            }
        )
        random.shuffle(specs)
        return specs

    if mode == 1:
        slots = _interior_slots(n)
        paired_prey, other_prey, last_prey = shuffled_colors
        change_patch, change_wanted = random.choice(CHANGE_PATCHES[:2])
        extra_change_patch, extra_change_wanted = random.choice(CHANGE_PATCHES)
        specs = [
            {
                "center": slots[0],
                "prey": paired_prey,
                "role_patch": _random_variant(STAY_ONE),
                "wanted_count": 1,
            },
            {
                "center": slots[1],
                "prey": paired_prey,
                "role_patch": _random_variant(change_patch),
                "wanted_count": change_wanted,
            },
            {
                "center": slots[2],
                "prey": other_prey,
                "role_patch": _random_variant(STAY_ONE),
                "wanted_count": 1,
            },
            {
                "center": slots[3],
                "prey": last_prey,
                "role_patch": _random_variant(extra_change_patch),
                "wanted_count": extra_change_wanted,
            },
        ]
        random.shuffle(specs)
        return specs

    last_patch, last_wanted = random.choice(CHANGE_PATCHES + STAY_PATCHES)
    boundary_specs = [
        {
            "center": (0, 0),
            "prey": shuffled_colors[0],
            "role_patch": np.rot90(CHANGE_ADJACENT, 2).copy(),
            "wanted_count": 2,
        },
        {
            "center": (0, n - 3),
            "prey": shuffled_colors[1],
            "role_patch": np.rot90(STAY_ONE, 2).copy(),
            "wanted_count": 1,
        },
        {
            "center": (n - 3, 0),
            "prey": shuffled_colors[2],
            "role_patch": np.rot90(CHANGE_ADJACENT, 2).copy(),
            "wanted_count": 2,
        },
        {
            "center": (n - 3, n - 3),
            "prey": random.choice(shuffled_colors),
            "role_patch": _random_variant(last_patch),
            "wanted_count": last_wanted,
        },
    ]
    return boundary_specs


def _is_balanced(grid: np.ndarray) -> bool:
    counts = [int(np.count_nonzero(grid == color)) for color in COLORS]
    return max(counts) - min(counts) <= grid.shape[0]


def _generate_attempt(mode: int) -> np.ndarray | None:
    n = random.randint(*GRID_SIZE_RANGE)
    grid = _build_base_grid(n)
    witnesses: list[dict[str, int]] = []

    for spec in _mode_specs(mode, n):
        witness = _stamp_patch(
            grid,
            tuple(spec["center"]),
            int(spec["prey"]),
            np.asarray(spec["role_patch"], dtype=int),
            int(spec["wanted_count"]),
        )
        if witness is None:
            return None
        witnesses.append(witness)

    rot_k = random.randrange(4)
    flip_lr = random.random() < 0.5
    grid, witnesses = _apply_symmetry(grid, witnesses, rot_k, flip_lr)

    if not _is_balanced(grid):
        return None

    out = main(grid)
    changed = int(np.count_nonzero(out != grid))
    if changed < 2:
        return None

    for witness in witnesses:
        r = witness["row"]
        c = witness["col"]
        prey = witness["prey"]
        predator = PREDATOR_OF[prey]
        wanted_count = witness["wanted_count"]
        if int(grid[r, c]) != prey:
            return None
        if _center_predator_count(grid, r, c, predator) != wanted_count:
            return None
        expected = predator if wanted_count >= 2 else prey
        if int(out[r, c]) != expected:
            return None

    if len(np.unique(grid)) != 3:
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """Update a full three-color board by letting cells be eaten when at least two orthogonal neighbors carry their predator color.

    Input structure:
    - The grid is an N x N board with no background; every cell is one of three recurring colors.
    - The same three colors are used across the task examples, and their counts stay roughly balanced.
    - Only four-neighbor adjacency matters; edge and corner cells simply have fewer neighbors to inspect.

    Rule:
    - The examples share one fixed cyclic food chain: red defeats green, green defeats blue, and blue defeats red.
    - For each cell, find the unique task color that defeats that cell's color and count how many orthogonal neighbors have that predator color.
    - If the predator count is at least two, replace the cell with the predator color; otherwise leave it unchanged.
    - Apply every replacement simultaneously from the original input state.

    Key insight:
    - Solving the task requires inferring both the hidden three-color cycle and the threshold: one predator neighbor is not enough, but two or more are.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w, "expected a square board"
    assert 1 <= h <= 30, "grid must stay within ARC size limits"

    colors = set(map(int, np.unique(g)))
    assert colors == set(map(int, COLORS)), "expected exactly the task's three colors"

    out = g.copy()
    for prey, predator in PREDATOR_OF.items():
        counts = _predator_neighbor_counts(g, predator)
        eaten = (g == prey) & (counts >= 2)
        out[eaten] = predator
    return out


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    mode = _GEN_CALL_COUNT % 3
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid grid for mode {mode}")
