from __future__ import annotations

import random

import numpy as np


BG = 0
DIVIDER = 9
PALETTE = tuple(range(1, 9))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver ---------------


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _find_divider_col(g: np.ndarray) -> int:
    candidates = [c for c in range(g.shape[1]) if np.all(g[:, c] == DIVIDER)]
    assert len(candidates) == 1, "expected exactly one full maroon divider column"
    divider = int(candidates[0])
    assert 0 < divider < g.shape[1] - 1, "divider must split the grid into two non-empty panels"
    return divider


def _parse_mapping(panel: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for comp in _components(panel != BG):
        assert len(comp) == 2, "each code entry must be a 2-cell block pair"
        (r0, c0), (r1, c1) = sorted(comp, key=lambda rc: (rc[0], rc[1]))
        assert r0 == r1, "each code entry must be horizontal"
        assert c1 == c0 + 1, "pair cells must be horizontally adjacent"

        src = int(panel[r0, c0])
        dst = int(panel[r0, c1])
        assert src not in {BG, DIVIDER} and dst not in {BG, DIVIDER}
        assert src not in mapping, "each source color must appear once in the code table"
        mapping[src] = dst

    assert mapping, "right panel must contain at least one code pair"
    return mapping


def _resolve_terminal(mapping: dict[int, int], color: int) -> int:
    seen: set[int] = set()
    cur = int(color)
    while cur in mapping:
        assert cur not in seen, "color mapping must be acyclic"
        seen.add(cur)
        cur = int(mapping[cur])
    return cur


def _recolor_left(panel: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    left = panel.copy()
    colors = set(map(int, np.unique(left))) - {BG}
    assert colors, "left panel must contain at least one non-black color"
    assert colors <= set(mapping), "every left-panel color must be defined in the code table"

    for color in colors:
        left[left == color] = _resolve_terminal(mapping, color)
    return left


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the left panel using the chained color-substitution table shown on the right.

    Input structure
    - One full maroon column splits the grid into left and right panels.
    - The left panel is a black-background picture made from several non-black colors.
    - The right panel is a black-background list of horizontal 2-cell color pairs.
      A pair [A, B] means "replace A with B".
    - The mapping can form chains such as A->B->C, but it never contains cycles.

    Rule
    - Read every 2-cell pair from the right panel and build the color mapping.
    - Resolve each mapping transitively until a terminal color is reached.
    - Replace each non-black color in the left panel by its terminal color.
    - Return only the decoded left panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    divider = _find_divider_col(g)
    left = g[:, :divider]
    right = g[:, divider + 1 :]
    assert left.shape[1] > 0 and right.shape[1] >= 2

    mapping = _parse_mapping(right)
    return _recolor_left(left, mapping)


# --------------- generator ---------------

MAX_GENERATION_ATTEMPTS = 400
GEN_RETRY_LIMIT = 80
PAIR_GAP_CHOICES = (1, 2)
MARGIN_CHOICES = (0, 1, 2)
LEFT_WIDTH_RANGE = (7, 14)
RIGHT_WIDTH_RANGE = (3, 7)
HEIGHT_FLOOR = 8
HEIGHT_EXTRA = 5
KEY_COUNT_RANGE = (2, 5)
SINK_COUNT_RANGE = (2, 3)
EXTRA_SHAPE_LIMIT = 2
PATCH_SIZE_MAX = 4
PATCH_RETRIES = 80


def _row_layout(n_pairs: int) -> tuple[int, list[int], int, int]:
    top = random.choice(MARGIN_CHOICES)
    gaps = [random.choice(PAIR_GAP_CHOICES) for _ in range(max(0, n_pairs - 1))]
    bottom = random.choice(MARGIN_CHOICES)
    required = top + n_pairs + sum(gaps) + bottom
    return top, gaps, bottom, required


def _mapping_with_active_colors() -> tuple[dict[int, int], list[int], bool] | None:
    require_chain = random.random() < 0.7
    expose_all_keys = random.random() < 0.5

    for _ in range(GEN_RETRY_LIMIT):
        n_keys = random.randint(*KEY_COUNT_RANGE)
        max_sinks = min(SINK_COUNT_RANGE[1], len(PALETTE) - n_keys)
        if max_sinks < SINK_COUNT_RANGE[0]:
            continue
        n_sinks = random.randint(SINK_COUNT_RANGE[0], max_sinks)

        nodes = random.sample(PALETTE, n_keys + n_sinks)
        keys = nodes[:n_keys]
        sinks = nodes[n_keys:]
        mapping: dict[int, int] = {}
        chain_root = None

        if require_chain:
            chain_root_index = random.randrange(n_keys - 1)
            chain_root = keys[chain_root_index]
            mapping[chain_root] = keys[random.randint(chain_root_index + 1, n_keys - 1)]

        for i, key in enumerate(keys):
            if key in mapping:
                continue
            choices = sinks if not require_chain else nodes[i + 1 :]
            mapping[key] = random.choice(choices)

        if expose_all_keys:
            active = set(keys)
        else:
            active_count = random.randint(2, n_keys)
            active = set(random.sample(keys, active_count))
        if chain_root is not None:
            active.add(chain_root)

        terminals = {key: _resolve_terminal(mapping, key) for key in keys}
        if len({terminals[key] for key in active}) < 2:
            continue
        if require_chain and not any(mapping[key] in mapping for key in active):
            continue

        active_colors = list(active)
        random.shuffle(active_colors)
        return mapping, active_colors, require_chain

    return None


def _grow_connected_patch(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    selected = {seed}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in selected:
                frontier.add((nr, nc))

    add_frontier(*seed)
    while len(selected) < area:
        if not frontier:
            break
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        selected.add(cell)
        add_frontier(*cell)

    assert len(selected) == area
    for r, c in selected:
        mask[r, c] = True
    return mask


def _random_patch_mask(max_h: int, max_w: int) -> np.ndarray:
    for _ in range(PATCH_RETRIES):
        ph = random.randint(1, min(PATCH_SIZE_MAX, max_h))
        pw = random.randint(1, min(PATCH_SIZE_MAX, max_w))
        if ph * pw == 1:
            continue
        min_area = max(2, (ph * pw + 1) // 2)
        area = random.randint(min_area, ph * pw)
        return _grow_connected_patch(ph, pw, area)
    raise RuntimeError("failed to sample a patch mask")


def _build_left_panel(height: int, width: int, active_colors: list[int]) -> np.ndarray | None:
    panel = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    shape_colors = list(active_colors)
    extra_shapes = random.randint(0, EXTRA_SHAPE_LIMIT)
    for _ in range(extra_shapes):
        shape_colors.append(random.choice(active_colors))
    random.shuffle(shape_colors)

    for color in shape_colors:
        placed = False
        for _ in range(PATCH_RETRIES):
            patch = _random_patch_mask(height, width)
            ph, pw = patch.shape
            r0 = random.randint(0, height - ph)
            c0 = random.randint(0, width - pw)

            rr0 = max(0, r0 - 1)
            rr1 = min(height, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            region = panel[r0 : r0 + ph, c0 : c0 + pw]
            region[patch] = color
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch
            placed = True
            break
        if not placed:
            return None

    filled = int((panel != BG).sum())
    if filled < 2 * len(active_colors):
        return None
    return panel


def _build_right_panel(
    height: int, width: int, mapping: dict[int, int], top: int, gaps: list[int]
) -> np.ndarray:
    panel = np.full((height, width), BG, dtype=int)
    entries = list(mapping.items())
    random.shuffle(entries)

    row = top
    for i, (src, dst) in enumerate(entries):
        col = random.randint(0, width - 2)
        panel[row, col] = src
        panel[row, col + 1] = dst
        if i < len(gaps):
            row += 1 + gaps[i]

    return panel


def _expected_output(left: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return _recolor_left(left, mapping)


def _generate_attempt() -> np.ndarray | None:
    spec = _mapping_with_active_colors()
    if spec is None:
        return None
    mapping, active_colors, _require_chain = spec

    top, gaps, _bottom, required_h = _row_layout(len(mapping))
    height = random.randint(max(HEIGHT_FLOOR, required_h), max(HEIGHT_FLOOR, required_h) + HEIGHT_EXTRA)
    left_width = random.randint(*LEFT_WIDTH_RANGE)
    right_width = random.randint(*RIGHT_WIDTH_RANGE)

    left = _build_left_panel(height, left_width, active_colors)
    if left is None:
        return None
    right = _build_right_panel(height, right_width, mapping, top, gaps)

    grid = np.concatenate(
        [left, np.full((height, 1), DIVIDER, dtype=int), right],
        axis=1,
    )
    expected = _expected_output(left, mapping)

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-color-substitution input")
