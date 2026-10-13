from __future__ import annotations

from collections import Counter

import numpy as np


_RNG = np.random.default_rng()

# Fixed visible roles.
_BG = 0
_SEP = 5
_OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# Generator bounds.
_MAX_GRID_SIDE = 30
_PANEL_SIZE_CHOICES = [5, 6, 7]
_MAX_GENERATION_ATTEMPTS = 300
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected non-empty values"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique mode"
    return int(ranked[0][0])


def _solid_line_colors(g: np.ndarray, axis: int) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    limit = g.shape[axis]
    for idx in range(limit):
        line = g[idx, :] if axis == 0 else g[:, idx]
        color = int(line[0])
        if np.all(line == color):
            mapping.setdefault(color, []).append(idx)
    return mapping


def _content_spans(separator_mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_separator in enumerate(separator_mask):
        if not is_separator and start is None:
            start = idx
        elif is_separator and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(separator_mask) - 1))
    return spans


def _panel_layout(g: np.ndarray) -> tuple[int, tuple[int, int], list[tuple[int, int]]]:
    h, w = g.shape
    row_map = _solid_line_colors(g, axis=0)
    col_map = _solid_line_colors(g, axis=1)

    candidates: list[tuple[int, tuple[int, int], list[tuple[int, int]]]] = []
    for color in sorted(set(row_map) | set(col_map)):
        row_mask = np.zeros(h, dtype=bool)
        col_mask = np.zeros(w, dtype=bool)
        row_mask[row_map.get(color, [])] = True
        col_mask[col_map.get(color, [])] = True

        row_spans = _content_spans(row_mask)
        col_spans = _content_spans(col_mask)
        if len(row_spans) != 1 or len(col_spans) < 2:
            continue

        (r0, r1) = row_spans[0]
        panel_h = r1 - r0 + 1
        widths = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(widths) != 1:
            continue
        panel_w = widths.pop()
        if panel_h != panel_w or panel_h < 2:
            continue
        if any(np.any(g[r0 : r1 + 1, c0 : c1 + 1] == color) for c0, c1 in col_spans):
            continue

        candidates.append((int(color), (int(r0), int(r1)), [(int(c0), int(c1)) for c0, c1 in col_spans]))

    assert len(candidates) == 1, "expected a unique square-panel separator layout"
    return candidates[0]


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    seen = np.zeros_like(mask, dtype=bool)
    start_r, start_c = map(int, coords[0])
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    count = 0

    while stack:
        r, c = stack.pop()
        count += 1
        for dr, dc in _DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((rr, cc))

    return count == int(mask.sum())


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _canonical_rotation_key(mask: np.ndarray) -> tuple[str, ...]:
    base = _crop_mask(mask)
    return min(_mask_key(np.rot90(base, k)) for k in range(4))


def _extract_panel_object(panel: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    fg = panel != bg
    assert fg.any(), "each panel must contain one foreground object"
    colors = {int(color) for color in panel[fg]}
    assert len(colors) == 1, "each panel must contain a monochrome object"

    tight = _crop_mask(fg)
    assert _is_connected(tight), "panel object must be 4-connected"
    return tight, next(iter(colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group square panels by rotation-equivalent object shape and output one voted color per group.

    Input structure:
    - A grey separator layout splits the grid into equal square panels in one row.
    - Each panel contains exactly one monochrome 4-connected object on a shared background.

    Rule:
    - Crop each panel object to its tight binary mask.
    - Canonicalize it over the four rotations and group panels with the same canonical mask.
    - Each group's output color is the most frequent panel color in that group; on a tie,
      use the color of that group's leftmost panel.
    - Output an N x 1 column, ordered by each group's first panel from left to right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, (r0, r1), col_spans = _panel_layout(g)
    panels = [g[r0 : r1 + 1, c0 : c1 + 1] for c0, c1 in col_spans]
    bg = _unique_mode(np.concatenate([panel.reshape(-1) for panel in panels]))
    assert bg != separator, "background must differ from the separator"

    ordered_keys: list[tuple[str, ...]] = []
    groups: dict[tuple[str, ...], list[int]] = {}
    for panel in panels:
        mask, color = _extract_panel_object(panel, bg)
        key = _canonical_rotation_key(mask)
        if key not in groups:
            ordered_keys.append(key)
            groups[key] = []
        groups[key].append(int(color))

    out_colors: list[int] = []
    for key in ordered_keys:
        colors = groups[key]
        counts = Counter(colors)
        max_count = max(counts.values())
        winners = [color for color, count in counts.items() if count == max_count]
        out_colors.append(int(colors[0] if len(winners) > 1 else winners[0]))

    return np.asarray(out_colors, dtype=int).reshape(-1, 1)


def _grid_width(panel_size: int, panel_count: int) -> int:
    return panel_count * panel_size + panel_count + 1


def _grow_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed_r = int(_RNG.integers(height))
    seed_c = int(_RNG.integers(width))
    mask[seed_r, seed_c] = True
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in _DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                frontier.add((rr, cc))

    add_frontier(seed_r, seed_c)
    while int(mask.sum()) < area:
        rr, cc = frontier.pop() if len(frontier) == 1 else tuple(frontier)[int(_RNG.integers(len(frontier)))]
        frontier.discard((rr, cc))
        mask[rr, cc] = True
        add_frontier(rr, cc)

    return _crop_mask(mask)


def _rotation_representatives(mask: np.ndarray) -> list[int]:
    reps: dict[tuple[str, ...], int] = {}
    for k in range(4):
        reps.setdefault(_mask_key(np.rot90(mask, k)), k)
    return list(reps.values())


def _sample_shape(panel_size: int, used_keys: set[tuple[str, ...]]) -> np.ndarray | None:
    for _ in range(200):
        raw_h = int(_RNG.integers(2, panel_size - 1))
        raw_w = int(_RNG.integers(2, panel_size - 1))
        max_area = min(raw_h * raw_w - 1, (panel_size * panel_size) // 3)
        if max_area < 3:
            continue

        area = int(_RNG.integers(3, max_area + 1))
        mask = _grow_mask(raw_h, raw_w, area)
        if mask.shape[0] > panel_size - 2 or mask.shape[1] > panel_size - 2:
            continue
        if len(_rotation_representatives(mask)) < 2:
            continue

        key = _canonical_rotation_key(mask)
        if key in used_keys:
            continue
        return mask
    return None


def _sequence_with_split_duplicates(sizes: list[int]) -> list[int]:
    labels = [group_id for group_id, size in enumerate(sizes) for _ in range(size)]
    best = labels[:]
    for _ in range(100):
        trial = labels[:]
        _RNG.shuffle(trial)
        positions: dict[int, list[int]] = {}
        for idx, group_id in enumerate(trial):
            positions.setdefault(group_id, []).append(idx)
        if any(len(pos) > 1 and pos[-1] - pos[0] + 1 != len(pos) for pos in positions.values()):
            return trial
        best = trial
    return best


def _pick_distinct_colors(count: int, exclude: set[int] | None = None) -> list[int]:
    blocked = set() if exclude is None else set(exclude)
    choices = [color for color in _OBJECT_COLORS if color not in blocked]
    sampled = _RNG.choice(choices, size=count, replace=False)
    return [int(color) for color in np.asarray(sampled)]


def _assign_group_colors(mode: str, sizes: list[int], sequence: list[int]) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = {}
    for idx, group_id in enumerate(sequence):
        positions.setdefault(group_id, []).append(idx)

    output_colors = {group_id: color for group_id, color in enumerate(_pick_distinct_colors(len(sizes)))}
    group_colors: dict[int, list[int]] = {}

    if mode == "majority":
        majority_group = next(group_id for group_id, size in enumerate(sizes) if size == 3)
        majority_color = output_colors[majority_group]
        minority_color = _pick_distinct_colors(1, exclude={*output_colors.values()})[0]
        group_colors[majority_group] = [minority_color, majority_color, majority_color]
        for group_id, size in enumerate(sizes):
            if group_id == majority_group:
                continue
            assert size == 1
            group_colors[group_id] = [output_colors[group_id]]
        return group_colors

    tie_candidates = [group_id for group_id, size in enumerate(sizes) if size == 2]
    tie_group = int(tie_candidates[int(_RNG.integers(len(tie_candidates)))])
    loser_color = _pick_distinct_colors(1, exclude={*output_colors.values()})[0]

    for group_id, size in enumerate(sizes):
        if size == 1:
            group_colors[group_id] = [output_colors[group_id]]
        elif group_id == tie_group:
            group_colors[group_id] = [output_colors[group_id], loser_color]
        else:
            group_colors[group_id] = [output_colors[group_id]] * size

    return group_colors


def _expected_output(sequence: list[int], group_colors: dict[int, list[int]]) -> np.ndarray:
    seen: list[int] = []
    out_colors: list[int] = []
    for group_id in sequence:
        if group_id in seen:
            continue
        seen.append(group_id)
        colors = group_colors[group_id]
        counts = Counter(colors)
        max_count = max(counts.values())
        winners = [color for color, count in counts.items() if count == max_count]
        out_colors.append(int(colors[0] if len(winners) > 1 else winners[0]))
    return np.asarray(out_colors, dtype=int).reshape(-1, 1)


def _panel_from_mask(panel_size: int, mask: np.ndarray, color: int, rotation: int) -> np.ndarray:
    rotated = np.rot90(mask, rotation)
    h, w = rotated.shape
    assert h <= panel_size - 2 and w <= panel_size - 2

    panel = np.full((panel_size, panel_size), _BG, dtype=int)
    row = int(_RNG.integers(1, panel_size - h))
    col = int(_RNG.integers(1, panel_size - w))
    panel[row : row + h, col : col + w][rotated] = int(color)
    return panel


def _sample_rotations(mask: np.ndarray, count: int) -> list[int]:
    reps = _rotation_representatives(mask)
    if count == 1:
        return [int(reps[int(_RNG.integers(len(reps)))])]

    _RNG.shuffle(reps)
    rotations = reps[: min(len(reps), count)]
    while len(rotations) < count:
        rotations.append(int(reps[int(_RNG.integers(len(reps)))]))
    _RNG.shuffle(rotations)
    return [int(rotation) for rotation in rotations]


def _build_input(panel_size: int, sequence: list[int], group_masks: dict[int, np.ndarray], group_colors: dict[int, list[int]]) -> np.ndarray:
    panel_count = len(sequence)
    grid = np.full((panel_size + 2, _grid_width(panel_size, panel_count)), _SEP, dtype=int)
    rotations_by_group = {
        group_id: _sample_rotations(group_masks[group_id], len(colors))
        for group_id, colors in group_colors.items()
    }
    used_by_group = {group_id: 0 for group_id in group_colors}

    for panel_idx, group_id in enumerate(sequence):
        occ_idx = used_by_group[group_id]
        used_by_group[group_id] += 1
        color = group_colors[group_id][occ_idx]
        rotation = rotations_by_group[group_id][occ_idx]
        panel = _panel_from_mask(panel_size, group_masks[group_id], color, rotation)
        c0 = 1 + panel_idx * (panel_size + 1)
        grid[1 : 1 + panel_size, c0 : c0 + panel_size] = panel

    return grid


def _gen_attempt() -> np.ndarray | None:
    mode = "majority" if bool(_RNG.integers(0, 2)) else "tie"
    if mode == "majority":
        panel_size = int(_RNG.choice([5, 6]))
        sizes = [3, 1]
    else:
        panel_size = int(_RNG.choice(_PANEL_SIZE_CHOICES))
        valid_counts = [count for count in (3, 4) if _grid_width(panel_size, count) <= _MAX_GRID_SIDE]
        if not valid_counts:
            return None
        panel_count = int(_RNG.choice(valid_counts))
        if panel_count == 3:
            sizes = [2, 1]
        else:
            sizes = [2, 2] if bool(_RNG.integers(0, 2)) else [2, 1, 1]

    if _grid_width(panel_size, sum(sizes)) > _MAX_GRID_SIDE:
        return None

    sequence = _sequence_with_split_duplicates(sizes)
    group_colors = _assign_group_colors(mode, sizes, sequence)

    used_keys: set[tuple[str, ...]] = set()
    group_masks: dict[int, np.ndarray] = {}
    for group_id in range(len(sizes)):
        mask = _sample_shape(panel_size, used_keys)
        if mask is None:
            return None
        used_keys.add(_canonical_rotation_key(mask))
        group_masks[group_id] = mask

    inp = _build_input(panel_size, sequence, group_masks, group_colors)
    expected = _expected_output(sequence, group_colors)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
