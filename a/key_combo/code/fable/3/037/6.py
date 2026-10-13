import random
from collections import Counter

import numpy as np


MASK = 6
USABLE_COLORS = [0, 1, 2, 3, 4, 5, 7, 8, 9]
TRANSFORMS = ("mirror_lr", "mirror_ud", "rot180")
GEN_ATTEMPTS = 400


def _transform(panel: np.ndarray, name: str) -> np.ndarray:
    if name == "mirror_lr":
        return np.fliplr(panel)
    if name == "mirror_ud":
        return np.flipud(panel)
    if name == "rot180":
        return np.rot90(panel, 2)
    raise ValueError(f"unknown transform: {name}")


def _transform_coord(shape: tuple[int, int], coord: tuple[int, int], name: str) -> tuple[int, int]:
    h, w = shape
    r, c = coord
    if name == "mirror_lr":
        return r, w - 1 - c
    if name == "mirror_ud":
        return h - 1 - r, c
    if name == "rot180":
        return h - 1 - r, w - 1 - c
    raise ValueError(f"unknown transform: {name}")


def _infer_color_maps(aligned_top: np.ndarray, bottom: np.ndarray):
    forward: dict[int, int] = {}
    backward: dict[int, int] = {}
    seen_pairs = 0
    for src, dst in zip(aligned_top.flat, bottom.flat):
        src = int(src)
        dst = int(dst)
        if src == MASK or dst == MASK:
            continue
        seen_pairs += 1
        if src in forward and forward[src] != dst:
            return None
        if dst in backward and backward[dst] != src:
            return None
        forward[src] = dst
        backward[dst] = src
    if seen_pairs == 0:
        return None
    return forward, backward


def _candidate_solutions(top: np.ndarray, bottom: np.ndarray):
    matches = []
    for name in TRANSFORMS:
        aligned_top = _transform(top, name)
        inferred = _infer_color_maps(aligned_top, bottom)
        if inferred is None:
            continue
        forward, backward = inferred
        matches.append((name, aligned_top, forward, backward))
    return matches


def _solve_panels(top: np.ndarray, bottom: np.ndarray):
    matches = _candidate_solutions(top, bottom)
    assert len(matches) == 1, "expected exactly one transform/permutation explanation"
    return matches[0]


def _fill_panels(top: np.ndarray, bottom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    name, aligned_top, forward, backward = _solve_panels(top, bottom)
    out_top = top.copy()
    out_bottom = bottom.copy()

    for r, c in zip(*np.where(bottom == MASK)):
        src = int(aligned_top[r, c])
        assert src != MASK, "corresponding top cell for a bottom mask must be visible"
        assert src in forward, "missing forward color mapping for bottom mask"
        out_bottom[r, c] = forward[src]

    for r, c in zip(*np.where(top == MASK)):
        rr, cc = _transform_coord(top.shape, (int(r), int(c)), name)
        dst = int(bottom[rr, cc])
        assert dst != MASK, "corresponding bottom cell for a top mask must be visible"
        assert dst in backward, "missing inverse color mapping for top mask"
        out_top[r, c] = backward[dst]

    return out_top, out_bottom


def _recolor(panel: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = panel.copy()
    for src, dst in mapping.items():
        out[panel == src] = dst
    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _make_shape_mask() -> np.ndarray:
    kind = random.choice(("rect", "lshape", "tee", "stair"))
    if kind == "rect":
        h = random.randint(1, 3)
        w = random.randint(2, 4)
        if h == 1 and w == 2:
            w = 3
        mask = np.ones((h, w), dtype=bool)
    elif kind == "lshape":
        h = random.randint(3, 4)
        w = random.randint(3, 4)
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
    elif kind == "tee":
        h = random.randint(2, 4)
        w = random.randint(3, 5)
        stem_col = w // 2
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[:, stem_col] = True
    else:
        width = random.randint(4, 5)
        mask = np.zeros((3, width), dtype=bool)
        mask[0, : width - 1] = True
        mask[1, 1:width] = True
        mask[2, 2:width] = True

    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    mask = np.rot90(mask, random.randint(0, 3))
    mask = _normalize_mask(mask)
    assert mask.sum() >= 3
    return mask


def _expanded(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_objects(h: int, w: int, bg: int, object_colors: list[int]) -> np.ndarray | None:
    panel = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    used_areas: set[int] = set()

    for color in object_colors:
        placed = False
        for _ in range(80):
            shape = _make_shape_mask()
            area = int(shape.sum())
            if area in used_areas:
                continue
            sh, sw = shape.shape
            if sh > h or sw > w:
                continue
            blocked = _expanded(occupied)
            r0 = random.randint(0, h - sh)
            c0 = random.randint(0, w - sw)
            window = blocked[r0 : r0 + sh, c0 : c0 + sw]
            if np.any(window & shape):
                continue
            panel[r0 : r0 + sh, c0 : c0 + sw][shape] = color
            occupied[r0 : r0 + sh, c0 : c0 + sw] |= shape
            used_areas.add(area)
            placed = True
            break
        if not placed:
            return None
    return panel


def _random_derangement(colors: list[int]) -> dict[int, int]:
    colors = list(colors)
    for _ in range(200):
        shuffled = colors.copy()
        random.shuffle(shuffled)
        if all(src != dst for src, dst in zip(colors, shuffled)):
            return dict(zip(colors, shuffled))
    raise RuntimeError("failed to sample a derangement")


def _safe_mask_position(
    order: list[tuple[int, int]],
    states: np.ndarray,
    remaining_visible: Counter,
    aligned_top: np.ndarray,
    bg: int,
    require_non_bg: bool,
) -> tuple[int, int] | None:
    for r, c in order:
        if states[r, c] != 0:
            continue
        color = int(aligned_top[r, c])
        if require_non_bg and color == bg:
            continue
        if remaining_visible[color] <= 1:
            continue
        return int(r), int(c)
    return None


def _apply_masks(
    top_full: np.ndarray,
    bottom_full: np.ndarray,
    transform_name: str,
    panel_colors: list[int],
    bg: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    aligned_top = _transform(top_full, transform_name)
    h, w = aligned_top.shape
    positions = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(positions)

    total_max = min(9, max(4, (h * w) // 4))
    total_masks = random.randint(4, total_max)
    top_target = random.randint(1, total_masks - 1)
    bottom_target = total_masks - top_target

    states = np.zeros((h, w), dtype=int)
    remaining_visible = Counter(map(int, aligned_top.flat))

    first_top = _safe_mask_position(positions, states, remaining_visible, aligned_top, bg, True)
    if first_top is None:
        first_top = _safe_mask_position(positions, states, remaining_visible, aligned_top, bg, False)
    if first_top is None:
        return None
    states[first_top] = 1
    remaining_visible[int(aligned_top[first_top])] -= 1

    first_bottom = _safe_mask_position(positions, states, remaining_visible, aligned_top, bg, True)
    if first_bottom is None:
        first_bottom = _safe_mask_position(positions, states, remaining_visible, aligned_top, bg, False)
    if first_bottom is None:
        return None
    states[first_bottom] = 2
    remaining_visible[int(aligned_top[first_bottom])] -= 1

    for side_code, target in ((1, top_target), (2, bottom_target)):
        already = int(np.count_nonzero(states == side_code))
        while already < target:
            pos = _safe_mask_position(positions, states, remaining_visible, aligned_top, bg, False)
            if pos is None:
                return None
            states[pos] = side_code
            remaining_visible[int(aligned_top[pos])] -= 1
            already += 1

    visible_colors = {int(aligned_top[r, c]) for r in range(h) for c in range(w) if states[r, c] == 0}
    if visible_colors != set(panel_colors):
        return None

    masked_top = top_full.copy()
    masked_bottom = bottom_full.copy()
    for r in range(h):
        for c in range(w):
            if states[r, c] == 1:
                rr, cc = _transform_coord((h, w), (r, c), transform_name)
                masked_top[rr, cc] = MASK
            elif states[r, c] == 2:
                masked_bottom[r, c] = MASK

    if not np.any(masked_top == MASK) or not np.any(masked_bottom == MASK):
        return None
    return masked_top, masked_bottom


def _assemble_grid(top: np.ndarray, separator_color: int, bottom: np.ndarray) -> np.ndarray:
    separator = np.full((1, top.shape[1]), separator_color, dtype=int)
    return np.vstack([top, separator, bottom]).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden cross-panel geometric transform and color permutation, then refill every pink masked cell.

    Input structure:
    - The grid is split by one full separator row into two equal-sized panels.
    - The lower panel is the upper panel after either a left-right mirror, an up-down mirror, or a 180-degree rotation, plus a color permutation.
    - Pink cells are masks and never appear on both sides of the same corresponding pair.

    Rule:
    - Test the three candidate geometric transforms by aligning the upper panel to the lower one.
    - The unique valid transform is the one whose visible aligned cell pairs induce a consistent one-to-one color dictionary in both directions.
    - Fill each bottom-panel pink cell by transforming the corresponding top color through the forward dictionary, and fill each top-panel pink cell by translating the visible bottom color through the inverse dictionary.

    Key insight:
    - Geometry and recoloring must be solved jointly: only the correct alignment makes the visible cell pairs collapse into a self-consistent bijection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h % 2 == 1 and h >= 3, "expected two equal panels separated by one row"

    mid = h // 2
    top = g[:mid].copy()
    bottom = g[mid + 1 :].copy()
    assert top.shape == bottom.shape and top.shape[1] == w, "panels must have equal size"

    filled_top, filled_bottom = _fill_panels(top, bottom)
    out = g.copy()
    out[:mid] = filled_top
    out[mid + 1 :] = filled_bottom
    return out


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(3, 5)
    h = random.randint(max(6, object_count + 3), 12)
    w = random.randint(max(7, object_count + 4), 14)

    sampled = random.sample(USABLE_COLORS, object_count + 2)
    panel_colors = sampled[:-1]
    separator_color = sampled[-1]
    bg = panel_colors[0]
    object_colors = panel_colors[1:]

    top_full = _place_objects(h, w, bg, object_colors)
    if top_full is None:
        return None

    if any(np.array_equal(top_full, _transform(top_full, name)) for name in TRANSFORMS):
        return None

    transform_name = random.choice(TRANSFORMS)
    color_map = _random_derangement(panel_colors)
    aligned_top = _transform(top_full, transform_name)
    bottom_full = _recolor(aligned_top, color_map)

    if np.any(aligned_top == bottom_full):
        return None

    if len(_candidate_solutions(top_full, bottom_full)) != 1:
        return None

    masked = _apply_masks(top_full, bottom_full, transform_name, panel_colors, bg)
    if masked is None:
        return None
    masked_top, masked_bottom = masked

    inp = _assemble_grid(masked_top, separator_color, masked_bottom)
    expected = _assemble_grid(top_full, separator_color, bottom_full)

    matches = _candidate_solutions(masked_top, masked_bottom)
    if len(matches) != 1 or matches[0][0] != transform_name:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
