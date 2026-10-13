from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BACKGROUND = 0
COLOR_POOL = (1, 2, 3, 7)
OBJECT_COUNT_OPTIONS = (3, 4)
ROW_GAP = 1
MIN_COL_GAP = 2
MAX_GEN_ATTEMPTS = 500
MAX_WANDER = 2

BASE_WIDTH_SEQS = (
    (2, 3, 4, 4),
    (2, 4, 5, 5),
    (2, 3, 5, 5),
    (2, 4, 6, 6),
    (2, 3, 4, 4, 5),
    (2, 3, 5, 5, 5),
    (2, 4, 5, 6, 6),
    (2, 4, 6, 6, 6),
    (2, 3, 4, 5, 5),
    (2, 3, 4, 4, 5, 5),
    (2, 3, 5, 5, 6, 6),
    (2, 4, 5, 5, 6, 6),
    (2, 4, 6, 6, 7, 7),
    (2, 4, 4, 2),
    (2, 3, 4, 3, 2),
    (2, 4, 6, 4, 2),
    (2, 3, 5, 3, 2),
    (2, 3, 4, 4, 3, 2),
    (2, 4, 5, 5, 4, 2),
    (2, 3, 5, 5, 3, 2),
    (2, 3, 4, 5, 4, 3, 2),
)


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_components(grid: np.ndarray, bg: int = BACKGROUND) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coord_arr = np.asarray(coords, dtype=int)
            r0, c0 = coord_arr.min(axis=0)
            r1, c1 = coord_arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return components


def _filled_patch(component: dict, bg: int = BACKGROUND) -> tuple[np.ndarray, int, int]:
    color = int(component["color"])
    coords = component["coords"]
    rows: dict[int, list[int]] = defaultdict(list)
    for r, c in coords:
        rows[int(r)].append(int(c))

    filled_cells = []
    for r in sorted(rows):
        row_cols = sorted(rows[r])
        assert len(row_cols) == 2, "each occupied row must contain exactly two boundary pixels"
        c0, c1 = row_cols
        filled_cells.extend((r, c) for c in range(c0, c1 + 1))

    filled = np.asarray(filled_cells, dtype=int)
    r0, c0 = filled.min(axis=0)
    r1, c1 = filled.max(axis=0)
    patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), bg, dtype=int)
    patch[filled[:, 0] - r0, filled[:, 1] - c0] = color
    return patch, int(r0), int(c0)


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, transparent: int = BACKGROUND) -> None:
    h, w = patch.shape
    block = dst[top : top + h, left : left + w]
    mask = patch != transparent
    block[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """Fill sparse row-paired outlines, sort the resulting solid shapes by area, and stack them tightly.

    Input structure:
    - The grid has black background and 3-4 monochromatic 8-connected outline objects.
    - Each outline occupies several rows, and on every occupied row it has exactly two pixels marking the row span.
    - Different objects use different colors and are spatially separated.

    Rule:
    - For each outline, fill every occupied row from its left boundary pixel to its right boundary pixel, inclusive.
    - Compute each filled shape's area, breaking ties by the outline's original top row.
    - Crop every filled shape to its minimal bounding box.
    - Build an output grid with the same width as the input, and stack the cropped shapes left-aligned from top to bottom in ascending area order, leaving one black row between neighbors.

    Key insight:
    - The sparse input hides the true sorting key: the row-span-filled area, not the outline pixel count or bounding-box size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _, w = g.shape

    components = _find_components(g, bg=BACKGROUND)
    assert 3 <= len(components) <= 4, "expected 3-4 outline objects"

    filled_shapes = []
    for component in components:
        patch, top_row, _left_col = _filled_patch(component, bg=BACKGROUND)
        filled_shapes.append(
            {
                "patch": patch,
                "area": int(np.count_nonzero(patch)),
                "top": int(top_row),
            }
        )

    ordered = sorted(filled_shapes, key=lambda item: (item["area"], item["top"]))
    out_h = sum(item["patch"].shape[0] for item in ordered) + ROW_GAP * (len(ordered) - 1)
    out = np.full((out_h, w), BACKGROUND, dtype=int)

    r = 0
    for idx, item in enumerate(ordered):
        patch = item["patch"]
        _paste(out, patch, r, 0)
        r += patch.shape[0]
        if idx != len(ordered) - 1:
            r += ROW_GAP
    return out


def _seq_area(widths: tuple[int, ...]) -> int:
    return int(sum(widths))


DECEPTIVE_PAIRS = [
    (a, b)
    for a in BASE_WIDTH_SEQS
    for b in BASE_WIDTH_SEQS
    if len(a) == len(b) and abs(_seq_area(a) - _seq_area(b)) >= 5
]


def _choose_extra_sequence(rng: random.Random, used_areas: set[int], allow_duplicate_area: bool) -> tuple[int, ...]:
    candidates = list(BASE_WIDTH_SEQS)
    rng.shuffle(candidates)
    for seq in candidates:
        area = _seq_area(seq)
        if allow_duplicate_area or area not in used_areas:
            return seq
    return candidates[0]


def _make_outline_patch(widths: tuple[int, ...], color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        widths = tuple(reversed(widths))

    bias = rng.choice((-1, 0, 1))
    for _ in range(40):
        lefts = [0]
        rights = [int(widths[0]) - 1]
        ok = True
        for idx in range(1, len(widths)):
            dw = int(widths[idx] - widths[idx - 1])
            options = []
            for dl in (-1, 0, 1):
                dr = dl + dw
                next_left = lefts[-1] + dl
                if dr in (-1, 0, 1) and -MAX_WANDER <= next_left <= MAX_WANDER:
                    options.append(dl)
            if not options:
                ok = False
                break
            min_cost = min(abs(dl - bias) for dl in options)
            preferred = [dl for dl in options if abs(dl - bias) == min_cost]
            dl = rng.choice(preferred if rng.random() < 0.7 else options)
            dr = dl + dw
            lefts.append(lefts[-1] + dl)
            rights.append(rights[-1] + dr)
        if not ok:
            continue

        shift = -min(lefts)
        lefts = [left + shift for left in lefts]
        rights = [right + shift for right in rights]
        bbox_w = max(rights) + 1
        if bbox_w > 11:
            continue

        outline = np.full((len(widths), bbox_w), BACKGROUND, dtype=int)
        filled = np.full((len(widths), bbox_w), BACKGROUND, dtype=int)
        for r, (left, right) in enumerate(zip(lefts, rights)):
            outline[r, left] = color
            outline[r, right] = color
            filled[r, left : right + 1] = color

        if rng.random() < 0.5:
            outline = outline[:, ::-1]
            filled = filled[:, ::-1]
        return outline, filled
    raise RuntimeError("failed to instantiate outline patch")


def _render_expected(shapes: list[dict], width: int) -> np.ndarray:
    ordered = sorted(shapes, key=lambda item: (item["area"], item["top"]))
    out_h = sum(item["filled"].shape[0] for item in ordered) + ROW_GAP * (len(ordered) - 1)
    out = np.full((out_h, width), BACKGROUND, dtype=int)
    r = 0
    for idx, item in enumerate(ordered):
        _paste(out, item["filled"], r, 0)
        r += item["filled"].shape[0]
        if idx != len(ordered) - 1:
            r += ROW_GAP
    return out


def _has_deceptive_pair(shapes: list[dict]) -> bool:
    for i, a in enumerate(shapes):
        for b in shapes[i + 1 :]:
            boundary_gap = abs(int(a["boundary_pixels"]) - int(b["boundary_pixels"]))
            area_gap = abs(int(a["area"]) - int(b["area"]))
            if boundary_gap <= 2 and area_gap >= 5:
                return True
    return False


def _choose_shapes(rng: random.Random) -> list[dict] | None:
    n_objects = rng.choice(OBJECT_COUNT_OPTIONS)
    deceptive_a, deceptive_b = rng.choice(DECEPTIVE_PAIRS)
    sequences = [deceptive_a, deceptive_b]

    include_tie = n_objects == 4 and rng.random() < 0.55
    used_areas = {_seq_area(deceptive_a), _seq_area(deceptive_b)}

    if include_tie:
        tie_seq = _choose_extra_sequence(rng, used_areas, allow_duplicate_area=False)
        sequences.extend([tie_seq, tie_seq])
    else:
        while len(sequences) < n_objects:
            seq = _choose_extra_sequence(rng, used_areas, allow_duplicate_area=False)
            sequences.append(seq)
            used_areas.add(_seq_area(seq))

    rng.shuffle(sequences)
    if sum(len(seq) for seq in sequences) + ROW_GAP * (len(sequences) - 1) > 30:
        return None

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    shapes = []
    for seq, color in zip(sequences, colors):
        try:
            outline, filled = _make_outline_patch(seq, color, rng)
        except RuntimeError:
            return None
        shapes.append(
            {
                "outline": outline,
                "filled": filled,
                "color": color,
                "area": int(np.count_nonzero(filled)),
                "height": int(filled.shape[0]),
                "width": int(filled.shape[1]),
                "boundary_pixels": 2 * int(filled.shape[0]),
            }
        )

    if not _has_deceptive_pair(shapes):
        return None
    return shapes


def _choose_canvas_and_positions(shapes: list[dict], rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    widths = [shape["width"] for shape in shapes]
    required_w = sum(widths) + MIN_COL_GAP * (len(shapes) - 1)
    if required_w > 22:
        return None

    max_h = max(shape["height"] for shape in shapes)
    canvas_h = rng.randint(max(12, max_h + 2), min(18, max_h + 8))
    canvas_w = rng.randint(max(12, required_w), min(22, required_w + 4))
    if canvas_h > 30 or canvas_w > 30:
        return None

    extra = canvas_w - required_w
    left_margin = rng.randint(0, extra)
    extra -= left_margin
    gaps = [MIN_COL_GAP] * (len(shapes) - 1)
    for _ in range(extra):
        gaps[rng.randrange(len(gaps))] += 1

    order = list(range(len(shapes)))
    rng.shuffle(order)
    x_positions = [0] * len(shapes)
    cursor = left_margin
    for idx, shape_idx in enumerate(order):
        x_positions[shape_idx] = cursor
        cursor += shapes[shape_idx]["width"]
        if idx < len(order) - 1:
            cursor += gaps[idx]

    top_limits = [canvas_h - shape["height"] for shape in shapes]
    if any(limit < 0 for limit in top_limits):
        return None

    available_tops = set(range(canvas_h))
    tops = [None] * len(shapes)
    sorted_indices = sorted(range(len(shapes)), key=lambda idx: top_limits[idx])
    for idx in sorted_indices:
        choices = [top for top in sorted(available_tops) if top <= top_limits[idx]]
        if not choices:
            return None
        top = rng.choice(choices)
        tops[idx] = top
        available_tops.remove(top)

    if len({(shape["area"], tops[idx]) for idx, shape in enumerate(shapes)}) != len(shapes):
        return None

    inp = np.full((canvas_h, canvas_w), BACKGROUND, dtype=int)
    placed = []
    for idx, shape in enumerate(shapes):
        top = int(tops[idx])
        left = int(x_positions[idx])
        _paste(inp, shape["outline"], top, left)
        placed.append({**shape, "top": top, "left": left})

    expected = _render_expected(placed, canvas_w)
    return inp, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shapes = _choose_shapes(rng)
    if shapes is None:
        return None

    rendered = _choose_canvas_and_positions(shapes, rng)
    if rendered is None:
        return None
    inp, expected = rendered

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
