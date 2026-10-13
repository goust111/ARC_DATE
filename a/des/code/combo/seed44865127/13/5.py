from __future__ import annotations

import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _reflect(kind: str, axis_sum: int, r: int, c: int) -> tuple[int, int]:
    if kind == "vertical":
        return r, axis_sum - c
    return axis_sum - r, c


def _evaluate_axis(
    grid: np.ndarray,
    color: int,
    bg: int,
    pts: list[tuple[int, int]],
    kind: str,
    axis_sum: int,
):
    h, w = grid.shape
    claims: set[tuple[int, int]] = set()
    same_pairs = 0
    axis_cells = 0

    for r, c in pts:
        rr, cc = _reflect(kind, axis_sum, r, c)
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        if rr == r and cc == c:
            axis_cells += 1
            continue

        value = int(grid[rr, cc])
        if value == bg:
            return None
        if value == color:
            same_pairs += 1
        else:
            claims.add((rr, cc))

    if same_pairs + axis_cells < 2:
        return None

    union = pts + list(claims)
    rows = [r for r, _c in union]
    cols = [c for _r, c in union]
    extent = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    score = (same_pairs, axis_cells, -extent, -len(claims))
    return {"kind": kind, "axis_sum": axis_sum, "claims": claims, "score": score}


def _best_axis(grid: np.ndarray, color: int, bg: int):
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
    if len(pts) < 2:
        return None

    h, w = grid.shape
    candidates = []
    for axis_sum in range(2 * w - 1):
        cand = _evaluate_axis(grid, color, bg, pts, "vertical", axis_sum)
        if cand is not None:
            candidates.append(cand)
    for axis_sum in range(2 * h - 1):
        cand = _evaluate_axis(grid, color, bg, pts, "horizontal", axis_sum)
        if cand is not None:
            candidates.append(cand)

    if not candidates:
        return None

    best = max(candidates, key=lambda cand: cand["score"])
    ties = [cand for cand in candidates if cand["score"] == best["score"]]
    if len(ties) > 1:
        claim_sets = [cand["claims"] for cand in ties]
        if any(claims != claim_sets[0] for claims in claim_sets[1:]):
            if claim_sets[0]:
                return None
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair foreground objects whose missing pixels can be inferred from horizontal or vertical symmetry.

    Input structure
    - The background is the dominant color.
    - Some foreground colors form objects that are symmetric about a horizontal or vertical axis.
    - A symmetric object may be partially hidden by pixels of another foreground color.

    Rule
    - For each foreground color, test horizontal and vertical reflection axes.
    - A valid axis must map every visible pixel either to the same color or to another foreground color.
    - If the mirror location is another foreground color, repaint that location with the current color.
    - Colors with no supported symmetric completion stay unchanged.

    Key insight
    The missing cells are not grown arbitrarily: they are exactly the occupied mirror positions required by a
    supported reflection axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    proposals: dict[tuple[int, int], list[tuple[tuple[int, int, int, int], int]]] = {}

    for color in colors:
        best = _best_axis(g, color, bg)
        if best is None:
            continue
        for pt in best["claims"]:
            proposals.setdefault(pt, []).append((best["score"], color))

    out = g.copy()
    for pt, options in proposals.items():
        if len(options) == 1:
            out[pt] = options[0][1]
            continue

        best_score = max(score for score, _color in options)
        winners = [color for score, color in options if score == best_score]
        if len(set(winners)) == 1:
            out[pt] = winners[0]

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (12, 18)
GRID_WIDTH_RANGE = (12, 18)
PALETTE = list(range(10))
MAX_ATTEMPTS = 400

HIDDEN_COUNT_CHOICES = (1, 2)
EXTRA_VISIBLE_PROB = 0.5
AXIS_KINDS = ("vertical", "horizontal")
ODD_SPANS = (5, 7, 9)
OTHER_SPAN_RANGE = (4, 7)
PROFILE_TRIES = 80
PLACEMENT_TRIES = 80


def _make_vertical_profile(height: int, half: int) -> list[int] | None:
    for _ in range(PROFILE_TRIES):
        widths = [random.randint(0, half)]
        for _idx in range(1, height):
            widths.append(max(0, min(half, widths[-1] + random.choice((-1, 0, 1)))))
        if max(widths) == 0:
            continue
        if len(set(widths)) < 2:
            continue
        if widths == widths[::-1]:
            continue
        if sum(2 * width + 1 for width in widths) < 9:
            continue
        return widths
    return None


def _make_horizontal_profile(width: int, half: int) -> list[int] | None:
    for _ in range(PROFILE_TRIES):
        heights = [random.randint(0, half)]
        for _idx in range(1, width):
            heights.append(max(0, min(half, heights[-1] + random.choice((-1, 0, 1)))))
        if max(heights) == 0:
            continue
        if len(set(heights)) < 2:
            continue
        if heights == heights[::-1]:
            continue
        if sum(2 * height + 1 for height in heights) < 9:
            continue
        return heights
    return None


def _make_object(kind: str):
    if kind == "vertical":
        height = random.randint(*OTHER_SPAN_RANGE)
        width = random.choice(ODD_SPANS)
        half = width // 2
        profile = _make_vertical_profile(height, half)
        if profile is None:
            return None
        mask = np.zeros((height, width), dtype=bool)
        axis = half
        for r, radius in enumerate(profile):
            mask[r, axis - radius : axis + radius + 1] = True
        return {"kind": kind, "mask": mask, "profile": profile}

    width = random.randint(*OTHER_SPAN_RANGE)
    height = random.choice(ODD_SPANS)
    half = height // 2
    profile = _make_horizontal_profile(width, half)
    if profile is None:
        return None
    mask = np.zeros((height, width), dtype=bool)
    axis = half
    for c, radius in enumerate(profile):
        mask[axis - radius : axis + radius + 1, c] = True
    return {"kind": kind, "mask": mask, "profile": profile}


def _mark_reserved(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(reserved.shape[0], r0 + h + 1)
    cc1 = min(reserved.shape[1], c0 + w + 1)
    reserved[rr0:rr1, cc0:cc1] = True


def _place_object(reserved: np.ndarray, obj: dict) -> tuple[int, int] | None:
    h, w = obj["mask"].shape
    grid_h, grid_w = reserved.shape
    if h > grid_h or w > grid_w:
        return None

    choices = [(r0, c0) for r0 in range(grid_h - h + 1) for c0 in range(grid_w - w + 1)]
    random.shuffle(choices)
    for r0, c0 in choices[:PLACEMENT_TRIES]:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(grid_h, r0 + h + 1)
        cc1 = min(grid_w, c0 + w + 1)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        _mark_reserved(reserved, r0, c0, h, w)
        return r0, c0
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _choose_occluder_cells(obj: dict, r0: int, c0: int) -> list[tuple[int, int]] | None:
    mask = obj["mask"]
    if obj["kind"] == "vertical":
        height, width = mask.shape
        axis = width // 2
        side = random.choice((-1, 1))
        offsets = list(range(1, axis + 1))
        random.shuffle(offsets)
        for offset in offsets:
            valid_rows = [r for r, radius in enumerate(obj["profile"]) if radius >= offset]
            if not valid_rows:
                continue
            runs = []
            start = valid_rows[0]
            prev = valid_rows[0]
            for row in valid_rows[1:]:
                if row == prev + 1:
                    prev = row
                    continue
                runs.append((start, prev))
                start = prev = row
            runs.append((start, prev))
            random.shuffle(runs)
            for lo, hi in runs:
                run_len = hi - lo + 1
                if run_len < 2:
                    continue
                bar_len = random.randint(2, run_len)
                start_row = random.randint(lo, hi - bar_len + 1)
                col = c0 + axis + side * offset
                cells = [(r0 + r, col) for r in range(start_row, start_row + bar_len)]
                return cells
        return None

    height, width = mask.shape
    axis = height // 2
    side = random.choice((-1, 1))
    offsets = list(range(1, axis + 1))
    random.shuffle(offsets)
    for offset in offsets:
        valid_cols = [c for c, radius in enumerate(obj["profile"]) if radius >= offset]
        if not valid_cols:
            continue
        runs = []
        start = valid_cols[0]
        prev = valid_cols[0]
        for col in valid_cols[1:]:
            if col == prev + 1:
                prev = col
                continue
            runs.append((start, prev))
            start = prev = col
        runs.append((start, prev))
        random.shuffle(runs)
        for lo, hi in runs:
            run_len = hi - lo + 1
            if run_len < 2:
                continue
            bar_len = random.randint(2, run_len)
            start_col = random.randint(lo, hi - bar_len + 1)
            row = r0 + axis + side * offset
            cells = [(row, c0 + c) for c in range(start_col, start_col + bar_len)]
            return cells
    return None


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bg = random.choice(PALETTE)

    hidden_count = random.choice(HIDDEN_COUNT_CHOICES)
    extra_visible = 1 if random.random() < EXTRA_VISIBLE_PROB else 0
    total_colors = hidden_count * 2 + extra_visible
    if total_colors > len(PALETTE) - 1:
        return None

    palette = [color for color in PALETTE if color != bg]
    chosen = random.sample(palette, total_colors)
    hidden_colors = chosen[:hidden_count]
    occluder_colors = chosen[hidden_count : 2 * hidden_count]
    visible_colors = chosen[2 * hidden_count :]

    reserved = np.zeros((height, width), dtype=bool)
    base = np.full((height, width), bg, dtype=int)
    inp = np.full((height, width), bg, dtype=int)

    hidden_objects = []
    for color in hidden_colors:
        kind = random.choice(AXIS_KINDS)
        obj = _make_object(kind)
        if obj is None:
            return None
        placement = _place_object(reserved, obj)
        if placement is None:
            return None
        r0, c0 = placement
        _paint_mask(base, obj["mask"], r0, c0, color)
        hidden_objects.append({"color": color, "r0": r0, "c0": c0, **obj})

    for color in visible_colors:
        kind = random.choice(AXIS_KINDS)
        obj = _make_object(kind)
        if obj is None:
            return None
        placement = _place_object(reserved, obj)
        if placement is None:
            return None
        r0, c0 = placement
        _paint_mask(base, obj["mask"], r0, c0, color)

    inp[:] = base
    changed = 0
    for obj, occ_color in zip(hidden_objects, occluder_colors):
        cells = _choose_occluder_cells(obj, obj["r0"], obj["c0"])
        if cells is None:
            return None
        for r, c in cells:
            if int(base[r, c]) != int(obj["color"]):
                return None
        for r, c in cells:
            inp[r, c] = occ_color
        changed += len(cells)

    if changed == 0 or np.array_equal(inp, base):
        return None
    if not np.array_equal(main(inp), base):
        return None
    return inp, base


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        scene = _build_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid symmetric-occlusion scene")
