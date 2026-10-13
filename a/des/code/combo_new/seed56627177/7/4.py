from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
ORANGE = 7
WINDOW_COLORS = (1, 2, 3, 6, 8, 9)

MAX_GRID_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400
WINDOW_COUNT_RANGE = (1, 4)
TARGET_SIDE_RANGE = (2, 4)
OFFSET_EXTRA_RANGE = (1, 4)
WINDOW_PADDING = 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _quadrant_specs(height: int, width: int, cross_r: int, cross_c: int) -> dict[str, dict]:
    specs = {
        "TL": {"rows": (0, cross_r), "cols": (0, cross_c), "v": "top", "h": "left"},
        "TR": {"rows": (0, cross_r), "cols": (cross_c + 1, width), "v": "top", "h": "right"},
        "BL": {"rows": (cross_r + 1, height), "cols": (0, cross_c), "v": "bottom", "h": "left"},
        "BR": {
            "rows": (cross_r + 1, height),
            "cols": (cross_c + 1, width),
            "v": "bottom",
            "h": "right",
        },
    }
    for spec in specs.values():
        spec["height"] = int(spec["rows"][1] - spec["rows"][0])
        spec["width"] = int(spec["cols"][1] - spec["cols"][0])
    return specs


def _target_quadrant(specs: dict[str, dict]) -> str:
    areas = {name: spec["height"] * spec["width"] for name, spec in specs.items()}
    smallest = min(areas.values())
    targets = [name for name, area in areas.items() if area == smallest]
    assert len(targets) == 1, "cross offset must define a unique smallest quadrant"
    return targets[0]


def _extract_fold_part(box: np.ndarray, specs: dict[str, dict], source: str, target: str) -> np.ndarray:
    src = specs[source]
    tgt = specs[target]
    part = box[src["rows"][0] : src["rows"][1], src["cols"][0] : src["cols"][1]]

    if src["v"] != tgt["v"]:
        if tgt["v"] == "top":
            part = part[: tgt["height"], :]
        else:
            part = part[-tgt["height"] :, :]
        part = np.flipud(part)

    if src["h"] != tgt["h"]:
        if tgt["h"] == "left":
            part = part[:, : tgt["width"]]
        else:
            part = part[:, -tgt["width"] :]
        part = np.fliplr(part)

    assert part.shape == (tgt["height"], tgt["width"])
    return part


def _solve_window(box: np.ndarray) -> np.ndarray:
    colors = set(map(int, np.unique(box)))
    assert BLACK not in colors, "window bbox must be fully filled"
    assert GREY in colors, "window must contain a grey cross"
    assert colors - {GREY, YELLOW}, "window must contain a base color"

    base_colors = colors - {GREY, YELLOW}
    assert len(base_colors) == 1, "window must use exactly one base color"
    base = next(iter(base_colors))
    assert np.all((box == base) | (box == GREY) | (box == YELLOW))

    grey_rows = [r for r in range(box.shape[0]) if np.all(box[r, :] == GREY)]
    grey_cols = [c for c in range(box.shape[1]) if np.all(box[:, c] == GREY)]
    assert len(grey_rows) == 1 and len(grey_cols) == 1, "window needs one horizontal and one vertical grey divider"
    cross_r = grey_rows[0]
    cross_c = grey_cols[0]

    specs = _quadrant_specs(int(box.shape[0]), int(box.shape[1]), cross_r, cross_c)
    assert all(spec["height"] > 0 and spec["width"] > 0 for spec in specs.values())
    target = _target_quadrant(specs)
    target_spec = specs[target]

    yellow_counts = np.zeros((target_spec["height"], target_spec["width"]), dtype=int)
    for source in ("TL", "TR", "BL", "BR"):
        folded = _extract_fold_part(box, specs, source, target)
        yellow_counts += folded == YELLOW

    out = np.full((target_spec["height"], target_spec["width"]), base, dtype=int)
    out[yellow_counts == 1] = YELLOW
    out[yellow_counts >= 2] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each offset-cross window into its smallest quadrant and overlay the marker colors.

    Input structure:
    - The background is black.
    - Each foreground object is a filled rectangle of one base color, forming a window on the black canvas.
    - Every window contains one full grey row and one full grey column that cross inside the rectangle, splitting it into four quadrants.
    - Some quadrant cells are recolored yellow as marker pixels, while the rest stay in the window's base color.

    Rule:
    - Find each rectangular window and locate its unique grey cross.
    - Identify the smallest quadrant, then fold the other three quadrants onto that quadrant by reflecting across the grey row and/or grey column, clipping to the smallest quadrant's size.
    - For each folded cell, keep the base color if no yellow marker lands there, keep yellow if exactly one marker lands there, and turn it orange if multiple markers overlap.
    - Replace the original window by clearing it to black and writing only the folded small rectangle at the window's top-left corner.

    Key insight:
    - The off-center grey cross determines both which quadrant is the folding target and which source strips survive the clipping after reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.full_like(g, BLACK)
    windows = _components(g != BLACK)
    assert windows, "input must contain at least one window"

    for coords in windows:
        r0, c0, r1, c1 = _bbox(coords)
        box = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(box != BLACK), "each foreground component must be a filled rectangle"
        folded = _solve_window(box)
        fh, fw = folded.shape
        out[r0 : r0 + fh, c0 : c0 + fw] = folded

    return out


def _inverse_fold_coord(specs: dict[str, dict], source: str, target: str, tr: int, tc: int) -> tuple[int, int]:
    src = specs[source]
    tgt = specs[target]

    if src["v"] == tgt["v"]:
        row = src["rows"][0] + tr
    elif src["v"] == "bottom" and tgt["v"] == "top":
        row = src["rows"][0] + (tgt["height"] - 1 - tr)
    else:
        row = src["rows"][1] - 1 - tr

    if src["h"] == tgt["h"]:
        col = src["cols"][0] + tc
    elif src["h"] == "right" and tgt["h"] == "left":
        col = src["cols"][0] + (tgt["width"] - 1 - tc)
    else:
        col = src["cols"][1] - 1 - tc

    return int(row), int(col)


def _make_window(base_color: int, target: str, orange_mode: bool) -> tuple[np.ndarray, np.ndarray]:
    target_h = random.randint(*TARGET_SIDE_RANGE)
    target_w = random.randint(*TARGET_SIDE_RANGE)

    if target[0] == "T":
        top_h = target_h
        bottom_h = target_h + random.randint(*OFFSET_EXTRA_RANGE)
    else:
        top_h = target_h + random.randint(*OFFSET_EXTRA_RANGE)
        bottom_h = target_h

    if target[1] == "L":
        left_w = target_w
        right_w = target_w + random.randint(*OFFSET_EXTRA_RANGE)
    else:
        left_w = target_w + random.randint(*OFFSET_EXTRA_RANGE)
        right_w = target_w

    height = top_h + bottom_h + 1
    width = left_w + right_w + 1
    cross_r = top_h
    cross_c = left_w

    box = np.full((height, width), base_color, dtype=int)
    box[cross_r, :] = GREY
    box[:, cross_c] = GREY

    specs = _quadrant_specs(height, width, cross_r, cross_c)
    target_spec = specs[target]
    counts = np.zeros((target_spec["height"], target_spec["width"]), dtype=int)

    cells = [(r, c) for r in range(target_spec["height"]) for c in range(target_spec["width"])]
    random.shuffle(cells)
    max_marks = min(len(cells), 1 + len(cells) // 2)
    mark_min = 2 if orange_mode else 1
    mark_count = random.randint(mark_min, max_marks)
    selected = cells[:mark_count]

    layer_plan: list[tuple[int, int, int]] = []
    if orange_mode:
        assert len(selected) >= 2
        layer_plan.append((selected[0][0], selected[0][1], random.randint(2, 4)))
        extra_yellow = selected[1]
        layer_plan.append((extra_yellow[0], extra_yellow[1], 1))
        for tr, tc in selected[2:]:
            layer_plan.append((tr, tc, random.randint(1, 4)))
    else:
        for tr, tc in selected:
            layer_plan.append((tr, tc, 1))

    applied: set[tuple[int, int]] = set()
    for tr, tc, layers in layer_plan:
        if (tr, tc) in applied:
            continue
        applied.add((tr, tc))
        counts[tr, tc] = layers
        sources = random.sample(("TL", "TR", "BL", "BR"), layers)
        for source in sources:
            sr, sc = _inverse_fold_coord(specs, source, target, tr, tc)
            box[sr, sc] = YELLOW

    expected = np.full((target_spec["height"], target_spec["width"]), base_color, dtype=int)
    expected[counts == 1] = YELLOW
    expected[counts >= 2] = ORANGE

    assert np.array_equal(_solve_window(box), expected)
    return box, expected


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - WINDOW_PADDING)
    c0 = max(0, left - WINDOW_PADDING)
    r1 = min(h, top + height + WINDOW_PADDING)
    c1 = min(w, left + width + WINDOW_PADDING)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _paste(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    height, width = patch.shape
    canvas[top : top + height, left : left + width] = patch


def _sample_canvas_dims(windows: list[np.ndarray]) -> tuple[int, int]:
    max_h = max(int(window.shape[0]) for window in windows)
    max_w = max(int(window.shape[1]) for window in windows)
    total_area = sum(int(window.shape[0] * window.shape[1]) for window in windows)
    min_side = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_area * 1.8))))
    min_side = max(12, min_side)
    high = MAX_GRID_SIZE
    low = min(min_side, high)
    return random.randint(low, high), random.randint(low, high)


def _place_windows(windows: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    raw_windows = [window for window, _expected in windows]
    grid_h, grid_w = _sample_canvas_dims(raw_windows)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    inp = np.full((grid_h, grid_w), BLACK, dtype=int)
    expected = np.full((grid_h, grid_w), BLACK, dtype=int)

    order = sorted(range(len(windows)), key=lambda i: windows[i][0].shape[0] * windows[i][0].shape[1], reverse=True)
    placements: list[tuple[int, int, int]] = []
    for idx in order:
        window, patch = windows[idx]
        wh, ww = map(int, window.shape)
        choices = [
            (r, c)
            for r in range(grid_h - wh + 1)
            for c in range(grid_w - ww + 1)
            if _can_place(occupied, r, c, wh, ww)
        ]
        if not choices:
            return None
        top, left = random.choice(choices)
        placements.append((idx, top, left))
        r0 = max(0, top - WINDOW_PADDING)
        c0 = max(0, left - WINDOW_PADDING)
        r1 = min(grid_h, top + wh + WINDOW_PADDING)
        c1 = min(grid_w, left + ww + WINDOW_PADDING)
        occupied[r0:r1, c0:c1] = True
        _paste(inp, window, top, left)
        _paste(expected, patch, top, left)

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    n_windows = random.randint(*WINDOW_COUNT_RANGE)
    base_colors = random.sample(WINDOW_COLORS, n_windows)
    targets = [random.choice(("TL", "TR", "BL", "BR")) for _ in range(n_windows)]

    require_orange = random.choice([False, True])
    orange_idx = random.randrange(n_windows) if require_orange else -1

    windows: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_windows):
        window, expected = _make_window(base_colors[i], targets[i], orange_mode=(i == orange_idx))
        windows.append((window, expected))

    placed = _place_windows(windows)
    if placed is None:
        return None
    inp, expected = placed

    if inp.ndim != 2 or max(inp.shape) > MAX_GRID_SIZE or max(expected.shape) > MAX_GRID_SIZE:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid offset-window folding input")
