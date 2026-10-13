from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEAM = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 600


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == BG:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords_arr = np.asarray(coords, dtype=int)
            r0, c0 = coords_arr.min(axis=0)
            r1, c1 = coords_arr.max(axis=0)
            box = (int(r0), int(c0), int(r1), int(c1))
            obj = np.full_like(grid, BG)
            obj[coords_arr[:, 0], coords_arr[:, 1]] = grid[coords_arr[:, 0], coords_arr[:, 1]]
            comps.append(
                {
                    "coords": coords_arr,
                    "bbox": box,
                    "grid": obj,
                    "area": int(coords_arr.shape[0]),
                }
            )
    return comps


def _group_consecutive(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    groups = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == SEAM))]
    groups = _group_consecutive(sep_cols)
    assert groups, "expected at least one solid grey separator column"
    panels = []
    start = 0
    for a, b in groups:
        assert start < a, "separator columns cannot touch"
        panels.append(np.asarray(grid[:, start:a], dtype=int))
        start = b + 1
    assert start < w, "missing final panel"
    panels.append(np.asarray(grid[:, start:], dtype=int))
    widths = {panel.shape[1] for panel in panels}
    assert len(widths) == 1 and next(iter(widths)) > 0, "panels must be non-empty and equal width"
    assert len(panels) >= 2, "expected multiple panels"
    return panels


def _describe_sprite(obj_grid: np.ndarray) -> dict:
    box = _bbox(obj_grid != BG)
    assert box is not None, "empty object"
    crop = _crop(obj_grid, box)
    colors = sorted({int(v) for v in crop.flatten() if v not in (BG, SEAM)})
    assert len(colors) == 1, "each object must have exactly one non-grey foreground color"
    color = colors[0]
    seam_coords = np.argwhere(crop == SEAM)
    assert seam_coords.shape[0] == 2, "each object must have exactly two grey seam pixels"
    left = np.argwhere(crop[:, 0] == SEAM)
    right = np.argwhere(crop[:, -1] == SEAM)
    assert left.shape[0] == 1, "expected one left-edge seam marker"
    assert right.shape[0] == 1, "expected one right-edge seam marker"
    filled = crop.copy()
    filled[filled == SEAM] = color
    return {
        "bbox": box,
        "crop": crop,
        "filled": filled,
        "color": color,
        "left_row": int(left[0, 0]),
        "right_row": int(right[0, 0]),
        "top": int(box[0]),
        "left": int(box[1]),
        "area": int(np.count_nonzero(crop)),
    }


def _ranked_panel_sprites(panel: np.ndarray, expected_count: int) -> list[dict]:
    objects = [_describe_sprite(item["grid"]) for item in _find_components(panel)]
    assert len(objects) == expected_count, "panel object count must match panel count"
    objects.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    for i in range(1, len(objects)):
        prev = objects[i - 1]["bbox"]
        curr = objects[i]["bbox"]
        assert prev[2] < curr[0], "panel objects must have non-overlapping vertical bbox ranges"
    return objects


def _assemble_selected(sprites: list[dict]) -> np.ndarray:
    tops = [0]
    for i in range(1, len(sprites)):
        prev = sprites[i - 1]
        curr = sprites[i]
        tops.append(tops[-1] + prev["right_row"] - curr["left_row"])
    min_top = min(tops)
    max_bottom = max(top + sprite["filled"].shape[0] - 1 for top, sprite in zip(tops, sprites))
    out_h = max_bottom - min_top + 1
    out_w = sum(int(sprite["filled"].shape[1]) for sprite in sprites)
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for top, sprite in zip(tops, sprites):
        patch = sprite["filled"]
        rr = top - min_top
        mask = patch != BG
        out[rr : rr + patch.shape[0], col : col + patch.shape[1]][mask] = patch[mask]
        col += patch.shape[1]
    return out


def _parse_grid(grid: np.ndarray) -> dict:
    panels = _split_panels(grid)
    n = len(panels)
    ranked = [_ranked_panel_sprites(panel, n) for panel in panels]
    selected = [ranked[i][i] for i in range(n)]
    return {"panels": panels, "ranked": ranked, "selected": selected}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the panel-indexed vertical-rank sprite from each panel and stitch the crops together by aligned seam markers.

    Input structure:
    - The grid has black background and is split into equal-width panels by solid grey separator columns.
    - Each panel contains exactly N 8-connected multicolor objects, where N is also the number of panels.
    - Every object has one non-black non-grey body color plus exactly two grey seam pixels: one on its left bbox edge and one on its right bbox edge.

    Rule:
    - Number the panels from left to right as 1..N, and within each panel rank the objects from top to bottom by bbox top row as 1..N.
    - Select rank k from panel k, crop each selected object to its bbox, and recolor its grey seam pixels to the object's body color.
    - Concatenate the cropped objects from left to right, keeping them flush horizontally while vertically shifting each later crop so its left seam row aligns with the previous crop's right seam row.

    Key insight:
    - The hidden selector is diagonal across the panel-by-rank matrix: panel index chooses vertical rank, and the grey edge pixels then determine the vertical offsets of the compact stitched output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    parsed = _parse_grid(g)
    return _assemble_selected(parsed["selected"])


def _random_distribution(total: int, buckets: int) -> list[int]:
    out = [0] * buckets
    for _ in range(total):
        out[random.randrange(buckets)] += 1
    return out


def _make_sprite(color: int, max_width: int) -> np.ndarray | None:
    total_cells = random.randint(5, 8)
    width = random.randint(3, min(max_width, 5))
    body_target = total_cells - 2
    if body_target > (width - 2) * 4:
        return None
    for _ in range(80):
        height = random.choices([2, 3, 4], weights=[4, 4, 2], k=1)[0]
        if body_target > (width - 2) * height:
            continue
        left_row = random.randrange(height)
        right_row = random.randrange(height)
        if width == 3:
            body = {(r, 1) for r in range(min(left_row, right_row), max(left_row, right_row) + 1)}
        else:
            if abs(left_row - right_row) > width - 3:
                continue
            body = {(left_row, 1)}
            row = left_row
            for col in range(2, width - 1):
                remaining = (width - 2) - col
                choices = [
                    nr
                    for nr in (row - 1, row, row + 1)
                    if 0 <= nr < height and abs(right_row - nr) <= remaining
                ]
                if not choices:
                    body = set()
                    break
                row = random.choice(choices)
                body.add((row, col))
            if not body or row != right_row:
                continue
        if not body or len(body) > body_target:
            continue
        while len(body) < body_target:
            candidates = []
            for r in range(height):
                for c in range(1, width - 1):
                    if (r, c) in body:
                        continue
                    if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in body):
                        candidates.append((r, c))
            if not candidates:
                break
            body.add(random.choice(candidates))
        if len(body) != body_target:
            continue
        sprite = np.full((height, width), BG, dtype=int)
        for r, c in body:
            sprite[r, c] = color
        sprite[left_row, 0] = SEAM
        sprite[right_row, width - 1] = SEAM
        return sprite
    return None


def _choose_heights(panel_sprites: list[list[np.ndarray]], n: int) -> int | None:
    min_h = max(sum(sprite.shape[0] for sprite in panel) + (n - 1) for panel in panel_sprites)
    base = 11 if n == 3 else 14
    high = min(20, max(base, min_h) + 4)
    low = max(base, min_h)
    if low > high:
        return None
    return random.randint(low, high)


def _choose_tops(heights: list[int], total_h: int) -> list[int] | None:
    required = sum(heights) + len(heights) - 1
    if required > total_h:
        return None
    slack = total_h - required
    extras = _random_distribution(slack, len(heights) + 1)
    tops = []
    row = extras[0]
    for i, height in enumerate(heights):
        tops.append(row)
        row += height
        if i < len(heights) - 1:
            row += 1 + extras[i + 1]
    return tops


def _render_panel(height: int, width: int, sprites: list[np.ndarray], tops: list[int], lefts: list[int]) -> np.ndarray:
    panel = np.full((height, width), BG, dtype=int)
    for sprite, top, left in zip(sprites, tops, lefts):
        h, w = sprite.shape
        panel[top : top + h, left : left + w][sprite != BG] = sprite[sprite != BG]
    return panel


def _wrong_strategy_outputs(parsed: dict) -> list[np.ndarray]:
    ranked = parsed["ranked"]
    strategies = []
    strategies.append([panel[0] for panel in ranked])
    strategies.append([max(panel, key=lambda item: (item["area"], -item["top"])) for panel in ranked])
    strategies.append([min(panel, key=lambda item: (item["area"], item["top"])) for panel in ranked])
    strategies.append([min(panel, key=lambda item: (item["left"], item["top"])) for panel in ranked])
    strategies.append([max(panel, key=lambda item: (item["left"], -item["top"])) for panel in ranked])
    return [_assemble_selected(choice) for choice in strategies]


def _anti_shortcut_checks(parsed: dict, expected: np.ndarray) -> bool:
    ranked = parsed["ranked"]
    selected = parsed["selected"]
    seam_diffs = [selected[i]["right_row"] - selected[i + 1]["left_row"] for i in range(len(selected) - 1)]
    if not any(diff != 0 for diff in seam_diffs):
        return False

    if not any(item["area"] < max(obj["area"] for obj in panel) for item, panel in zip(selected, ranked)):
        return False
    if not any(item["area"] > min(obj["area"] for obj in panel) for item, panel in zip(selected, ranked)):
        return False

    if not any(
        [obj["left"] for obj in panel] != sorted(obj["left"] for obj in panel)
        for panel in ranked
    ):
        return False

    distractor_colors = []
    for panel_index, panel in enumerate(ranked):
        for rank, obj in enumerate(panel):
            if rank != panel_index:
                distractor_colors.append(obj["color"])
    if not any(obj["color"] in distractor_colors for obj in selected):
        return False

    for wrong in _wrong_strategy_outputs(parsed):
        if np.array_equal(wrong, expected):
            return False
    return True


def _validate_instance(inp: np.ndarray, intended_n: int) -> np.ndarray | None:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return None
    try:
        parsed = _parse_grid(inp)
    except AssertionError:
        return None
    if len(parsed["panels"]) != intended_n:
        return None
    expected = _assemble_selected(parsed["selected"])
    if max(expected.shape) > 30:
        return None
    if not _anti_shortcut_checks(parsed, expected):
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return expected


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    panel_w = random.randint(5, 7 if n == 3 else 6)
    panel_sprites = []
    panel_colors = [random.sample(PALETTE, n) for _ in range(n)]
    for colors in panel_colors:
        sprites = []
        for color in colors:
            sprite = _make_sprite(color, panel_w)
            if sprite is None:
                return None
            sprites.append(sprite)
        panel_sprites.append(sprites)

    global_h = _choose_heights(panel_sprites, n)
    if global_h is None or global_h > 30:
        return None

    panel_tops = []
    panel_lefts = []
    for sprites in panel_sprites:
        tops = _choose_tops([sprite.shape[0] for sprite in sprites], global_h)
        if tops is None:
            return None
        lefts = [random.randint(0, panel_w - sprite.shape[1]) for sprite in sprites]
        panel_tops.append(tops)
        panel_lefts.append(lefts)

    panels = [
        _render_panel(global_h, panel_w, sprites, tops, lefts)
        for sprites, tops, lefts in zip(panel_sprites, panel_tops, panel_lefts)
    ]
    inp = np.full((global_h, n * panel_w + (n - 1)), BG, dtype=int)
    col = 0
    for i, panel in enumerate(panels):
        inp[:, col : col + panel_w] = panel
        col += panel_w
        if i < n - 1:
            inp[:, col] = SEAM
            col += 1

    return inp if _validate_instance(inp, n) is not None else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
