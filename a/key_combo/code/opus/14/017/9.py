from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
FRAME = 5
MARKER_COLORS = (1, 2, 3)
BODY_COLORS = (6, 7, 9)
MARKER_SET = set(MARKER_COLORS)

GEN_RETRIES = 400
PLACE_RETRIES = 300
LOCAL_SHAPE_RETRIES = 80


def _neighbors4(r: int, c: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yield r + dr, c + dc


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == FRAME)
    assert len(coords) > 0, "legend frame missing"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    assert r1 > r0 and c1 > c0, "legend frame must be rectangular"
    assert np.all(grid[r0, c0 : c1 + 1] == FRAME)
    assert np.all(grid[r1, c0 : c1 + 1] == FRAME)
    assert np.all(grid[r0 : r1 + 1, c0] == FRAME)
    assert np.all(grid[r0 : r1 + 1, c1] == FRAME)
    return r0, c0, r1, c1


def _legend_keys(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    key_col = None
    for c in range(c0 + 1, c1):
        col_colors = [int(grid[r, c]) for r in range(r0 + 1, r1) if int(grid[r, c]) in MARKER_SET]
        if col_colors:
            key_col = c
            break
    assert key_col is not None, "legend key column missing"
    keys = []
    for r in range(r0 + 1, r1):
        color = int(grid[r, key_col])
        if color in MARKER_SET:
            keys.append((r, color))
    assert len(keys) == 3, "legend must have exactly three key rows"
    assert {color for _, color in keys} == MARKER_SET
    return key_col, keys


def _classify_scene(grid: np.ndarray) -> dict:
    bbox = _legend_bbox(grid)
    key_col, keys = _legend_keys(grid, bbox)
    r0, c0, r1, c1 = bbox

    fg_mask = (grid != BG) & (grid != FRAME)
    comps = _find_components(fg_mask)

    sprites = []
    singles = []
    marker_pixels_outside = Counter()
    distractors_outside = Counter()
    bodies_by_marker: dict[int, set[int]] = defaultdict(set)
    markers_by_body: dict[int, set[int]] = defaultdict(set)

    for comp in comps:
        colors = Counter(int(grid[r, c]) for r, c in comp)
        inside_legend = all(r0 < r < r1 and c0 < c < c1 for r, c in comp)
        if len(comp) == 1:
            r, c = comp[0]
            color = int(grid[r, c])
            if color in MARKER_SET:
                singles.append((r, c, color, inside_legend))
                if not inside_legend:
                    distractors_outside[color] += 1
                    marker_pixels_outside[color] += 1
            continue

        marker_colors = [color for color in colors if color in MARKER_SET]
        body_colors = [color for color in colors if color not in MARKER_SET]
        if len(marker_colors) == 1 and colors[marker_colors[0]] == 1 and len(body_colors) == 1:
            marker_color = int(marker_colors[0])
            body_color = int(body_colors[0])
            sprites.append(
                {
                    "coords": comp,
                    "marker": marker_color,
                    "body": body_color,
                    "inside_legend": inside_legend,
                }
            )
            if not inside_legend:
                marker_pixels_outside[marker_color] += 1
                bodies_by_marker[marker_color].add(body_color)
                markers_by_body[body_color].add(marker_color)

    sprite_counts = Counter(sprite["marker"] for sprite in sprites if not sprite["inside_legend"])
    return {
        "bbox": bbox,
        "key_col": key_col,
        "keys": keys,
        "sprites": sprites,
        "singles": singles,
        "sprite_counts": sprite_counts,
        "distractors_outside": distractors_outside,
        "marker_pixels_outside": marker_pixels_outside,
        "bodies_by_marker": bodies_by_marker,
        "markers_by_body": markers_by_body,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Count marked composite sprites by marker color and fill those tallies into the legend rows.

    Input structure:
    - The background is black, and a gray rectangular frame near the lower-left acts as a legend box.
    - Inside the legend, one red, one blue, and one green key pixel mark the rows for the three marker classes.
    - Outside the legend are 4-connected multicolor sprites made of one body color plus one adjacent marker pixel, along with isolated red/blue/green distractor pixels.

    Rule:
    - Ignore the gray frame, then find every 4-connected non-background component.
    - A component counts as a sprite iff it has exactly one red/blue/green pixel and every other pixel is the same non-marker body color; isolated single marker pixels are distractors and do not count.
    - For each legend row, keep the key pixel and place that same color starting two cells to the right, then every other cell, once per counted sprite of that marker color.

    Key insight:
    - The same colors appear both as embedded sprite markers and as isolated noise, so the solver must classify whole components before counting by color.
    """
    g = np.asarray(grid, dtype=int)
    info = _classify_scene(g)
    r0, c0, r1, c1 = info["bbox"]
    key_col = info["key_col"]

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = BG
    for row, color in info["keys"]:
        out[row, key_col] = color
        count = int(info["sprite_counts"][color])
        for idx in range(count):
            col = key_col + 2 * (idx + 1)
            assert col < c1, "legend width is insufficient for the tally"
            out[row, col] = color
    return out


def _random_body_shape(body_size: int) -> set[tuple[int, int]]:
    for _ in range(LOCAL_SHAPE_RETRIES):
        cells = {(3, 3)}
        while len(cells) < body_size:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 7 and 0 <= nc < 7 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if len(cells) == body_size and max(rows) > min(rows) and max(cols) > min(cols):
            return cells
    raise RuntimeError("failed to build a non-degenerate body shape")


def _make_sprite_patch(body_color: int, marker_color: int) -> np.ndarray:
    body_size = random.randint(3, 6)
    body = _random_body_shape(body_size)
    frontier = []
    for r, c in body:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < 7 and 0 <= nc < 7 and (nr, nc) not in body:
                frontier.append((nr, nc))
    marker = random.choice(frontier)
    cells = set(body)
    cells.add(marker)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = body_color
    patch[marker[0] - r0, marker[1] - c0] = marker_color
    return patch


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(gh, r0 + h + padding)
    cc1 = min(gw, c0 + w + padding)
    if np.any(grid[rr0:rr1, cc0:cc1] != BG):
        return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray) -> bool:
    gh, gw = grid.shape
    h, w = patch.shape
    for _ in range(PLACE_RETRIES):
        r0 = random.randint(0, gh - h)
        c0 = random.randint(0, gw - w)
        if not _can_place_patch(grid, patch, r0, c0):
            continue
        mask = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        return True
    return False


def _draw_legend(grid: np.ndarray, key_rows: list[int], key_order: list[int]) -> tuple[int, int, int, int]:
    gh, gw = grid.shape
    outer_h = 7
    outer_w = random.randint(12, 14)
    r0 = gh - outer_h - random.randint(1, 2)
    c0 = random.randint(1, 2)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    for row, color in zip(key_rows, key_order):
        grid[r0 + row, c0 + 1] = color
    return r0, c0, r1, c1


def _sample_sprite_counts() -> dict[int, int]:
    while True:
        counts = {color: random.randint(1, 4) for color in MARKER_COLORS}
        vals = list(counts.values())
        if sum(vals) < 5 or sum(vals) > 8:
            continue
        if max(vals) < 2:
            continue
        if len(set(vals)) == 1:
            continue
        return counts


def _assign_bodies(sprite_counts: dict[int, int]) -> list[tuple[int, int]]:
    varied_markers = [color for color, count in sprite_counts.items() if count >= 2]
    varied_marker = random.choice(varied_markers)
    shared_body = random.choice(BODY_COLORS)
    shared_markers = random.sample(MARKER_COLORS, 2)

    slots = {color: [None] * sprite_counts[color] for color in MARKER_COLORS}
    slots[shared_markers[0]][0] = shared_body
    slots[shared_markers[1]][0] = shared_body

    body_a, body_b = random.sample(BODY_COLORS, 2)
    slots[varied_marker][0] = body_a
    slots[varied_marker][1] = body_b

    for color in MARKER_COLORS:
        for idx in range(len(slots[color])):
            if slots[color][idx] is None:
                slots[color][idx] = random.choice(BODY_COLORS)

    assignments = []
    for color in MARKER_COLORS:
        for body in slots[color]:
            assignments.append((color, int(body)))

    bodies_by_marker: dict[int, set[int]] = defaultdict(set)
    markers_by_body: dict[int, set[int]] = defaultdict(set)
    for marker, body in assignments:
        bodies_by_marker[marker].add(body)
        markers_by_body[body].add(marker)
    assert any(len(v) >= 2 for v in bodies_by_marker.values())
    assert any(len(v) >= 2 for v in markers_by_body.values())
    random.shuffle(assignments)
    return assignments


def _sample_distractors() -> dict[int, int]:
    while True:
        counts = {color: random.randint(0, 2) for color in MARKER_COLORS}
        if sum(counts.values()) < 2 or sum(counts.values()) > 4:
            continue
        if sum(1 for count in counts.values() if count > 0) < 2:
            continue
        return counts


def _expected_output(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    keys: list[tuple[int, int]],
    counts: dict[int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    key_col = c0 + 1
    out = grid.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = BG
    for row, color in keys:
        out[row, key_col] = color
        for idx in range(counts[color]):
            out[row, key_col + 2 * (idx + 1)] = color
    return out


def _semantic_checks(
    grid: np.ndarray,
    sprite_counts: dict[int, int],
    distractor_counts: dict[int, int],
) -> bool:
    info = _classify_scene(grid)
    parsed_counts = {color: int(info["sprite_counts"][color]) for color in MARKER_COLORS}
    parsed_distractors = {color: int(info["distractors_outside"][color]) for color in MARKER_COLORS}

    if parsed_counts != sprite_counts:
        return False
    if parsed_distractors != distractor_counts:
        return False
    if {color for _, color in info["keys"]} != MARKER_SET:
        return False
    if len(info["keys"]) != 3:
        return False
    if sum(parsed_distractors.values()) < 2:
        return False
    if sum(1 for color in MARKER_COLORS if info["marker_pixels_outside"][color] > info["sprite_counts"][color]) < 2:
        return False
    if not any(len(v) >= 2 for v in info["bodies_by_marker"].values()):
        return False
    if not any(len(v) >= 2 for v in info["markers_by_body"].values()):
        return False
    if len(set(parsed_counts.values())) == 1:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    sprite_counts = _sample_sprite_counts()
    distractor_counts = _sample_distractors()
    total_objects = sum(sprite_counts.values()) + sum(distractor_counts.values())

    min_h = min(21 + total_objects // 4, 26)
    min_w = min(22 + total_objects // 4, 28)
    h = random.randint(min_h, 26)
    w = random.randint(min_w, 28)
    grid = np.zeros((h, w), dtype=int)

    key_rows = sorted(random.sample([1, 2, 3, 4, 5], 3))
    key_order = random.sample(list(MARKER_COLORS), 3)
    bbox = _draw_legend(grid, key_rows, key_order)
    keys = [(bbox[0] + row, color) for row, color in zip(key_rows, key_order)]

    patches = []
    for marker, body in _assign_bodies(sprite_counts):
        patches.append(_make_sprite_patch(body, marker))
    random.shuffle(patches)
    patches.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)

    for patch in patches:
        if not _place_patch(grid, patch):
            return None

    distractor_patches = [np.array([[color]], dtype=int) for color, count in distractor_counts.items() for _ in range(count)]
    random.shuffle(distractor_patches)
    for patch in distractor_patches:
        if not _place_patch(grid, patch):
            return None

    if not _semantic_checks(grid, sprite_counts, distractor_counts):
        return None

    expected = _expected_output(grid, bbox, keys, sprite_counts)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
