import random
from collections import defaultdict
from itertools import permutations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
PRIMARY_COLORS = (BLUE, RED, YELLOW)
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_tower_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return components


def _decode_tower(component: dict) -> dict:
    color = int(component["color"])
    r0, c0, r1, c1 = component["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert int(component["area"]) == 11, "each tower must have 11 pixels"

    mask = np.zeros((height, width), dtype=bool)
    for r, c in component["coords"]:
        mask[r - r0, c - c0] = True

    if (height, width) == (3, 5):
        if mask[:, :3].all() and mask[1, 3] and mask[1, 4] and int(mask.sum()) == 11:
            return {"color": color, "direction": "R", "tip": (r0 + 1, c0 + 4)}
        if mask[:, 2:].all() and mask[1, 0] and mask[1, 1] and int(mask.sum()) == 11:
            return {"color": color, "direction": "L", "tip": (r0 + 1, c0)}
    if (height, width) == (5, 3):
        if mask[:3, :].all() and mask[3, 1] and mask[4, 1] and int(mask.sum()) == 11:
            return {"color": color, "direction": "D", "tip": (r0 + 4, c0 + 1)}
        if mask[2:, :].all() and mask[0, 1] and mask[1, 1] and int(mask.sum()) == 11:
            return {"color": color, "direction": "U", "tip": (r0, c0 + 1)}
    raise AssertionError("foreground component is not a valid signal tower")


def _best_matching(sources: list[dict], targets: list[dict], gap_fn) -> list[tuple[dict, dict]]:
    assert len(sources) == len(targets), "opposite directions must have equal counts on each axis lane"
    if not sources:
        return []

    best = None
    for perm in permutations(targets):
        gaps = []
        for src, dst in zip(sources, perm):
            gap = gap_fn(src, dst)
            if gap < 1:
                break
            gaps.append(gap)
        else:
            target_signature = tuple(dst["tip"] for dst in perm)
            candidate = (sum(gaps), tuple(gaps), target_signature, perm)
            if best is None or candidate[:-1] < best[:-1]:
                best = candidate

    assert best is not None, "expected a valid nearest pairing"
    return list(zip(sources, best[-1]))


def _pair_towers(towers: list[dict]) -> list[tuple[dict, dict]]:
    left_by_row: dict[int, list[dict]] = defaultdict(list)
    right_by_row: dict[int, list[dict]] = defaultdict(list)
    up_by_col: dict[int, list[dict]] = defaultdict(list)
    down_by_col: dict[int, list[dict]] = defaultdict(list)

    for tower in towers:
        r, c = tower["tip"]
        if tower["direction"] == "L":
            left_by_row[r].append(tower)
        elif tower["direction"] == "R":
            right_by_row[r].append(tower)
        elif tower["direction"] == "U":
            up_by_col[c].append(tower)
        else:
            assert tower["direction"] == "D"
            down_by_col[c].append(tower)

    pairs = []

    rows = sorted(set(left_by_row) | set(right_by_row))
    for row in rows:
        lefts = sorted(left_by_row.get(row, []), key=lambda tower: tower["tip"][1])
        rights = sorted(right_by_row.get(row, []), key=lambda tower: tower["tip"][1])
        pairs.extend(_best_matching(rights, lefts, lambda src, dst: dst["tip"][1] - src["tip"][1] - 1))

    cols = sorted(set(up_by_col) | set(down_by_col))
    for col in cols:
        ups = sorted(up_by_col.get(col, []), key=lambda tower: tower["tip"][0])
        downs = sorted(down_by_col.get(col, []), key=lambda tower: tower["tip"][0])
        pairs.extend(_best_matching(downs, ups, lambda src, dst: dst["tip"][0] - src["tip"][0] - 1))

    used = {id(tower) for pair in pairs for tower in pair}
    assert len(used) == len(towers), "every tower must belong to exactly one pair"
    return pairs


def _mix_line_color(color_a: int, color_b: int) -> int:
    if color_a == color_b:
        return color_a
    missing = [color for color in PRIMARY_COLORS if color not in {color_a, color_b}]
    assert len(missing) == 1, "different tower colors must determine the third primary color"
    return missing[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair opposite-facing signal towers and connect each pair with a mixed-color straight segment.

    Input structure:
    - The grid has a black background and several separated monochrome signal towers.
    - Each tower is a solid 3x3 square plus a same-color 2-pixel antenna extending from the midpoint of one side.
    - Antennas point left, right, up, or down, and only red, blue, and yellow are used as tower colors.

    Rule:
    - Keep every tower unchanged.
    - Pair right-pointing towers with left-pointing towers on the same row, and pair down-pointing towers with up-pointing towers on the same column.
    - When a row or column has multiple opposite-facing candidates, choose the pairing with the smallest antenna-tip gap.
    - Draw the open horizontal or vertical segment between the two antenna tips.
    - If the paired towers share a color, use that color for the segment; otherwise use the third primary color not present on the towers.

    Key insight:
    - The transformation is determined jointly by directional opposition and by a visible three-color complement rule for the connector.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    towers = [_decode_tower(component) for component in _find_tower_components(g, bg)]
    assert towers and len(towers) % 2 == 0, "expected an even number of towers"
    assert all(tower["color"] in PRIMARY_COLORS for tower in towers), "unexpected tower color"

    out = g.copy()
    for first, second in _pair_towers(towers):
        line_color = _mix_line_color(int(first["color"]), int(second["color"]))
        r1, c1 = first["tip"]
        r2, c2 = second["tip"]
        if r1 == r2:
            left = min(c1, c2)
            right = max(c1, c2)
            assert np.all(out[r1, left + 1 : right] == bg), "horizontal path must be clear"
            out[r1, left + 1 : right] = line_color
        else:
            assert c1 == c2, "vertical pairs must share a column"
            top = min(r1, r2)
            bottom = max(r1, r2)
            assert np.all(out[top + 1 : bottom, c1] == bg), "vertical path must be clear"
            out[top + 1 : bottom, c1] = line_color
    return out


def _paint_tower(grid: np.ndarray, center_r: int, center_c: int, direction: str, color: int) -> None:
    grid[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = color
    if direction == "R":
        grid[center_r, center_c + 2 : center_c + 4] = color
    elif direction == "L":
        grid[center_r, center_c - 3 : center_c - 1] = color
    elif direction == "D":
        grid[center_r + 2 : center_r + 4, center_c] = color
    else:
        assert direction == "U"
        grid[center_r - 3 : center_r - 1, center_c] = color


def _pair_colors(palette: tuple[int, int], mixed: bool) -> tuple[int, int]:
    if mixed:
        colors = list(palette)
        random.shuffle(colors)
        return colors[0], colors[1]
    color = random.choice(palette)
    return color, color


def _place_horizontal_pair(grid: np.ndarray, reserved: np.ndarray, colors: tuple[int, int]) -> bool:
    h, w = grid.shape
    if h < 6 or w < 13:
        return False
    for _ in range(PLACEMENT_RETRIES):
        row = random.randint(2, h - 3)
        left_center = random.randint(2, w - 11)
        max_sep = min(w - 3 - left_center, 15)
        if max_sep < 8:
            continue
        sep = random.randint(8, max_sep)
        right_center = left_center + sep

        r0 = row - 2
        r1 = row + 2
        c0 = left_center - 2
        c1 = right_center + 2
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue

        _paint_tower(grid, row, left_center, "R", colors[0])
        _paint_tower(grid, row, right_center, "L", colors[1])
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        return True
    return False


def _place_vertical_pair(grid: np.ndarray, reserved: np.ndarray, colors: tuple[int, int]) -> bool:
    h, w = grid.shape
    if h < 13 or w < 6:
        return False
    for _ in range(PLACEMENT_RETRIES):
        col = random.randint(2, w - 3)
        top_center = random.randint(2, h - 11)
        max_sep = min(h - 3 - top_center, 15)
        if max_sep < 8:
            continue
        sep = random.randint(8, max_sep)
        bottom_center = top_center + sep

        r0 = top_center - 2
        r1 = bottom_center + 2
        c0 = col - 2
        c1 = col + 2
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue

        _paint_tower(grid, top_center, col, "D", colors[0])
        _paint_tower(grid, bottom_center, col, "U", colors[1])
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    mixed_count = pair_count // 2
    if pair_count % 2 == 1 and random.random() < 0.5:
        mixed_count += 1

    kinds = [random.choice(("H", "V")) for _ in range(pair_count)]
    mixed_flags = [True] * mixed_count + [False] * (pair_count - mixed_count)
    random.shuffle(mixed_flags)
    specs = list(zip(kinds, mixed_flags))
    random.shuffle(specs)

    palette = tuple(random.sample(PRIMARY_COLORS, 2))
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    grid = np.full((height, width), BLACK, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    for kind, mixed in specs:
        colors = _pair_colors(palette, mixed)
        ok = _place_horizontal_pair(grid, reserved, colors) if kind == "H" else _place_vertical_pair(grid, reserved, colors)
        if not ok:
            return None

    towers = [_decode_tower(component) for component in _find_tower_components(grid, BLACK)]
    if len(towers) != pair_count * 2:
        return None
    if set(int(tower["color"]) for tower in towers) != set(palette):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower puzzle instance")
