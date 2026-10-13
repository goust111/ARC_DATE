from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BLUE = 1

TOP_LEFT = 2
TOP_RIGHT = 3
BOTTOM_LEFT = 4
BOTTOM_RIGHT = 6

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")

UNIQUE_SIDE_TO_CORNER = {
    "top": "ur",
    "bottom": "ll",
    "left": "ul",
    "right": "lr",
}

ADJACENT_PAIR_TO_CORNER = {
    frozenset({"top", "left"}): "ul",
    frozenset({"top", "right"}): "ur",
    frozenset({"bottom", "left"}): "ll",
    frozenset({"bottom", "right"}): "lr",
}

CORNER_TO_COLOR = {
    "ul": TOP_LEFT,
    "ur": TOP_RIGHT,
    "ll": BOTTOM_LEFT,
    "lr": BOTTOM_RIGHT,
}

CORNER_TO_SLICE = {
    "ul": (slice(0, 3), slice(0, 3)),
    "ur": (slice(0, 3), slice(-3, None)),
    "ll": (slice(-3, None), slice(0, 3)),
    "lr": (slice(-3, None), slice(-3, None)),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbor_count(mask: np.ndarray, r: int, c: int) -> int:
    total = 0
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if _in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]):
            total += 1
    return total


def _mode(values: list[int], *, prefer: str = "min") -> int:
    counts = Counter(values)
    best = max(counts.values())
    candidates = [v for v, cnt in counts.items() if cnt == best]
    return min(candidates) if prefer == "min" else max(candidates)


def _blue_mask(grid: np.ndarray) -> np.ndarray:
    counts = Counter(int(v) for v in grid.flat if int(v) != BG)
    if BLUE in counts:
        color = BLUE
    else:
        color = max(counts, key=counts.get)
    return grid == color


def _body_rect(mask: np.ndarray) -> tuple[int, int, int, int]:
    row_counts = [int(np.count_nonzero(mask[r])) for r in range(mask.shape[0])]
    col_counts = [int(np.count_nonzero(mask[:, c])) for c in range(mask.shape[1])]

    wide_rows = [count for count in row_counts if count >= 2]
    tall_cols = [count for count in col_counts if count >= 2]
    assert wide_rows and tall_cols, "expected a thick foreground body"

    body_width = _mode(wide_rows, prefer="max")
    body_height = _mode(tall_cols, prefer="max")

    body_rows = [
        r
        for r, count in enumerate(row_counts)
        if body_width - 1 <= count <= body_width + 1
    ]
    body_cols = [
        c
        for c, count in enumerate(col_counts)
        if body_height - 1 <= count <= body_height + 1
    ]
    assert body_rows and body_cols, "failed to recover body rectangle"
    top, bottom = min(body_rows), max(body_rows)
    left, right = min(body_cols), max(body_cols)
    return top, bottom, left, right


def _detect_features(mask: np.ndarray, body: tuple[int, int, int, int]) -> dict[str, dict[str, list[int]]]:
    top, bottom, left, right = body
    features = {side: {"tabs": [], "notches": []} for side in SIDES}
    tab_cells: set[tuple[int, int]] = set()

    for c in range(left + 1, right):
        if top - 1 >= 0 and bool(mask[top - 1, c]) and _neighbor_count(mask, top - 1, c) == 1:
            features["top"]["tabs"].append(c)
            tab_cells.add((top - 1, c))
        if bottom + 1 < mask.shape[0] and bool(mask[bottom + 1, c]) and _neighbor_count(mask, bottom + 1, c) == 1:
            features["bottom"]["tabs"].append(c)
            tab_cells.add((bottom + 1, c))

    for r in range(top + 1, bottom):
        if left - 1 >= 0 and bool(mask[r, left - 1]) and _neighbor_count(mask, r, left - 1) == 1:
            features["left"]["tabs"].append(r)
            tab_cells.add((r, left - 1))
        if right + 1 < mask.shape[1] and bool(mask[r, right + 1]) and _neighbor_count(mask, r, right + 1) == 1:
            features["right"]["tabs"].append(r)
            tab_cells.add((r, right + 1))

    def adjacent_to_tab(cell: tuple[int, int]) -> bool:
        r, c = cell
        return any(abs(r - tr) + abs(c - tc) == 1 for tr, tc in tab_cells)

    for c in range(left + 1, right):
        if not bool(mask[top, c]) and _neighbor_count(mask, top, c) == 3 and not adjacent_to_tab((top, c)):
            features["top"]["notches"].append(c)
        if not bool(mask[bottom, c]) and _neighbor_count(mask, bottom, c) == 3 and not adjacent_to_tab((bottom, c)):
            features["bottom"]["notches"].append(c)

    for r in range(top + 1, bottom):
        if not bool(mask[r, left]) and _neighbor_count(mask, r, left) == 3 and not adjacent_to_tab((r, left)):
            features["left"]["notches"].append(r)
        if not bool(mask[r, right]) and _neighbor_count(mask, r, right) == 3 and not adjacent_to_tab((r, right)):
            features["right"]["notches"].append(r)

    return features


def _completion_counts(mask: np.ndarray, body: tuple[int, int, int, int], features: dict[str, dict[str, list[int]]]) -> dict[str, int]:
    top, bottom, left, right = body
    out = mask.copy()
    counts = {side: 0 for side in SIDES}

    for c in features["top"]["tabs"]:
        rr, cc = bottom + 1, c
        if _in_bounds(out.shape, rr, cc) and not bool(out[rr, cc]):
            out[rr, cc] = True
            counts["top"] += 1
    for c in features["top"]["notches"]:
        rr, cc = bottom, c
        if _in_bounds(out.shape, rr, cc) and bool(out[rr, cc]):
            out[rr, cc] = False
            counts["top"] += 1

    for c in features["bottom"]["tabs"]:
        rr, cc = top - 1, c
        if _in_bounds(out.shape, rr, cc) and not bool(out[rr, cc]):
            out[rr, cc] = True
            counts["bottom"] += 1
    for c in features["bottom"]["notches"]:
        rr, cc = top, c
        if _in_bounds(out.shape, rr, cc) and bool(out[rr, cc]):
            out[rr, cc] = False
            counts["bottom"] += 1

    for r in features["left"]["tabs"]:
        rr, cc = r, right + 1
        if _in_bounds(out.shape, rr, cc) and not bool(out[rr, cc]):
            out[rr, cc] = True
            counts["left"] += 1
    for r in features["left"]["notches"]:
        rr, cc = r, right
        if _in_bounds(out.shape, rr, cc) and bool(out[rr, cc]):
            out[rr, cc] = False
            counts["left"] += 1

    for r in features["right"]["tabs"]:
        rr, cc = r, left - 1
        if _in_bounds(out.shape, rr, cc) and not bool(out[rr, cc]):
            out[rr, cc] = True
            counts["right"] += 1
    for r in features["right"]["notches"]:
        rr, cc = r, left
        if _in_bounds(out.shape, rr, cc) and bool(out[rr, cc]):
            out[rr, cc] = False
            counts["right"] += 1

    return counts


def _corner_color(grid: np.ndarray, corner: str) -> int:
    if corner == "ul":
        return int(grid[0, 0])
    if corner == "ur":
        return int(grid[0, -1])
    if corner == "ll":
        return int(grid[-1, 0])
    return int(grid[-1, -1])


def _choose_corner(grid: np.ndarray, counts: dict[str, int]) -> str:
    max_count = max(counts.values())
    assert max_count > 0, "expected at least one mirrored completion"
    leaders = [side for side in SIDES if counts[side] == max_count]

    if len(leaders) == 1:
        return UNIQUE_SIDE_TO_CORNER[leaders[0]]

    if len(leaders) == 2:
        pair = frozenset(leaders)
        if pair in ADJACENT_PAIR_TO_CORNER:
            return ADJACENT_PAIR_TO_CORNER[pair]

        candidates = [UNIQUE_SIDE_TO_CORNER[side] for side in leaders]
        freqs = {corner: int(np.count_nonzero(grid == _corner_color(grid, corner))) for corner in candidates}
        ranked = sorted(candidates, key=lambda corner: (freqs[corner], "ul ur ll lr".split().index(corner)))
        return ranked[0]

    raise AssertionError("generator should avoid 3-way or 4-way maximum ties")


def _crop_corner(grid: np.ndarray, corner: str) -> np.ndarray:
    rs, cs = CORNER_TO_SLICE[corner]
    return grid[rs, cs].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the corner selected by which body side creates the most symmetric-completion changes.

    Input structure:
    - A black background contains one large thick blue rectilinear body near the center.
    - The body's four sides may contain one-pixel tabs and one-pixel notches.
    - The four grid corners contain solid 3x3 color blocks: red at top-left, green at top-right, yellow at bottom-left, and magenta at bottom-right.

    Rule:
    - First mirror-complete the blue body: a tab adds one blue pixel just outside the opposite side, while a notch removes the opposite edge pixel to create the mirrored notch.
    - Count how many actual mirrored changes were triggered by sources on each of the four sides.
    - If one side is strictly largest, use the task's side-to-corner map; if two adjacent sides tie, take their shared geometric corner; if two opposite sides tie, compare the tied candidate corner colors' whole-grid frequencies and pick the rarer one.

    Key insight:
    - The geometric completion is only the first step; the decisive signal is the per-side count of mirrored edits, and only after that count is known can the correct 3x3 corner crop be chosen.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3

    mask = _blue_mask(g)
    body = _body_rect(mask)
    features = _detect_features(mask, body)
    counts = _completion_counts(mask, body, features)
    corner = _choose_corner(g, counts)
    return _crop_corner(g, corner)


GEN_ATTEMPTS = 400
GRID_RANGE = (17, 21)
BODY_HEIGHT_RANGE = (8, 11)
BODY_WIDTH_RANGE = (8, 11)


def _spaced_subsets(candidates: list[int], count: int) -> list[tuple[int, ...]]:
    if count == 0:
        return [tuple()]
    out: list[tuple[int, ...]] = []

    def rec(start: int, chosen: list[int]) -> None:
        if len(chosen) == count:
            out.append(tuple(chosen))
            return
        for idx in range(start, len(candidates)):
            value = candidates[idx]
            if chosen and value - chosen[-1] < 2:
                continue
            chosen.append(value)
            rec(idx + 1, chosen)
            chosen.pop()

    rec(0, [])
    return out


def _sample_disjoint_opposites(candidates: list[int], count_a: int, count_b: int) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    left_sets = _spaced_subsets(candidates, count_a)
    right_sets = _spaced_subsets(candidates, count_b)
    pairs = [(a, b) for a in left_sets for b in right_sets if set(a).isdisjoint(b)]
    if not pairs:
        return None
    return random.choice(pairs)


def _base_grid(shape: tuple[int, int], body: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    top, bottom, left, right = body
    grid = np.zeros((h, w), dtype=int)
    grid[:3, :3] = TOP_LEFT
    grid[:3, -3:] = TOP_RIGHT
    grid[-3:, :3] = BOTTOM_LEFT
    grid[-3:, -3:] = BOTTOM_RIGHT
    grid[top : bottom + 1, left : right + 1] = BLUE
    return grid


def _branch_counts() -> tuple[str, dict[str, int], str]:
    branch = random.choice(["unique", "adjacent_tie", "opposite_tie"])

    if branch == "unique":
        winner = random.choice(list(SIDES))
        counts = {side: 0 for side in SIDES}
        counts[winner] = 2
        others = [side for side in SIDES if side != winner]
        random.shuffle(others)
        counts[others[0]] = random.choice([0, 1])
        counts[others[1]] = random.choice([0, 1])
        counts[others[2]] = 0
        expected_corner = UNIQUE_SIDE_TO_CORNER[winner]
        return branch, counts, expected_corner

    if branch == "adjacent_tie":
        pair = random.choice(list(ADJACENT_PAIR_TO_CORNER))
        counts = {side: 0 for side in SIDES}
        for side in pair:
            counts[side] = 2
        others = [side for side in SIDES if side not in pair]
        counts[others[0]] = random.choice([0, 1])
        counts[others[1]] = random.choice([0, 1])
        return branch, counts, ADJACENT_PAIR_TO_CORNER[pair]

    pair = random.choice([("top", "bottom"), ("left", "right")])
    counts = {side: 0 for side in SIDES}
    counts[pair[0]] = 2
    counts[pair[1]] = 2
    others = [side for side in SIDES if side not in pair]
    counts[others[0]] = random.choice([0, 1])
    counts[others[1]] = random.choice([0, 1])
    expected_corner = random.choice([UNIQUE_SIDE_TO_CORNER[pair[0]], UNIQUE_SIDE_TO_CORNER[pair[1]]])
    return branch, counts, expected_corner


def _feature_records(body: tuple[int, int, int, int], counts: dict[str, int]) -> list[tuple[str, int, str]] | None:
    top, bottom, left, right = body
    horizontal_candidates = list(range(left + 1, right))
    vertical_candidates = list(range(top + 1, bottom))

    top_bottom = _sample_disjoint_opposites(horizontal_candidates, counts["top"], counts["bottom"])
    left_right = _sample_disjoint_opposites(vertical_candidates, counts["left"], counts["right"])
    if top_bottom is None or left_right is None:
        return None

    positions = {
        "top": list(top_bottom[0]),
        "bottom": list(top_bottom[1]),
        "left": list(left_right[0]),
        "right": list(left_right[1]),
    }

    bare_records: list[tuple[str, int]] = []
    for side in SIDES:
        for pos in positions[side]:
            bare_records.append((side, pos))
    random.shuffle(bare_records)

    if not bare_records:
        return None

    if len(bare_records) == 1:
        kinds = ["tab"]
    else:
        kinds = ["tab", "notch"] + [random.choice(["tab", "notch"]) for _ in range(len(bare_records) - 2)]
        random.shuffle(kinds)

    return [(side, pos, kind) for (side, pos), kind in zip(bare_records, kinds)]


def _apply_records(grid: np.ndarray, body: tuple[int, int, int, int], records: list[tuple[str, int, str]]) -> None:
    top, bottom, left, right = body
    for side, pos, kind in records:
        if side == "top":
            if kind == "tab":
                grid[top - 1, pos] = BLUE
            else:
                grid[top, pos] = BG
        elif side == "bottom":
            if kind == "tab":
                grid[bottom + 1, pos] = BLUE
            else:
                grid[bottom, pos] = BG
        elif side == "left":
            if kind == "tab":
                grid[pos, left - 1] = BLUE
            else:
                grid[pos, left] = BG
        else:
            if kind == "tab":
                grid[pos, right + 1] = BLUE
            else:
                grid[pos, right] = BG


def _clear_for_blob(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if grid[r, c] != BG:
            return False
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if _in_bounds(grid.shape, rr, cc) and (rr, cc) not in cell_set and grid[rr, cc] != BG:
                return False
    return True


def _place_frequency_blob(grid: np.ndarray, color: int) -> bool:
    lengths = [3, 2, 1]
    candidates: list[list[tuple[int, int]]] = []
    for length in lengths:
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1] - length + 1):
                cells = [(r, c + dc) for dc in range(length)]
                if _clear_for_blob(grid, cells):
                    candidates.append(cells)
        for r in range(grid.shape[0] - length + 1):
            for c in range(grid.shape[1]):
                cells = [(r + dr, c) for dr in range(length)]
                if _clear_for_blob(grid, cells):
                    candidates.append(cells)
        if candidates:
            break
    if not candidates:
        return False
    for r, c in random.choice(candidates):
        grid[r, c] = color
    return True


def _expected_crop(grid: np.ndarray, corner: str) -> np.ndarray:
    rs, cs = CORNER_TO_SLICE[corner]
    return grid[rs, cs]


def _attempt() -> np.ndarray | None:
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    body_h = random.randint(*BODY_HEIGHT_RANGE)
    body_w = random.randint(*BODY_WIDTH_RANGE)
    if body_h > h - 8 or body_w > w - 8:
        return None

    top = random.randint(4, h - body_h - 4)
    left = random.randint(4, w - body_w - 4)
    body = (top, top + body_h - 1, left, left + body_w - 1)

    branch, counts, expected_corner = _branch_counts()
    records = _feature_records(body, counts)
    if records is None:
        return None

    grid = _base_grid((h, w), body)
    _apply_records(grid, body, records)

    if branch == "opposite_tie":
        tied_sides = [side for side, value in counts.items() if value == 2]
        candidate_corners = [UNIQUE_SIDE_TO_CORNER[side] for side in tied_sides]
        loser_corner = candidate_corners[0] if candidate_corners[1] == expected_corner else candidate_corners[1]
        if not _place_frequency_blob(grid, CORNER_TO_COLOR[loser_corner]):
            return None

    mask = _blue_mask(grid)
    found_body = _body_rect(mask)
    found_features = _detect_features(mask, found_body)
    found_counts = _completion_counts(mask, found_body, found_features)

    if found_counts != counts:
        return None
    if _choose_corner(grid, found_counts) != expected_corner:
        return None
    if not np.array_equal(main(grid), _expected_crop(grid, expected_corner)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")
