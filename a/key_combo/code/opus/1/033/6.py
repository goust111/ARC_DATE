from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
L_COLOR = 8
ANCHOR_COLOR = 1
LINE_COLOR = 2

GRID_MIN = 12
GRID_MAX = 20
OBJECT_MIN = 4
OBJECT_MAX = 8
GENERATION_ATTEMPTS = 500
PAIR_ATTEMPTS = 200
EXTRA_OBJECT_ATTEMPTS = 200
MIN_CELL_DISTANCE = 3

NORMAL_GAPS = (4, 5)
WIDE_GAPS = (8, 9)


@dataclass(frozen=True)
class LObject:
    r0: int
    c0: int
    dr: int
    dc: int

    @property
    def anchor(self) -> tuple[int, int]:
        return self.r0 + self.dr, self.c0 + self.dc

    @property
    def orientation(self) -> tuple[int, int]:
        return self.dr, self.dc

    def bbox(self) -> tuple[int, int, int, int]:
        return self.r0, self.c0, self.r0 + 1, self.c0 + 1

    def cells(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (self.r0 + rr, self.c0 + cc)
            for rr in range(2)
            for cc in range(2)
            if (rr, cc) != (self.dr, self.dc)
        )


def _components_of_color(g: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _extract_objects(g: np.ndarray) -> list[dict[str, object]]:
    assert g.ndim == 2, "expected a 2D grid"
    colors = {int(v) for v in np.unique(g)}
    assert colors <= {BG, L_COLOR}, "input must contain only black background and teal L objects"

    objects: list[dict[str, object]] = []
    for coords in _components_of_color(g, L_COLOR):
        assert len(coords) == 3, "each foreground object must have exactly three pixels"
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 == 1 and c1 - c0 == 1, "each object must fit a 2x2 bounding box"

        present = set(coords)
        box_cells = {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}
        missing = box_cells - present
        assert len(missing) == 1, "each 2x2 box must have exactly one missing corner"
        anchor = next(iter(missing))
        assert int(g[anchor]) == BG, "the missing corner must be black in the input"

        objects.append(
            {
                "coords": tuple(sorted(present)),
                "bbox": (r0, c0, r1, c1),
                "anchor": anchor,
                "orientation": (anchor[0] - r0, anchor[1] - c0),
            }
        )

    assert objects, "expected at least one L object"
    anchors = [obj["anchor"] for obj in objects]
    assert len(set(anchors)) == len(anchors), "anchor positions must be unique"
    return sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))


def _adjacent_anchor_pairs(
    anchors: list[tuple[int, int]]
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c in anchors:
        by_row[r].append(c)
        by_col[c].append(r)

    row_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    col_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, cols in by_row.items():
        ordered = sorted(cols)
        for c0, c1 in zip(ordered, ordered[1:]):
            row_pairs.append(((r, c0), (r, c1)))
    for c, rows in by_col.items():
        ordered = sorted(rows)
        for r0, r1 in zip(ordered, ordered[1:]):
            col_pairs.append(((r0, c), (r1, c)))
    return row_pairs, col_pairs


def _paint_segment(
    base: np.ndarray,
    out: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
) -> tuple[int, bool]:
    painted = 0
    blocked = False
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        for c in range(c0 + 1, c1):
            if int(base[r, c]) != BG:
                blocked = True
                break
            out[r, c] = LINE_COLOR
            painted += 1
    else:
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        for r in range(r0 + 1, r1):
            if int(base[r, c]) != BG:
                blocked = True
                break
            out[r, c] = LINE_COLOR
            painted += 1
    return painted, blocked


def _solve_from_anchors(g: np.ndarray, anchors: list[tuple[int, int]]) -> np.ndarray:
    base = g.copy()
    for r, c in anchors:
        assert int(base[r, c]) == BG, "anchor must start on a black cell"
        base[r, c] = ANCHOR_COLOR

    out = base.copy()
    row_pairs, col_pairs = _adjacent_anchor_pairs(anchors)
    for pair in row_pairs + col_pairs:
        _paint_segment(base, out, pair[0], pair[1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each teal L, then connect the new blue missing corners with red segments between adjacent aligned anchors.

    Input structure:
    - The grid has a black background and several separated teal 3-pixel foreground objects.
    - Each foreground object is a 4-connected L occupying three cells of a 2x2 bounding box.
    - The one missing corner of each 2x2 box is black in the input and becomes the hidden anchor.

    Rule:
    - For every teal L object, find the unique missing corner of its 2x2 bounding box and recolor that cell blue.
    - Collect all blue cells as anchors.
    - For each row and each column, sort the anchors and connect only adjacent anchors with a red straight segment.
    - Paint only black cells along the segment path; stop immediately before the first non-black blocker, and never overwrite teal or blue cells.

    Key insight:
    - The connectable markers are not explicit in the input: they are revealed only after locally completing each L into a 2x2 square.
    """
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    anchors = [obj["anchor"] for obj in objects]
    return _solve_from_anchors(g, anchors)


def _objects_compatible(a: LObject, b: LObject) -> bool:
    ar0, ac0, ar1, ac1 = a.bbox()
    br0, bc0, br1, bc1 = b.bbox()
    if not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0):
        return False

    for r0, c0 in a.cells():
        for r1, c1 in b.cells():
            if abs(r0 - r1) + abs(c0 - c1) < MIN_CELL_DISTANCE:
                return False
    return True


def _placement_valid(objects: list[LObject], h: int, w: int) -> bool:
    anchors = set()
    for obj in objects:
        if not (0 <= obj.r0 <= h - 2 and 0 <= obj.c0 <= w - 2):
            return False
        if obj.anchor in anchors:
            return False
        anchors.add(obj.anchor)

    for i, obj in enumerate(objects):
        for other in objects[i + 1 :]:
            if not _objects_compatible(obj, other):
                return False
    return True


def _render_input(h: int, w: int, objects: list[LObject]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        for r, c in obj.cells():
            grid[r, c] = L_COLOR
    return grid


def _oracle_output(h: int, w: int, objects: list[LObject]) -> tuple[np.ndarray, dict[str, object]]:
    inp = _render_input(h, w, objects)
    anchors = [obj.anchor for obj in objects]
    base = inp.copy()
    for r, c in anchors:
        base[r, c] = ANCHOR_COLOR

    out = base.copy()
    row_pairs, col_pairs = _adjacent_anchor_pairs(anchors)
    blocked_pairs = 0
    clear_pairs = 0
    for pair in row_pairs + col_pairs:
        _painted, blocked = _paint_segment(base, out, pair[0], pair[1])
        if blocked:
            blocked_pairs += 1
        else:
            clear_pairs += 1

    return out, {
        "anchors": sorted(anchors),
        "adjacent_pair_count": len(row_pairs) + len(col_pairs),
        "blocked_pairs": blocked_pairs,
        "clear_pairs": clear_pairs,
        "orientation_count": len({obj.orientation for obj in objects}),
    }


def _sample_horizontal_pair(
    h: int,
    w: int,
    existing: list[LObject],
    used_rows: set[int],
    used_cols: set[int],
    wide_gap: bool,
) -> dict[str, object] | None:
    gap_choices = WIDE_GAPS if wide_gap else NORMAL_GAPS
    row_candidates = list(range(1, h - 1))
    random.shuffle(row_candidates)
    for anchor_row in row_candidates:
        if anchor_row in used_rows:
            continue
        for _ in range(PAIR_ATTEMPTS):
            gap = random.choice(gap_choices)
            max_left = w - 2 - gap
            if max_left < 1:
                continue
            left_col = random.randint(1, max_left)
            right_col = left_col + gap
            if left_col in used_cols or right_col in used_cols:
                continue

            dr_left = random.randint(0, 1)
            dr_right = random.randint(0, 1)
            left = LObject(anchor_row - dr_left, left_col - 1, dr_left, 1)
            right = LObject(anchor_row - dr_right, right_col, dr_right, 0)
            if _placement_valid(existing + [left, right], h, w):
                return {
                    "orientation": "H",
                    "objects": [left, right],
                    "anchors": (left.anchor, right.anchor),
                }
    return None


def _sample_vertical_pair(
    h: int,
    w: int,
    existing: list[LObject],
    used_rows: set[int],
    used_cols: set[int],
    wide_gap: bool,
) -> dict[str, object] | None:
    gap_choices = WIDE_GAPS if wide_gap else NORMAL_GAPS
    col_candidates = list(range(1, w - 1))
    random.shuffle(col_candidates)
    for anchor_col in col_candidates:
        if anchor_col in used_cols:
            continue
        for _ in range(PAIR_ATTEMPTS):
            gap = random.choice(gap_choices)
            max_top = h - 2 - gap
            if max_top < 1:
                continue
            top_row = random.randint(1, max_top)
            bottom_row = top_row + gap
            if top_row in used_rows or bottom_row in used_rows:
                continue

            dc_top = random.randint(0, 1)
            dc_bottom = random.randint(0, 1)
            top = LObject(top_row - 1, anchor_col - dc_top, 1, dc_top)
            bottom = LObject(bottom_row, anchor_col - dc_bottom, 0, dc_bottom)
            if _placement_valid(existing + [top, bottom], h, w):
                return {
                    "orientation": "V",
                    "objects": [top, bottom],
                    "anchors": (top.anchor, bottom.anchor),
                }
    return None


def _sample_blocker(
    h: int,
    w: int,
    existing: list[LObject],
    used_rows: set[int],
    used_cols: set[int],
    pair: dict[str, object],
) -> LObject | None:
    a0, a1 = pair["anchors"]
    if pair["orientation"] == "H":
        row = a0[0]
        row_options: list[tuple[int, int]] = []
        if row - 1 not in used_rows:
            row_options.append((row - 1, 0))
        if row + 1 not in used_rows and row <= h - 2:
            row_options.append((row, 1))

        col_candidates = list(range(a0[1] + 2, a1[1] - 2))
        random.shuffle(row_options)
        random.shuffle(col_candidates)
        for box_row, dr in row_options:
            if not (0 <= box_row <= h - 2):
                continue
            for box_col in col_candidates:
                for dc in random.sample((0, 1), 2):
                    blocker = LObject(box_row, box_col, dr, dc)
                    if blocker.anchor[1] in used_cols:
                        continue
                    if _placement_valid(existing + [blocker], h, w):
                        return blocker
        return None

    col = a0[1]
    col_options: list[tuple[int, int]] = []
    if col - 1 not in used_cols:
        col_options.append((col - 1, 0))
    if col + 1 not in used_cols and col <= w - 2:
        col_options.append((col, 1))

    row_candidates = list(range(a0[0] + 2, a1[0] - 2))
    random.shuffle(col_options)
    random.shuffle(row_candidates)
    for box_col, dc in col_options:
        if not (0 <= box_col <= w - 2):
            continue
        for box_row in row_candidates:
            for dr in random.sample((0, 1), 2):
                blocker = LObject(box_row, box_col, dr, dc)
                if blocker.anchor[0] in used_rows:
                    continue
                if _placement_valid(existing + [blocker], h, w):
                    return blocker
    return None


def _sample_isolated_object(
    h: int,
    w: int,
    existing: list[LObject],
    used_rows: set[int],
    used_cols: set[int],
) -> LObject | None:
    available_rows = [r for r in range(1, h - 1) if r not in used_rows]
    available_cols = [c for c in range(1, w - 1) if c not in used_cols]
    if not available_rows or not available_cols:
        return None

    for _ in range(EXTRA_OBJECT_ATTEMPTS):
        anchor_row = random.choice(available_rows)
        anchor_col = random.choice(available_cols)
        dr = random.randint(0, 1)
        dc = random.randint(0, 1)
        obj = LObject(anchor_row - dr, anchor_col - dc, dr, dc)
        if _placement_valid(existing + [obj], h, w):
            return obj
    return None


def _inspect_generated_input(g: np.ndarray) -> dict[str, object]:
    objects = _extract_objects(g)
    assert OBJECT_MIN <= len(objects) <= OBJECT_MAX, "generated input must have 4-8 objects"
    assert len({obj["orientation"] for obj in objects}) >= 2, "generated input should use multiple L orientations"

    anchors = [obj["anchor"] for obj in objects]
    row_pairs, col_pairs = _adjacent_anchor_pairs(anchors)

    coords_per_object = [obj["coords"] for obj in objects]
    for i, coords_a in enumerate(coords_per_object):
        for coords_b in coords_per_object[i + 1 :]:
            for r0, c0 in coords_a:
                for r1, c1 in coords_b:
                    assert abs(r0 - r1) + abs(c0 - c1) >= MIN_CELL_DISTANCE, "objects are too close"

    base = g.copy()
    for r, c in anchors:
        base[r, c] = ANCHOR_COLOR
    blocked_pairs = 0
    clear_pairs = 0
    scratch = base.copy()
    for pair in row_pairs + col_pairs:
        _painted, blocked = _paint_segment(base, scratch, pair[0], pair[1])
        if blocked:
            blocked_pairs += 1
        else:
            clear_pairs += 1

    return {
        "anchors": sorted(anchors),
        "adjacent_pair_count": len(row_pairs) + len(col_pairs),
        "blocked_pairs": blocked_pairs,
        "clear_pairs": clear_pairs,
    }


PAIR_TEMPLATES = (
    ("H", "H", "H"),
    ("V", "V", "V"),
    ("H", "H", "V"),
    ("H", "V", "V"),
    ("V", "H", "H"),
    ("V", "V", "H"),
)


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    want_blocker = random.random() < 0.55
    template = random.choice(PAIR_TEMPLATES)
    blocker_index = random.randrange(len(template)) if want_blocker else -1

    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    objects: list[LObject] = []
    pairs: list[dict[str, object]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    for idx, orientation in enumerate(template):
        wide_gap = want_blocker and idx == blocker_index
        if orientation == "H":
            pair = _sample_horizontal_pair(h, w, objects, used_rows, used_cols, wide_gap)
        else:
            pair = _sample_vertical_pair(h, w, objects, used_rows, used_cols, wide_gap)
        if pair is None:
            return None
        objects.extend(pair["objects"])
        pairs.append(pair)
        for r, c in pair["anchors"]:
            used_rows.add(r)
            used_cols.add(c)

    if want_blocker:
        blocker = _sample_blocker(h, w, objects, used_rows, used_cols, pairs[blocker_index])
        if blocker is None:
            return None
        objects.append(blocker)
        used_rows.add(blocker.anchor[0])
        used_cols.add(blocker.anchor[1])

    target_count = random.randint(len(objects), OBJECT_MAX)
    while len(objects) < target_count:
        extra = _sample_isolated_object(h, w, objects, used_rows, used_cols)
        if extra is None:
            break
        objects.append(extra)
        used_rows.add(extra.anchor[0])
        used_cols.add(extra.anchor[1])

    if not _placement_valid(objects, h, w):
        return None

    inp = _render_input(h, w, objects)
    expected, oracle_info = _oracle_output(h, w, objects)
    try:
        inspected = _inspect_generated_input(inp)
        assert inspected["anchors"] == oracle_info["anchors"], "metadata anchors must match rendered input"
        assert inspected["adjacent_pair_count"] == oracle_info["adjacent_pair_count"], "pair counts must match"
        assert inspected["blocked_pairs"] == oracle_info["blocked_pairs"], "blocked-pair counts must match"
        assert inspected["clear_pairs"] == oracle_info["clear_pairs"], "clear-pair counts must match"
        actual = main(inp)
    except AssertionError:
        return None

    if oracle_info["adjacent_pair_count"] != 3:
        return None
    if want_blocker:
        if oracle_info["blocked_pairs"] != 1 or oracle_info["clear_pairs"] != 2:
            return None
    else:
        if oracle_info["blocked_pairs"] != 0 or oracle_info["clear_pairs"] != 3:
            return None

    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        sample = _sample_case()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid input for puzzle 6")
