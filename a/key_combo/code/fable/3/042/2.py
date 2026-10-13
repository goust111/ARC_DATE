from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import random

import numpy as np
from scipy.ndimage import label


BACKGROUND = 0
GREY = 5
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
LEDGER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
AREA_SHAPES = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    10: [(2, 5), (5, 2)],
    12: [(3, 4), (4, 3), (2, 6), (6, 2)],
    15: [(3, 5), (5, 3)],
}


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    spans = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            spans.append((start, prev))
            start = prev = idx
    spans.append((start, prev))
    return spans


def _find_ledger(g: np.ndarray) -> tuple[list[tuple[int, int]], dict[int, int], list[int]]:
    candidates = []
    h, w = g.shape
    for r in (0, h - 1):
        colored_cols = [
            c for c in range(w) if int(g[r, c]) not in (BACKGROUND, GREY)
        ]
        for c0, c1 in _runs(colored_cols):
            if c0 != 0 and c1 != w - 1:
                continue
            if c1 - c0 + 1 < 3:
                continue
            cells = [(r, c) for c in range(c0, c1 + 1)]
            values = [int(g[r, c]) for c in range(c0, c1 + 1)]
            if any(v in (BACKGROUND, GREY) for v in values):
                continue

            seen_segments = set()
            order = []
            prev = None
            repeated_segment = False
            for color in values:
                if color == prev:
                    continue
                if color in seen_segments:
                    repeated_segment = True
                    break
                seen_segments.add(color)
                order.append(color)
                prev = color
            if repeated_segment or len(order) < 2:
                continue
            candidates.append((cells, dict(Counter(values)), order))

    assert len(candidates) == 1, "expected one corner ledger bar"
    return candidates[0]


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _find_frames(g: np.ndarray) -> list[dict]:
    grey_lab, n_grey = label(g == GREY, structure=STRUCT4)
    frames = []

    for cid in range(1, n_grey + 1):
        pts = np.argwhere(grey_lab == cid)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a cavity"

        border = _border_cells(r0, c0, r1, c1)
        markers = []
        grey_count = 0
        for r, c in border:
            color = int(g[r, c])
            if color == GREY:
                grey_count += 1
            else:
                assert color not in (BACKGROUND, GREY), "frame border must have one colored marker"
                markers.append((r, c, color))

        assert len(markers) == 1, "each frame must have exactly one border marker"
        assert grey_count == len(border) - 1, "all non-marker border cells must be grey"
        marker_r, marker_c, marker_color = markers[0]

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0
        assert np.all(interior == BACKGROUND), "frame cavity must be black in the input"

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "area": int(interior.size),
                "marker": int(marker_color),
                "marker_pos": (int(marker_r), int(marker_c)),
            }
        )

    assert frames, "expected at least one hollow frame"
    frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][1]))
    return frames


def _all_assignments(
    frames: list[dict],
    counts: dict[int, int],
    color_order: list[int],
    *,
    use_markers: bool,
    stop_after: int | None = None,
) -> list[dict[int, int]]:
    frame_order = sorted(
        range(len(frames)),
        key=lambda i: (
            sum(
                counts[color] == frames[i]["area"]
                and (not use_markers or color != frames[i]["marker"])
                for color in color_order
            ),
            frames[i]["bbox"],
        ),
    )
    assignments: list[dict[int, int]] = []

    def rec(pos: int, unused: set[int], partial: dict[int, int]) -> None:
        if stop_after is not None and len(assignments) >= stop_after:
            return
        if pos == len(frame_order):
            assignments.append(dict(partial))
            return
        fi = frame_order[pos]
        frame = frames[fi]
        candidates = [
            color
            for color in color_order
            if color in unused
            and counts[color] == frame["area"]
            and (not use_markers or color != frame["marker"])
        ]
        for color in candidates:
            unused.remove(color)
            partial[fi] = color
            rec(pos + 1, unused, partial)
            partial.pop(fi)
            unused.add(color)

    rec(0, set(color_order), {})
    return assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame cavity with the ledger color whose count equals the cavity area.

    Input structure
    - The background is black, and one horizontal one-cell-high ledger bar sits
      flush with a canvas corner.
    - Away from the ledger are grey one-pixel-thick hollow rectangular frames.
      Each frame has one non-grey colored marker embedded in its border and a
      black rectangular cavity inside.
    - The ledger contains one contiguous segment for each ledger color; segment
      length is the amount of that color available.

    Rule
    - Count how many cells of each color appear in the ledger bar.
    - Assign ledger colors to frame cavities as a global one-to-one matching:
      a color may fill a cavity only when its count equals the cavity area, and
      it may not equal that frame's marker color.
    - Fill each whole cavity with its assigned color, clear the ledger bar to
      black, and preserve all grey frame borders and colored markers.

    Key insight
    - The bar is a resource ledger, not a shape to copy; equal-size ledger
      entries and equal-area cavities are disambiguated only by the border
      marker exclusions under a global bijection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    ledger_cells, counts, color_order = _find_ledger(g)
    frames = _find_frames(g)
    assert len(frames) == len(counts), "ledger colors and cavities must match one-to-one"

    assignments = _all_assignments(frames, counts, color_order, use_markers=True, stop_after=2)
    assert len(assignments) == 1, "expected a unique marker-constrained assignment"
    assignment = assignments[0]

    out = g.copy()
    for r, c in ledger_cells:
        out[r, c] = BACKGROUND
    for fi, color in assignment.items():
        r0, c0, r1, c1 = frames[fi]["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(color)
    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 17, 30
FRAME_PADDING = 1
MAX_GENERATION_ATTEMPTS = 500
FRAME_PLACEMENT_ATTEMPTS = 200


def _area_choices(width: int) -> list[list[int]]:
    choices = []
    areas = list(AREA_SHAPES)
    for n_frames in (3, 4):
        for dup in areas:
            for extras in itertools.combinations([a for a in areas if a != dup], n_frames - 2):
                vals = [dup, dup, *extras]
                if sum(vals) <= width:
                    choices.append(vals)
    return choices


def _choose_entries(width: int) -> list[dict] | None:
    choices = _area_choices(width)
    if not choices:
        return None
    areas = random.choice(choices)
    random.shuffle(areas)

    colors = random.sample(LEDGER_COLORS, len(areas))
    entries = [{"color": color, "area": area, "marker": None} for color, area in zip(colors, areas)]

    area_to_idxs: dict[int, list[int]] = defaultdict(list)
    for i, entry in enumerate(entries):
        area_to_idxs[entry["area"]].append(i)
    duplicate_groups = [idxs for idxs in area_to_idxs.values() if len(idxs) == 2]
    if not duplicate_groups:
        return None
    pair = random.choice(duplicate_groups)
    a, b = pair
    entries[a]["marker"] = entries[b]["color"]
    entries[b]["marker"] = entries[a]["color"]

    all_colors = [entry["color"] for entry in entries]
    for i, entry in enumerate(entries):
        if entry["marker"] is not None:
            continue
        blockers = [
            other["color"]
            for other in entries
            if other["color"] != entry["color"] and other["area"] != entry["area"]
        ]
        if not blockers:
            blockers = [c for c in all_colors if c != entry["color"]]
        entry["marker"] = random.choice(blockers)

    return entries


def _ledger_cells(row: int, side: str, width: int, entries: list[dict]) -> list[tuple[int, int, int]]:
    order = entries[:]
    random.shuffle(order)
    length = sum(entry["area"] for entry in order)
    start = 0 if side == "left" else width - length
    cells = []
    c = start
    for entry in order:
        for _ in range(entry["area"]):
            cells.append((row, c, int(entry["color"])))
            c += 1
    return cells


def _marker_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return _border_cells(r0, c0, r1, c1)


def _place_frame(
    grid: np.ndarray,
    blocked: np.ndarray,
    entry: dict,
    ledger_row: int,
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    inner_h, inner_w = random.choice(AREA_SHAPES[entry["area"]])
    outer_h, outer_w = inner_h + 2, inner_w + 2

    for _ in range(FRAME_PLACEMENT_ATTEMPTS):
        if ledger_row == 0:
            r_min, r_max = 2, h - outer_h - 1
        else:
            r_min, r_max = 1, h - outer_h - 2
        c_min, c_max = 1, w - outer_w - 1
        if r_min > r_max or c_min > c_max:
            return None
        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        r1, c1 = r0 + outer_h - 1, c0 + outer_w - 1

        pr0, pc0 = max(0, r0 - FRAME_PADDING), max(0, c0 - FRAME_PADDING)
        pr1, pc1 = min(h, r1 + FRAME_PADDING + 1), min(w, c1 + FRAME_PADDING + 1)
        if blocked[pr0:pr1, pc0:pc1].any():
            continue

        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        mr, mc = random.choice(_marker_positions(r0, c0, r1, c1))
        grid[mr, mc] = int(entry["marker"])
        blocked[pr0:pr1, pc0:pc1] = True
        return (r0, c0, r1, c1)

    return None


def _expected_output(inp: np.ndarray, entries: list[dict], frame_bboxes: list[tuple[int, int, int, int]], ledger: list[tuple[int, int, int]]) -> np.ndarray:
    out = inp.copy()
    for r, c, _color in ledger:
        out[r, c] = BACKGROUND
    for entry, (r0, c0, r1, c1) in zip(entries, frame_bboxes):
        out[r0 + 1 : r1, c0 + 1 : c1] = int(entry["color"])
    return out


def _semantic_checks(inp: np.ndarray) -> bool:
    ledger_cells, counts, color_order = _find_ledger(inp)
    frames = _find_frames(inp)
    if len(counts) != len(frames):
        return False

    area_counts = Counter(frame["area"] for frame in frames)
    ledger_count_counts = Counter(counts.values())
    if max(area_counts.values()) < 2 or max(ledger_count_counts.values()) < 2:
        return False

    without_markers = _all_assignments(frames, counts, color_order, use_markers=False, stop_after=3)
    with_markers = _all_assignments(frames, counts, color_order, use_markers=True, stop_after=2)
    if len(without_markers) < 2 or len(with_markers) != 1:
        return False

    out = main(inp)
    if np.any(out[[r for r, _c in ledger_cells], [c for _r, c in ledger_cells]] != BACKGROUND):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(20, GRID_MAX)
    height = random.randint(GRID_MIN, GRID_MAX)
    entries = _choose_entries(width)
    if entries is None:
        return None

    ledger_row = 0 if random.random() < 0.5 else height - 1
    ledger_side = "left" if random.random() < 0.5 else "right"
    ledger = _ledger_cells(ledger_row, ledger_side, width, entries)

    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    blocked[ledger_row, :] = True
    if ledger_row == 0:
        blocked[1, :] = True
    else:
        blocked[height - 2, :] = True

    for r, c, color in ledger:
        grid[r, c] = color

    frame_entries = entries[:]
    random.shuffle(frame_entries)
    frame_bboxes = []
    for entry in frame_entries:
        bbox = _place_frame(grid, blocked, entry, ledger_row)
        if bbox is None:
            return None
        frame_bboxes.append(bbox)

    expected = _expected_output(grid, frame_entries, frame_bboxes, ledger)
    try:
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return None
        if not _semantic_checks(grid):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid hollow ledger input")
