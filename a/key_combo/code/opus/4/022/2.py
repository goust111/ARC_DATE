from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
TEAL = 8

GEN_MAX_ATTEMPTS = 400
PATTERN_LIBRARY = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 1), (2, 1), (1, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (1, 1), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _chamber_key(chamber: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = chamber["bbox"]
    return int(r0), int(c0)


def _analyze_grid(grid: np.ndarray) -> tuple[list[dict], tuple[tuple[int, int], ...]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all(g[0, :] == TEAL) and np.all(g[-1, :] == TEAL), "top and bottom borders must be teal"
    assert np.all(g[:, 0] == TEAL) and np.all(g[:, -1] == TEAL), "left and right borders must be teal"

    chambers: list[dict] = []
    for coords in _components_mask(g != TEAL):
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert int(coords.shape[0]) == h * w, "each chamber interior must be a solid rectangle"
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        green = np.argwhere(sub == GREEN)
        red = np.argwhere(sub == RED)
        assert not (green.size and red.size), "a chamber cannot mix green template pixels and red markers"
        chambers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (h, w),
                "green": tuple(sorted((int(r), int(c)) for r, c in green)),
                "red": tuple(sorted((int(r), int(c)) for r, c in red)),
            }
        )

    assert chambers, "expected at least one chamber"
    sizes = {tuple(map(int, chamber["shape"])) for chamber in chambers}
    assert len(sizes) == 1, "all chambers must have the same interior size"

    templates = [chamber for chamber in chambers if chamber["green"]]
    assert len(templates) == 1, "expected exactly one template chamber"
    pattern = templates[0]["green"]
    assert 2 <= len(pattern) <= 4, "template pattern must contain 2 to 4 cells"
    return sorted(chambers, key=_chamber_key), pattern


def main(grid: np.ndarray) -> np.ndarray:
    """
    Process each teal-bounded chamber by branching on its red marker count.

    Input structure:
    - Black background with a 1-pixel-thick teal grid skeleton that partitions the canvas into equal rectangular chambers.
    - Exactly one chamber contains a small 4-connected green pattern and no red markers; this is the template chamber.
    - Every other chamber contains only black cells and 0 to 3 red marker pixels.

    Rule:
    - Identify the unique green template chamber and record the green pattern's coordinates relative to that chamber's top-left interior cell.
    - For every non-template chamber, count its red markers: if the count is 0, stamp the green template pattern into the chamber at the same relative positions; if the count is 1, fill the entire chamber interior red; if the count is 2 or more, erase all red markers to black.
    - Leave the teal skeleton unchanged and keep the template chamber unchanged.

    Key insight:
    - The same local measurement, red-marker count inside one chamber, acts as a three-way selector that chooses between stamping, filling, and erasing.
    """
    chambers, pattern = _analyze_grid(grid)
    out = np.asarray(grid, dtype=int).copy()

    template_bbox = None
    for chamber in chambers:
        if chamber["green"]:
            template_bbox = chamber["bbox"]
            break
    assert template_bbox is not None

    for chamber in chambers:
        bbox = chamber["bbox"]
        if bbox == template_bbox:
            continue
        r0, c0, r1, c1 = bbox
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        red_count = len(chamber["red"])
        if red_count == 0:
            for dr, dc in pattern:
                sub[dr, dc] = GREEN
        elif red_count == 1:
            sub[:, :] = RED
        else:
            for dr, dc in chamber["red"]:
                sub[dr, dc] = BLACK

    return out


def _grid_shape(room_rows: int, room_cols: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return room_rows * cell_h + room_rows + 1, room_cols * cell_w + room_cols + 1


def _room_bboxes(room_rows: int, room_cols: int, cell_h: int, cell_w: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    stride_r = cell_h + 1
    stride_c = cell_w + 1
    for rr in range(room_rows):
        for cc in range(room_cols):
            r0 = 1 + rr * stride_r
            c0 = 1 + cc * stride_c
            boxes.append((r0, c0, r0 + cell_h - 1, c0 + cell_w - 1))
    return boxes


def _shift_pattern(base: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r + dr, c + dc) for r, c in base))


def _connected(pattern: tuple[tuple[int, int], ...]) -> bool:
    cells = set(pattern)
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _choose_pattern(cell_h: int, cell_w: int) -> tuple[tuple[int, int], ...]:
    candidates = []
    for base in PATTERN_LIBRARY:
        max_r = max(r for r, _c in base)
        max_c = max(c for _r, c in base)
        if max_r >= cell_h or max_c >= cell_w:
            continue
        for off_r in range(cell_h - max_r):
            for off_c in range(cell_w - max_c):
                shifted = _shift_pattern(base, off_r, off_c)
                if _connected(shifted):
                    candidates.append(shifted)
    assert candidates, "expected at least one pattern that fits the chamber"
    return random.choice(candidates)


def _marker_coords(count: int, cell_h: int, cell_w: int) -> tuple[tuple[int, int], ...]:
    positions = [(r, c) for r in range(cell_h) for c in range(cell_w)]
    chosen = random.sample(positions, count)
    return tuple(sorted(chosen))


def _paint_room(arr: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    arr[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _build_pair(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    room_rows = int(spec["room_rows"])
    room_cols = int(spec["room_cols"])
    cell_h = int(spec["cell_h"])
    cell_w = int(spec["cell_w"])
    template_idx = int(spec["template_idx"])
    pattern = tuple(spec["pattern"])
    counts = list(map(int, spec["counts"]))
    markers = [tuple(room) for room in spec["markers"]]

    h, w = _grid_shape(room_rows, room_cols, cell_h, cell_w)
    inp = np.zeros((h, w), dtype=int)
    inp[:: cell_h + 1, :] = TEAL
    inp[:, :: cell_w + 1] = TEAL
    out = inp.copy()

    boxes = _room_bboxes(room_rows, room_cols, cell_h, cell_w)
    for idx, bbox in enumerate(boxes):
        r0, c0, _r1, _c1 = bbox
        if idx == template_idx:
            for dr, dc in pattern:
                inp[r0 + dr, c0 + dc] = GREEN
                out[r0 + dr, c0 + dc] = GREEN
            continue

        for dr, dc in markers[idx]:
            inp[r0 + dr, c0 + dc] = RED

        count = counts[idx]
        if count == 0:
            for dr, dc in pattern:
                out[r0 + dr, c0 + dc] = GREEN
        elif count == 1:
            _paint_room(out, bbox, RED)
        else:
            pass

    return inp, out


def _sample_counts(non_template_count: int) -> list[int]:
    assert non_template_count >= 3
    counts = [0, 1, random.choice((2, 3))]
    while len(counts) < non_template_count:
        counts.append(random.choices((0, 1, 2, 3), weights=(3, 3, 2, 2), k=1)[0])
    random.shuffle(counts)
    return counts


def _sample_spec() -> dict:
    cell_h = random.choice((3, 4))
    cell_w = random.choice((3, 4))
    feasible: list[tuple[int, int]] = []
    for room_rows in range(2, 6):
        for room_cols in range(2, 6):
            if room_rows * room_cols < 4:
                continue
            h, w = _grid_shape(room_rows, room_cols, cell_h, cell_w)
            if h <= 30 and w <= 30:
                feasible.append((room_rows, room_cols))
    room_rows, room_cols = random.choice(feasible)

    boxes = _room_bboxes(room_rows, room_cols, cell_h, cell_w)
    template_idx = random.randrange(len(boxes))
    pattern = _choose_pattern(cell_h, cell_w)

    counts = [-1] * len(boxes)
    markers: list[tuple[tuple[int, int], ...]] = [tuple() for _ in boxes]
    non_template_indices = [idx for idx in range(len(boxes)) if idx != template_idx]
    sampled_counts = _sample_counts(len(non_template_indices))

    for idx, count in zip(non_template_indices, sampled_counts):
        counts[idx] = int(count)
        markers[idx] = _marker_coords(int(count), cell_h, cell_w)

    return {
        "room_rows": room_rows,
        "room_cols": room_cols,
        "cell_h": cell_h,
        "cell_w": cell_w,
        "template_idx": template_idx,
        "pattern": pattern,
        "counts": counts,
        "markers": markers,
    }


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, spec: dict) -> bool:
    room_rows = int(spec["room_rows"])
    room_cols = int(spec["room_cols"])
    cell_h = int(spec["cell_h"])
    cell_w = int(spec["cell_w"])
    template_idx = int(spec["template_idx"])
    pattern = tuple(spec["pattern"])
    counts = list(map(int, spec["counts"]))
    markers = [tuple(room) for room in spec["markers"]]
    boxes = _room_bboxes(room_rows, room_cols, cell_h, cell_w)

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.all(inp[:: cell_h + 1, :] == TEAL) or not np.all(inp[:, :: cell_w + 1] == TEAL):
        return False
    if np.count_nonzero(inp == GREEN) != len(pattern):
        return False

    seen_categories = set()
    for idx, bbox in enumerate(boxes):
        r0, c0, r1, c1 = bbox
        sub_in = inp[r0 : r1 + 1, c0 : c1 + 1]
        sub_out = expected[r0 : r1 + 1, c0 : c1 + 1]
        green_rel = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(sub_in == GREEN)))
        red_rel = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(sub_in == RED)))
        if idx == template_idx:
            if green_rel != pattern or red_rel:
                return False
            if not np.array_equal(sub_in, sub_out):
                return False
            continue

        count = counts[idx]
        if green_rel:
            return False
        if red_rel != markers[idx]:
            return False
        seen_categories.add(2 if count >= 2 else count)

        if count == 0:
            want = np.zeros((cell_h, cell_w), dtype=int)
            for dr, dc in pattern:
                want[dr, dc] = GREEN
            if not np.array_equal(sub_out, want):
                return False
        elif count == 1:
            if not np.all(sub_out == RED):
                return False
        else:
            if np.any(sub_out == RED) or np.any(sub_out == GREEN) or np.any(sub_out == TEAL):
                return False

    if seen_categories != {0, 1, 2}:
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        spec = _sample_spec()
        inp, expected = _build_pair(spec)
        if _validate_semantics(inp, expected, spec):
            return inp
    raise RuntimeError(f"failed to generate a valid sample after {GEN_MAX_ATTEMPTS} attempts")
