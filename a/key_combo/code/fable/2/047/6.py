from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
HORIZONTAL_SIDES = {"top", "bottom"}
VERTICAL_SIDES = {"left", "right"}
ALL_SIDES = ("top", "bottom", "left", "right")

GEN_MAX_ATTEMPTS = 800
GEN_CANVAS_MAX = 30
GEN_GAP_COUNT_RANGE = (2, 4)
GEN_GAP_LENGTH_CHOICES = np.arange(2, 8)
GEN_RECT_MIN = 5
GEN_RECT_MAX = 11
GEN_PADDING = 1

_GEN_SEEN_INPUTS: set[tuple[tuple[int, int], bytes]] = set()
_GEN_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean mask, returned as (N,2) coord arrays."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _grid_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    g = np.asarray(grid, dtype=int)
    return (int(g.shape[0]), int(g.shape[1])), g.tobytes()


def _border_mask(h: int, w: int) -> np.ndarray:
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[h - 1, :] = True
    border[:, 0] = True
    border[:, w - 1] = True
    return border


def _gap_from_missing(missing: np.ndarray):
    coords = np.argwhere(missing)
    if coords.size == 0:
        return None

    h, w = missing.shape
    sides = []
    if np.all(coords[:, 0] == 0):
        sides.append("top")
    if np.all(coords[:, 0] == h - 1):
        sides.append("bottom")
    if np.all(coords[:, 1] == 0):
        sides.append("left")
    if np.all(coords[:, 1] == w - 1):
        sides.append("right")
    assert len(sides) == 1, "missing border cells must form one side gap"

    side = sides[0]
    axis = coords[:, 1] if side in HORIZONTAL_SIDES else coords[:, 0]
    axis = np.sort(axis.astype(int))
    limit = w if side in HORIZONTAL_SIDES else h
    assert axis[0] >= 1 and axis[-1] <= limit - 2, "gap cannot remove a corner"
    assert np.array_equal(axis, np.arange(axis[0], axis[-1] + 1)), "gap must be contiguous"

    if side == "top":
        local_gap = tuple((0, int(c)) for c in axis)
    elif side == "bottom":
        local_gap = tuple((h - 1, int(c)) for c in axis)
    elif side == "left":
        local_gap = tuple((int(r), 0) for r in axis)
    else:
        local_gap = tuple((int(r), w - 1) for r in axis)

    return {
        "side": side,
        "gap_start": int(axis[0]),
        "gap_length": int(axis.size),
        "orientation": "horizontal" if side in HORIZONTAL_SIDES else "vertical",
        "local_gap": local_gap,
    }


def _detect_locks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    locks: list[dict] = []
    for coords in _components_mask(g == GREY):
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "rectangle outline must have an interior"

        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        border = _border_mask(h, w)
        assert not np.any(local & ~border), "gray object must be a thin rectangle outline"

        inside = g[r0 + 1 : r1, c0 + 1 : c1]
        assert np.all(inside == BLACK), "rectangle interior must be black in the input"

        missing = border & ~local
        gap = _gap_from_missing(missing)
        if gap is None:
            locks.append(
                {
                    "bbox": bbox,
                    "complete": True,
                    "side": None,
                    "gap_start": None,
                    "gap_length": 0,
                    "orientation": None,
                    "gap_coords": (),
                }
            )
            continue

        locks.append(
            {
                "bbox": bbox,
                "complete": False,
                "side": gap["side"],
                "gap_start": gap["gap_start"],
                "gap_length": gap["gap_length"],
                "orientation": gap["orientation"],
                "gap_coords": tuple((r0 + rr, c0 + cc) for rr, cc in gap["local_gap"]),
            }
        )

    assert locks, "expected at least one gray hollow rectangle"
    return sorted(
        locks,
        key=lambda lock: (
            lock["bbox"],
            lock["complete"],
            lock["side"] if lock["side"] is not None else "",
            -lock["gap_length"],
        ),
    )


def _detect_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bars: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY)):
        for coords in _components_mask(g == color):
            bbox = _bbox_coords(coords)
            r0, c0, r1, c1 = bbox
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            area = int(coords.shape[0])
            if h == 1 and area == w:
                orientation = "horizontal"
                length = w
            elif w == 1 and area == h:
                orientation = "vertical"
                length = h
            else:
                raise AssertionError("each colored object must be a straight one-cell-thick bar")
            bars.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "coords": coords,
                    "orientation": orientation,
                    "length": int(length),
                }
            )

    return sorted(bars, key=lambda bar: (bar["length"], bar["orientation"], bar["color"], bar["bbox"]))


def _pair_bars_to_locks(locks: list[dict], bars: list[dict]) -> list[tuple[dict, dict]]:
    gapped = [lock for lock in locks if not lock["complete"]]
    assert len(gapped) == len(bars) and len(gapped) >= 1, "bars must match gapped rectangles one-to-one"

    gap_lengths = [int(lock["gap_length"]) for lock in gapped]
    bar_lengths = [int(bar["length"]) for bar in bars]
    assert len(set(gap_lengths)) == len(gap_lengths), "gap widths must be unique"
    assert len(set(bar_lengths)) == len(bar_lengths), "bar lengths must be unique"
    assert sorted(gap_lengths) == sorted(bar_lengths), "bar lengths must match gap widths exactly"

    by_length = {int(bar["length"]): bar for bar in bars}
    pairs = []
    for lock in gapped:
        bar = by_length[int(lock["gap_length"])]
        assert bar["orientation"] == lock["orientation"], "bar orientation must match its gap"
        pairs.append((lock, bar))
    return pairs


def _apply_rule(grid: np.ndarray, locks: list[dict], bars: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for bar in bars:
        out[bar["coords"][:, 0], bar["coords"][:, 1]] = BLACK

    for lock, bar in _pair_bars_to_locks(locks, bars):
        color = int(bar["color"])
        for r, c in lock["gap_coords"]:
            out[int(r), int(c)] = color
        r0, c0, r1, c1 = lock["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each colored key bar into the uniquely matching rectangle gap, then fill that lock with the key color.

    Input structure:
    - The grid has a black background with several gray hollow rectangular outlines.
    - Most gray rectangles have exactly one contiguous gap on one side, and at most one rectangle may already be complete.
    - Separate colored objects are straight horizontal or vertical bars whose lengths uniquely match the gap widths.

    Rule:
    - Detect every gray hollow rectangle and classify whether it is complete or has one side gap.
    - Detect every colored bar, pair it to the unique gap with the same length, and translate the bar onto the gap while clearing its original cells to black.
    - After a gap is sealed, fill the entire interior of that rectangle with the key bar's color; complete rectangles that never receive a key stay unfilled.

    Key insight:
    - Filling is not caused by mere closure: only a colored key bar entering a matching lock both seals the outline and determines the interior fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    locks = _detect_locks(g)
    bars = _detect_bars(g)
    return _apply_rule(g, locks, bars)


def _randint(low: int, high: int) -> int:
    return int(np.random.randint(int(low), int(high) + 1))


def _sample_gap_sides(count: int) -> list[str]:
    sides = [
        str(np.random.choice(sorted(HORIZONTAL_SIDES))),
        str(np.random.choice(sorted(VERTICAL_SIDES))),
    ]
    while len(sides) < count:
        sides.append(str(np.random.choice(ALL_SIDES)))
    order = np.random.permutation(count)
    return [sides[int(i)] for i in order]


def _choose_dims_for_gap(side: str, gap_length: int) -> tuple[int, int]:
    need_h = max(GEN_RECT_MIN, gap_length + 2 if side in VERTICAL_SIDES else GEN_RECT_MIN)
    need_w = max(GEN_RECT_MIN, gap_length + 2 if side in HORIZONTAL_SIDES else GEN_RECT_MIN)
    h_hi = min(GEN_RECT_MAX, need_h + 3)
    w_hi = min(GEN_RECT_MAX, need_w + 3)
    return _randint(need_h, h_hi), _randint(need_w, w_hi)


def _choose_shared_dims(sides: list[str], lengths: list[int]) -> tuple[int, int]:
    need_h = GEN_RECT_MIN
    need_w = GEN_RECT_MIN
    for side, gap_length in zip(sides, lengths):
        if side in VERTICAL_SIDES:
            need_h = max(need_h, gap_length + 2)
        else:
            need_w = max(need_w, gap_length + 2)
    h_hi = min(GEN_RECT_MAX, need_h + 2)
    w_hi = min(GEN_RECT_MAX, need_w + 2)
    return _randint(need_h, h_hi), _randint(need_w, w_hi)


def _make_lock_patch(h: int, w: int, side: str | None, gap_length: int) -> tuple[np.ndarray, dict]:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = GREY
    patch[h - 1, :] = GREY
    patch[:, 0] = GREY
    patch[:, w - 1] = GREY

    if side is None:
        return patch, {"side": None, "gap_start": None, "gap_length": 0, "orientation": None, "local_gap": ()}

    if side in HORIZONTAL_SIDES:
        start = _randint(1, w - gap_length - 1)
        cols = np.arange(start, start + gap_length, dtype=int)
        row = 0 if side == "top" else h - 1
        patch[row, cols] = BLACK
        local_gap = tuple((int(row), int(c)) for c in cols)
        orientation = "horizontal"
    else:
        start = _randint(1, h - gap_length - 1)
        rows = np.arange(start, start + gap_length, dtype=int)
        col = 0 if side == "left" else w - 1
        patch[rows, col] = BLACK
        local_gap = tuple((int(r), int(col)) for r in rows)
        orientation = "vertical"

    return patch, {
        "side": side,
        "gap_start": int(start),
        "gap_length": int(gap_length),
        "orientation": orientation,
        "local_gap": local_gap,
    }


def _make_bar_patch(length: int, orientation: str, color: int) -> np.ndarray:
    if orientation == "horizontal":
        return np.full((1, length), int(color), dtype=int)
    return np.full((length, 1), int(color), dtype=int)


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int, padding: int) -> bool:
    hh, ww = occupied.shape
    if top < 0 or left < 0 or top + h > hh or left + w > ww:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(hh, top + h + padding)
    c1 = min(ww, left + w + padding)
    return not occupied[r0:r1, c0:c1].any()


def _place_boxes(shape: tuple[int, int], sizes: list[tuple[int, int]], padding: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros(shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    for idx in order:
        h, w = sizes[idx]
        placed = False
        for _ in range(400):
            top = int(np.random.randint(0, shape[0] - h + 1))
            left = int(np.random.randint(0, shape[1] - w + 1))
            if not _can_place(occupied, top, left, h, w, padding):
                continue
            occupied[top : top + h, left : left + w] = True
            placements[idx] = (top, left)
            placed = True
            break
        if not placed:
            return None
    return [p for p in placements if p is not None]


def _render_specs(
    shape: tuple[int, int],
    lock_specs: list[dict],
    bar_specs: list[dict],
    placements: list[tuple[int, int]],
) -> tuple[np.ndarray, list[dict], list[dict]]:
    grid = np.zeros(shape, dtype=int)
    placed_locks = []
    placed_bars = []

    for spec, (top, left) in zip(lock_specs, placements[: len(lock_specs)]):
        patch = spec["patch"]
        h, w = patch.shape
        grid[top : top + h, left : left + w] = np.maximum(grid[top : top + h, left : left + w], patch)
        meta = {
            "bbox": (top, left, top + h - 1, left + w - 1),
            "complete": spec["complete"],
            "side": spec["side"],
            "gap_start": spec["gap_start"],
            "gap_length": spec["gap_length"],
            "orientation": spec["orientation"],
            "gap_coords": tuple((top + rr, left + cc) for rr, cc in spec["local_gap"]),
        }
        placed_locks.append(meta)

    for spec, (top, left) in zip(bar_specs, placements[len(lock_specs) :]):
        patch = spec["patch"]
        h, w = patch.shape
        grid[top : top + h, left : left + w] = patch
        coords = np.argwhere(patch != BLACK) + np.array([top, left], dtype=int)
        placed_bars.append(
            {
                "color": spec["color"],
                "bbox": (top, left, top + h - 1, left + w - 1),
                "coords": coords.astype(int),
                "orientation": spec["orientation"],
                "length": spec["length"],
            }
        )

    return grid, placed_locks, sorted(
        placed_bars, key=lambda bar: (bar["length"], bar["orientation"], bar["color"], bar["bbox"])
    )


def _lock_signature(lock: dict):
    return (
        tuple(int(v) for v in lock["bbox"]),
        bool(lock["complete"]),
        lock["side"],
        -1 if lock["gap_start"] is None else int(lock["gap_start"]),
        int(lock["gap_length"]),
    )


def _bar_signature(bar: dict):
    return (
        int(bar["color"]),
        tuple(int(v) for v in bar["bbox"]),
        bar["orientation"],
        int(bar["length"]),
    )


def _validate_scene(input_grid: np.ndarray, expected: np.ndarray, locks: list[dict], bars: list[dict]) -> None:
    parsed_locks = _detect_locks(input_grid)
    parsed_bars = _detect_bars(input_grid)

    assert sorted(_lock_signature(lock) for lock in parsed_locks) == sorted(_lock_signature(lock) for lock in locks)
    assert sorted(_bar_signature(bar) for bar in parsed_bars) == sorted(_bar_signature(bar) for bar in bars)

    gapped = [lock for lock in parsed_locks if not lock["complete"]]
    complete = [lock for lock in parsed_locks if lock["complete"]]
    assert len(gapped) >= 2, "generator must produce multiple lock-key pairs"
    assert len(complete) <= 1, "generator keeps at most one naturally complete rectangle"
    assert len({lock["gap_length"] for lock in gapped}) == len(gapped), "gap lengths must stay unique"
    assert len({bar["length"] for bar in parsed_bars}) == len(parsed_bars), "bar lengths must stay unique"
    assert {"horizontal", "vertical"} <= {lock["orientation"] for lock in gapped}, "both orientations must appear"

    by_length = {int(bar["length"]): bar for bar in parsed_bars}
    for lock in gapped:
        bar = by_length[int(lock["gap_length"])]
        r0, c0, r1, c1 = lock["bbox"]
        fill = expected[r0 + 1 : r1, c0 + 1 : c1]
        assert fill.size > 0 and np.all(fill == int(bar["color"])), "sealed lock interior must match key color"
        assert all(int(expected[r, c]) == int(bar["color"]) for r, c in lock["gap_coords"]), "gap must be sealed"
        assert np.all(expected[bar["coords"][:, 0], bar["coords"][:, 1]] == BLACK), "key must move away"

    for lock in complete:
        r0, c0, r1, c1 = lock["bbox"]
        assert np.all(expected[r0 + 1 : r1, c0 + 1 : c1] == BLACK), "complete lock must remain unfilled"

    assert np.array_equal(main(input_grid), expected), "solver must realize the intended scene semantics"


def _build_scene_attempt():
    gap_count = _randint(*GEN_GAP_COUNT_RANGE)
    include_complete = bool(np.random.rand() < (0.65 if gap_count < 4 else 0.35))

    gap_lengths = np.random.choice(GEN_GAP_LENGTH_CHOICES, size=gap_count, replace=False).astype(int).tolist()
    gap_sides = _sample_gap_sides(gap_count)
    key_colors = np.random.choice(np.array(KEY_COLORS, dtype=int), size=gap_count, replace=False).astype(int).tolist()

    lock_specs = []
    shared_h, shared_w = _choose_shared_dims(gap_sides[:2], gap_lengths[:2])
    for idx, (side, gap_length) in enumerate(zip(gap_sides, gap_lengths)):
        if idx < 2:
            h, w = shared_h, shared_w
        else:
            h, w = _choose_dims_for_gap(side, gap_length)
        patch, gap_meta = _make_lock_patch(h, w, side, gap_length)
        lock_specs.append(
            {
                "patch": patch,
                "complete": False,
                "side": gap_meta["side"],
                "gap_start": gap_meta["gap_start"],
                "gap_length": gap_meta["gap_length"],
                "orientation": gap_meta["orientation"],
                "local_gap": gap_meta["local_gap"],
            }
        )

    if include_complete:
        if np.random.rand() < 0.75:
            ref = lock_specs[_randint(0, len(lock_specs) - 1)]
            h, w = ref["patch"].shape
        else:
            h, w = _choose_dims_for_gap(str(np.random.choice(ALL_SIDES)), _randint(2, 5))
        patch, gap_meta = _make_lock_patch(h, w, None, 0)
        lock_specs.append(
            {
                "patch": patch,
                "complete": True,
                "side": gap_meta["side"],
                "gap_start": gap_meta["gap_start"],
                "gap_length": gap_meta["gap_length"],
                "orientation": gap_meta["orientation"],
                "local_gap": gap_meta["local_gap"],
            }
        )

    bar_specs = []
    for gap_length, side, color in zip(gap_lengths, gap_sides, key_colors):
        orientation = "horizontal" if side in HORIZONTAL_SIDES else "vertical"
        bar_specs.append(
            {
                "patch": _make_bar_patch(gap_length, orientation, color),
                "orientation": orientation,
                "length": int(gap_length),
                "color": int(color),
            }
        )

    lock_sizes = [tuple(spec["patch"].shape) for spec in lock_specs]
    bar_sizes = [tuple(spec["patch"].shape) for spec in bar_specs]
    min_side = max(18, 12 + 2 * len(lock_specs) + gap_count)
    shape = (_randint(min_side, GEN_CANVAS_MAX), _randint(min_side, GEN_CANVAS_MAX))
    placements = _place_boxes(shape, lock_sizes + bar_sizes, padding=GEN_PADDING)
    if placements is None:
        return None

    grid, locks, bars = _render_specs(shape, lock_specs, bar_specs, placements)
    expected = _apply_rule(grid, locks, bars)
    return grid, expected, locks, bars


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        attempt = _build_scene_attempt()
        if attempt is None:
            continue

        grid, expected, locks, bars = attempt
        in_sig = _grid_signature(grid)
        out_sig = _grid_signature(expected)
        if in_sig in _GEN_SEEN_INPUTS or out_sig in _GEN_SEEN_OUTPUTS:
            continue

        try:
            _validate_scene(grid, expected, locks, bars)
        except AssertionError:
            continue

        _GEN_SEEN_INPUTS.add(in_sig)
        _GEN_SEEN_OUTPUTS.add(out_sig)
        return grid

    raise RuntimeError("failed to generate a valid key-lock ARC input")
