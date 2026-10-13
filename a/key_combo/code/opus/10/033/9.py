import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = random.Random()
MAX_GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
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
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary_coords(frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords += [(r1, c) for c in range(c0, c1 + 1)]
    coords += [(r, c0) for r in range(r0 + 1, r1)]
    coords += [(r, c1) for r in range(r0 + 1, r1)]
    return coords


def _make_frame(r0: int, c0: int, h: int, w: int, name: str = "") -> dict:
    return {
        "name": name,
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r0 + h - 1),
        "c1": int(c0 + w - 1),
        "h": int(h),
        "w": int(w),
        "area": int(h * w),
        "size_key": (int(w), int(h)),
    }


def _extract_frame(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 5 and w >= 5, "frame too small"

    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True

    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    assert np.array_equal(patch, expected), "expected a one-cell-thick hollow rectangle"

    return _make_frame(r0, c0, h, w)


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, GREY}, "expected only black background and grey frames"

    comps = _components(g == GREY)
    assert comps, "expected at least one grey frame"
    frames = [_extract_frame(g, comp) for comp in comps]
    return frames


def _shell_color(depth: int, theme: int) -> int:
    phase = depth % 4
    if phase in (0, 2):
        return GREY
    if phase == 1:
        return theme
    return BLACK


def _frame_color(frame: dict, theme: int, r: int, c: int) -> int:
    depth = min(r - frame["r0"], frame["r1"] - r, c - frame["c0"], frame["c1"] - c)
    return _shell_color(depth, theme)


def _render(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    size_counts = Counter(frame["size_key"] for frame in frames)
    boundary_mask = np.zeros(shape, dtype=bool)
    out = np.full(shape, BLACK, dtype=int)

    for frame in frames:
        for r, c in _boundary_coords(frame):
            boundary_mask[r, c] = True
            out[r, c] = GREY

    draw_order = sorted(frames, key=lambda f: (f["area"], f["r0"], f["c0"]), reverse=True)
    for frame in draw_order:
        theme = RED if size_counts[frame["size_key"]] == 1 else BLUE
        for r in range(frame["r0"], frame["r1"] + 1):
            for c in range(frame["c0"], frame["c1"] + 1):
                if boundary_mask[r, c]:
                    continue
                out[r, c] = _frame_color(frame, theme, r, c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey hollow rectangle with concentric distance shells, using red for unique sizes and blue for repeated sizes.

    Input structure:
    - The grid has black background and several one-cell-thick grey hollow rectangular frames.
    - Each frame is an axis-aligned closed rectangle ring, and different frame outlines do not share pixels.
    - Frames may be disjoint or nested so that their interior regions overlap.

    Rule:
    - Extract every grey frame and record its exact bounding-box size as the ordered pair (width, height).
    - Frames whose size pair appears once use the shell cycle grey, red, grey, black; frames whose size pair appears multiple times use grey, blue, grey, black.
    - Apply the cycle by Chebyshev distance from each frame border, keep all boundary pixels grey, and resolve overlapping non-boundary cells by giving priority to the frame with smaller area, then smaller top row, then smaller left column.

    Key insight:
    - Size frequency chooses the color theme, but overlap priority is independent of that theme and depends only on frame area and position.
    """
    frames = _extract_frames(np.asarray(grid, dtype=int))
    size_counts = Counter(frame["size_key"] for frame in frames)
    assert any(count == 1 for count in size_counts.values()), "expected at least one unique size"
    assert any(count >= 2 for count in size_counts.values()), "expected at least one repeated size"
    return _render(np.asarray(grid, dtype=int).shape, frames)


def _strictly_contains(outer: dict, inner: dict) -> bool:
    return (
        outer["r0"] + 1 < inner["r0"]
        and outer["c0"] + 1 < inner["c0"]
        and outer["r1"] - 1 > inner["r1"]
        and outer["c1"] - 1 > inner["c1"]
    )


def _bboxes_overlap(a: dict, b: dict) -> bool:
    return not (
        a["r1"] < b["r0"]
        or b["r1"] < a["r0"]
        or a["c1"] < b["c0"]
        or b["c1"] < a["c0"]
    )


def _expanded_overlap(a: dict, b: dict, margin: int = 1) -> bool:
    return not (
        a["r1"] + margin < b["r0"] - margin
        or b["r1"] + margin < a["r0"] - margin
        or a["c1"] + margin < b["c0"] - margin
        or b["c1"] + margin < a["c0"] - margin
    )


def _outlines_intersect(a: dict, b: dict) -> bool:
    return bool(set(_boundary_coords(a)) & set(_boundary_coords(b)))


def _interior_overlap_cells(a: dict, b: dict) -> list[tuple[int, int]]:
    r0 = max(a["r0"] + 1, b["r0"] + 1)
    r1 = min(a["r1"] - 1, b["r1"] - 1)
    c0 = max(a["c0"] + 1, b["c0"] + 1)
    c1 = min(a["c1"] - 1, b["c1"] - 1)
    if r0 > r1 or c0 > c1:
        return []
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _valid_with_existing(candidate: dict, placed: list[dict], allowed_nested: frozenset[str]) -> bool:
    for other in placed:
        if _outlines_intersect(candidate, other):
            return False
        contains = _strictly_contains(candidate, other) or _strictly_contains(other, candidate)
        if contains:
            if frozenset({candidate["name"], other["name"]}) != allowed_nested:
                return False
            continue
        if _bboxes_overlap(candidate, other):
            return False
        if _expanded_overlap(candidate, other, margin=1):
            return False
    return True


def _choose_simple_scenario() -> tuple[list[dict], tuple[str, str]]:
    unique_inner = RNG.random() < 0.5
    total = RNG.choice([3, 4])
    repeated_count = 2 if total == 3 else 3

    if unique_inner:
        rep_h = RNG.randint(9, 13)
        rep_w = RNG.randint(9, 13)
        uniq_h = RNG.randint(5, rep_h - 4)
        uniq_w = RNG.randint(5, rep_w - 4)
        while (uniq_w, uniq_h) == (rep_w, rep_h):
            uniq_h = RNG.randint(5, rep_h - 4)
            uniq_w = RNG.randint(5, rep_w - 4)
        specs = [_make_frame(0, 0, rep_h, rep_w, "rep_anchor"), _make_frame(0, 0, uniq_h, uniq_w, "unique")]
        for idx in range(repeated_count - 1):
            specs.append(_make_frame(0, 0, rep_h, rep_w, f"rep_extra_{idx}"))
        return specs, ("rep_anchor", "unique")

    uniq_h = RNG.randint(10, 13)
    uniq_w = RNG.randint(10, 13)
    rep_h = RNG.randint(5, uniq_h - 4)
    rep_w = RNG.randint(5, uniq_w - 4)
    specs = [_make_frame(0, 0, uniq_h, uniq_w, "unique"), _make_frame(0, 0, rep_h, rep_w, "rep_anchor")]
    for idx in range(repeated_count - 1):
        specs.append(_make_frame(0, 0, rep_h, rep_w, f"rep_extra_{idx}"))
    return specs, ("unique", "rep_anchor")


def _choose_transpose_scenario() -> tuple[list[dict], tuple[str, str]]:
    rep_a_h = RNG.randint(5, 9)
    rep_a_w = RNG.randint(5, 9)
    while rep_a_h == rep_a_w:
        rep_a_w = RNG.randint(5, 9)

    uniq_h, uniq_w = rep_a_w, rep_a_h
    outer_h = RNG.randint(max(uniq_h + 4, 10), 13)
    outer_w = RNG.randint(max(uniq_w + 4, 10), 13)

    specs = [
        _make_frame(0, 0, outer_h, outer_w, "rep_c_anchor"),
        _make_frame(0, 0, uniq_h, uniq_w, "unique"),
        _make_frame(0, 0, outer_h, outer_w, "rep_c_extra"),
        _make_frame(0, 0, rep_a_h, rep_a_w, "rep_a_0"),
        _make_frame(0, 0, rep_a_h, rep_a_w, "rep_a_1"),
    ]
    return specs, ("rep_c_anchor", "unique")


def _choose_scenario() -> tuple[list[dict], tuple[str, str]]:
    roll = RNG.random()
    if roll < 0.4:
        return _choose_simple_scenario()
    if roll < 0.8:
        return _choose_simple_scenario()
    return _choose_transpose_scenario()


def _place_specs(specs: list[dict], nested_pair: tuple[str, str]) -> list[dict] | None:
    count = len(specs)
    if count == 3:
        h_range = (18, 24)
        w_range = (18, 24)
    elif count == 4:
        h_range = (20, 27)
        w_range = (20, 27)
    else:
        h_range = (23, 30)
        w_range = (23, 30)

    allowed_nested = frozenset(nested_pair)
    parent_name, child_name = nested_pair
    parent_spec = next(spec for spec in specs if spec["name"] == parent_name)
    child_spec = next(spec for spec in specs if spec["name"] == child_name)
    extras = [spec for spec in specs if spec["name"] not in allowed_nested]

    for _ in range(200):
        h = RNG.randint(*h_range)
        w = RNG.randint(*w_range)
        placed: list[dict] = []

        parent_r0 = RNG.randint(0, h - parent_spec["h"])
        parent_c0 = RNG.randint(0, w - parent_spec["w"])
        parent = _make_frame(parent_r0, parent_c0, parent_spec["h"], parent_spec["w"], parent_name)
        placed.append(parent)

        child_r0 = RNG.randint(parent["r0"] + 2, parent["r1"] - child_spec["h"] - 1)
        child_c0 = RNG.randint(parent["c0"] + 2, parent["c1"] - child_spec["w"] - 1)
        child = _make_frame(child_r0, child_c0, child_spec["h"], child_spec["w"], child_name)
        if not _valid_with_existing(child, placed, allowed_nested):
            continue
        placed.append(child)

        ok = True
        order = extras[:]
        RNG.shuffle(order)
        for spec in order:
            found = False
            for _ in range(120):
                cand = _make_frame(
                    RNG.randint(0, h - spec["h"]),
                    RNG.randint(0, w - spec["w"]),
                    spec["h"],
                    spec["w"],
                    spec["name"],
                )
                if _valid_with_existing(cand, placed, allowed_nested):
                    placed.append(cand)
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok:
            return placed
    return None


def _draw_input(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for frame in frames:
        for r, c in _boundary_coords(frame):
            grid[r, c] = GREY
    return grid


def _priority_key(frame: dict) -> tuple[int, int, int]:
    return frame["area"], frame["r0"], frame["c0"]


def _has_visible_priority_conflict(frames: list[dict], out: np.ndarray) -> bool:
    size_counts = Counter(frame["size_key"] for frame in frames)
    boundary_mask = np.zeros(out.shape, dtype=bool)
    for frame in frames:
        for r, c in _boundary_coords(frame):
            boundary_mask[r, c] = True

    for i, a in enumerate(frames):
        for b in frames[i + 1 :]:
            cells = _interior_overlap_cells(a, b)
            if not cells:
                continue
            winner = a if _priority_key(a) < _priority_key(b) else b
            loser = b if winner is a else a
            winner_theme = RED if size_counts[winner["size_key"]] == 1 else BLUE
            loser_theme = RED if size_counts[loser["size_key"]] == 1 else BLUE
            for r, c in cells:
                if boundary_mask[r, c]:
                    continue
                wc = _frame_color(winner, winner_theme, r, c)
                lc = _frame_color(loser, loser_theme, r, c)
                if wc != lc and int(out[r, c]) == wc:
                    return True
    return False


def _semantic_ok(grid: np.ndarray, frames: list[dict]) -> bool:
    extracted = _extract_frames(grid)
    if len(extracted) != len(frames):
        return False

    size_counts = Counter(frame["size_key"] for frame in extracted)
    if sum(count == 1 for count in size_counts.values()) != 1:
        return False
    if sum(count >= 2 for count in size_counts.values()) < 1:
        return False

    if not any(_interior_overlap_cells(a, b) for i, a in enumerate(extracted) for b in extracted[i + 1 :]):
        return False

    out = main(grid)
    if not np.any(out == RED) or not np.any(out == BLUE):
        return False
    if not _has_visible_priority_conflict(extracted, out):
        return False

    overlap_count = sum(bool(_interior_overlap_cells(a, b)) for i, a in enumerate(extracted) for b in extracted[i + 1 :])
    return overlap_count >= 1


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        specs, nested_pair = _choose_scenario()
        placed = _place_specs(specs, nested_pair)
        if placed is None:
            continue
        h = max(frame["r1"] for frame in placed) + 1
        w = max(frame["c1"] for frame in placed) + 1
        grid = _draw_input((h, w), placed)
        if _semantic_ok(grid, placed):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
