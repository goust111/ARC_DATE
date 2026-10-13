from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLUE = 1
MAX_GENERATION_ATTEMPTS = 400
BACKGROUND_CHOICES = (0, 7)
INPUT_COLOR_POOL = (2, 3, 4, 5, 6, 8, 9)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = cr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(comp)
    return comps


def _segment_from_coords(coords: list[tuple[int, int]], color: int) -> dict:
    pts = sorted((int(r), int(c)) for r, c in coords)
    assert len(pts) >= 2, "expected each diagonal bar to have length >= 2"

    same_diff = len({r - c for r, c in pts}) == 1
    same_sum = len({r + c for r, c in pts}) == 1
    assert same_diff ^ same_sum, "expected each component to be a single diagonal segment"

    if same_diff:
        orientation = "backslash"
        r0, c0 = pts[0]
        expected = [(r0 + i, c0 + i) for i in range(len(pts))]
        line_key = r0 - c0
    else:
        orientation = "slash"
        r0, c0 = pts[0]
        expected = [(r0 + i, c0 - i) for i in range(len(pts))]
        line_key = r0 + c0

    assert pts == expected, "expected every component to be gap-free"
    return {
        "color": int(color),
        "coords": tuple(expected),
        "coord_set": frozenset(expected),
        "length": len(expected),
        "orientation": orientation,
        "line_key": int(line_key),
        "endpoints": (expected[0], expected[-1]),
    }


def _extract_segments(grid: np.ndarray, bg: int) -> list[dict]:
    segments: list[dict] = []
    used_lines = set()
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components8(grid == color):
            seg = _segment_from_coords(comp, color)
            line_id = (seg["orientation"], seg["line_key"])
            assert line_id not in used_lines, "collinear same-orientation bars are ambiguous"
            used_lines.add(line_id)
            segments.append(seg)
    return segments


def _pair_intersection(backslash: dict, slash: dict, shape: tuple[int, int]) -> tuple[str, tuple[int, int] | None]:
    d = backslash["line_key"]
    s = slash["line_key"]
    if (d - s) % 2 != 0:
        return "parity_skip", None
    r = (s + d) // 2
    c = (s - d) // 2
    h, w = shape
    if not (0 <= r < h and 0 <= c < w):
        return "off_grid", None
    return "grid", (int(r), int(c))


def _cells_to_intersection(segment: dict, target: tuple[int, int]) -> list[tuple[int, int]]:
    e0, e1 = segment["endpoints"]
    dist0 = abs(target[0] - e0[0])
    dist1 = abs(target[0] - e1[0])
    assert dist0 != dist1, "intersection should be closer to exactly one endpoint"
    start = e0 if dist0 < dist1 else e1
    dr = _sign(target[0] - start[0])
    dc = _sign(target[1] - start[1])
    cells: list[tuple[int, int]] = []
    r, c = start
    while (r, c) != target:
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _claim(claims: dict[tuple[int, int], int], cell: tuple[int, int], color: int) -> None:
    prev = claims.get(cell)
    if prev is None or prev == color:
        claims[cell] = color
        return
    raise AssertionError(f"conflicting paint claims at {cell}: {prev} vs {color}")


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    segments = _extract_segments(g, bg)

    by_orientation = {"backslash": [], "slash": []}
    for seg in segments:
        by_orientation[seg["orientation"]].append(seg)
    assert by_orientation["backslash"] and by_orientation["slash"], "both diagonal orientations are required"

    claims: dict[tuple[int, int], int] = {}
    events: list[dict] = []

    for backslash in by_orientation["backslash"]:
        for slash in by_orientation["slash"]:
            status, intersection = _pair_intersection(backslash, slash, g.shape)
            event = {
                "backslash_color": backslash["color"],
                "slash_color": slash["color"],
                "backslash_length": backslash["length"],
                "slash_length": slash["length"],
                "kind": status,
            }
            if status != "grid":
                events.append(event)
                continue

            r, c = intersection
            event["intersection"] = intersection
            if int(g[r, c]) != bg:
                event["kind"] = "blocked"
                events.append(event)
                continue

            if backslash["length"] > slash["length"]:
                path = _cells_to_intersection(backslash, intersection)
                for cell in path:
                    assert int(g[cell]) == bg
                    _claim(claims, cell, backslash["color"])
                event["kind"] = "decisive"
                event["winner_orientation"] = "backslash"
                event["winner_color"] = backslash["color"]
            elif slash["length"] > backslash["length"]:
                path = _cells_to_intersection(slash, intersection)
                for cell in path:
                    assert int(g[cell]) == bg
                    _claim(claims, cell, slash["color"])
                event["kind"] = "decisive"
                event["winner_orientation"] = "slash"
                event["winner_color"] = slash["color"]
            else:
                backslash_path = _cells_to_intersection(backslash, intersection)
                slash_path = _cells_to_intersection(slash, intersection)
                for cell in backslash_path[:-1]:
                    assert int(g[cell]) == bg
                    _claim(claims, cell, backslash["color"])
                for cell in slash_path[:-1]:
                    assert int(g[cell]) == bg
                    _claim(claims, cell, slash["color"])
                _claim(claims, intersection, BLUE)
                event["kind"] = "tie"
            events.append(event)

    out = g.copy()
    for (r, c), color in claims.items():
        out[r, c] = color
    return out, events, segments, bg


def main(grid: np.ndarray) -> np.ndarray:
    """Extend only the winning diagonal bars from their virtual crossovers.

    Input structure
    - A background grid contains several one-cell-thick diagonal line segments in
      two orientations: NW-SE and NE-SW.
    - Each colored component is a single gap-free diagonal bar, and the original
      bars do not touch each other.

    Rule
    - Treat each bar as evidence for its full invisible diagonal line.
    - For every opposite-orientation pair, compute the virtual intersection of
      those two full lines. If the line parities disagree or the intersection is
      outside the grid or already occupied in the input, skip that pair.
    - Otherwise compare the two original bar lengths only. The longer bar extends
      along its own diagonal from its nearer endpoint to the intersection, using
      its own color. If the lengths tie, both bars extend to that cell and the
      intersection itself becomes blue.
    - Newly drawn cells never affect any later comparison; every duel is judged
      from the original input bars alone.

    Key insight
    - The visible segments are only partial witnesses: the real interaction is
      between their unseen full diagonal lines, but every winner is decided by
      the input bar lengths rather than by any growth that happens later.
    """
    out, _events, _segments, _bg = _solve(grid)
    return out


def _halo(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    halo.add((rr, cc))
    return halo


def _segment_between_backslashes(
    a: tuple[int, int], b: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    steps = b[0] - a[0]
    assert steps == b[1] - a[1] and steps >= length + 1
    start = random.randint(1, steps - length)
    return [(a[0] + i, a[1] + i) for i in range(start, start + length)]


def _segment_between_slashes(
    a: tuple[int, int], b: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    steps = b[0] - a[0]
    assert steps == a[1] - b[1] and steps >= length + 1
    start = random.randint(1, steps - length)
    return [(a[0] + i, a[1] - i) for i in range(start, start + length)]


def _place_skip_slash(
    shape: tuple[int, int],
    forbidden_halo: set[tuple[int, int]],
    parity: int,
    existing_slash_keys: set[int],
) -> list[tuple[int, int]] | None:
    h, w = shape
    for _ in range(200):
        length = random.randint(2, 5)
        r0 = random.randint(0, h - length)
        c0 = random.randint(length - 1, w - 1)
        if (r0 + c0) % 2 != parity:
            continue
        if (r0 + c0) in existing_slash_keys:
            continue
        cells = [(r0 + i, c0 - i) for i in range(length)]
        if any(cell in forbidden_halo for cell in cells):
            continue
        return cells
    return None


def _apply_symmetry(grid: np.ndarray) -> np.ndarray:
    out = grid
    k = random.randint(0, 3)
    if k:
        out = np.rot90(out, k=k)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=int)


def _build_candidate() -> np.ndarray | None:
    tie_len = random.randint(4, 6)
    medium_len = random.randint(3, tie_len - 1)
    short_len = random.randint(2, medium_len - 1)

    step_s = random.randint(tie_len + 1, tie_len + 3)
    step_d = random.randint(tie_len + 1, tie_len + 2)

    min_h = step_s + step_d + 3
    min_w = step_s + step_d + 4
    h = random.randint(max(18, min_h), min(24, 30))
    w = random.randint(max(18, min_w), min(24, 30))

    base_r = random.randint(1, h - (step_s + step_d) - 2)
    base_c = random.randint(step_d + 1, w - step_s - 2)
    i00 = (base_r, base_c)
    i01 = (base_r + step_s, base_c + step_s)
    i10 = (base_r + step_d, base_c - step_d)
    i11 = (base_r + step_s + step_d, base_c + step_s - step_d)

    bg = random.choice(BACKGROUND_CHOICES)
    colors = random.sample([c for c in INPUT_COLOR_POOL if c != bg], 5)
    b0_color, b1_color, s0_color, s1_color, s2_color = colors

    b0 = _segment_between_backslashes(i00, i01, tie_len)
    b1 = _segment_between_backslashes(i10, i11, short_len)
    s0 = _segment_between_slashes(i00, i10, tie_len)
    s1 = _segment_between_slashes(i01, i11, medium_len)

    occupied = b0 + b1 + s0 + s1
    halo = _halo(occupied, (h, w))
    parity = (i00[0] - i00[1]) % 2
    s2 = _place_skip_slash((h, w), halo, 1 - parity, {i00[0] + i00[1], i01[0] + i01[1]})
    if s2 is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for color, cells in (
        (b0_color, b0),
        (b1_color, b1),
        (s0_color, s0),
        (s1_color, s1),
        (s2_color, s2),
    ):
        for r, c in cells:
            if int(grid[r, c]) != bg:
                return None
            grid[r, c] = color
    return _apply_symmetry(grid)


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        out, events, segments, bg = _solve(grid)
    except AssertionError:
        return False

    if len(segments) != 5:
        return False
    if len({seg["color"] for seg in segments}) != 5:
        return False

    counts = Counter(seg["orientation"] for seg in segments)
    if counts["backslash"] != 2 or counts["slash"] != 3:
        return False

    if any(event["kind"] in {"off_grid", "blocked"} for event in events):
        return False

    ties = [event for event in events if event["kind"] == "tie"]
    decisive = [event for event in events if event["kind"] == "decisive"]
    parity_skips = [event for event in events if event["kind"] == "parity_skip"]

    if len(ties) != 1:
        return False
    if np.count_nonzero(out == BLUE) != 1:
        return False
    if not any(event["winner_orientation"] == "backslash" for event in decisive):
        return False
    if not any(event["winner_orientation"] == "slash" for event in decisive):
        return False
    if len(parity_skips) < 2:
        return False

    changed = out != grid
    if int(np.count_nonzero(changed)) < 4:
        return False
    if not np.all(grid[changed] == bg):
        return False

    tie_cell = ties[0]["intersection"]
    if int(out[tie_cell]) != BLUE:
        return False

    for event in decisive:
        if int(out[event["intersection"]]) != int(event["winner_color"]):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None and _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid invisible-diagonal-duel input")
