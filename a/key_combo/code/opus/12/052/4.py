from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 7)
QUADRANT_ORDER = ("TL", "TR", "BL", "BR")
CONFLICT_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(
            (coords[:, 0] == 0)
            | (coords[:, 1] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == w - 1)
        )
    )


def _has_enclosed_background(grid: np.ndarray) -> bool:
    for coords in _components(grid == BG):
        if not _touches_edge(coords, grid.shape):
            return True
    return False


def _quadrants(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h % 2 == 0 and w % 2 == 0, "input must split evenly into four quadrants"
    hh, ww = h // 2, w // 2
    return [
        g[:hh, :ww],
        g[:hh, ww:],
        g[hh:, :ww],
        g[hh:, ww:],
    ]


def _fold_overlay(grid: np.ndarray) -> np.ndarray:
    quads = _quadrants(grid)
    h, w = quads[0].shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            vals = [int(quad[r, c]) for quad in quads]
            if WALL in vals:
                out[r, c] = WALL
                continue
            color = BG
            for val in vals:
                if val not in (BG, WALL):
                    color = val
                    break
            out[r, c] = color
    return out


def _chamber_votes(overlay: np.ndarray) -> list[dict]:
    info: list[dict] = []
    for coords in _components(overlay == BG):
        adjacent_markers: dict[tuple[int, int], int] = {}
        for r_raw, c_raw in coords:
            r, c = int(r_raw), int(c_raw)
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(overlay.shape, nr, nc):
                    value = int(overlay[nr, nc])
                    if value not in (BG, WALL):
                        adjacent_markers[(nr, nc)] = value
        assert adjacent_markers, "each chamber must touch at least one visible marker"
        counts = Counter(adjacent_markers.values())
        top = max(counts.values())
        winners = {color for color, count in counts.items() if count == top}
        if len(winners) == 1:
            fill = next(iter(winners))
        else:
            fill = next(
                color
                for pos, color in sorted(adjacent_markers.items())
                if color in winners
            )
        info.append(
            {
                "coords": coords,
                "markers": adjacent_markers,
                "counts": counts,
                "fill": int(fill),
            }
        )
    return info


def _fill_overlay(overlay: np.ndarray) -> np.ndarray:
    out = np.array(overlay, copy=True)
    for chamber in _chamber_votes(overlay):
        coords = chamber["coords"]
        out[coords[:, 0], coords[:, 1]] = int(chamber["fill"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold four quadrant wall fragments into one canvas, then fill each emergent chamber by the majority color of its adjacent visible markers.

    Input structure:
    - The input is an even-by-even black grid conceptually split into four equal quadrants.
    - Each quadrant contains grey wall fragments and 1-3 colored marker pixels chosen from blue, red, green, yellow, and orange.
    - No single quadrant is sufficient on its own; the intended chamber topology appears only after all four quadrants are folded onto the same coordinates.

    Rule:
    - Fold corresponding cells from TL, TR, BL, and BR onto one half-size canvas.
    - If any folded cell is grey, the output cell is a grey wall; otherwise the first visible marker under priority TL > TR > BL > BR wins, and empty folds stay black.
    - On the folded canvas, find every maximal 4-connected black chamber and collect the orthogonally adjacent visible marker pixels.
    - Fill the whole chamber with the majority adjacent marker color; when colors tie, use the color of the topmost, then leftmost, tied adjacent marker pixel. Preserve walls and visible markers.

    Key insight:
    - The walls and the voting markers both have to be reasoned about after folding: folding creates the chamber geometry, while priority-resolved marker overlaps create the chamber votes.
    """
    overlay = _fold_overlay(np.asarray(grid, dtype=int))
    return _fill_overlay(overlay)


def _choose_layout() -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], set[int], set[int]]:
    kind = random.choice(("vertical2", "horizontal2", "grid4"))
    if kind == "vertical2":
        n = random.randint(9, 12)
        cut = random.randint(4, n - 5)
        row_bands = [(1, n - 2)]
        col_bands = [(1, cut - 1), (cut + 1, n - 2)]
        return n, row_bands, col_bands, {0, n - 1}, {0, cut, n - 1}
    if kind == "horizontal2":
        n = random.randint(9, 12)
        cut = random.randint(4, n - 5)
        row_bands = [(1, cut - 1), (cut + 1, n - 2)]
        col_bands = [(1, n - 2)]
        return n, row_bands, col_bands, {0, cut, n - 1}, {0, n - 1}

    n = random.randint(8, 12)
    hcut = random.randint(3, n - 4)
    vcut = random.randint(3, n - 4)
    row_bands = [(1, hcut - 1), (hcut + 1, n - 2)]
    col_bands = [(1, vcut - 1), (vcut + 1, n - 2)]
    return n, row_bands, col_bands, {0, hcut, n - 1}, {0, vcut, n - 1}


def _build_chambers(
    n: int,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> list[dict]:
    chambers: list[dict] = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            candidates: list[tuple[int, int]] = []
            if r0 == 1:
                candidates.extend((0, c) for c in range(c0, c1 + 1))
            if r1 == n - 2:
                candidates.extend((n - 1, c) for c in range(c0, c1 + 1))
            if c0 == 1:
                candidates.extend((r, 0) for r in range(r0, r1 + 1))
            if c1 == n - 2:
                candidates.extend((r, n - 1) for r in range(r0, r1 + 1))
            chambers.append(
                {
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "candidates": sorted(set(candidates)),
                    "probe": (r0, c0),
                }
            )
    return chambers


def _pick_distinct_positions(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if len(candidates) < count:
        return None
    if count == 2:
        return [candidates[0], random.choice(candidates[1:])]
    if count == 3:
        middle = candidates[1:-1]
        if not middle:
            return None
        return [candidates[0], random.choice(middle), candidates[-1]]
    raise AssertionError("unsupported marker count")


def _marker_specs(chambers: list[dict]) -> tuple[list[dict], int, tuple[int, int], tuple[int, int]]:
    n_chambers = len(chambers)
    special_idx = random.randrange(n_chambers)
    majority_choices = [i for i in range(n_chambers) if i != special_idx]
    majority_idx = random.choice(majority_choices)

    specs: list[dict] = []
    fill_targets: list[int] = []
    conflict_pos: tuple[int, int] | None = None
    alt_info: tuple[int, int] | None = None

    for idx, chamber in enumerate(chambers):
        candidates = chamber["candidates"]
        if idx == special_idx:
            positions = _pick_distinct_positions(candidates, 2)
            assert positions is not None
            visible, alt = random.sample(MARKER_COLORS, 2)
            specs.append(
                {
                    "pos": positions[0],
                    "visible": visible,
                    "alt": alt,
                    "chamber": idx,
                    "conflict": True,
                }
            )
            specs.append(
                {
                    "pos": positions[1],
                    "visible": alt,
                    "alt": None,
                    "chamber": idx,
                    "conflict": False,
                }
            )
            fill_targets.append(visible)
            conflict_pos = positions[0]
            alt_info = (visible, alt)
            continue

        if idx == majority_idx:
            positions = _pick_distinct_positions(candidates, 3)
            assert positions is not None
            major, minor = random.sample(MARKER_COLORS, 2)
            colors = [major, major, minor]
            random.shuffle(colors)
            for pos, color in zip(positions, colors):
                specs.append({"pos": pos, "visible": color, "alt": None, "chamber": idx, "conflict": False})
            fill_targets.append(major)
            continue

        pattern = random.choice(("single", "pair_same", "triple_majority"))
        needed = 1 if pattern == "single" else 2 if pattern == "pair_same" else 3
        positions = _pick_distinct_positions(candidates, needed)
        if positions is None:
            positions = _pick_distinct_positions(candidates, 2)
            pattern = "pair_same"
        assert positions is not None
        if pattern == "single":
            color = random.choice(MARKER_COLORS)
            specs.append({"pos": positions[0], "visible": color, "alt": None, "chamber": idx, "conflict": False})
            fill_targets.append(color)
        elif pattern == "pair_same":
            color = random.choice(MARKER_COLORS)
            for pos in positions:
                specs.append({"pos": pos, "visible": color, "alt": None, "chamber": idx, "conflict": False})
            fill_targets.append(color)
        else:
            major, minor = random.sample(MARKER_COLORS, 2)
            colors = [major, major, minor]
            random.shuffle(colors)
            for pos, color in zip(positions, colors):
                specs.append({"pos": pos, "visible": color, "alt": None, "chamber": idx, "conflict": False})
            fill_targets.append(major)

    assert conflict_pos is not None and alt_info is not None
    if len(set(fill_targets)) < 2:
        raise AssertionError("chambers must not all share the same fill color")
    return specs, special_idx, conflict_pos, alt_info


def _build_overlay(
    n: int,
    sep_rows: set[int],
    sep_cols: set[int],
    specs: list[dict],
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    overlay = np.zeros((n, n), dtype=int)
    marker_map = {spec["pos"]: int(spec["visible"]) for spec in specs}
    wall_positions: set[tuple[int, int]] = set()
    for r in range(n):
        for c in range(n):
            if r in sep_rows or c in sep_cols:
                if (r, c) in marker_map:
                    overlay[r, c] = marker_map[(r, c)]
                else:
                    overlay[r, c] = WALL
                    wall_positions.add((r, c))
    return overlay, wall_positions


def _assign_marker_sources(specs: list[dict]) -> tuple[dict[tuple[int, int], list[tuple[int, int]]], list[int]]:
    conflict_spec = next(spec for spec in specs if spec["conflict"])
    hi, lo = random.choice(CONFLICT_PAIRS)
    counts = [0, 0, 0, 0]
    sources: dict[tuple[int, int], list[tuple[int, int]]] = {spec["pos"]: [] for spec in specs}
    sources[conflict_spec["pos"]] = [(hi, int(conflict_spec["visible"])), (lo, int(conflict_spec["alt"]))]
    counts[hi] += 1
    counts[lo] += 1

    remaining = [spec for spec in specs if not spec["conflict"]]
    for q in [idx for idx, count in enumerate(counts) if count == 0]:
        choices = [spec for spec in remaining if not sources[spec["pos"]]]
        if not choices:
            raise AssertionError("not enough visible markers to cover all quadrants")
        spec = random.choice(choices)
        sources[spec["pos"]].append((q, int(spec["visible"])))
        counts[q] += 1

    for spec in remaining:
        if sources[spec["pos"]]:
            continue
        available = [q for q, count in enumerate(counts) if count < 3]
        if not available:
            raise AssertionError("marker counts exceed quadrant capacity")
        min_count = min(counts[q] for q in available)
        q = random.choice([q for q in available if counts[q] == min_count])
        sources[spec["pos"]].append((q, int(spec["visible"])))
        counts[q] += 1

    if any(not (1 <= count <= 3) for count in counts):
        raise AssertionError("each quadrant must receive 1-3 markers")
    return sources, counts


def _assign_walls(
    n: int,
    wall_positions: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    pattern = random.randrange(3)
    parity_offset = random.randrange(2)
    pairings = {
        0: ((0, 3), (1, 2)),
        1: ((0, 1), (2, 3)),
        2: ((0, 2), (1, 3)),
    }
    walls_by_q: list[set[tuple[int, int]]] = [set() for _ in range(4)]
    even_pair, odd_pair = pairings[pattern]
    for r, c in wall_positions:
        qset = even_pair if ((r + c + parity_offset) % 2 == 0) else odd_pair
        for q in qset:
            walls_by_q[q].add((r, c))
    for walls in walls_by_q:
        if len(walls) < n:
            raise AssertionError("each quadrant should keep a visible wall fragment set")
    return walls_by_q


def _render_input(
    n: int,
    walls_by_q: list[set[tuple[int, int]]],
    marker_sources: dict[tuple[int, int], list[tuple[int, int]]],
) -> np.ndarray:
    quads = [np.zeros((n, n), dtype=int) for _ in range(4)]
    for q, walls in enumerate(walls_by_q):
        for r, c in walls:
            quads[q][r, c] = WALL
    for pos, assignments in marker_sources.items():
        r, c = pos
        for q, color in assignments:
            assert quads[q][r, c] == BG
            quads[q][r, c] = int(color)

    inp = np.zeros((2 * n, 2 * n), dtype=int)
    inp[:n, :n] = quads[0]
    inp[:n, n:] = quads[1]
    inp[n:, :n] = quads[2]
    inp[n:, n:] = quads[3]
    return inp


def _validate_generated_sample(
    inp: np.ndarray,
    expected_overlay: np.ndarray,
    chambers: list[dict],
    conflict_pos: tuple[int, int],
    alt_info: tuple[int, int],
) -> bool:
    n = expected_overlay.shape[0]
    if inp.shape != (2 * n, 2 * n):
        return False

    quads = _quadrants(inp)
    for quad in quads:
        marker_count = int(np.isin(quad, MARKER_COLORS).sum())
        if not (1 <= marker_count <= 3):
            return False
        if int((quad == WALL).sum()) == 0:
            return False
        if _has_enclosed_background(quad):
            return False

    overlay = _fold_overlay(inp)
    if not np.array_equal(overlay, expected_overlay):
        return False

    chamber_info = _chamber_votes(overlay)
    if len(chamber_info) != len(chambers):
        return False
    if any(_touches_edge(info["coords"], overlay.shape) for info in chamber_info):
        return False

    tie_seen = False
    mixed_majority_seen = False
    special_info = None
    for info in chamber_info:
        counts = info["counts"]
        top = max(counts.values())
        winners = [color for color, count in counts.items() if count == top]
        if len(winners) > 1:
            tie_seen = True
        elif len(counts) > 1:
            mixed_majority_seen = True
        if any(pos == conflict_pos for pos in info["markers"]):
            special_info = info

    if not tie_seen or not mixed_majority_seen or special_info is None:
        return False

    solved = main(inp)
    expected_output = _fill_overlay(overlay)
    if not np.array_equal(solved, expected_output):
        return False

    fills = {int(expected_output[r, c]) for chamber in chamber_info for r, c in chamber["coords"]}
    if len(fills) < 2:
        return False

    visible_color, alt_color = alt_info
    if visible_color == alt_color:
        return False
    overlay_alt = overlay.copy()
    overlay_alt[conflict_pos] = alt_color
    alt_output = _fill_overlay(overlay_alt)
    probe_r, probe_c = map(int, special_info["coords"][0])
    if int(expected_output[probe_r, probe_c]) == int(alt_output[probe_r, probe_c]):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    n, row_bands, col_bands, sep_rows, sep_cols = _choose_layout()
    chambers = _build_chambers(n, row_bands, col_bands)
    try:
        specs, _special_idx, conflict_pos, alt_info = _marker_specs(chambers)
        overlay, wall_positions = _build_overlay(n, sep_rows, sep_cols, specs)
        marker_sources, _marker_counts = _assign_marker_sources(specs)
        walls_by_q = _assign_walls(n, wall_positions)
    except AssertionError:
        return None

    inp = _render_input(n, walls_by_q, marker_sources)
    if not _validate_generated_sample(inp, overlay, chambers, conflict_pos, alt_info):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _attempt_generate()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid four-quadrant chamber-overlay input")
