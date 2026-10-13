from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
YELLOW = 4
MARKER_POOL = (1, 2, 3, 5, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 1600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _rect_bounds(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, r1 = sorted((int(p[0]), int(q[0])))
    c0, c1 = sorted((int(p[1]), int(q[1])))
    return r0, c0, r1, c1


def _rect_mask(shape: tuple[int, int], p: tuple[int, int], q: tuple[int, int]) -> np.ndarray:
    r0, c0, r1, c1 = _rect_bounds(p, q)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[r0 : r1 + 1, c0 : c1 + 1] = 1
    return mask


def _xor_shadow(
    shape: tuple[int, int], rects: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    shadow = np.zeros(shape, dtype=np.uint8)
    cover = np.zeros(shape, dtype=np.uint8)
    for rect in rects:
        mask = _rect_mask(shape, rect["p"], rect["q"])
        shadow ^= mask
        cover += mask
    return shadow, cover


def _xor_boundary_points(mask: np.ndarray) -> set[tuple[int, int]]:
    m = np.asarray(mask, dtype=np.uint8)
    diff = np.zeros((m.shape[0] + 1, m.shape[1] + 1), dtype=np.uint8)
    diff[:-1, :-1] ^= m
    diff[1:, :-1] ^= m
    diff[:-1, 1:] ^= m
    diff[1:, 1:] ^= m
    return {tuple(map(int, rc)) for rc in np.argwhere(diff == 1)}


def _candidate_rectangles(
    boundary_points: set[tuple[int, int]],
    marker: tuple[int, int, int],
    shape: tuple[int, int],
) -> list[tuple[tuple[int, int], frozenset[tuple[int, int]]]]:
    _color, r, c = marker
    H, W = map(int, shape)
    candidates: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}

    for row_side in (0, 1):
        for col_side in (0, 1):
            known_boundary = (int(r + row_side), int(c + col_side))
            if known_boundary not in boundary_points:
                continue

            for opp_boundary in boundary_points:
                if opp_boundary[0] == known_boundary[0] or opp_boundary[1] == known_boundary[1]:
                    continue

                qr = int(opp_boundary[0] - (1 - row_side))
                qc = int(opp_boundary[1] - (1 - col_side))
                if not (0 <= qr < H and 0 <= qc < W):
                    continue
                if qr == r or qc == c:
                    continue
                if row_side == 0 and qr <= r:
                    continue
                if row_side == 1 and qr >= r:
                    continue
                if col_side == 0 and qc <= c:
                    continue
                if col_side == 1 and qc >= c:
                    continue

                cross_a = (known_boundary[0], opp_boundary[1])
                cross_b = (opp_boundary[0], known_boundary[1])
                if cross_a not in boundary_points or cross_b not in boundary_points:
                    continue

                q = (qr, qc)
                toggles = frozenset((known_boundary, cross_a, cross_b, opp_boundary))
                candidates[q] = toggles

    return [(q, toggles) for q, toggles in candidates.items()]


def _solve_solutions(
    observed_shadow: np.ndarray,
    markers: list[tuple[int, int, int]],
    *,
    max_solutions: int = 2,
) -> list[dict[int, tuple[int, int]]]:
    shadow = np.asarray(observed_shadow, dtype=np.uint8)
    H, W = shadow.shape
    solutions: list[tuple[tuple[int, tuple[int, int]], ...]] = []

    for bitmask in range(1 << len(markers)):
        completed_shadow = shadow.copy()
        for i, (_color, r, c) in enumerate(markers):
            completed_shadow[int(r), int(c)] = (bitmask >> i) & 1

        boundary_points = _xor_boundary_points(completed_shadow)

        def rec(
            points: set[tuple[int, int]],
            remaining: tuple[tuple[int, int, int], ...],
            partial: dict[int, tuple[int, int]],
        ) -> None:
            if len(solutions) >= max_solutions:
                return
            if not remaining:
                if not points:
                    solutions.append(tuple(sorted(partial.items())))
                return

            best_idx = -1
            best_marker = None
            best_candidates = None
            for idx, marker in enumerate(remaining):
                candidates = _candidate_rectangles(points, marker, (H, W))
                if not candidates:
                    return
                if best_candidates is None or len(candidates) < len(best_candidates):
                    best_idx = idx
                    best_marker = marker
                    best_candidates = candidates

            assert best_marker is not None and best_candidates is not None
            color = int(best_marker[0])
            next_remaining = remaining[:best_idx] + remaining[best_idx + 1 :]
            for q, toggles in best_candidates:
                partial[color] = q
                rec(points.symmetric_difference(toggles), next_remaining, partial)
                del partial[color]

        rec(boundary_points, tuple(markers), {})

    unique: list[dict[int, tuple[int, int]]] = []
    seen: set[tuple[tuple[int, tuple[int, int]], ...]] = set()
    for solution in solutions:
        if solution in seen:
            continue
        seen.add(solution)
        unique.append({int(color): tuple(pos) for color, pos in solution})
    return unique


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each hidden opposite corner whose rectangle participates in the yellow XOR shadow.

    Input structure:
    - The dominant background contains a yellow region plus several isolated one-pixel colored markers.
    - Each marker color appears exactly once and marks one corner of a different axis-aligned filled rectangle.
    - The rectangles themselves are invisible; only the cells covered by an odd number of rectangles remain visible as yellow, and the colored markers are drawn on top of that shadow.

    Rule:
    - Treat the yellow cells as the XOR overlay of all hidden filled rectangles, with every colored marker fixing one known rectangle corner.
    - Reconstruct the unique opposite corner for each marker so that the XOR of all completed rectangles matches the yellow shadow exactly.
    - Output the input grid unchanged except for drawing each recovered opposite corner in the same color as its marker.

    Key insight:
    - The yellow shape is not an object to extend directly; it is the parity residue of several overlapped filled rectangles, so the missing corners must be inferred by inverting that rectangle-XOR construction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    marker_colors = sorted(set(map(int, np.unique(g))) - {bg, YELLOW})
    assert marker_colors, "expected at least one marker color"

    markers: list[tuple[int, int, int]] = []
    for color in marker_colors:
        coords = np.argwhere(g == int(color))
        assert coords.shape == (1, 2), "each marker color must appear exactly once"
        r, c = map(int, coords[0])
        markers.append((int(color), r, c))

    observed_shadow = (g == YELLOW).astype(np.uint8)
    solutions = _solve_solutions(observed_shadow, markers, max_solutions=2)
    assert len(solutions) == 1, "puzzle must admit a unique reconstruction"

    out = g.copy()
    for color, q in solutions[0].items():
        out[int(q[0]), int(q[1])] = int(color)
    return out


def _row_or_col_has_multiple_spans(mask: np.ndarray) -> bool:
    for line in np.asarray(mask, dtype=np.uint8):
        spans = 0
        prev = 0
        for val in line.tolist():
            if val and not prev:
                spans += 1
            prev = int(val)
        if spans >= 2:
            return True
    return False


def _shadow_is_structured(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask == 1)
    if coords.size == 0:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    if int(mask.sum()) < bbox_area:
        return True
    return _row_or_col_has_multiple_spans(mask) or _row_or_col_has_multiple_spans(mask.T)


def _intersection_area(a: dict, b: dict) -> int:
    ar0, ac0, ar1, ac1 = _rect_bounds(a["p"], a["q"])
    br0, bc0, br1, bc1 = _rect_bounds(b["p"], b["q"])
    r0 = max(ar0, br0)
    c0 = max(ac0, bc0)
    r1 = min(ar1, br1)
    c1 = min(ac1, bc1)
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _expected_output(inp: np.ndarray, rects: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for rect in rects:
        qr, qc = rect["q"]
        out[int(qr), int(qc)] = int(rect["color"])
    return out


def _build_input(shape: tuple[int, int], rects: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shadow, cover = _xor_shadow(shape, rects)
    inp = np.full(shape, BLACK, dtype=int)
    inp[shadow == 1] = YELLOW
    for rect in rects:
        pr, pc = rect["p"]
        inp[int(pr), int(pc)] = int(rect["color"])
    return inp, shadow, cover


def _sample_rectangles(rng: np.random.Generator) -> tuple[tuple[int, int], list[dict], bool] | None:
    H = int(rng.integers(12, 19))
    W = int(rng.integers(12, 19))
    shape = (H, W)
    rect_count = int(rng.choice([2, 3], p=[0.45, 0.55]))
    require_mixed_marker_shadow = bool(rng.random() < 0.7)

    colors = rng.choice(np.array(MARKER_POOL, dtype=int), size=rect_count, replace=False)
    used_marker_rows: set[int] = set()
    used_marker_cols: set[int] = set()
    used_vertices: set[tuple[int, int]] = set()
    rects: list[dict] = []

    for color in map(int, colors.tolist()):
        placed = False
        for _ in range(500):
            h = int(rng.integers(3, min(8, H) + 1))
            w = int(rng.integers(3, min(8, W) + 1))
            r0 = int(rng.integers(0, H - h + 1))
            c0 = int(rng.integers(0, W - w + 1))
            r1 = r0 + h - 1
            c1 = c0 + w - 1

            corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
            p = corners[int(rng.integers(0, 4))]
            q = {
                (r0, c0): (r1, c1),
                (r0, c1): (r1, c0),
                (r1, c0): (r0, c1),
                (r1, c1): (r0, c0),
            }[p]

            if p[0] in used_marker_rows or p[1] in used_marker_cols:
                continue
            if p in used_vertices or q in used_vertices:
                continue

            rects.append({"color": color, "p": p, "q": q})
            used_marker_rows.add(int(p[0]))
            used_marker_cols.add(int(p[1]))
            used_vertices.add(tuple(p))
            used_vertices.add(tuple(q))
            placed = True
            break

        if not placed:
            return None

    return shape, rects, require_mixed_marker_shadow


def _scene_is_valid(
    shape: tuple[int, int],
    rects: list[dict],
    require_mixed_marker_shadow: bool,
) -> bool:
    inp, shadow, cover = _build_input(shape, rects)
    if int(cover.max()) > 2:
        return False
    if not _shadow_is_structured(shadow):
        return False

    overlaps = [_intersection_area(rects[i], rects[j]) for i in range(len(rects)) for j in range(i + 1, len(rects))]
    if not overlaps or max(overlaps) == 0:
        return False

    total_area = sum(int(_rect_mask(shape, rect["p"], rect["q"]).sum()) for rect in rects)
    if total_area == int(shadow.sum()):
        return False

    yellow_count = int(shadow.sum())
    if yellow_count < 12 or yellow_count > int(shape[0] * shape[1] * 0.45):
        return False

    hidden_bits = [int(shadow[rect["p"][0], rect["p"][1]]) for rect in rects]
    if require_mixed_marker_shadow and len(set(hidden_bits)) < 2:
        return False

    marker_positions = {tuple(rect["p"]) for rect in rects}
    target_positions = {tuple(rect["q"]) for rect in rects}
    if len(marker_positions) != len(rects) or len(target_positions) != len(rects):
        return False
    if marker_positions & target_positions:
        return False

    expected = _expected_output(inp, rects)
    solutions = _solve_solutions((inp == YELLOW).astype(np.uint8), [(rect["color"], *rect["p"]) for rect in rects], max_solutions=3)
    if len(solutions) != 1:
        return False

    truth = {int(rect["color"]): tuple(rect["q"]) for rect in rects}
    if solutions[0] != truth:
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    sampled = _sample_rectangles(rng)
    if sampled is None:
        return None

    shape, rects, require_mixed_marker_shadow = sampled
    if not _scene_is_valid(shape, rects, require_mixed_marker_shadow):
        return None
    inp, _shadow, _cover = _build_input(shape, rects)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
