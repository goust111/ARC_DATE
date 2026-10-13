from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _same_color_component(grid: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    color = int(grid[sr, sc])
    comp = {(sr, sc)}
    queue: deque[tuple[int, int]] = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in comp and int(grid[nr, nc]) == color:
                comp.add((nr, nc))
                queue.append((nr, nc))
    return comp


def _bbox(points: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_frame(points: set[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return points == expected and r1 - r0 >= 4 and c1 - c0 >= 4


def _plus_arm_length(mask: np.ndarray) -> int | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    span_r = int(r1 - r0)
    span_c = int(c1 - c0)
    if span_r != span_c or span_r % 2 != 0:
        return None
    cr = int((r0 + r1) // 2)
    cc = int((c0 + c1) // 2)
    expected = {
        (cr, c) for c in range(int(c0), int(c1) + 1)
    } | {
        (r, cc) for r in range(int(r0), int(r1) + 1)
    }
    actual = {tuple(map(int, pt)) for pt in pts}
    if actual != expected:
        return None
    return int((span_r + 1) // 2)


def _find_reference(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int]:
    h, w = grid.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for corner in corners:
        if int(grid[corner]) == bg:
            continue
        comp = _same_color_component(grid, corner)
        box = _bbox(comp)
        if not _is_rectangular_frame(comp, box):
            continue
        r0, c0, r1, c1 = box
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner_colors = set(map(int, inner.flatten())) - {bg, int(grid[corner])}
        if len(inner_colors) != 1:
            continue
        sample_color = int(next(iter(inner_colors)))
        arm_len = _plus_arm_length(inner == sample_color)
        if arm_len is not None:
            return box, arm_len
    raise AssertionError("failed to locate corner reference cross")


def _simulate_growth(
    shape: tuple[int, int],
    seeds: list[dict[str, object]],
    arm_len: int,
    blocked_mask: np.ndarray | None = None,
) -> list[dict[str, object]]:
    h, w = shape
    if blocked_mask is None:
        blocked_mask = np.zeros(shape, dtype=bool)
    occupied = {tuple(seed["pos"]) for seed in seeds}
    records: list[dict[str, object]] = []
    for seed in seeds:
        records.append(
            {
                "pos": tuple(seed["pos"]),
                "color": int(seed["color"]),
                "cells": set(),
                "dir_lengths": [0, 0, 0, 0],
                "dir_reasons": [None, None, None, None],
                "active": [True, True, True, True],
            }
        )

    for _step in range(1, arm_len + 1):
        proposals: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for idx, rec in enumerate(records):
            r, c = rec["pos"]
            for didx, (dr, dc) in enumerate(DIRECTIONS):
                if not rec["active"][didx]:
                    continue
                dist = int(rec["dir_lengths"][didx]) + 1
                nr = r + dr * dist
                nc = c + dc * dist
                if not (0 <= nr < h and 0 <= nc < w):
                    rec["active"][didx] = False
                    rec["dir_reasons"][didx] = "boundary"
                    continue
                if bool(blocked_mask[nr, nc]):
                    rec["active"][didx] = False
                    rec["dir_reasons"][didx] = "blocked"
                    continue
                if (nr, nc) in occupied:
                    rec["active"][didx] = False
                    rec["dir_reasons"][didx] = "occupied"
                    continue
                proposals[(nr, nc)].append((idx, didx))

        accepted: list[tuple[int, int, int, int]] = []
        for (nr, nc), claimants in proposals.items():
            if len(claimants) > 1:
                for idx, didx in claimants:
                    records[idx]["active"][didx] = False
                    records[idx]["dir_reasons"][didx] = "conflict"
                continue
            idx, didx = claimants[0]
            accepted.append((idx, didx, nr, nc))

        for idx, didx, nr, nc in accepted:
            occupied.add((nr, nc))
            records[idx]["cells"].add((nr, nc))
            records[idx]["dir_lengths"][didx] = int(records[idx]["dir_lengths"][didx]) + 1

    return records


def _render_output(
    grid: np.ndarray,
    seeds: list[dict[str, object]],
    records: list[dict[str, object]],
    ref_box: tuple[int, int, int, int],
    bg: int,
    arm_len: int,
) -> np.ndarray:
    out = np.full_like(grid, bg)
    r0, c0, r1, c1 = ref_box
    out[r0 : r1 + 1, c0 : c1 + 1] = grid[r0 : r1 + 1, c0 : c1 + 1]

    for seed, rec in zip(seeds, records):
        full_count = sum(int(length) == arm_len for length in rec["dir_lengths"])
        if full_count % 2 != 0:
            continue
        sr, sc = seed["pos"]
        out[sr, sc] = int(seed["color"])
        for r, c in rec["cells"]:
            out[r, c] = int(seed["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into a blocked cross, then keep only seeds with an even number of full arms.

    Input structure:
    - bg is the most common color.
    - One corner contains a rectangular grey frame whose interior shows a complete sample cross.
    - Outside that reference box, every non-background pixel is a single colored seed.

    Rule:
    - Read the sample cross inside the corner frame to determine the common arm length.
    - Every seed grows outward in the four cardinal directions by that many cells, stopping before the grid edge,
      the reference box, another seed center, or any previously claimed arm cell; simultaneous claims to the same
      cell are blocked for both sides.
    - After growth, keep only the seeds whose number of full-length arms is even, along with all arm cells they
      successfully grew; erase odd-arm seeds back to background. The corner reference stays unchanged.

    Key insight:
    - The framed example supplies one global arm length, and the hidden selection rule depends only on the parity
      of each seed's fully completed directions, not on its color or position.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    ref_box, arm_len = _find_reference(g, bg)
    r0, c0, r1, c1 = ref_box

    blocked_mask = np.zeros_like(g, dtype=bool)
    blocked_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    seeds: list[dict[str, object]] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if blocked_mask[r, c]:
                continue
            color = int(g[r, c])
            if color == bg:
                continue
            seeds.append({"pos": (r, c), "color": color})

    records = _simulate_growth(g.shape, seeds, arm_len, blocked_mask)
    return _render_output(g, seeds, records, ref_box, bg, arm_len)


def _reference_bbox(shape: tuple[int, int], corner: str, arm_len: int) -> tuple[int, int, int, int]:
    h, w = shape
    size = 2 * arm_len + 3
    if corner == "TL":
        return 0, 0, size - 1, size - 1
    if corner == "TR":
        return 0, w - size, size - 1, w - 1
    if corner == "BL":
        return h - size, 0, h - 1, size - 1
    if corner == "BR":
        return h - size, w - size, h - 1, w - 1
    raise ValueError(corner)


def _expanded_forbidden(
    shape: tuple[int, int], bbox: tuple[int, int, int, int], margin: int = 1
) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    mask = np.zeros(shape, dtype=bool)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def _position_ok(
    pos: tuple[int, int],
    placed: list[tuple[int, int]],
    arm_len: int,
    forbidden: np.ndarray,
    allow: set[int] | None = None,
) -> bool:
    if allow is None:
        allow = set()
    r, c = pos
    if bool(forbidden[r, c]):
        return False
    for idx, (pr, pc) in enumerate(placed):
        if idx in allow:
            continue
        if r == pr and abs(c - pc) <= 2 * arm_len:
            return False
        if c == pc and abs(r - pr) <= 2 * arm_len:
            return False
        if abs(r - pr) <= arm_len and abs(c - pc) <= arm_len:
            return False
    return True


def _pick_position(
    rng: random.Random,
    candidates: list[tuple[int, int]],
    placed: list[tuple[int, int]],
    arm_len: int,
    forbidden: np.ndarray,
    allow: set[int] | None = None,
) -> tuple[int, int] | None:
    rng.shuffle(candidates)
    for pos in candidates:
        if _position_ok(pos, placed, arm_len, forbidden, allow=allow):
            return pos
    return None


def _edge_candidates(h: int, w: int, arm_len: int, edge: str) -> list[tuple[int, int]]:
    if edge == "top":
        return [(r, c) for r in range(0, arm_len) for c in range(arm_len, w - arm_len)]
    if edge == "bottom":
        return [(r, c) for r in range(h - arm_len, h) for c in range(arm_len, w - arm_len)]
    if edge == "left":
        return [(r, c) for r in range(arm_len, h - arm_len) for c in range(0, arm_len)]
    if edge == "right":
        return [(r, c) for r in range(arm_len, h - arm_len) for c in range(w - arm_len, w)]
    raise ValueError(edge)


def _corner_candidates(h: int, w: int, arm_len: int, corner: str) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(r, c) for r in range(0, arm_len) for c in range(0, arm_len)]
    if corner == "TR":
        return [(r, c) for r in range(0, arm_len) for c in range(w - arm_len, w)]
    if corner == "BL":
        return [(r, c) for r in range(h - arm_len, h) for c in range(0, arm_len)]
    if corner == "BR":
        return [(r, c) for r in range(h - arm_len, h) for c in range(w - arm_len, w)]
    raise ValueError(corner)


def _interior_candidates(h: int, w: int, arm_len: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(arm_len, h - arm_len)
        for c in range(arm_len, w - arm_len)
    ]


def _pair_positions(
    rng: random.Random,
    h: int,
    w: int,
    arm_len: int,
    placed: list[tuple[int, int]],
    forbidden: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    gap_lo = max(4, arm_len + 2)
    gap_hi = 2 * arm_len
    if gap_lo > gap_hi:
        return None
    orientations = ["horizontal", "vertical"]
    rng.shuffle(orientations)
    for orientation in orientations:
        gaps = list(range(gap_lo, gap_hi + 1))
        rng.shuffle(gaps)
        for gap in gaps:
            if orientation == "horizontal":
                candidates = [
                    ((r, c), (r, c + gap))
                    for r in range(arm_len, h - arm_len)
                    for c in range(arm_len, w - arm_len - gap)
                ]
            else:
                candidates = [
                    ((r, c), (r + gap, c))
                    for r in range(arm_len, h - arm_len - gap)
                    for c in range(arm_len, w - arm_len)
                ]
            rng.shuffle(candidates)
            for first, second in candidates:
                if not _position_ok(first, placed, arm_len, forbidden):
                    continue
                if not _position_ok(second, placed + [first], arm_len, forbidden, allow={len(placed)}):
                    continue
                return first, second
    return None


def _build_reference(grid: np.ndarray, corner: str, arm_len: int, color: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _reference_bbox(grid.shape, corner, arm_len)
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    for step in range(-arm_len, arm_len + 1):
        grid[cr + step, cc] = color
        grid[cr, cc + step] = color
    return r0, c0, r1, c1


def _instance_metrics(
    shape: tuple[int, int],
    seeds: list[dict[str, object]],
    arm_len: int,
    blocked_mask: np.ndarray,
) -> tuple[list[int], list[dict[str, object]]]:
    records = _simulate_growth(shape, seeds, arm_len, blocked_mask)
    full_counts = [sum(int(length) == arm_len for length in rec["dir_lengths"]) for rec in records]
    return full_counts, records


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    arm_len = rng.randint(2, 4)
    size = 2 * arm_len + 3
    h = rng.randint(max(18, size + 8), 30)
    w = rng.randint(max(18, size + 8), 30)
    grid = np.full((h, w), BG, dtype=int)

    ref_corner = rng.choice(["TL", "TR", "BL", "BR"])
    ref_color = rng.choice(PALETTE)
    ref_box = _build_reference(grid, ref_corner, arm_len, ref_color)
    forbidden = _expanded_forbidden(grid.shape, ref_box, margin=1)

    seed_colors = [c for c in PALETTE if c != ref_color]
    rng.shuffle(seed_colors)

    placed: list[tuple[int, int]] = []

    pair = _pair_positions(rng, h, w, arm_len, placed, forbidden)
    if pair is None:
        return None
    placed.extend(pair)

    open_pos = _pick_position(
        rng, _interior_candidates(h, w, arm_len), placed, arm_len, forbidden
    )
    if open_pos is None:
        return None
    placed.append(open_pos)

    odd_edge = _pick_position(
        rng,
        _edge_candidates(h, w, arm_len, rng.choice(["top", "bottom", "left", "right"])),
        placed,
        arm_len,
        forbidden,
    )
    if odd_edge is None:
        alt_edges = ["top", "bottom", "left", "right"]
        rng.shuffle(alt_edges)
        for edge in alt_edges:
            odd_edge = _pick_position(rng, _edge_candidates(h, w, arm_len, edge), placed, arm_len, forbidden)
            if odd_edge is not None:
                break
    if odd_edge is None:
        return None
    placed.append(odd_edge)

    even_corners = [corner for corner in ["TL", "TR", "BL", "BR"] if corner != ref_corner]
    rng.shuffle(even_corners)
    even_corner_pos = None
    for corner in even_corners:
        even_corner_pos = _pick_position(
            rng, _corner_candidates(h, w, arm_len, corner), placed, arm_len, forbidden
        )
        if even_corner_pos is not None:
            break
    if even_corner_pos is None:
        return None
    placed.append(even_corner_pos)

    extras = rng.randint(0, 2)
    for _ in range(extras):
        mode = rng.choice(["open", "edge", "corner"])
        if mode == "open":
            candidates = _interior_candidates(h, w, arm_len)
        elif mode == "edge":
            candidates = _edge_candidates(h, w, arm_len, rng.choice(["top", "bottom", "left", "right"]))
        else:
            candidates = _corner_candidates(h, w, arm_len, rng.choice(["TL", "TR", "BL", "BR"]))
        pos = _pick_position(rng, candidates, placed, arm_len, forbidden)
        if pos is not None:
            placed.append(pos)

    if len(placed) > len(seed_colors):
        return None

    seeds = [{"pos": pos, "color": seed_colors[idx]} for idx, pos in enumerate(placed)]
    for seed in seeds:
        r, c = seed["pos"]
        grid[r, c] = int(seed["color"])

    blocked_mask = np.zeros_like(grid, dtype=bool)
    r0, c0, r1, c1 = ref_box
    blocked_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    full_counts, records = _instance_metrics(grid.shape, seeds, arm_len, blocked_mask)
    kept = sum(count % 2 == 0 for count in full_counts)
    removed = len(full_counts) - kept
    if kept == 0 or removed == 0:
        return None
    if not any(reason in {"occupied", "conflict"} for rec in records for reason in rec["dir_reasons"] if reason):
        return None
    if not any(reason == "boundary" for rec in records for reason in rec["dir_reasons"] if reason):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
