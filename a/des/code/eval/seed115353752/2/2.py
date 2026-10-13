from __future__ import annotations

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "right", "bottom", "left")
BG = 0

GRID_SIZE_RANGE = (22, 31)
FRAME_COUNT_RANGE = (2, 5)
FRAME_SIDE_RANGE = (6, 9)
FRAGMENT_AREA_RANGE = (2, 6)
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _frame_info(comp: dict) -> dict | None:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    expected: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    actual = set(coords)
    if len(actual) != len(expected) - 1 or not actual.issubset(expected):
        return None

    gap = next(iter(expected - actual))
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        side = "top"
        align = gc
    elif gc == c1 and r0 < gr < r1:
        side = "right"
        align = gr
    elif gr == r1 and c0 < gc < c1:
        side = "bottom"
        align = gc
    elif gc == c0 and r0 < gr < r1:
        side = "left"
        align = gr
    else:
        return None

    return {
        "color": int(comp["color"]),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": (int(gr), int(gc)),
        "side": side,
        "align": int(align),
    }


def _edge_anchor(coords: list[tuple[int, int]], side: str) -> tuple[int, int]:
    if side == "top":
        edge = min(r for r, _ in coords)
        candidates = sorted((r, c) for r, c in coords if r == edge)
        return candidates[len(candidates) // 2]
    if side == "bottom":
        edge = max(r for r, _ in coords)
        candidates = sorted((r, c) for r, c in coords if r == edge)
        return candidates[len(candidates) // 2]
    if side == "left":
        edge = min(c for _, c in coords)
        candidates = sorted((r, c) for r, c in coords if c == edge)
        return candidates[len(candidates) // 2]
    edge = max(c for _, c in coords)
    candidates = sorted((r, c) for r, c in coords if c == edge)
    return candidates[len(candidates) // 2]


def _translated_fragment(
    coords: list[tuple[int, int]],
    frame: dict,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["side"]
    align = frame["align"]
    anchor_r, anchor_c = _edge_anchor(coords, side)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]

    if side == "top":
        dr = (r1 - 1) - max(rows)
        dc = align - anchor_c
    elif side == "bottom":
        dr = (r0 + 1) - min(rows)
        dc = align - anchor_c
    elif side == "left":
        dr = align - anchor_r
        dc = (c1 - 1) - max(cols)
    else:
        dr = align - anchor_r
        dc = (c0 + 1) - min(cols)

    moved = [(int(r + dr), int(c + dc)) for r, c in coords]
    for r, c in moved:
        assert r0 < r < r1 and c0 < c < c1
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each scattered fragment into the same-color frame.

    The missing border pixel gives both the entry side and the alignment row/column.
    The fragment is translated until it touches the opposite interior wall while keeping
    its chosen edge-anchor aligned with the gap coordinate.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)

    frames: dict[int, dict] = {}
    fragments: dict[int, list[dict]] = {}
    for comp in comps:
        info = _frame_info(comp)
        if info is not None:
            frames[info["color"]] = info
        else:
            fragments.setdefault(int(comp["color"]), []).append(comp)

    assert frames, "expected at least one broken rectangular frame"
    out = g.copy()

    for color, parts in fragments.items():
        assert len(parts) == 1, "each frame color should have exactly one matching fragment"
        for r, c in parts[0]["coords"]:
            out[r, c] = bg

    for color, frame in frames.items():
        assert color in fragments and len(fragments[color]) == 1
        moved = _translated_fragment(fragments[color][0]["coords"], frame)
        for r, c in moved:
            out[r, c] = color

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    out = sorted((r - min_r, c - min_c) for r, c in cells)
    return [(int(r), int(c)) for r, c in out]


def _shape_dims(cells: list[tuple[int, int]]) -> tuple[int, int]:
    r1 = max(r for r, _ in cells)
    c1 = max(c for _, c in cells)
    return int(r1 + 1), int(c1 + 1)


def _sample_connected_shape(
    rng: np.random.Generator,
    max_h: int,
    max_w: int,
    area: int,
) -> list[tuple[int, int]] | None:
    dims = [(h, w) for h in range(1, max_h + 1) for w in range(1, max_w + 1) if h * w >= area]
    if not dims:
        return None
    h, w = dims[int(rng.integers(0, len(dims)))]
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            return None
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return _normalize_cells(cells)


def _unique_side_anchor(cells: list[tuple[int, int]], side: str) -> tuple[int, int] | None:
    if side == "top":
        edge = min(r for r, _ in cells)
        points = [(r, c) for r, c in cells if r == edge]
    elif side == "bottom":
        edge = max(r for r, _ in cells)
        points = [(r, c) for r, c in cells if r == edge]
    elif side == "left":
        edge = min(c for _, c in cells)
        points = [(r, c) for r, c in cells if c == edge]
    else:
        edge = max(c for _, c in cells)
        points = [(r, c) for r, c in cells if c == edge]
    if len(points) != 1:
        return None
    return points[0]


def _sample_fragment_for_side(
    rng: np.random.Generator,
    interior_h: int,
    interior_w: int,
    side: str,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    max_h = interior_h - 1 if side in ("top", "bottom") else interior_h
    max_w = interior_w - 1 if side in ("left", "right") else interior_w
    max_h = min(max_h, 4)
    max_w = min(max_w, 4)
    if max_h < 1 or max_w < 1:
        return None

    for _ in range(120):
        area = int(rng.integers(*FRAGMENT_AREA_RANGE))
        shape = _sample_connected_shape(rng, max_h, max_w, area)
        if shape is None:
            continue
        anchor = _unique_side_anchor(shape, side)
        if anchor is None:
            continue
        return shape, anchor
    return None


def _gap_choices(
    cells: list[tuple[int, int]],
    anchor: tuple[int, int],
    frame_h: int,
    frame_w: int,
    side: str,
) -> list[int]:
    ar, ac = anchor
    if side in ("top", "bottom"):
        rel = [c - ac for _, c in cells]
        lo = 1 - min(rel)
        hi = (frame_w - 2) - max(rel)
    else:
        rel = [r - ar for r, _ in cells]
        lo = 1 - min(rel)
        hi = (frame_h - 2) - max(rel)
    return [int(v) for v in range(lo, hi + 1)]


def _final_local_cells(
    cells: list[tuple[int, int]],
    anchor: tuple[int, int],
    frame_h: int,
    frame_w: int,
    side: str,
    gap_value: int,
) -> list[tuple[int, int]]:
    ar, ac = anchor
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if side == "top":
        dr = (frame_h - 2) - max(rows)
        dc = gap_value - ac
    elif side == "bottom":
        dr = 1 - min(rows)
        dc = gap_value - ac
    elif side == "left":
        dr = gap_value - ar
        dc = (frame_w - 2) - max(cols)
    else:
        dr = gap_value - ar
        dc = 1 - min(cols)
    moved = [(int(r + dr), int(c + dc)) for r, c in cells]
    for r, c in moved:
        assert 1 <= r <= frame_h - 2 and 1 <= c <= frame_w - 2
    return moved


def _frame_patch(frame_h: int, frame_w: int, color: int, side: str, gap_value: int) -> np.ndarray:
    patch = np.full((frame_h, frame_w), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if side == "top":
        patch[0, gap_value] = BG
    elif side == "bottom":
        patch[-1, gap_value] = BG
    elif side == "left":
        patch[gap_value, 0] = BG
    else:
        patch[gap_value, -1] = BG
    return patch


def _can_place_rect(
    occ: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    *,
    padding: int,
    border: int,
) -> bool:
    rows, cols = occ.shape
    if r0 < border or c0 < border or r0 + h > rows - border or c0 + w > cols - border:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(rows, r0 + h + padding)
    cc1 = min(cols, c0 + w + padding)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _mark_rect(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _place_rectangle(
    rng: np.random.Generator,
    occ: np.ndarray,
    h: int,
    w: int,
    *,
    padding: int,
    border: int,
) -> tuple[int, int] | None:
    rows, cols = occ.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(border, rows - border - h + 1):
        for c0 in range(border, cols - border - w + 1):
            if _can_place_rect(occ, r0, c0, h, w, padding=padding, border=border):
                candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _patch_from_cells(cells: list[tuple[int, int]], color: int) -> np.ndarray:
    h, w = _shape_dims(cells)
    patch = np.full((h, w), BG, dtype=int)
    for r, c in cells:
        patch[r, c] = color
    return patch


def _sample_spec(rng: np.random.Generator, color: int, side: str) -> dict | None:
    for _ in range(160):
        frame_h = int(rng.integers(*FRAME_SIDE_RANGE))
        frame_w = int(rng.integers(*FRAME_SIDE_RANGE))
        fragment = _sample_fragment_for_side(rng, frame_h - 2, frame_w - 2, side)
        if fragment is None:
            continue
        cells, anchor = fragment
        gaps = _gap_choices(cells, anchor, frame_h, frame_w, side)
        if not gaps:
            continue
        gap_value = gaps[int(rng.integers(0, len(gaps)))]
        final_cells = _final_local_cells(cells, anchor, frame_h, frame_w, side, gap_value)
        return {
            "color": int(color),
            "side": side,
            "frame_h": frame_h,
            "frame_w": frame_w,
            "gap_value": gap_value,
            "fragment_cells": cells,
            "final_cells": final_cells,
        }
    return None


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng()
    h = int(rng.integers(*GRID_SIZE_RANGE))
    w = int(rng.integers(*GRID_SIZE_RANGE))
    n_frames = int(rng.integers(*FRAME_COUNT_RANGE))
    colors = [int(c) for c in rng.choice(np.arange(1, 10), size=n_frames, replace=False)]

    side_order = list(rng.permutation(SIDES))
    if n_frames > len(side_order):
        extra = [SIDES[int(i)] for i in rng.integers(0, len(SIDES), size=n_frames - len(side_order))]
        side_order.extend(extra)
    sides = side_order[:n_frames]

    specs: list[dict] = []
    for color, side in zip(colors, sides):
        spec = _sample_spec(rng, color, side)
        if spec is None:
            return None
        specs.append(spec)

    frame_occ = np.zeros((h, w), dtype=bool)
    for spec in sorted(specs, key=lambda s: s["frame_h"] * s["frame_w"], reverse=True):
        pos = _place_rectangle(
            rng,
            frame_occ,
            spec["frame_h"],
            spec["frame_w"],
            padding=1,
            border=1,
        )
        if pos is None:
            return None
        spec["frame_pos"] = pos
        _mark_rect(frame_occ, pos[0], pos[1], spec["frame_h"], spec["frame_w"])

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    frag_occ = frame_occ.copy()

    for spec in specs:
        color = spec["color"]
        r0, c0 = spec["frame_pos"]
        frame_patch = _frame_patch(spec["frame_h"], spec["frame_w"], color, spec["side"], spec["gap_value"])
        out[r0 : r0 + spec["frame_h"], c0 : c0 + spec["frame_w"]] = frame_patch
        inp[r0 : r0 + spec["frame_h"], c0 : c0 + spec["frame_w"]] = frame_patch
        for rr, cc in spec["final_cells"]:
            out[r0 + rr, c0 + cc] = color

    for spec in specs:
        patch = _patch_from_cells(spec["fragment_cells"], spec["color"])
        ph, pw = patch.shape
        pos = _place_rectangle(rng, frag_occ, ph, pw, padding=1, border=1)
        if pos is None:
            return None
        fr, fc = pos
        _mark_rect(frag_occ, fr, fc, ph, pw)
        mask = patch != BG
        inp[fr : fr + ph, fc : fc + pw][mask] = spec["color"]

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is None:
            continue
        inp, out = candidate
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
