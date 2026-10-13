from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SOURCE_COLORS = (GREY, RED)
COPY_COLOR = {GREY: YELLOW, RED: GREEN}
DIAGONAL_STEP = 4
L_CORNERS = {(0, 0), (0, 1), (1, 0), (1, 1)}
MAX_GENERATION_ATTEMPTS = 800


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _components(grid: np.ndarray) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    comps: list[dict] = []
    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color not in SOURCE_COLORS or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords.sort()
            b = _bbox(coords)
            comps.append({"color": color, "coords": coords, "bbox": b, "area": len(coords)})
    return comps


def _missing_corner(comp: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    assert comp["area"] == 3
    assert r1 - r0 == 1 and c1 - c0 == 1
    rel = {(r - r0, c - c0) for r, c in comp["coords"]}
    missing = L_CORNERS - rel
    assert len(missing) == 1
    return next(iter(missing))


def _diag_direction(missing: tuple[int, int]) -> tuple[int, int]:
    mr, mc = missing
    return (-1 if mr == 0 else 1, -1 if mc == 0 else 1)


def _bbox_center(b: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = b
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _nearest_pixel(coords: list[tuple[int, int]], target: tuple[float, float]) -> tuple[int, int]:
    tr, tc = target
    best_dist = None
    best = None
    for cell in sorted(coords):
        r, c = cell
        dist = (r - tr) ** 2 + (c - tc) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = cell
    assert best is not None
    return best


def _copy_shift(color: int, missing: tuple[int, int]) -> tuple[int, int]:
    dr, dc = _diag_direction(missing)
    if color == RED:
        dr, dc = -dr, -dc
    return dr * DIAGONAL_STEP, dc * DIAGONAL_STEP


def _copy_component(comp: dict) -> dict:
    missing = _missing_corner(comp)
    dr, dc = _copy_shift(int(comp["color"]), missing)
    coords = sorted((r + dr, c + dc) for r, c in comp["coords"])
    return {"color": COPY_COLOR[int(comp["color"])], "coords": coords, "bbox": _bbox(coords), "area": 3}


def _path_cells(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    pr, pc = p
    qr, qc = q
    cells: list[tuple[int, int]] = []
    if pc != qc:
        step = 1 if qc > pc else -1
        for c in range(pc + step, qc + step, step):
            cells.append((pr, c))
    if pr != qr:
        step = 1 if qr > pr else -1
        for r in range(pr + step, qr + step, step):
            cells.append((r, qc))
    if cells and cells[-1] == q:
        cells.pop()
    return cells


def _sort_key(comp: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = comp["bbox"]
    return (c0, r0)


def _simulate(grid: np.ndarray, *, trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({BLACK, RED, GREY})

    comps = _components(g)
    assert 6 <= len(comps) <= 8
    assert all(comp["area"] == 3 for comp in comps)

    out = g.copy()
    copies_by_source: dict[int, dict] = {}
    for idx, comp in enumerate(comps):
        copy_comp = _copy_component(comp)
        copies_by_source[idx] = copy_comp
        for r, c in copy_comp["coords"]:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
            assert int(out[r, c]) == BLACK
            out[r, c] = int(copy_comp["color"])

    traces: list[dict] = []
    ordered: list[tuple[int, dict]] = []
    gray_items = sorted(
        [(idx, comp) for idx, comp in enumerate(comps) if int(comp["color"]) == GREY],
        key=lambda item: _sort_key(item[1]),
    )
    red_items = sorted(
        [(idx, comp) for idx, comp in enumerate(comps) if int(comp["color"]) == RED],
        key=lambda item: _sort_key(item[1]),
    )
    ordered.extend(gray_items)
    ordered.extend(red_items)

    for idx, comp in ordered:
        copy_comp = copies_by_source[idx]
        p = _nearest_pixel(comp["coords"], _bbox_center(copy_comp["bbox"]))
        q = _nearest_pixel(copy_comp["coords"], _bbox_center(comp["bbox"]))
        attempted = _path_cells(p, q)
        drawn: list[tuple[int, int]] = []
        blocked_at = None
        blocked_by = None
        for cell in attempted:
            r, c = cell
            if int(out[r, c]) != BLACK:
                blocked_at = cell
                blocked_by = int(out[r, c])
                break
            out[r, c] = int(copy_comp["color"])
            drawn.append(cell)
        traces.append(
            {
                "src_color": int(comp["color"]),
                "path_color": int(copy_comp["color"]),
                "bbox": comp["bbox"],
                "copy_bbox": copy_comp["bbox"],
                "p": p,
                "q": q,
                "attempted": attempted,
                "drawn": drawn,
                "blocked_at": blocked_at,
                "blocked_by": blocked_by,
                "completed": blocked_at is None,
            }
        )

    if trace:
        return out, traces
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a recolored diagonal copy of each L triomino, then connect it back with a clipped right-angle path.

    Input structure:
    - The grid has a black background.
    - The only foreground objects are 4-connected grey and red L triominoes, each
      occupying three cells of a 2x2 box, with several objects of each color.

    Rule:
    - Keep every source triomino, and add a recolored diagonal copy one diagonal
      placement step away: grey moves toward its missing corner and becomes yellow, while
      red moves away from its missing corner and becomes green.
    - For each source-copy pair, choose the source endpoint as the source pixel
      closest to the copy bbox center, and the copy endpoint as the copy pixel
      closest to the source bbox center; equal distances are broken in reading order.
    - Starting from the source endpoint, try to draw in the copy color horizontally
      until the copy column, then vertically until the copy row. If the next cell is
      already non-black, stop there without overwriting it.
    - Draw all yellow paths first, then all green paths; within each family, sort
      sources by bbox top-left from left to right and then top to bottom.

    Key insight:
    - The missing corner fixes the copy direction, which in turn fixes the two path
      endpoints and the right-angle corner; later paths can be shortened because
      earlier copies or earlier paths already occupy those cells.
    """
    return _simulate(grid, trace=False)


def _source_cells(r0: int, c0: int, missing: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(
        (r0 + rr, c0 + cc)
        for rr in (0, 1)
        for cc in (0, 1)
        if (rr, cc) != missing
    )


def _make_comp(color: int, r0: int, c0: int, missing: tuple[int, int]) -> dict:
    coords = _source_cells(r0, c0, missing)
    return {"color": color, "coords": coords, "bbox": (r0, c0, r0 + 1, c0 + 1), "area": 3}


def _gap_ok(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    r0a, c0a, r1a, c1a = b1
    r0b, c0b, r1b, c1b = b2
    row_gap = max(0, r0b - r1a - 1, r0a - r1b - 1)
    col_gap = max(0, c0b - c1a - 1, c0a - c1b - 1)
    return row_gap >= 4 or col_gap >= 4


def _can_place(comp: dict, placed: list[dict], occupied: set[tuple[int, int]], H: int, W: int) -> bool:
    for r, c in comp["coords"]:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if (r, c) in occupied:
            return False
    copy_comp = _copy_component(comp)
    for r, c in copy_comp["coords"]:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if (r, c) in occupied:
            return False
    for other in placed:
        if not _gap_ok(comp["bbox"], other["bbox"]):
            return False
    return True


def _build_input(comps: list[dict], H: int, W: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for comp in comps:
        for r, c in comp["coords"]:
            grid[r, c] = int(comp["color"])
    return grid


def _raw_path_for_comp(comp: dict) -> list[tuple[int, int]]:
    copy_comp = _copy_component(comp)
    p = _nearest_pixel(comp["coords"], _bbox_center(copy_comp["bbox"]))
    q = _nearest_pixel(copy_comp["coords"], _bbox_center(comp["bbox"]))
    return _path_cells(p, q)


def _place_comp(comp: dict, placed: list[dict], occupied: set[tuple[int, int]]) -> None:
    placed.append(comp)
    occupied.update(comp["coords"])
    occupied.update(_copy_component(comp)["coords"])


def _find_random_comp(color: int, placed: list[dict], occupied: set[tuple[int, int]], H: int, W: int) -> dict | None:
    for _ in range(300):
        missing = random.choice(sorted(L_CORNERS))
        r0 = random.randint(0, H - 2)
        c0 = random.randint(0, W - 2)
        comp = _make_comp(color, r0, c0, missing)
        if _can_place(comp, placed, occupied, H, W):
            return comp
    return None


def _find_blocker(
    color: int,
    target: tuple[int, int],
    keep_free: tuple[int, int],
    placed: list[dict],
    occupied: set[tuple[int, int]],
    H: int,
    W: int,
) -> dict | None:
    tr, tc = target
    for missing in sorted(L_CORNERS):
        shift_r, shift_c = _copy_shift(color, missing)
        for rr in (0, 1):
            for cc in (0, 1):
                if (rr, cc) == missing:
                    continue
                r0 = tr - shift_r - rr
                c0 = tc - shift_c - cc
                comp = _make_comp(color, r0, c0, missing)
                if not _can_place(comp, placed, occupied, H, W):
                    continue
                copy_cells = set(_copy_component(comp)["coords"])
                all_cells = copy_cells | set(comp["coords"])
                if target not in copy_cells:
                    continue
                if keep_free in all_cells:
                    continue
                return comp
    return None


def _semantic_checks(grid: np.ndarray) -> bool:
    out, traces = _simulate(grid, trace=True)
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return False

    yellow = [t for t in traces if t["path_color"] == YELLOW]
    green = [t for t in traces if t["path_color"] == GREEN]
    if len(yellow) not in {3, 4} or len(green) not in {3, 4}:
        return False

    if any(len(t["attempted"]) == 0 for t in traces):
        return False
    if any(len(t["drawn"]) == 0 for t in traces):
        return False
    if not any(t["completed"] for t in traces):
        return False
    if not any(not t["completed"] for t in traces):
        return False
    if not any(not t["completed"] and t["blocked_by"] in {YELLOW, GREEN} for t in traces):
        return False

    outputs = [tuple(t["drawn"]) for t in traces]
    if len(set(outputs)) < len(outputs) // 2:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    H = random.randint(22, 28)
    W = random.randint(22, 28)
    grey_count = random.randint(3, 4)
    red_count = random.randint(3, 4)
    colors = [GREY] * grey_count + [RED] * red_count
    random.shuffle(colors)

    placed: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    victim_idx = random.randrange(len(colors))
    victim_color = colors.pop(victim_idx)
    victim = _find_random_comp(victim_color, placed, occupied, H, W)
    if victim is None:
        return None
    victim_path = _raw_path_for_comp(victim)
    if len(victim_path) < 2:
        return None
    _place_comp(victim, placed, occupied)

    target = victim_path[-1]
    keep_free = victim_path[0]
    blocker = None
    for idx, color in enumerate(colors):
        blocker = _find_blocker(color, target, keep_free, placed, occupied, H, W)
        if blocker is not None:
            colors.pop(idx)
            break
    if blocker is None:
        return None
    _place_comp(blocker, placed, occupied)

    for color in colors:
        candidate = _find_random_comp(color, placed, occupied, H, W)
        if candidate is None:
            return None
        _place_comp(candidate, placed, occupied)

    grid = _build_input(placed, H, W)

    try:
        if not np.array_equal(main(grid), _simulate(grid, trace=False)):
            return None
        if not _semantic_checks(grid):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
