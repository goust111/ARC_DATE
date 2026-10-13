from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import numpy as np


BG = 0
ARROW_COLORS = (1, 2, 3, 4)
DECO_COLORS = (5, 6, 7, 8, 9)
GEN_ATTEMPTS = 500


ARROW_SPECS = {
    "left": {
        "shape": (3, 4),
        "cells": {(1, 0), (1, 1), (1, 2), (1, 3), (0, 1), (2, 1)},
        "tip": (1, 0),
        "dir": (0, -1),
    },
    "right": {
        "shape": (3, 4),
        "cells": {(1, 0), (1, 1), (1, 2), (1, 3), (0, 2), (2, 2)},
        "tip": (1, 3),
        "dir": (0, 1),
    },
    "up": {
        "shape": (4, 3),
        "cells": {(0, 1), (1, 1), (2, 1), (3, 1), (1, 0), (1, 2)},
        "tip": (0, 1),
        "dir": (-1, 0),
    },
    "down": {
        "shape": (4, 3),
        "cells": {(0, 1), (1, 1), (2, 1), (3, 1), (2, 0), (2, 2)},
        "tip": (3, 1),
        "dir": (1, 0),
    },
}


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _decode_arrow(coords: np.ndarray) -> dict[str, object] | None:
    pts = {(int(r), int(c)) for r, c in coords}
    if len(pts) != 6:
        return None

    row_counts = Counter(r for r, _c in pts)
    col_counts = Counter(c for _r, c in pts)
    long_rows = [r for r, n in row_counts.items() if n == 4]
    long_cols = [c for c, n in col_counts.items() if n == 4]
    cross_rows = [r for r, n in row_counts.items() if n == 3]
    cross_cols = [c for c, n in col_counts.items() if n == 3]
    bbox = _bbox_coords(coords)

    if len(long_rows) == 1 and len(cross_cols) == 1 and not long_cols:
        axis_r = int(long_rows[0])
        cross_c = int(cross_cols[0])
        line_cols = sorted(c for r, c in pts if r == axis_r)
        cross_rows_here = sorted(r for r, c in pts if c == cross_c)
        if line_cols != list(range(line_cols[0], line_cols[0] + 4)):
            return None
        if cross_rows_here != [axis_r - 1, axis_r, axis_r + 1]:
            return None
        left_dist = cross_c - line_cols[0]
        right_dist = line_cols[-1] - cross_c
        if {left_dist, right_dist} != {1, 2}:
            return None
        direction = "left" if left_dist < right_dist else "right"
        tip = (axis_r, line_cols[0] if direction == "left" else line_cols[-1])
        return {"direction": direction, "tip": tip, "bbox": bbox, "dirvec": ARROW_SPECS[direction]["dir"]}

    if len(long_cols) == 1 and len(cross_rows) == 1 and not long_rows:
        axis_c = int(long_cols[0])
        cross_r = int(cross_rows[0])
        line_rows = sorted(r for r, c in pts if c == axis_c)
        cross_cols_here = sorted(c for r, c in pts if r == cross_r)
        if line_rows != list(range(line_rows[0], line_rows[0] + 4)):
            return None
        if cross_cols_here != [axis_c - 1, axis_c, axis_c + 1]:
            return None
        up_dist = cross_r - line_rows[0]
        down_dist = line_rows[-1] - cross_r
        if {up_dist, down_dist} != {1, 2}:
            return None
        direction = "up" if up_dist < down_dist else "down"
        tip = (line_rows[0] if direction == "up" else line_rows[-1], axis_c)
        return {"direction": direction, "tip": tip, "bbox": bbox, "dirvec": ARROW_SPECS[direction]["dir"]}

    return None


def _find_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != BG):
        for coords in _components4(grid == int(color)):
            info = _decode_arrow(coords)
            if info is None:
                continue
            info = dict(info)
            info["color"] = int(color)
            info["coords"] = coords
            arrows.append(info)
    arrows.sort(key=lambda a: (a["bbox"][0], a["bbox"][1], a["color"]))
    return arrows


def _diagonal_copy(grid: np.ndarray) -> np.ndarray:
    h, w = map(int, grid.shape)
    out = np.full((2 * h, 2 * w), BG, dtype=int)
    _paste(out, grid, 0, 0)
    _paste(out, grid, h, w)
    return out


def _enumerate_rays(grid: np.ndarray, *, ignore_blockers: bool = False) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    base = _diagonal_copy(g)
    arrows = _find_arrows(g)

    rays: list[dict[str, object]] = []
    for arrow in arrows:
        tip_r, tip_c = map(int, arrow["tip"])
        dr, dc = map(int, arrow["dirvec"])
        b0, b1, b2, b3 = map(int, arrow["bbox"])
        for copy_rank, (roff, coff) in enumerate(((0, 0), (h, w))):
            cells: list[tuple[int, int]] = []
            r = tip_r + roff + dr
            c = tip_c + coff + dc
            stop_reason = "boundary"
            while 0 <= r < 2 * h and 0 <= c < 2 * w:
                if not ignore_blockers and int(base[r, c]) != BG:
                    stop_reason = "blocked"
                    break
                cells.append((int(r), int(c)))
                r += dr
                c += dc
            rays.append(
                {
                    "color": int(arrow["color"]),
                    "copy_rank": int(copy_rank),
                    "bbox": (b0 + roff, b1 + coff, b2 + roff, b3 + coff),
                    "cells": cells,
                    "stop_reason": stop_reason,
                    "local_bbox": tuple(map(int, arrow["bbox"])),
                    "direction": str(arrow["direction"]),
                }
            )

    candidates: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for ray in rays:
        for cell in ray["cells"]:
            candidates[cell].append(ray)

    return {"base": base, "rays": rays, "candidates": candidates, "shape": (h, w)}


def _priority_key(ray: dict[str, object]) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = map(int, ray["bbox"])
    return int(ray["copy_rank"]), int(r0), int(c0)


def _priority_key_cross_reversed(ray: dict[str, object]) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = map(int, ray["bbox"])
    return 1 - int(ray["copy_rank"]), int(r0), int(c0)


def _priority_key_same_reversed(ray: dict[str, object]) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = map(int, ray["bbox"])
    return int(ray["copy_rank"]), -int(r0), -int(c0)


def _render_from_candidates(base: np.ndarray, candidates: dict[tuple[int, int], list[dict[str, object]]], key_fn) -> np.ndarray:
    out = base.copy()
    for cell, rays in candidates.items():
        winner = min(rays, key=key_fn)
        out[cell] = int(winner["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diagonally duplicate the input and extend arrow rays to bridge the empty off-diagonal quadrants.

    Input structure:
    - The input is a black-background grid, roughly 8x8 to 10x10.
    - It contains 1 to 3 monochrome six-pixel asymmetric T arrows, each using a distinct color from the small task palette.
    - Single-pixel foreground decorations may also appear, but they do not form arrow components.

    Rule:
    - Create a 2H x 2W black canvas and paste the input into the upper-left and lower-right quadrants.
    - In each copied sprite set, detect every six-pixel arrow by its length-4 axis and length-3 crossbar; the crossbar is one cell closer to the pointed end.
    - From each arrow tip, extend a same-color ray one pixel at a time along the pointing direction until the next cell would be non-black in the copied canvas or would leave the canvas.
    - Preserve all copied input pixels; if multiple rays want the same black cell, upper-left-copy rays win over lower-right-copy rays, and ties within one copy use arrow-bbox top-left scan order.

    Key insight:
    - The diagonal duplication creates two empty off-diagonal quadrants, and the arrows in the copied quadrants shoot straight into that empty space to form visible bridges.
    """
    g = np.asarray(grid, dtype=int)
    info = _enumerate_rays(g, ignore_blockers=False)
    return _render_from_candidates(info["base"], info["candidates"], _priority_key)


def _build_arrow_patch(direction: str, color: int) -> np.ndarray:
    spec = ARROW_SPECS[direction]
    patch = np.full(spec["shape"], BG, dtype=int)
    for r, c in spec["cells"]:
        patch[r, c] = int(color)
    return patch


def _candidate_arrows(h: int, w: int, direction: str, color: int) -> list[dict[str, object]]:
    patch = _build_arrow_patch(direction, color)
    ph, pw = patch.shape
    out: list[dict[str, object]] = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            cells = {(top + r, left + c) for r, c in ARROW_SPECS[direction]["cells"]}
            tip = (top + ARROW_SPECS[direction]["tip"][0], left + ARROW_SPECS[direction]["tip"][1])
            out.append(
                {
                    "direction": direction,
                    "color": int(color),
                    "top": int(top),
                    "left": int(left),
                    "patch": patch,
                    "cells": cells,
                    "tip": tip,
                    "bbox": (top, left, top + ph - 1, left + pw - 1),
                }
            )
    return out


def _well_separated(a_cells: set[tuple[int, int]], b_cells: set[tuple[int, int]]) -> bool:
    for ra, ca in a_cells:
        for rb, cb in b_cells:
            if max(abs(ra - rb), abs(ca - cb)) <= 1:
                return False
    return True


def _intersects_ray_same_copy(a: dict[str, object], b: dict[str, object]) -> bool:
    da = str(a["direction"])
    db = str(b["direction"])
    if {da, db} == {"right", "down"}:
        right = a if da == "right" else b
        down = a if db == "down" else b
        cell = (int(right["tip"][0]), int(down["tip"][1]))
        return (
            int(right["tip"][1]) < int(down["tip"][1])
            and int(right["tip"][0]) > int(down["tip"][0])
            and cell not in right["cells"]
            and cell not in down["cells"]
        )
    if {da, db} == {"up", "left"}:
        up = a if da == "up" else b
        left = a if db == "left" else b
        cell = (int(left["tip"][0]), int(up["tip"][1]))
        return (
            int(left["tip"][1]) > int(up["tip"][1])
            and int(left["tip"][0]) < int(up["tip"][0])
            and cell not in up["cells"]
            and cell not in left["cells"]
        )
    return False


def _random_nonblocking_decor_cells(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    ray_forbidden: set[tuple[int, int]],
    rng: np.random.Generator,
    count: int,
) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    attempts = 0
    while len(cells) < count and attempts < 200:
        attempts += 1
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        if (r, c) in occupied or (r, c) in ray_forbidden:
            continue
        ok = True
        for rr, cc in occupied | {(dr, dc) for dr, dc, _color in cells}:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                ok = False
                break
        if not ok:
            continue
        color = int(DECO_COLORS[int(rng.integers(0, len(DECO_COLORS)))])
        cells.append((r, c, color))
    return cells


def _make_grid(h: int, w: int, arrows: list[dict[str, object]], decorations: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for arrow in arrows:
        _paste(grid, arrow["patch"], int(arrow["top"]), int(arrow["left"]), transparent=BG)
    for r, c, color in decorations:
        grid[int(r), int(c)] = int(color)
    return grid


def _new_cells_by_direction(grid: np.ndarray) -> set[tuple[int, int]]:
    out = main(grid)
    base = _diagonal_copy(grid)
    return {(int(r), int(c)) for r, c in np.argwhere(out != base)}


def _validate_generated(grid: np.ndarray, mode: str) -> bool:
    arrows = _find_arrows(grid)
    if not (1 <= len(arrows) <= 3):
        return False
    if len({int(a["color"]) for a in arrows}) != len(arrows):
        return False

    info = _enumerate_rays(grid, ignore_blockers=False)
    actual = _render_from_candidates(info["base"], info["candidates"], _priority_key)
    if not np.array_equal(actual, main(grid)):
        return False

    h, w = map(int, grid.shape)
    new_cells = {(int(r), int(c)) for r, c in np.argwhere(actual != info["base"])}
    if not new_cells:
        return False
    bridge_cells = {
        cell
        for cell in new_cells
        if (cell[0] < h and cell[1] >= w) or (cell[0] >= h and cell[1] < w)
    }
    if not bridge_cells:
        return False

    per_local_arrow = defaultdict(int)
    for ray in info["rays"]:
        per_local_arrow[(ray["local_bbox"], ray["direction"])] += int(bool(ray["cells"]))
    if any(count == 0 for count in per_local_arrow.values()):
        return False

    cross_overlaps = []
    same_overlaps = []
    for cell, rays in info["candidates"].items():
        copy_ranks = {int(ray["copy_rank"]) for ray in rays}
        if len(copy_ranks) > 1:
            cross_overlaps.append(cell)
        elif len(rays) > 1:
            same_overlaps.append(cell)

    if mode == "cross_copy":
        if not cross_overlaps:
            return False
        wrong = _render_from_candidates(info["base"], info["candidates"], _priority_key_cross_reversed)
        if np.array_equal(actual, wrong):
            return False
    elif mode == "same_copy":
        if not same_overlaps:
            return False
        wrong = _render_from_candidates(info["base"], info["candidates"], _priority_key_same_reversed)
        if np.array_equal(actual, wrong):
            return False
    elif mode == "blocked":
        blocked_rays = [ray for ray in info["rays"] if ray["stop_reason"] == "blocked" and ray["cells"]]
        if not blocked_rays:
            return False
        obstacle_free = _enumerate_rays(grid, ignore_blockers=True)
        wrong = _render_from_candidates(obstacle_free["base"], obstacle_free["candidates"], _priority_key)
        if np.array_equal(actual, wrong):
            return False
    else:
        return False

    return True


def _gen_cross_copy(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(8, 11))
    w = int(rng.integers(8, 11))
    pair = ("right", "up") if int(rng.integers(0, 2)) == 0 else ("down", "left")
    n_arrows = 2 if int(rng.integers(0, 3)) else 3
    directions = [pair[0], pair[1]]
    remaining = [d for d in ("left", "right", "up", "down") if d not in directions]
    if n_arrows == 3:
        directions.append(remaining[int(rng.integers(0, len(remaining)))])

    colors = list(rng.choice(np.array(ARROW_COLORS), size=len(directions), replace=False))
    candidates_by_dir = {
        direction: _candidate_arrows(h, w, direction, int(color))
        for direction, color in zip(directions, colors)
    }
    for cand_list in candidates_by_dir.values():
        rng.shuffle(cand_list)

    chosen: list[dict[str, object]] = []
    for direction in directions:
        placed = None
        for cand in candidates_by_dir[direction]:
            if all(_well_separated(cand["cells"], other["cells"]) for other in chosen):
                placed = cand
                break
        if placed is None:
            return None
        chosen.append(placed)

    occupied = set().union(*(arrow["cells"] for arrow in chosen))
    probe = _make_grid(h, w, chosen, [])
    predicted = _new_cells_by_direction(probe)
    ray_forbidden = {
        (r, c)
        for r, c in predicted
        if 0 <= r < h and 0 <= c < w
    }
    decorations = _random_nonblocking_decor_cells(h, w, occupied, ray_forbidden, rng, int(rng.integers(0, 3)))
    grid = _make_grid(h, w, chosen, decorations)
    return grid if _validate_generated(grid, "cross_copy") else None


def _gen_same_copy(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(9, 11))
    w = int(rng.integers(9, 11))
    pair = ("right", "down") if int(rng.integers(0, 2)) == 0 else ("up", "left")
    n_arrows = 2 if int(rng.integers(0, 3)) else 3
    directions = [pair[0], pair[1]]
    remaining = [d for d in ("left", "right", "up", "down") if d not in directions]
    if n_arrows == 3:
        directions.append(remaining[int(rng.integers(0, len(remaining)))])

    colors = list(rng.choice(np.array(ARROW_COLORS), size=len(directions), replace=False))
    pair_a = _candidate_arrows(h, w, pair[0], int(colors[0]))
    pair_b = _candidate_arrows(h, w, pair[1], int(colors[1]))
    rng.shuffle(pair_a)
    rng.shuffle(pair_b)

    chosen_pair: tuple[dict[str, object], dict[str, object]] | None = None
    for a in pair_a:
        for b in pair_b:
            if not _well_separated(a["cells"], b["cells"]):
                continue
            if not _intersects_ray_same_copy(a, b):
                continue
            chosen_pair = (a, b)
            break
        if chosen_pair is not None:
            break
    if chosen_pair is None:
        return None

    chosen = [chosen_pair[0], chosen_pair[1]]
    if n_arrows == 3:
        extra = _candidate_arrows(h, w, directions[2], int(colors[2]))
        rng.shuffle(extra)
        placed = None
        for cand in extra:
            if all(_well_separated(cand["cells"], other["cells"]) for other in chosen):
                placed = cand
                break
        if placed is None:
            return None
        chosen.append(placed)

    occupied = set().union(*(arrow["cells"] for arrow in chosen))
    probe = _make_grid(h, w, chosen, [])
    predicted = _new_cells_by_direction(probe)
    ray_forbidden = {
        (r, c)
        for r, c in predicted
        if 0 <= r < h and 0 <= c < w
    }
    decorations = _random_nonblocking_decor_cells(h, w, occupied, ray_forbidden, rng, int(rng.integers(0, 2)))
    grid = _make_grid(h, w, chosen, decorations)
    return grid if _validate_generated(grid, "same_copy") else None


def _find_blocker_cell(
    h: int,
    w: int,
    blocked_arrow: dict[str, object],
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    direction = str(blocked_arrow["direction"])
    tip_r, tip_c = map(int, blocked_arrow["tip"])
    if direction == "right":
        candidates = [(tip_r, c) for c in range(tip_c + 2, w)]
    elif direction == "left":
        candidates = [(tip_r, c) for c in range(0, tip_c - 1)]
    elif direction == "down":
        candidates = [(r, tip_c) for r in range(tip_r + 2, h)]
    else:
        candidates = [(r, tip_c) for r in range(0, tip_r - 1)]
    rng.shuffle(candidates)
    for r, c in candidates:
        if (r, c) in occupied:
            continue
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in occupied):
            continue
        return int(r), int(c)
    return None


def _gen_blocked(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(9, 11))
    w = int(rng.integers(9, 11))
    directions = ["right", "up"]
    if int(rng.integers(0, 2)):
        directions = ["down", "left"]
    if int(rng.integers(0, 2)):
        directions = ["right", "up"]
    n_arrows = 2 if int(rng.integers(0, 3)) else 3
    if n_arrows == 3:
        extra_choices = [d for d in ("left", "right", "up", "down") if d not in directions]
        directions.append(extra_choices[int(rng.integers(0, len(extra_choices)))])

    colors = list(rng.choice(np.array(ARROW_COLORS), size=len(directions), replace=False))
    candidates_by_dir = {
        direction: _candidate_arrows(h, w, direction, int(color))
        for direction, color in zip(directions, colors)
    }
    for cand_list in candidates_by_dir.values():
        rng.shuffle(cand_list)

    chosen: list[dict[str, object]] = []
    for direction in directions:
        placed = None
        for cand in candidates_by_dir[direction]:
            if all(_well_separated(cand["cells"], other["cells"]) for other in chosen):
                placed = cand
                break
        if placed is None:
            return None
        chosen.append(placed)

    blocked_arrow = chosen[0]
    occupied = set().union(*(arrow["cells"] for arrow in chosen))
    blocker = _find_blocker_cell(h, w, blocked_arrow, occupied, rng)
    if blocker is None:
        return None
    decorations = [(blocker[0], blocker[1], int(DECO_COLORS[int(rng.integers(0, len(DECO_COLORS)))]))]
    occupied.add(blocker)

    probe = _make_grid(h, w, chosen, decorations)
    predicted = _new_cells_by_direction(probe)
    ray_forbidden = {
        (r, c)
        for r, c in predicted
        if 0 <= r < h and 0 <= c < w
    }
    extras = _random_nonblocking_decor_cells(h, w, occupied, ray_forbidden, rng, int(rng.integers(0, 2)))
    grid = _make_grid(h, w, chosen, decorations + extras)
    return grid if _validate_generated(grid, "blocked") else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    modes = ("cross_copy", "same_copy", "blocked")
    for _ in range(GEN_ATTEMPTS):
        mode = modes[int(rng.integers(0, len(modes)))]
        if mode == "cross_copy":
            grid = _gen_cross_copy(rng)
        elif mode == "same_copy":
            grid = _gen_same_copy(rng)
        else:
            grid = _gen_blocked(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-copy arrow-bridge input")
