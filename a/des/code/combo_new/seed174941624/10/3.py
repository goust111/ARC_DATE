import random
from typing import Iterable

import numpy as np


DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ORIENTATIONS = {
    "tl": {
        "adj": [(-1, 0), (0, -1), (-1, -1)],
        "empty": [(1, 0), (0, 1)],
    },
    "tr": {
        "adj": [(-1, 0), (0, 1), (-1, 1)],
        "empty": [(1, 0), (0, -1)],
    },
    "bl": {
        "adj": [(1, 0), (0, -1), (1, -1)],
        "empty": [(-1, 0), (0, 1)],
    },
    "br": {
        "adj": [(1, 0), (0, 1), (1, 1)],
        "empty": [(-1, 0), (0, -1)],
    },
}

GEN_ATTEMPTS = 400


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _marker_position(r0: int, c0: int, r1: int, c1: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return (r0 + 1, c0 + 1)
    if corner == "tr":
        return (r0 + 1, c1 - 1)
    if corner == "bl":
        return (r1 - 1, c0 + 1)
    if corner == "br":
        return (r1 - 1, c1 - 1)
    raise ValueError(corner)


def _gate_position(r0: int, c0: int, r1: int, c1: int, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return (r0, c0 + offset)
    if side == "bottom":
        return (r1, c0 + offset)
    if side == "left":
        return (r0 + offset, c0)
    if side == "right":
        return (r0 + offset, c1)
    raise ValueError(side)


def _spec_nonzero_cells(spec: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = spec["bbox"]
    gate_positions = {gate["pos"] for gate in spec["gates"]}
    cells = {cell for cell in _border_cells(r0, c0, r1, c1) if cell not in gate_positions}
    cells.add(spec["marker"])
    return cells


def _can_place(spec: dict, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    cells = _spec_nonzero_cells(spec)
    for r, c in cells:
        if occupied[r, c]:
            return False
        rr0 = max(0, r - 1)
        rr1 = min(h, r + 2)
        cc0 = max(0, c - 1)
        cc1 = min(w, c + 2)
        if bool(occupied[rr0:rr1, cc0:cc1].any()):
            return False
    return True


def _stamp_spec(grid: np.ndarray, spec: dict) -> None:
    r0, c0, r1, c1 = spec["bbox"]
    fence_color = int(spec["fence_color"])
    for r, c in _border_cells(r0, c0, r1, c1):
        grid[r, c] = fence_color
    for gate in spec["gates"]:
        gr, gc = gate["pos"]
        grid[gr, gc] = 0
    mr, mc = spec["marker"]
    grid[mr, mc] = int(spec["marker_color"])


def _validate_fence(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    fence_color: int,
    marker: tuple[int, int],
) -> dict | None:
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    h, w = g.shape
    if not (_in_bounds(h, w, r0, c0) and _in_bounds(h, w, r1, c1)):
        return None

    mr, mc = marker
    if not (r0 < mr < r1 and c0 < mc < c1):
        return None
    if (mr, mc) not in {
        (r0 + 1, c0 + 1),
        (r0 + 1, c1 - 1),
        (r1 - 1, c0 + 1),
        (r1 - 1, c1 - 1),
    }:
        return None

    marker_color = int(g[mr, mc])
    if marker_color == 0 or marker_color == fence_color:
        return None

    for rr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            val = int(g[rr, cc])
            if (rr, cc) == (mr, mc):
                continue
            if val != 0:
                return None

    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    for rr, cc in corners:
        if int(g[rr, cc]) != fence_color:
            return None

    border_count = 4
    gates: list[tuple[int, int, int, int]] = []
    for cc in range(c0 + 1, c1):
        top = int(g[r0, cc])
        bottom = int(g[r1, cc])
        if top not in (0, fence_color) or bottom not in (0, fence_color):
            return None
        if top == 0:
            if int(g[r0, cc - 1]) != fence_color or int(g[r0, cc + 1]) != fence_color:
                return None
            gates.append((r0, cc, -1, 0))
        else:
            border_count += 1
        if bottom == 0:
            if int(g[r1, cc - 1]) != fence_color or int(g[r1, cc + 1]) != fence_color:
                return None
            gates.append((r1, cc, 1, 0))
        else:
            border_count += 1

    for rr in range(r0 + 1, r1):
        left = int(g[rr, c0])
        right = int(g[rr, c1])
        if left not in (0, fence_color) or right not in (0, fence_color):
            return None
        if left == 0:
            if int(g[rr - 1, c0]) != fence_color or int(g[rr + 1, c0]) != fence_color:
                return None
            gates.append((rr, c0, 0, -1))
        else:
            border_count += 1
        if right == 0:
            if int(g[rr - 1, c1]) != fence_color or int(g[rr + 1, c1]) != fence_color:
                return None
            gates.append((rr, c1, 0, 1))
        else:
            border_count += 1

    if not gates:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "fence_color": fence_color,
        "marker": (mr, mc),
        "marker_color": marker_color,
        "gates": gates,
        "score": (border_count, (r1 - r0 + 1) * (c1 - c0 + 1)),
    }


def _candidate_boxes(
    g: np.ndarray, r: int, c: int, orientation: str, fence_color: int
) -> Iterable[tuple[int, int, int, int]]:
    h, w = g.shape
    if orientation == "tl":
        r0, c0 = r - 1, c - 1
        if r0 < 0 or c0 < 0:
            return []
        row_targets = [cc for cc in range(c + 1, w) if int(g[r0, cc]) == fence_color]
        col_targets = [rr for rr in range(r + 1, h) if int(g[rr, c0]) == fence_color]
        return [(r0, c0, rr, cc) for rr in col_targets for cc in row_targets]
    if orientation == "tr":
        r0, c1 = r - 1, c + 1
        if r0 < 0 or c1 >= w:
            return []
        row_targets = [cc for cc in range(0, c) if int(g[r0, cc]) == fence_color]
        col_targets = [rr for rr in range(r + 1, h) if int(g[rr, c1]) == fence_color]
        return [(r0, cc, rr, c1) for rr in col_targets for cc in row_targets]
    if orientation == "bl":
        r1, c0 = r + 1, c - 1
        if r1 >= h or c0 < 0:
            return []
        row_targets = [cc for cc in range(c + 1, w) if int(g[r1, cc]) == fence_color]
        col_targets = [rr for rr in range(0, r) if int(g[rr, c0]) == fence_color]
        return [(rr, c0, r1, cc) for rr in col_targets for cc in row_targets]
    if orientation == "br":
        r1, c1 = r + 1, c + 1
        if r1 >= h or c1 >= w:
            return []
        row_targets = [cc for cc in range(0, c) if int(g[r1, cc]) == fence_color]
        col_targets = [rr for rr in range(0, r) if int(g[rr, c1]) == fence_color]
        return [(rr, cc, r1, c1) for rr in col_targets for cc in row_targets]
    raise ValueError(orientation)


def _find_fences(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    fences: list[dict] = []
    seen_markers: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            marker_color = int(g[r, c])
            if marker_color == 0 or (r, c) in seen_markers:
                continue

            best: dict | None = None
            for orientation, config in ORIENTATIONS.items():
                adj_values = []
                valid = True
                for dr, dc in config["adj"]:
                    rr, cc = r + dr, c + dc
                    if not _in_bounds(h, w, rr, cc):
                        valid = False
                        break
                    adj_values.append(int(g[rr, cc]))
                if not valid:
                    continue

                fence_color = adj_values[0]
                if fence_color == 0 or fence_color == marker_color or any(v != fence_color for v in adj_values):
                    continue

                for dr, dc in config["empty"]:
                    rr, cc = r + dr, c + dc
                    if not _in_bounds(h, w, rr, cc) or int(g[rr, cc]) != 0:
                        valid = False
                        break
                if not valid:
                    continue

                for bbox in _candidate_boxes(g, r, c, orientation, fence_color):
                    cand = _validate_fence(g, *bbox, fence_color, (r, c))
                    if cand is None:
                        continue
                    if best is None or cand["score"] > best["score"]:
                        best = cand

            if best is not None:
                fences.append(best)
                seen_markers.add((r, c))

    return fences


def _emit_rays(g: np.ndarray, fences: list[dict]) -> np.ndarray:
    out = g.copy()
    h, w = g.shape

    for fence in fences:
        ray_color = int(fence["marker_color"])
        ray_len = len(fence["gates"])
        for gr, gc, dr, dc in fence["gates"]:
            for step in range(ray_len):
                rr = gr + dr * step
                cc = gc + dc * step
                if not _in_bounds(h, w, rr, cc):
                    break
                if int(g[rr, cc]) != 0:
                    break
                if int(out[rr, cc]) == 0:
                    out[rr, cc] = ray_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect gapped rectangular fences and cast outward rays whose length is the fence's gate count.

    Input structure:
    - Background is black.
    - Each fence is a one-pixel-thick rectangular ring of one non-black color with one or more single-pixel border gaps.
    - The fence interior is black except for one colored pixel placed at one interior corner; that pixel gives the ray color.
    - Fences do not touch each other, and gap cells lie on sides rather than corners.

    Rule:
    - For every fence, identify all black single-pixel gaps on its border.
    - Use the interior corner pixel's color as that fence's ray color.
    - From each gap cell, draw a straight ray starting at the gap and extending outward, perpendicular to the side that contains the gap.
    - Every ray from the same fence has length equal to the total number of gaps on that fence, and a ray stops early if the next cell is outside the grid or already non-black in the input.

    Key insight:
    - The missing border cells are the meaningful markers: they encode both the ray directions and, via their count, the shared ray length, while the lone interior corner pixel supplies the drawing color.
    """
    g = np.asarray(grid, dtype=int)
    fences = _find_fences(g)
    return _emit_rays(g, fences)


def _forbidden_offset(side: str, marker_corner: str, h: int, w: int) -> int | None:
    if marker_corner == "tl":
        if side == "top":
            return 1
        if side == "left":
            return 1
    if marker_corner == "tr":
        if side == "top":
            return w - 2
        if side == "right":
            return 1
    if marker_corner == "bl":
        if side == "bottom":
            return 1
        if side == "left":
            return h - 2
    if marker_corner == "br":
        if side == "bottom":
            return w - 2
        if side == "right":
            return h - 2
    return None


def _random_gate_offset(rng: random.Random, side: str, h: int, w: int, marker_corner: str) -> int:
    if side in ("top", "bottom"):
        candidates = list(range(1, w - 1))
    else:
        candidates = list(range(1, h - 1))
    forbidden = _forbidden_offset(side, marker_corner, h, w)
    if forbidden in candidates:
        candidates.remove(forbidden)
    return rng.choice(candidates)


def _make_spec(
    bbox: tuple[int, int, int, int],
    fence_color: int,
    marker_color: int,
    marker_corner: str,
    gate_sides: list[str],
    rng: random.Random,
    forced_offsets: dict[str, int] | None = None,
) -> dict:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    gates = []
    forced_offsets = forced_offsets or {}
    for side in gate_sides:
        offset = forced_offsets.get(side)
        if offset is None:
            offset = _random_gate_offset(rng, side, h, w, marker_corner)
        gates.append({"side": side, "pos": _gate_position(r0, c0, r1, c1, side, offset)})
    return {
        "bbox": bbox,
        "fence_color": int(fence_color),
        "marker_color": int(marker_color),
        "marker": _marker_position(r0, c0, r1, c1, marker_corner),
        "gates": gates,
    }


def _choose_gate_sides(rng: random.Random, k: int, must_include: str | None = None) -> list[str]:
    sides = ["top", "bottom", "left", "right"]
    if must_include is None:
        return rng.sample(sides, k)
    remaining = [side for side in sides if side != must_include]
    return [must_include] + rng.sample(remaining, k - 1)


def _pair_specs(rng: random.Random, h: int, w: int, colors: list[int]) -> list[dict] | None:
    direction = rng.choice(["top", "bottom", "left", "right"])
    fence_a, marker_a, fence_b, marker_b = colors[:4]
    marker_corner_a = rng.choice(list(ORIENTATIONS))
    marker_corner_b = rng.choice(list(ORIENTATIONS))

    if direction in ("left", "right"):
        rect_h = rng.randint(5, 7)
        rect_w_a = rng.randint(5, 8)
        rect_w_b = rng.randint(5, 8)
        k_a = rng.randint(2, 4)
        d = rng.randint(2, k_a)
        if rect_h >= h or rect_w_a + rect_w_b + d > w:
            return None
        top = rng.randint(0, h - rect_h)
        start_col = rng.randint(0, w - (rect_w_a + rect_w_b + d))
        if direction == "right":
            bbox_a = (top, start_col, top + rect_h - 1, start_col + rect_w_a - 1)
            bbox_b = (
                top,
                bbox_a[3] + d,
                top + rect_h - 1,
                bbox_a[3] + d + rect_w_b - 1,
            )
        else:
            bbox_b = (top, start_col, top + rect_h - 1, start_col + rect_w_b - 1)
            bbox_a = (
                top,
                bbox_b[3] + d,
                top + rect_h - 1,
                bbox_b[3] + d + rect_w_a - 1,
            )
        shared_candidates = list(range(1, rect_h - 1))
        forbidden = _forbidden_offset(direction, marker_corner_a, rect_h, rect_w_a)
        if forbidden in shared_candidates:
            shared_candidates.remove(forbidden)
        shared_offset = rng.choice(shared_candidates)
        sides_a = _choose_gate_sides(rng, k_a, must_include=direction)
        sides_b = _choose_gate_sides(rng, rng.randint(1, 4))
        spec_a = _make_spec(
            bbox_a,
            fence_a,
            marker_a,
            marker_corner_a,
            sides_a,
            rng,
            forced_offsets={direction: shared_offset},
        )
        spec_b = _make_spec(
            bbox_b,
            fence_b,
            marker_b,
            marker_corner_b,
            sides_b,
            rng,
        )
        return [spec_a, spec_b]

    rect_w = rng.randint(5, 7)
    rect_h_a = rng.randint(5, 8)
    rect_h_b = rng.randint(5, 8)
    k_a = rng.randint(2, 4)
    d = rng.randint(2, k_a)
    if rect_w >= w or rect_h_a + rect_h_b + d > h:
        return None
    left = rng.randint(0, w - rect_w)
    start_row = rng.randint(0, h - (rect_h_a + rect_h_b + d))
    if direction == "bottom":
        bbox_a = (start_row, left, start_row + rect_h_a - 1, left + rect_w - 1)
        bbox_b = (
            bbox_a[2] + d,
            left,
            bbox_a[2] + d + rect_h_b - 1,
            left + rect_w - 1,
        )
    else:
        bbox_b = (start_row, left, start_row + rect_h_b - 1, left + rect_w - 1)
        bbox_a = (
            bbox_b[2] + d,
            left,
            bbox_b[2] + d + rect_h_a - 1,
            left + rect_w - 1,
        )
    shared_candidates = list(range(1, rect_w - 1))
    forbidden = _forbidden_offset(direction, marker_corner_a, rect_h_a, rect_w)
    if forbidden in shared_candidates:
        shared_candidates.remove(forbidden)
    shared_offset = rng.choice(shared_candidates)
    sides_a = _choose_gate_sides(rng, k_a, must_include=direction)
    sides_b = _choose_gate_sides(rng, rng.randint(1, 4))
    spec_a = _make_spec(
        bbox_a,
        fence_a,
        marker_a,
        marker_corner_a,
        sides_a,
        rng,
        forced_offsets={direction: shared_offset},
    )
    spec_b = _make_spec(
        bbox_b,
        fence_b,
        marker_b,
        marker_corner_b,
        sides_b,
        rng,
    )
    return [spec_a, spec_b]


def _random_spec(
    rng: random.Random,
    h: int,
    w: int,
    fence_color: int,
    marker_color: int,
    force_multi_gate: bool,
) -> dict:
    rect_h = rng.randint(4, 8)
    rect_w = rng.randint(4, 8)
    row_min = 0 if rng.random() < 0.35 else 1
    col_min = 0 if rng.random() < 0.35 else 1
    row_max = h - rect_h
    col_max = w - rect_w
    r0 = rng.randint(row_min, row_max)
    c0 = rng.randint(col_min, col_max)
    k = rng.randint(2, 4) if force_multi_gate else rng.randint(1, 4)
    sides = _choose_gate_sides(rng, k)
    return _make_spec(
        (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1),
        fence_color,
        marker_color,
        rng.choice(list(ORIENTATIONS)),
        sides,
        rng,
    )


def _gate_trace(spec: dict) -> list[tuple[int, int]]:
    ray_len = len(spec["gates"])
    cells = []
    for gate in spec["gates"]:
        gr, gc = gate["pos"]
        dr, dc = DIRS[gate["side"]]
        for step in range(ray_len):
            cells.append((gr + dr * step, gc + dc * step))
    return cells


def _has_blocked_gate(grid: np.ndarray, spec: dict) -> bool:
    h, w = grid.shape
    ray_len = len(spec["gates"])
    for gate in spec["gates"]:
        gr, gc = gate["pos"]
        dr, dc = DIRS[gate["side"]]
        for step in range(ray_len):
            rr = gr + dr * step
            cc = gc + dc * step
            if not _in_bounds(h, w, rr, cc):
                break
            if int(grid[rr, cc]) != 0:
                return step > 0
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 28)
    w = rng.randint(14, 28)
    num_fences = rng.randint(1, 3)
    use_blocker = num_fences >= 2 and rng.random() < 0.5

    colors = rng.sample(list(range(1, 10)), 2 * num_fences)
    color_idx = 0
    specs: list[dict] = []
    occupied = np.zeros((h, w), dtype=bool)

    if use_blocker:
        pair = _pair_specs(rng, h, w, colors[color_idx : color_idx + 4])
        if pair is None:
            return None
        for spec in pair:
            if not _can_place(spec, occupied):
                return None
            specs.append(spec)
            for r, c in _spec_nonzero_cells(spec):
                occupied[r, c] = True
        color_idx += 4

    while len(specs) < num_fences:
        fence_color = colors[color_idx]
        marker_color = colors[color_idx + 1]
        force_multi_gate = len(specs) == 0 and not use_blocker
        placed = False
        for _ in range(80):
            spec = _random_spec(rng, h, w, fence_color, marker_color, force_multi_gate=force_multi_gate)
            if not _can_place(spec, occupied):
                continue
            specs.append(spec)
            for r, c in _spec_nonzero_cells(spec):
                occupied[r, c] = True
            placed = True
            break
        if not placed:
            return None
        color_idx += 2

    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        _stamp_spec(grid, spec)

    fences = _find_fences(grid)
    if len(fences) != len(specs):
        return None

    out = main(grid)
    if out.shape != grid.shape or not np.any(out != grid):
        return None

    if use_blocker and not any(_has_blocked_gate(grid, spec) for spec in specs[:2]):
        return None

    if rng.random() < 0.25 and len(specs) >= 2:
        changed = int(np.count_nonzero(out != grid))
        if changed < 4:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fence-ray puzzle input")
