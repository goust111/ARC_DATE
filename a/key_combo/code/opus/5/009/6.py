from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(cells)),
                }
            )
    return out


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _is_rect_outline(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * (h + w) - 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])
    want = np.zeros((h, w), dtype=bool)
    want[0, :] = True
    want[-1, :] = True
    want[:, 0] = True
    want[:, -1] = True
    return bool(np.array_equal(patch, want))


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    comps = _find_components(g, bg=BG)
    frames = [comp for comp in comps if _is_rect_outline(g, comp)]
    frames.sort(key=lambda comp: (comp["bbox"], comp["color"]))

    erased_idx: set[int] = set()
    ghost_bboxes: list[tuple[int, int, int, int]] = []
    for i, comp in enumerate(frames):
        for j, other in enumerate(frames):
            if i == j or int(other["color"]) == int(comp["color"]):
                continue
            if int(other["area"]) <= int(comp["area"]):
                continue
            if _bbox_contains(other["bbox"], comp["bbox"]):
                erased_idx.add(i)
                ghost_bboxes.append(comp["bbox"])
                break

    survivors = [comp for i, comp in enumerate(frames) if i not in erased_idx]
    erased = [comp for i, comp in enumerate(frames) if i in erased_idx]
    survivor_colors = {int(comp["color"]) for comp in survivors}
    marker_rows = {
        int(color): [int(r) for r in np.where(g[:, 0] == int(color))[0]]
        for color, count in Counter(map(int, g[:, 0])).items()
        if int(color) != BG and int(count) == 2 and int(color) in survivor_colors
    }
    return {
        "frames": frames,
        "survivors": survivors,
        "erased": erased,
        "ghost_bboxes": ghost_bboxes,
        "marker_rows": marker_rows,
    }


def _build_masks(shape: tuple[int, int], survivors: list[dict], ghost_bboxes: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    survivor_mask = np.zeros((h, w), dtype=bool)
    for comp in survivors:
        coords = comp["coords"]
        survivor_mask[coords[:, 0], coords[:, 1]] = True
    ghost_mask = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in ghost_bboxes:
        ghost_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return survivor_mask, ghost_mask


def _first_blocker(row: int, survivor_mask: np.ndarray, ghost_mask: np.ndarray) -> tuple[str, int]:
    _h, w = survivor_mask.shape
    for c in range(1, w):
        if ghost_mask[row, c]:
            return "ghost", c
        if survivor_mask[row, c]:
            return "object", c
    return "edge", w


def _render(
    grid: np.ndarray,
    *,
    erase_contained: bool = True,
    use_ghost: bool = True,
) -> tuple[np.ndarray, dict, dict[int, list[tuple[int, str, int]]]]:
    g = np.asarray(grid, dtype=int)
    structure = _extract_structure(g)
    h, _w = g.shape
    out = g.copy()

    if erase_contained:
        for comp in structure["erased"]:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = BG
        survivors = structure["survivors"]
        ghost_bboxes = structure["ghost_bboxes"] if use_ghost else []
    else:
        survivors = structure["frames"]
        ghost_bboxes = []

    survivor_mask, ghost_mask = _build_masks((h, g.shape[1]), survivors, ghost_bboxes)
    ray_details: dict[int, list[tuple[int, str, int]]] = {}
    for color in sorted(structure["marker_rows"]):
        rows = sorted(structure["marker_rows"][color])
        if len(rows) != 2:
            continue
        r1, r2 = rows
        period = r2 - r1
        if period <= 0:
            continue
        ray_details[int(color)] = []
        for r in range(r1, h, period):
            reason, stop_c = _first_blocker(r, survivor_mask, ghost_mask)
            ray_details[int(color)].append((int(r), reason, int(stop_c)))
            for c in range(1, stop_c):
                if int(out[r, c]) == BG:
                    out[r, c] = int(color)
    return out, structure, ray_details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase bbox-contained small frames, then cast periodic left-edge rays that are blocked by both visible frames and invisible ghost boxes.

    Input structure:
    - Black background with 4-6 single-color 8-connected rectangular outline objects.
    - One or two smaller frames have bounding boxes fully contained in a larger frame of a different color.
    - On the left edge, some surviving-frame colors appear exactly twice as marker pixels; the row gap of each same-color pair defines a period.

    Rule:
    - Detect all monochromatic rectangular outline components.
    - Erase every frame whose bbox is contained in a larger different-color frame, but remember the erased frame's whole bbox as a ghost blocker.
    - For each left-edge color pair at rows r1 < r2, cast horizontal rays from column 1 on rows r1, r1 + P, r1 + 2P, ... where P = r2 - r1.
    - Paint only background cells with that marker color until the ray would enter a surviving frame pixel or any ghost bbox cell; the stopping cell stays unchanged.

    Key insight:
    - The deleted inner frames still matter geometrically: their now-invisible bounding boxes terminate periodic rays earlier than any visible paint would suggest.
    """
    return _render(grid, erase_contained=True, use_ghost=True)[0]


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_free_frame(
    h: int,
    w: int,
    reserved: list[tuple[int, int, int, int]],
    *,
    min_h: int,
    max_h: int,
    min_w: int,
    max_w: int,
    min_c0: int,
) -> tuple[int, int, int, int] | None:
    for _ in range(200):
        fh = random.randint(min_h, max_h)
        fw = random.randint(min_w, max_w)
        if fh >= h - 1 or fw >= w - min_c0:
            continue
        r0 = random.randint(1, h - fh - 2)
        c0 = random.randint(min_c0, w - fw - 1)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if any(_overlaps(bbox, other, pad=1) for other in reserved):
            continue
        return bbox
    return None


def _place_nested_pair(
    h: int,
    w: int,
    reserved: list[tuple[int, int, int, int]],
    *,
    min_c0: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    outer = _place_free_frame(h, w, reserved, min_h=7, max_h=10, min_w=7, max_w=10, min_c0=min_c0)
    if outer is None:
        return None
    r0, c0, r1, c1 = outer
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    for _ in range(100):
        max_ih = min(5, outer_h - 4)
        max_iw = min(5, outer_w - 4)
        if max_ih < 3 or max_iw < 3:
            return None
        ih = random.randint(3, max_ih)
        iw = random.randint(3, max_iw)
        ir0 = random.randint(r0 + 2, r1 - ih - 1)
        ic0 = random.randint(c0 + 2, c1 - iw - 1)
        inner = (ir0, ic0, ir0 + ih - 1, ic0 + iw - 1)
        if inner[2] < r1 and inner[3] < c1:
            return outer, inner
    return None


def _row_reasons(grid: np.ndarray) -> tuple[list[str], np.ndarray]:
    _, structure, _ = _render(grid, erase_contained=True, use_ghost=True)
    survivor_mask, ghost_mask = _build_masks(grid.shape, structure["survivors"], structure["ghost_bboxes"])
    reasons = [_first_blocker(r, survivor_mask, ghost_mask)[0] for r in range(grid.shape[0])]
    return reasons, ghost_mask


def _find_marker_pair(
    grid: np.ndarray,
    reasons: list[str],
    *,
    require_ghost: bool,
    used_marker_rows: set[int],
    used_emitted_rows: set[int],
) -> tuple[int, int, list[int]] | None:
    h, _w = grid.shape
    free_rows = [r for r in range(h) if int(grid[r, 0]) == BG and r not in used_marker_rows]
    random.shuffle(free_rows)
    periods = [3, 4, 5, 6, 2]
    for period in periods:
        for r1 in free_rows:
            r2 = r1 + period
            if r2 >= h or int(grid[r2, 0]) != BG or r2 in used_marker_rows:
                continue
            seq = list(range(r1, h, period))
            if set(seq) & used_emitted_rows:
                continue
            seq_reasons = [reasons[r] for r in seq]
            if require_ghost:
                if "ghost" not in seq_reasons or not any(reason != "ghost" for reason in seq_reasons):
                    continue
            else:
                if not any(reason != "ghost" for reason in seq_reasons):
                    continue
            return int(r1), int(r2), seq
    return None


def _semantic_checks(grid: np.ndarray) -> bool:
    out, structure, ray_details = _render(grid, erase_contained=True, use_ghost=True)
    frames = structure["frames"]
    erased = structure["erased"]
    markers = structure["marker_rows"]
    if not (4 <= len(frames) <= 6):
        return False
    if not (1 <= len(erased) <= 2):
        return False
    if not (2 <= len(markers) <= 3):
        return False
    if not all(len(rows) == 2 for rows in markers.values()):
        return False

    reasons = [reason for rows in ray_details.values() for _r, reason, _c in rows]
    if "ghost" not in reasons:
        return False
    if not any(reason != "ghost" for reason in reasons):
        return False

    no_ghost = _render(grid, erase_contained=True, use_ghost=False)[0]
    if np.array_equal(out, no_ghost):
        return False

    no_erase = _render(grid, erase_contained=False, use_ghost=False)[0]
    if np.array_equal(out, no_erase):
        return False

    erase_only = np.asarray(grid, dtype=int).copy()
    for comp in erased:
        coords = comp["coords"]
        erase_only[coords[:, 0], coords[:, 1]] = BG
    if np.array_equal(out, erase_only):
        return False

    return True


def _generate_case() -> np.ndarray | None:
    h = random.randint(19, 28)
    w = random.randint(19, 28)
    grid = np.zeros((h, w), dtype=int)

    total_frames = random.choice([4, 5, 5, 6])
    contained_count = 1 if total_frames == 4 else random.choice([1, 2])
    survivor_target = total_frames - contained_count
    extra_survivors = survivor_target - 1
    if extra_survivors < 2:
        return None

    color_count = total_frames
    colors = random.sample(PALETTE, color_count)
    color_it = iter(colors)

    reserved: list[tuple[int, int, int, int]] = []
    specs: list[tuple[tuple[int, int, int, int], int]] = []

    outer_h = random.randint(8, min(12, h - 6))
    outer_w = random.randint(7, min(11, w - 8))
    top_min = 3
    top_max = h - outer_h - 3
    if top_max < top_min:
        return None
    outer_r0 = random.randint(top_min, top_max)
    outer = (outer_r0, 0, outer_r0 + outer_h - 1, outer_w - 1)

    max_inner_h = min(5, outer_h - 4)
    max_inner_w = min(5, outer_w - 4)
    if max_inner_h < 3 or max_inner_w < 3:
        return None
    inner_h = random.randint(3, max_inner_h)
    inner_w = random.randint(3, max_inner_w)
    inner_r0 = random.randint(outer[0] + 2, outer[2] - inner_h - 1)
    inner_c0 = random.randint(2, outer[3] - inner_w - 1)
    inner = (inner_r0, inner_c0, inner_r0 + inner_h - 1, inner_c0 + inner_w - 1)

    outer_color = next(color_it)
    inner_color = next(color_it)
    specs.append((outer, outer_color))
    specs.append((inner, inner_color))
    reserved.append(outer)

    if contained_count == 2:
        pair = _place_nested_pair(h, w, reserved, min_c0=max(outer[3] + 2, 6))
        if pair is None:
            return None
        outer2, inner2 = pair
        specs.append((outer2, next(color_it)))
        specs.append((inner2, next(color_it)))
        reserved.append(outer2)

    survivor_placed = 1 + (1 if contained_count == 2 else 0)
    survivors_needed = survivor_target - survivor_placed
    while survivors_needed > 0:
        bbox = _place_free_frame(h, w, reserved, min_h=4, max_h=8, min_w=4, max_w=8, min_c0=5)
        if bbox is None:
            return None
        specs.append((bbox, next(color_it)))
        reserved.append(bbox)
        survivors_needed -= 1

    if len(specs) != total_frames:
        return None

    for bbox, color in specs:
        _draw_frame(grid, bbox, color)

    out0, structure0, _details0 = _render(grid, erase_contained=True, use_ghost=True)
    if len(structure0["frames"]) != total_frames or len(structure0["erased"]) != contained_count:
        return None

    survivor_colors = [int(comp["color"]) for comp in structure0["survivors"] if int(comp["bbox"][1]) > 0]
    if len(survivor_colors) < 2:
        return None
    marker_colors = random.sample(survivor_colors, k=2 if len(survivor_colors) == 2 else random.choice([2, 3]))

    reasons, _ghost_mask = _row_reasons(grid)
    used_marker_rows: set[int] = set()
    used_emitted_rows: set[int] = set()

    ghost_pair = _find_marker_pair(
        grid,
        reasons,
        require_ghost=True,
        used_marker_rows=used_marker_rows,
        used_emitted_rows=used_emitted_rows,
    )
    if ghost_pair is None:
        return None
    r1, r2, seq = ghost_pair
    grid[r1, 0] = marker_colors[0]
    grid[r2, 0] = marker_colors[0]
    used_marker_rows.update({r1, r2})
    used_emitted_rows.update(seq)

    nonghost_pair = _find_marker_pair(
        grid,
        reasons,
        require_ghost=False,
        used_marker_rows=used_marker_rows,
        used_emitted_rows=used_emitted_rows,
    )
    if nonghost_pair is None:
        return None
    r1, r2, seq = nonghost_pair
    grid[r1, 0] = marker_colors[1]
    grid[r2, 0] = marker_colors[1]
    used_marker_rows.update({r1, r2})
    used_emitted_rows.update(seq)

    if len(marker_colors) == 3:
        extra_pair = _find_marker_pair(
            grid,
            reasons,
            require_ghost=False,
            used_marker_rows=used_marker_rows,
            used_emitted_rows=used_emitted_rows,
        )
        if extra_pair is None:
            return None
        r1, r2, seq = extra_pair
        grid[r1, 0] = marker_colors[2]
        grid[r2, 0] = marker_colors[2]
        used_marker_rows.update({r1, r2})
        used_emitted_rows.update(seq)

    if not _semantic_checks(grid):
        return None

    if np.array_equal(grid, out0):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
