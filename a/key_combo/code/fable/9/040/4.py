from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


OFFSETS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OFFSETS8 = OFFSETS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
SIDES = ("L", "R", "T", "B")


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    offsets = OFFSETS4 if connectivity == 4 else OFFSETS8
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        if seen[r, c]:
            continue
        stack = [(int(r), int(c))]
        seen[r, c] = True
        coords = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in offsets:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _find_frame(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    H, W = grid.shape
    candidates = []
    for comp in _components(grid == bg, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
            continue
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        fr0, fc0, fr1, fc1 = r0 - 1, c0 - 1, r1 + 1, c1 + 1
        if fr0 < 0 or fc0 < 0 or fr1 >= H or fc1 >= W:
            continue
        ring = _ring_mask(grid.shape, (fr0, fc0, fr1, fc1))
        if np.any(grid[ring] == bg):
            continue
        candidates.append(((r1 - r0 + 1) * (c1 - c0 + 1), (r0, c0, r1, c1), (fr0, fc0, fr1, fc1)))
    assert candidates, "expected one enclosed frame interior"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, interior_bbox, frame_bbox = candidates[0]
    return interior_bbox, frame_bbox


def _find_divider(grid: np.ndarray, bg: int, frame_left: int) -> tuple[int, int]:
    candidates = []
    for c in range(frame_left):
        col = grid[:, c]
        if col[0] != bg and np.all(col == col[0]):
            candidates.append((c, int(col[0])))
    assert candidates, "expected a full-height divider column"
    return max(candidates, key=lambda item: item[0])


def _extract_sprite(grid: np.ndarray, bg: int, divider_c: int) -> dict:
    left_mask = grid[:, :divider_c] != bg
    comps = _components(left_mask, connectivity=8)
    assert comps, "expected one sprite in the left strip"
    comp = max(comps, key=len)
    values = {int(grid[r, c]) for r, c in comp}
    assert len(values) == 1, "sprite must be monochromatic"
    color = next(iter(values))
    r0, c0, r1, c1 = _bbox(comp)
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    pixels = [(int(r - center_r), int(c - center_c)) for r, c in comp]
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "center": (int(center_r), int(center_c)),
        "pixels": pixels,
        "coords": comp,
    }


def _marker_cell(frame_bbox: tuple[int, int, int, int], marker: tuple[str, int]) -> tuple[int, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    side, pos = marker
    if side == "L":
        return int(pos), int(fc0)
    if side == "R":
        return int(pos), int(fc1)
    if side == "T":
        return int(fr0), int(pos)
    if side == "B":
        return int(fr1), int(pos)
    raise ValueError(f"unknown side {side}")


def _collect_roles(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> dict:
    fr0, fc0, fr1, fc1 = frame_bbox
    border_cells = []
    for c in range(fc0, fc1 + 1):
        border_cells.append((fr0, c, "T"))
        border_cells.append((fr1, c, "B"))
    for r in range(fr0 + 1, fr1):
        border_cells.append((r, fc0, "L"))
        border_cells.append((r, fc1, "R"))

    base_color = Counter(int(grid[r, c]) for r, c, _ in border_cells).most_common(1)[0][0]
    by_color: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for r, c, side in border_cells:
        color = int(grid[r, c])
        if color != base_color:
            by_color[color].append((int(r), int(c), side))

    row_targets = []
    col_targets = []
    singletons = []
    for color, cells in by_color.items():
        if len(cells) == 1:
            r, c, side = cells[0]
            singletons.append({"color": int(color), "r": r, "c": c, "side": side})
            continue
        assert len(cells) == 2, "marker color must occur once or twice"
        sides = {side for _, _, side in cells}
        if sides == {"L", "R"}:
            rows = {r for r, _, _ in cells}
            assert len(rows) == 1, "paired side markers must share one row"
            row_targets.append(int(next(iter(rows))))
        elif sides == {"T", "B"}:
            cols = {c for _, c, _ in cells}
            assert len(cols) == 1, "paired top/bottom markers must share one column"
            col_targets.append(int(next(iter(cols))))
        else:
            raise AssertionError("paired markers must be on opposite sides")

    assert row_targets and col_targets, "need both target rows and target cols"
    assert len(singletons) >= 2, "need at least two unpaired markers"
    return {
        "base_color": int(base_color),
        "row_targets": sorted(row_targets),
        "col_targets": sorted(col_targets),
        "singletons": singletons,
    }


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    interior_bbox, frame_bbox = _find_frame(g, bg)
    _, divider_color = _find_divider(g, bg, frame_bbox[1])
    divider_c, _ = _find_divider(g, bg, frame_bbox[1])
    sprite = _extract_sprite(g, bg, divider_c)
    roles = _collect_roles(g, frame_bbox)
    single_rows = [m["r"] for m in roles["singletons"]]
    single_cols = [m["c"] for m in roles["singletons"]]
    forbidden_bbox = (min(single_rows), min(single_cols), max(single_rows), max(single_cols))
    targets = [(r, c) for r in roles["row_targets"] for c in roles["col_targets"]]
    cancelled = []
    kept = []
    rr0, cc0, rr1, cc1 = forbidden_bbox
    for r, c in targets:
        if rr0 <= r <= rr1 and cc0 <= c <= cc1:
            cancelled.append((int(r), int(c)))
        else:
            kept.append((int(r), int(c)))
    return {
        "grid": g,
        "bg": int(bg),
        "interior_bbox": interior_bbox,
        "frame_bbox": frame_bbox,
        "divider_c": int(divider_c),
        "divider_color": int(divider_color),
        "sprite": sprite,
        "roles": roles,
        "forbidden_bbox": forbidden_bbox,
        "targets": targets,
        "kept_targets": kept,
        "cancelled_targets": cancelled,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the left sprite onto marker-defined frame intersections, except inside the unpaired-marker window.

    Input structure:
    - A narrow left strip contains one monochrome sprite, separated from a right-side hollow frame by a full-height divider column.
    - The frame border is mostly one base color, with embedded marker pixels on its edges.
    - Some marker colors appear exactly twice on opposite sides of the frame, while other marker colors appear only once.

    Rule:
    - A color paired across the left and right edges defines one target row; a color paired across the top and bottom edges defines one target column.
    - The sprite is erased from the left strip, then copied into the frame interior with its center aligned to every row-column intersection.
    - The bounding box of all unpaired markers is a forbidden window, and any copy whose center falls inside that window is cancelled.

    Key insight:
    - Edge markers split by role: opposite-side pairs create stamp coordinates, while singletons do not place anything and instead carve out the no-stamp region.
    """
    info = _solve_details(grid)
    g = info["grid"]
    bg = info["bg"]
    ir0, ic0, ir1, ic1 = info["interior_bbox"]
    sprite = info["sprite"]

    out = g.copy()
    for r, c in sprite["coords"]:
        out[int(r), int(c)] = int(bg)

    for tr, tc in info["kept_targets"]:
        for dr, dc in sprite["pixels"]:
            rr = int(tr + dr)
            cc = int(tc + dc)
            if ir0 <= rr <= ir1 and ic0 <= cc <= ic1:
                out[rr, cc] = int(sprite["color"])

    return out


def _make_sprite() -> np.ndarray:
    h = random.choice((3, 5))
    w = random.choice((3, 5))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, w // 2] = True
    mask[h // 2, :] = True
    extra_cap = max(0, min(2, 9 - int(mask.sum())))
    extras = random.randint(0, extra_cap)
    candidates = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            near = False
            for dr, dc in OFFSETS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    near = True
                    break
            if near:
                candidates.append((r, c))
    random.shuffle(candidates)
    for r, c in candidates[:extras]:
        mask[r, c] = True
    return mask


def _sample_spaced(points: list[int], count: int, min_gap: int) -> list[int] | None:
    order = points[:]
    random.shuffle(order)
    chosen = []
    for value in order:
        if all(abs(value - prev) >= min_gap for prev in chosen):
            chosen.append(int(value))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _forbidden_targets(
    frame_bbox: tuple[int, int, int, int],
    markers: list[tuple[str, int]],
    rows: list[int],
    cols: list[int],
) -> list[tuple[int, int]]:
    cells = [_marker_cell(frame_bbox, marker) for marker in markers]
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    return [(r, c) for r in rows for c in cols if r0 <= r <= r1 and c0 <= c <= c1]


def _singleton_candidates(
    frame_bbox: tuple[int, int, int, int],
    row_targets: list[int],
    col_targets: list[int],
    free_rows: list[int],
    free_cols: list[int],
) -> list[list[tuple[str, int]]]:
    total = len(row_targets) * len(col_targets)
    out = []

    for a in free_rows:
        for b in free_rows:
            if a > b:
                continue
            markers = [("L", a), ("R", b)]
            cancelled = _forbidden_targets(frame_bbox, markers, row_targets, col_targets)
            if 0 < len(cancelled) < total:
                out.append(markers)

    for a in free_cols:
        for b in free_cols:
            if a > b:
                continue
            markers = [("T", a), ("B", b)]
            cancelled = _forbidden_targets(frame_bbox, markers, row_targets, col_targets)
            if 0 < len(cancelled) < total:
                out.append(markers)

    for side_row in ("L", "R"):
        for side_col in ("T", "B"):
            for r in free_rows:
                for c in free_cols:
                    markers = [(side_row, r), (side_col, c)]
                    cancelled = _forbidden_targets(frame_bbox, markers, row_targets, col_targets)
                    if 0 < len(cancelled) < total:
                        out.append(markers)

    for a in free_rows:
        for b in free_rows:
            if a > b:
                continue
            for side in ("T", "B"):
                for c in free_cols:
                    markers = [("L", a), ("R", b), (side, c)]
                    cancelled = _forbidden_targets(frame_bbox, markers, row_targets, col_targets)
                    if 0 < len(cancelled) < total:
                        out.append(markers)

    for a in free_cols:
        for b in free_cols:
            if a > b:
                continue
            for side in ("L", "R"):
                for r in free_rows:
                    markers = [("T", a), ("B", b), (side, r)]
                    cancelled = _forbidden_targets(frame_bbox, markers, row_targets, col_targets)
                    if 0 < len(cancelled) < total:
                        out.append(markers)

    random.shuffle(out)
    uniq = []
    seen = set()
    for markers in out:
        key = tuple(sorted(markers))
        if key not in seen:
            seen.add(key)
            uniq.append(markers)
    return uniq


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    sprite_mask = _make_sprite()
    sh, sw = sprite_mask.shape
    half_h, half_w = sh // 2, sw // 2

    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    interior_h = random.randint(max(8, sh + 4), 13)
    interior_w = random.randint(max(9, sw + 4), 14)
    left_w = random.randint(max(4, sw + 1), 7)
    right_margin = random.randint(0, 2)

    H = top_margin + interior_h + 2 + bottom_margin
    W = left_w + 1 + interior_w + 2 + right_margin
    if H > 30 or W > 30:
        return None

    fr0 = top_margin
    fc0 = left_w + 1
    fr1 = fr0 + interior_h + 1
    fc1 = fc0 + interior_w + 1
    ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
    frame_bbox = (fr0, fc0, fr1, fc1)

    center_rows = list(range(ir0 + half_h + 1, ir1 - half_h))
    center_cols = list(range(ic0 + half_w + 1, ic1 - half_w))
    if not center_rows:
        center_rows = list(range(ir0 + half_h, ir1 - half_h + 1))
    if not center_cols:
        center_cols = list(range(ic0 + half_w, ic1 - half_w + 1))

    count_options = []
    for nr in range(1, 4):
        for nc in range(1, 4):
            if nr * nc < 2 or nr + nc > 5:
                continue
            if _sample_spaced(center_rows, nr, sh + 1) is None:
                continue
            if _sample_spaced(center_cols, nc, sw + 1) is None:
                continue
            count_options.append((nr, nc))
    if not count_options:
        return None

    nr, nc = random.choice(count_options)
    row_targets = _sample_spaced(center_rows, nr, sh + 1)
    col_targets = _sample_spaced(center_cols, nc, sw + 1)
    if row_targets is None or col_targets is None:
        return None

    free_rows = [r for r in range(ir0, ir1 + 1) if r not in row_targets]
    free_cols = [c for c in range(ic0, ic1 + 1) if c not in col_targets]
    marker_options = [
        markers
        for markers in _singleton_candidates(frame_bbox, row_targets, col_targets, free_rows, free_cols)
        if 2 + nr + nc + len(markers) <= 9
    ]
    if not marker_options:
        return None
    singleton_markers = random.choice(marker_options)
    singleton_count = len(singleton_markers)

    palette = list(range(1, 10))
    random.shuffle(palette)
    needed = 2 + nr + nc + singleton_count
    if needed > len(palette):
        return None
    chosen = palette[:needed]
    base_color = chosen[0]
    sprite_color = chosen[1]
    row_colors = chosen[2 : 2 + nr]
    col_colors = chosen[2 + nr : 2 + nr + nc]
    singleton_colors = chosen[2 + nr + nc :]

    grid = np.full((H, W), bg, dtype=int)
    ring = _ring_mask(grid.shape, frame_bbox)
    grid[ring] = base_color
    grid[:, left_w] = base_color

    sprite_r0 = random.randint(0, H - sh)
    sprite_c0 = random.randint(0, left_w - sw)
    for r, c in np.argwhere(sprite_mask):
        grid[sprite_r0 + int(r), sprite_c0 + int(c)] = sprite_color

    for row, color in zip(row_targets, row_colors):
        grid[row, fc0] = color
        grid[row, fc1] = color
    for col, color in zip(col_targets, col_colors):
        grid[fr0, col] = color
        grid[fr1, col] = color
    for marker, color in zip(singleton_markers, singleton_colors):
        r, c = _marker_cell(frame_bbox, marker)
        if grid[r, c] != base_color:
            return None
        grid[r, c] = color

    info = _solve_details(grid)
    if sorted(info["roles"]["row_targets"]) != sorted(row_targets):
        return None
    if sorted(info["roles"]["col_targets"]) != sorted(col_targets):
        return None
    if len(info["roles"]["singletons"]) != singleton_count:
        return None
    if not info["kept_targets"] or not info["cancelled_targets"]:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out == sprite_color) <= np.count_nonzero(grid == sprite_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
