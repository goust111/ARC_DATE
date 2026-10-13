from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _count_holes_in_component(grid: np.ndarray, comp: list[tuple[int, int]], frame_color: int) -> int:
    r0, c0, r1, c1 = _bbox(comp)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    solid = sub == int(frame_color)
    h, w = solid.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0

    for r in range(h):
        for c in range(w):
            if solid[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_boundary = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_boundary = True
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not solid[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_boundary:
                holes += 1
    return int(holes)


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for c in range(w):
        column = grid[:, c]
        if int(column[0]) == int(bg):
            continue
        if not np.all(column == column[0]):
            continue
        if c == 0 or c == w - 1:
            continue
        if not np.any(grid[:, :c] != int(bg)):
            continue
        if not np.any(grid[:, c + 1 :] != int(bg)):
            continue
        candidates.append(int(c))
    assert len(candidates) == 1, "expected exactly one full separator column"
    return candidates[0]


def _palette_colors(grid: np.ndarray, sep_col: int, bg: int) -> list[int]:
    right = grid[:, sep_col + 1 :]
    coords = np.argwhere(right != int(bg))
    coords = sorted((int(r), int(c)) for r, c in coords)
    assert coords, "expected a non-empty palette"

    mask = right != int(bg)
    comps = _components(mask)
    assert all(len(comp) == 1 for comp in comps), "palette entries must be singleton pixels"

    colors = [int(right[r, c]) for r, c in coords]
    assert len(colors) == len(set(colors)), "palette colors must be distinct"
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame's outer halo with the palette color indexed by its number of enclosed background holes.

    Input structure:
    - The background is the most common color.
    - A single full-height separator column splits the grid into a left frame area and a right palette area.
    - The left area contains several same-colored rectangular frame components; internal walls of that same color may divide a frame interior into multiple enclosed background holes.
    - The right area contains a vertical palette made of distinct non-background singleton pixels, ordered from top to bottom.

    Rule:
    - Read the palette top-to-bottom so its first pixel corresponds to 1 hole, the second to 2 holes, and so on.
    - For each left-side frame component, count the 4-connected background regions inside its bbox that do not touch that bbox boundary.
    - Expand the component bbox by one cell in every direction, clipped to the grid, and recolor only the background cells in that outer band with the palette color for that hole count.
    - Keep the separator, the palette pixels, the frame pixels, and the interiors of the frames unchanged.

    Key insight:
    The interior hole count is not the output itself; it is an index into a separate palette, and that looked-up color is then applied to a one-cell halo outside the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep_col = _find_separator_col(g, bg)
    palette = _palette_colors(g, sep_col, bg)

    left = g[:, :sep_col]
    left_non_bg = [int(v) for v in np.unique(left) if int(v) != int(bg)]
    assert len(left_non_bg) == 1, "left side must use one frame color"
    frame_color = left_non_bg[0]

    comps = _components(left == int(frame_color))
    assert comps, "expected at least one frame component"

    out = g.copy()
    h, w = g.shape
    for comp in comps:
        holes = _count_holes_in_component(left, comp, frame_color)
        if holes < 1 or holes > len(palette):
            continue
        color = int(palette[holes - 1])
        r0, c0, r1, c1 = _bbox(comp)
        er0 = max(0, r0 - 1)
        ec0 = max(0, c0 - 1)
        er1 = min(h - 1, r1 + 1)
        ec1 = min(sep_col - 1, c1 + 1)
        for r in range(er0, er1 + 1):
            for c in range(ec0, ec1 + 1):
                inside_bbox = r0 <= r <= r1 and c0 <= c <= c1
                if inside_bbox or int(g[r, c]) != int(bg):
                    continue
                out[r, c] = color
    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out: list[int] = []
    for cut in cuts + [int(total)]:
        out.append(int(cut - prev))
        prev = cut
    return out


def _frame_patch(hole_count: int, orient: str, frame_color: int, bg: int) -> np.ndarray:
    assert hole_count >= 1
    if orient == "vertical":
        inner_h = random.randint(2, 6)
        inner_w = 2 * hole_count - 1 + random.randint(0, 3)
        chamber_sum = inner_w - (hole_count - 1)
        chamber_widths = _positive_composition(chamber_sum, hole_count)
        patch = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
        patch[0, :] = frame_color
        patch[-1, :] = frame_color
        patch[:, 0] = frame_color
        patch[:, -1] = frame_color
        x = 1
        for width in chamber_widths[:-1]:
            x += int(width)
            patch[1:-1, x] = frame_color
            x += 1
        return patch

    assert orient == "horizontal"
    inner_h = 2 * hole_count - 1 + random.randint(0, 3)
    inner_w = random.randint(2, 6)
    chamber_sum = inner_h - (hole_count - 1)
    chamber_heights = _positive_composition(chamber_sum, hole_count)
    patch = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    y = 1
    for height in chamber_heights[:-1]:
        y += int(height)
        patch[y, 1:-1] = frame_color
        y += 1
    return patch


def _ring_cells_for_bbox(
    top: int,
    left: int,
    height: int,
    width: int,
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]]:
    r0, c0 = int(top), int(left)
    r1 = r0 + int(height) - 1
    c1 = c0 + int(width) - 1
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(int(grid_h) - 1, r1 + 1)
    ec1 = min(int(grid_w) - 1, c1 + 1)
    cells = []
    for r in range(er0, er1 + 1):
        for c in range(ec0, ec1 + 1):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            cells.append((int(r), int(c)))
    return cells


def _place_frames(
    grid_h: int,
    left_w: int,
    patches: list[np.ndarray],
    want_edge_touch: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    blocked = np.zeros((grid_h, left_w), dtype=bool)
    placements: list[tuple[int, int, np.ndarray]] = []
    edge_used = False

    order = sorted(patches, key=lambda p: int(p.shape[0] * p.shape[1]), reverse=True)
    for patch in order:
        ph, pw = map(int, patch.shape)
        candidates: list[tuple[int, int, bool]] = []
        for top in range(grid_h - ph + 1):
            for left in range(left_w - pw + 1):
                rr0 = max(0, top - 1)
                cc0 = max(0, left - 1)
                rr1 = min(grid_h - 1, top + ph)
                cc1 = min(left_w - 1, left + pw)
                if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue
                touches_edge = top == 0 or left == 0 or top + ph == grid_h or left + pw == left_w
                candidates.append((int(top), int(left), bool(touches_edge)))
        if not candidates:
            return None

        preferred = candidates
        if want_edge_touch and not edge_used:
            edge_candidates = [cand for cand in candidates if cand[2]]
            if edge_candidates:
                preferred = edge_candidates
        top, left, touches_edge = random.choice(preferred)
        rr0 = max(0, top - 1)
        cc0 = max(0, left - 1)
        rr1 = min(grid_h - 1, top + ph)
        cc1 = min(left_w - 1, left + pw)
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        placements.append((int(top), int(left), patch))
        edge_used = edge_used or bool(touches_edge)

    if want_edge_touch and not edge_used:
        return None
    return placements


def _build_candidate() -> np.ndarray | None:
    bg = random.randint(0, 9)
    frame_color_choices = [c for c in range(10) if c != bg]
    frame_color = random.choice(frame_color_choices)
    remaining = [c for c in range(10) if c not in {bg, frame_color}]
    sep_color = random.choice(remaining)
    remaining.remove(sep_color)

    palette_len = random.randint(2, min(5, len(remaining)))
    palette_colors = random.sample(remaining, palette_len)

    frame_count = random.randint(2, 4)
    hole_counts = [random.randint(1, palette_len) for _ in range(frame_count)]
    if len(set(hole_counts)) == 1:
        hole_counts[0] = 1
        hole_counts[-1] = palette_len

    patches = [
        _frame_patch(hole_count, random.choice(["vertical", "horizontal"]), frame_color, bg)
        for hole_count in hole_counts
    ]

    max_patch_h = max(int(p.shape[0]) for p in patches)
    max_patch_w = max(int(p.shape[1]) for p in patches)
    right_w = random.randint(3, 5)

    palette_rows = sorted(random.sample(range(max(palette_len + 1, 6), 30), palette_len))
    min_grid_h = max(max_patch_h + 1, palette_rows[-1] + 1, 10)
    if min_grid_h > 30:
        return None
    grid_h = random.randint(min_grid_h, 30)

    min_left_w = max(max_patch_w, 8)
    max_left_w = 30 - 1 - right_w
    if min_left_w > max_left_w:
        return None
    left_w = random.randint(min_left_w, max_left_w)
    placements = _place_frames(grid_h, left_w, patches, want_edge_touch=(random.random() < 0.5))
    if placements is None:
        return None

    grid = np.full((grid_h, left_w + 1 + right_w), bg, dtype=int)
    for top, left, patch in placements:
        ph, pw = patch.shape
        window = grid[top : top + ph, left : left + pw]
        window[patch == int(frame_color)] = int(frame_color)

    sep_col = left_w
    grid[:, sep_col] = int(sep_color)

    palette_c = sep_col + 1 + random.randint(0, right_w - 1)
    for row, color in zip(palette_rows, palette_colors, strict=False):
        if row >= grid_h:
            return None
        grid[row, palette_c] = int(color)

    # Guard against accidental extra separator-like columns.
    uniform_non_bg_cols = [
        c
        for c in range(grid.shape[1])
        if int(grid[0, c]) != int(bg) and np.all(grid[:, c] == grid[0, c])
    ]
    if uniform_non_bg_cols != [sep_col]:
        return None

    # Every frame should visibly change at least one ring cell.
    for top, left, patch in placements:
        ring = _ring_cells_for_bbox(top, left, int(patch.shape[0]), int(patch.shape[1]), grid_h, left_w)
        if not any(int(grid[r, c]) == int(bg) for r, c in ring):
            return None

    # Hole counts must stay inside the palette range and show at least two cases.
    observed_holes = []
    for top, left, patch in placements:
        ph, pw = patch.shape
        comp_grid = grid[:, :left_w]
        coords = [(top + r, left + c) for r, c in np.argwhere(patch == int(frame_color))]
        observed_holes.append(_count_holes_in_component(comp_grid, coords, frame_color))
    if min(observed_holes) < 1 or max(observed_holes) > palette_len:
        return None
    if len(set(observed_holes)) < 2:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
