from __future__ import annotations

import random
from collections import Counter

import numpy as np


SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    out: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        out.add((r0, c))
        out.add((r1, c))
    for r in range(r0 + 1, r1):
        out.add((r, c0))
        out.add((r, c1))
    return out


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _ring_candidates(grid: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int], set[tuple[int, int]]]]:
    candidates: list[tuple[int, tuple[int, int, int, int], set[tuple[int, int]]]] = []
    for color in sorted(set(map(int, grid.flat)) - {int(bg)}):
        for comp in _components_of_color(grid, color):
            box = _bbox(comp)
            ring = _border_cells(box)
            h = box[2] - box[0] + 1
            w = box[3] - box[1] + 1
            if h >= 4 and w >= 4 and set(comp) == ring:
                candidates.append((int(color), box, ring))
    return candidates


def _valid_centers(grid: np.ndarray, mask: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    visible_fg = np.argwhere((grid != int(bg)) & (~mask))
    hidden = np.argwhere(mask)
    centers: list[tuple[int, int]] = []

    for sr in range(2 * h - 1):
        for sc in range(2 * w - 1):
            ok = True

            for r, c in visible_fg:
                rr = sr - int(r)
                cc = sc - int(c)
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                if mask[rr, cc]:
                    continue
                if int(grid[rr, cc]) == int(bg) or int(grid[rr, cc]) != int(grid[r, c]):
                    ok = False
                    break
            if not ok:
                continue

            for r, c in hidden:
                rr = sr - int(r)
                cc = sc - int(c)
                if not (0 <= rr < h and 0 <= cc < w) or mask[rr, cc]:
                    ok = False
                    break
            if ok:
                centers.append((sr, sc))

    return centers


def _restore_from_center(grid: np.ndarray, mask: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    sr, sc = center
    for r, c in np.argwhere(mask):
        out[int(r), int(c)] = int(grid[sr - int(r), sc - int(c)])
    return out


def _foreground_point_symmetric(grid: np.ndarray, center: tuple[int, int], bg: int) -> bool:
    h, w = grid.shape
    sr, sc = center
    for r, c in np.argwhere(grid != int(bg)):
        rr = sr - int(r)
        cc = sc - int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if int(grid[rr, cc]) != int(grid[r, c]):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the rectangular ring occluder and restore its cells from off-center 180-degree symmetry.

    Input structure:
    - A sparse multicolor mosaic sits on a dominant background and is 180-degree rotationally symmetric
      around one off-center point.
    - A one-cell-thick hollow rectangle ring is overlaid on top of the mosaic; all of its visible cells
      share one color, but that same color can also appear elsewhere as genuine mosaic pixels.
    - The occluded cells' rotational counterparts are still visible, and the visible local arrangement near
      the ring can look almost left-right mirrored except for at least one contradiction.

    Rule:
    - Identify the occluder by its complete rectangular-ring shape, not by color alone.
    - Remove that ring from the evidence set, then find the unique rotation center that makes the remaining
      foreground pixels self-consistent under a 180-degree turn.
    - Recolor every ring cell with the value at its rotational counterpart, making the ring disappear.

    Key insight:
    The decisive cue is the full hollow-rectangle silhouette; once those pixels are discounted, the true
    symmetry is a unique off-center point reflection rather than the tempting local mirror pattern.
    """
    g = np.array(grid, dtype=int, copy=False)
    bg = _get_bg(g)

    valid_outputs: list[np.ndarray] = []
    for _color, _box, ring in _ring_candidates(g, bg):
        mask = np.zeros(g.shape, dtype=bool)
        for r, c in ring:
            mask[r, c] = True
        centers = _valid_centers(g, mask, bg)
        if len(centers) != 1:
            continue
        restored = _restore_from_center(g, mask, centers[0])
        if _foreground_point_symmetric(restored, centers[0], bg):
            valid_outputs.append(restored)

    assert len(valid_outputs) == 1, f"expected exactly one valid occluder, got {len(valid_outputs)}"
    return valid_outputs[0]


def _mate(cell: tuple[int, int], center: tuple[int, int]) -> tuple[int, int]:
    sr, sc = center
    r, c = cell
    return sr - r, sc - c


def _make_generator(
    h: int,
    w: int,
    bg: int,
    center: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    frame_color: int,
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    grid = np.full((h, w), bg, dtype=int)
    ring = _border_cells(frame_box)
    ring_adjacent: set[tuple[int, int]] = set()
    for r, c in ring:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in ring:
                ring_adjacent.add((nr, nc))
    return grid, ring, ring_adjacent


def _place_orbit(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
    center: tuple[int, int],
    ring: set[tuple[int, int]],
    ring_adjacent: set[tuple[int, int]],
    frame_color: int,
    occupied: set[tuple[int, int]],
    *,
    allow_on_ring: bool = False,
) -> bool:
    h, w = grid.shape
    pts: list[tuple[int, int]] = []
    for cell in cells:
        for p in (cell, _mate(cell, center)):
            r, c = p
            if not (0 <= r < h and 0 <= c < w):
                return False
            if (not allow_on_ring) and p in ring:
                return False
            if int(color) == int(frame_color) and p not in ring and p in ring_adjacent:
                return False
            pts.append(p)
    if len(set(pts)) != len(pts):
        return False
    for p in pts:
        if p in occupied and int(grid[p]) != int(color):
            return False
    for p in pts:
        grid[p] = int(color)
        occupied.add(p)
    return True


def _sample_frame_and_center(h: int, w: int) -> tuple[tuple[int, int], tuple[int, int, int, int]] | None:
    for _ in range(200):
        frame_h = random.randint(5, min(8, h - 6))
        frame_w = random.randint(6, min(9, w - 5))
        c0 = random.randint(2, w - frame_w - 3)
        c1 = c0 + frame_w - 1
        sc = c0 + c1

        sr_choices = []
        for sr in range(3, 2 * h - 4):
            r1_max = min(h - 4, (sr - 2) // 2)
            if r1_max < frame_h - 1:
                continue
            r1_min = max(frame_h - 1, sr - (h - 1))
            if r1_min > r1_max:
                continue
            if sr == h - 1 and sc == w - 1:
                continue
            sr_choices.append(sr)
        if not sr_choices:
            continue

        sr = random.choice(sr_choices)
        r1_max = min(h - 4, (sr - 2) // 2)
        r1_min = max(frame_h - 1, sr - (h - 1))
        if r1_min > r1_max:
            continue
        r1 = random.randint(r1_min, r1_max)
        r0 = r1 - frame_h + 1
        frame_box = (r0, c0, r1, c1)
        ring = _border_cells(frame_box)
        rotated = {_mate(p, (sr, sc)) for p in ring}
        if any(not (0 <= r < h and 0 <= c < w) for r, c in rotated):
            continue
        if rotated & ring:
            continue
        return (sr, sc), frame_box
    return None


def _validate_generated(
    inp: np.ndarray,
    out: np.ndarray,
    bg: int,
    center: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    frame_color: int,
) -> bool:
    ring = _border_cells(frame_box)
    if not _foreground_point_symmetric(out, center, bg):
        return False
    if not np.array_equal(main(inp), out):
        return False

    mask = np.zeros(inp.shape, dtype=bool)
    for r, c in ring:
        mask[r, c] = True
    if _valid_centers(inp, mask, bg) != [center]:
        return False

    candidates = _ring_candidates(inp, bg)
    candidate_hits = 0
    for _color, _box, cand_ring in candidates:
        cand_mask = np.zeros(inp.shape, dtype=bool)
        for r, c in cand_ring:
            cand_mask[r, c] = True
        if len(_valid_centers(inp, cand_mask, bg)) == 1:
            candidate_hits += 1
    if candidate_hits != 1:
        return False

    changed = sum(int(out[p]) != int(frame_color) for p in ring)
    if changed < max(6, len(ring) // 3):
        return False

    visible_frame_reuse = any(
        int(inp[r, c]) == int(frame_color) and (r, c) not in ring for r, c in np.argwhere(inp == frame_color)
    )
    if not visible_frame_reuse:
        return False

    r0, c0, r1, c1 = frame_box
    matches = 0
    mismatches = 0
    sc = center[1]
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            cm = sc - c
            if not (c0 + 1 <= cm <= c1 - 1):
                continue
            if c >= cm:
                continue
            a = int(inp[r, c])
            b = int(inp[r, cm])
            if a == int(bg) or b == int(bg):
                continue
            if a == b:
                matches += 1
            else:
                mismatches += 1
    if matches < 3 or mismatches < 1:
        return False

    if len(set(map(int, inp.flat)) - {int(bg)}) < 3:
        return False
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    bg = random.randint(0, 9)

    sampled = _sample_frame_and_center(h, w)
    if sampled is None:
        return None
    center, frame_box = sampled

    palette = random.sample([c for c in range(10) if c != bg], random.randint(4, 5))
    frame_color = random.choice(palette)
    grid, ring, ring_adjacent = _make_generator(h, w, bg, center, frame_box, frame_color)
    occupied: set[tuple[int, int]] = set()

    ring_cells = list(ring)
    random.shuffle(ring_cells)
    border_colors: list[int] = []
    for i, cell in enumerate(ring_cells):
        if i < 8 or random.random() < 0.85:
            color = random.choice(palette)
            if not _place_orbit(
                grid,
                [cell],
                color,
                center,
                ring,
                ring_adjacent,
                frame_color,
                occupied,
                allow_on_ring=True,
            ):
                return None
            border_colors.append(color)
    if len(set(border_colors)) < 3:
        return None

    r0, c0, r1, c1 = frame_box
    sc = center[1]
    interior_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            cm = sc - c
            if not (c0 + 1 <= cm <= c1 - 1):
                continue
            if c >= cm:
                continue
            a = (r, c)
            b = (r, cm)
            if _mate(a, center) == b or _mate(b, center) == a:
                continue
            interior_pairs.append((a, b))
    random.shuffle(interior_pairs)
    if len(interior_pairs) < 4:
        return None

    same_count = random.randint(3, min(5, len(interior_pairs) - 1))
    chosen = interior_pairs[: same_count + 1]
    mismatch_pair = chosen[-1]
    safe_colors = [c for c in palette if c != frame_color]
    if len(safe_colors) < 2:
        return None

    for a, b in chosen[:-1]:
        color = random.choice(safe_colors)
        if not _place_orbit(grid, [a], color, center, ring, ring_adjacent, frame_color, occupied):
            return None
        if not _place_orbit(grid, [b], color, center, ring, ring_adjacent, frame_color, occupied):
            return None

    left_color, right_color = random.sample(safe_colors, 2)
    if not _place_orbit(grid, [mismatch_pair[0]], left_color, center, ring, ring_adjacent, frame_color, occupied):
        return None
    if not _place_orbit(grid, [mismatch_pair[1]], right_color, center, ring, ring_adjacent, frame_color, occupied):
        return None

    extra_needed = random.randint(5, 8)
    placed_extra = 0
    tries = 0
    while placed_extra < extra_needed and tries < 500:
        tries += 1
        shape = random.choice(SHAPES)
        color = random.choice(palette)
        anchor_r = random.randint(0, h - 1)
        anchor_c = random.randint(0, w - 1)
        cells = [(anchor_r + dr, anchor_c + dc) for dr, dc in shape]
        if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
            continue
        if any(_mate(cell, center) == cell for cell in cells):
            continue
        if _place_orbit(grid, cells, color, center, ring, ring_adjacent, frame_color, occupied):
            placed_extra += 1
    if placed_extra < 5:
        return None

    visible_frame_reuse = any(
        int(grid[r, c]) == int(frame_color) and (r, c) not in ring and (r, c) not in ring_adjacent
        for r, c in occupied
    )
    if not visible_frame_reuse:
        for _ in range(150):
            cell = (random.randint(0, h - 1), random.randint(0, w - 1))
            if _place_orbit(grid, [cell], frame_color, center, ring, ring_adjacent, frame_color, occupied):
                visible_frame_reuse = True
                break
    if not visible_frame_reuse:
        return None

    out = grid.copy()
    inp = grid.copy()
    for r, c in ring:
        inp[r, c] = int(frame_color)

    if not _validate_generated(inp, out, bg, center, frame_box, frame_color):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _gen_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input grid")
