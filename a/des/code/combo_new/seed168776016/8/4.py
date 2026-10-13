from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
BG = 0
RAIL = 5
REGION_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_straight_line(coords: list[tuple[int, int]]) -> bool:
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1:
        ordered = sorted(c for _, c in coords)
        return ordered == list(range(ordered[0], ordered[-1] + 1))
    if len(cols) == 1:
        ordered = sorted(r for r, _ in coords)
        return ordered == list(range(ordered[0], ordered[-1] + 1))
    return False


def _find_rail_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[tuple[int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        comps = _components(grid == color)
        if comps and all(_is_straight_line(comp) for comp in comps):
            total = sum(len(comp) for comp in comps)
            candidates.append((int(total), len(comps), int(color)))
    assert candidates, "expected a straight-line rail color"
    candidates.sort()
    return candidates[-1][2]


def _enclosed_rel_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    empty = ~mask
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = [(int(r), int(c)) for r in range(h) for c in range(w) if empty[r, c] and not seen[r, c]]
    return holes


def _adjacent_rail_info(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    rail_labels: np.ndarray,
    rail_components: list[list[tuple[int, int]]],
    rail_color: int,
) -> tuple[str, int]:
    touches = {"up": [], "down": [], "left": [], "right": []}
    comp_ids: set[int] = set()
    h, w = grid.shape

    for r, c in coords:
        for side, (dr, dc) in SIDE_DIRS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == int(rail_color):
                touches[side].append((int(nr), int(nc)))
                comp_ids.add(int(rail_labels[nr, nc]))

    active_sides = [side for side, pts in touches.items() if pts]
    assert len(comp_ids) == 1, "expected one adjacent rail"
    assert len(active_sides) == 1, "expected rail contact on one side"

    rail_coords = rail_components[next(iter(comp_ids))]
    side = active_sides[0]
    rail_rows = sorted(r for r, _ in rail_coords)
    rail_cols = sorted(c for _, c in rail_coords)

    if len(set(rail_rows)) == 1:
        assert side in {"up", "down"}, "horizontal rail must be above or below the region"
        contact_cols = sorted({c for _, c in touches[side]})
        left_ext = contact_cols[0] - rail_cols[0]
        right_ext = rail_cols[-1] - contact_cols[-1]
        assert left_ext != right_ext, "rail extension must choose a direction"
        return "h", 1 if right_ext > left_ext else -1

    assert len(set(rail_cols)) == 1, "rail must be a straight segment"
    assert side in {"left", "right"}, "vertical rail must be left or right of the region"
    contact_rows = sorted({r for r, _ in touches[side]})
    up_ext = contact_rows[0] - rail_rows[0]
    down_ext = rail_rows[-1] - contact_rows[-1]
    assert up_ext != down_ext, "rail extension must choose a direction"
    return "v", 1 if down_ext > up_ext else -1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each colored region along its adjacent grey rail, repeating it by the number of enclosed black hole pixels.

    Input structure:
    - The background is black.
    - Several distinct-colored 4-connected filled regions sit on the grid; some regions contain 1-3 enclosed black hole pixels.
    - Each region touches exactly one grey straight rail segment on one cardinal side.

    Rule:
    - For each colored region, count the black pixels inside its bbox that are enclosed by the region.
    - Find the unique adjacent grey rail and read its orientation from the rail geometry.
    - Copy the whole region shape along the rail axis, moving toward the side where the rail extends away from the contact span.
    - The translation step is the region bbox size along that rail axis plus one, and the number of copies equals the enclosed-hole count.
    - The original region and the grey rail stay unchanged, and copied hole positions stay black.

    Key insight:
    - The hole count acts as the repetition counter, while the grey rail gives both the movement axis and the travel direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    rail_color = _find_rail_color(g, bg)

    rail_components = _components(g == rail_color)
    rail_labels = np.full((h, w), -1, dtype=int)
    for idx, comp in enumerate(rail_components):
        for r, c in comp:
            rail_labels[r, c] = idx

    out = g.copy()
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {int(bg), int(rail_color)}):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            rel_pixels: list[tuple[int, int]] = []
            for r, c in comp:
                rr, cc = r - r0, c - c0
                mask[rr, cc] = True
                rel_pixels.append((int(rr), int(cc)))

            hole_rel = _enclosed_rel_cells(mask)
            hole_count = len(hole_rel)
            if hole_count == 0:
                continue

            axis, sign = _adjacent_rail_info(g, comp, rail_labels, rail_components, rail_color)
            step = mask.shape[1] + 1 if axis == "h" else mask.shape[0] + 1

            for k in range(1, hole_count + 1):
                dr = sign * k * step if axis == "v" else 0
                dc = sign * k * step if axis == "h" else 0
                for rr, cc in hole_rel:
                    tr, tc = r0 + rr + dr, c0 + cc + dc
                    if 0 <= tr < h and 0 <= tc < w:
                        out[tr, tc] = bg
                for rr, cc in rel_pixels:
                    tr, tc = r0 + rr + dr, c0 + cc + dc
                    if 0 <= tr < h and 0 <= tc < w:
                        out[tr, tc] = color

    return out


H0_TEMPLATES = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
]


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=bool)


def _flush_sides(mask: np.ndarray) -> list[str]:
    sides: list[str] = []
    if bool(mask[0, :].all()):
        sides.append("up")
    if bool(mask[-1, :].all()):
        sides.append("down")
    if bool(mask[:, 0].all()):
        sides.append("left")
    if bool(mask[:, -1].all()):
        sides.append("right")
    return sides


def _hole_mask(hole_count: int) -> np.ndarray:
    width = hole_count + 2
    mask = np.ones((3, width), dtype=bool)
    mask[1, 1:-1] = False
    if random.random() < 0.7:
        corners = [(0, 0), (0, width - 1), (2, 0), (2, width - 1)]
        rr, cc = random.choice(corners)
        mask[rr, cc] = False
    return mask


def _sample_region_mask(target_holes: int) -> np.ndarray:
    for _ in range(100):
        base = random.choice(H0_TEMPLATES) if target_holes == 0 else _hole_mask(target_holes)
        mask = _random_transform(base)
        if len(_enclosed_rel_cells(mask)) != target_holes:
            continue
        if _flush_sides(mask):
            return mask
    raise RuntimeError("failed to sample a region mask")


def _build_group(mask: np.ndarray, color: int, rail_side: str, move_sign: int) -> np.ndarray:
    holes = _enclosed_rel_cells(mask)
    hole_count = len(holes)
    h, w = mask.shape
    axis = "h" if rail_side in {"up", "down"} else "v"
    span = w if axis == "h" else h
    step = span + 1
    rail_len = random.randint(2 * span, 3 * span)

    if axis == "h":
        total_h = h + 1
        total_w = max(w + hole_count * step, rail_len)
        obj_left = 0 if move_sign > 0 else total_w - w
        obj_top = 1 if rail_side == "up" else 0
        rail_row = 0 if rail_side == "up" else h
        rail_left = obj_left if move_sign > 0 else obj_left + w - rail_len

        grid = np.full((total_h, total_w), BG, dtype=int)
        grid[rail_row, rail_left : rail_left + rail_len] = RAIL
        for rr in range(h):
            for cc in range(w):
                if mask[rr, cc]:
                    grid[obj_top + rr, obj_left + cc] = color
    else:
        total_h = max(h + hole_count * step, rail_len)
        total_w = w + 1
        obj_top = 0 if move_sign > 0 else total_h - h
        obj_left = 1 if rail_side == "left" else 0
        rail_col = 0 if rail_side == "left" else w
        rail_top = obj_top if move_sign > 0 else obj_top + h - rail_len

        grid = np.full((total_h, total_w), BG, dtype=int)
        grid[rail_top : rail_top + rail_len, rail_col] = RAIL
        for rr in range(h):
            for cc in range(w):
                if mask[rr, cc]:
                    grid[obj_top + rr, obj_left + cc] = color

    return grid


def _sample_hole_counts(n_groups: int) -> list[int]:
    if n_groups == 4:
        counts = [random.choice([0, 1, 2]) for _ in range(n_groups)]
    else:
        counts = [random.choice([0, 1, 2, 3]) for _ in range(n_groups)]
    if all(v == 0 for v in counts):
        counts[random.randrange(n_groups)] = random.choice([1, 2, 3])
    if all(v > 0 for v in counts) and random.random() < 0.7:
        counts[random.randrange(n_groups)] = 0
    return counts


def _pack_groups(groups: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.full((30, 30), BG, dtype=int)
    reserved = np.zeros((30, 30), dtype=bool)

    for group in sorted(groups, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        gh, gw = group.shape
        placed = False
        for _ in range(300):
            top = random.randint(0, 30 - gh)
            left = random.randint(0, 30 - gw)
            r0 = max(0, top - 1)
            r1 = min(30, top + gh + 1)
            c0 = max(0, left - 1)
            c1 = min(30, left + gw + 1)
            if reserved[r0:r1, c0:c1].any():
                continue
            patch = group != BG
            canvas[top : top + gh, left : left + gw][patch] = group[patch]
            reserved[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None

    coords = np.argwhere(canvas != BG)
    if coords.size == 0:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_margin = random.randint(0, min(int(r0), 3))
    left_margin = random.randint(0, min(int(c0), 3))
    bottom_margin = random.randint(0, min(int(29 - r1), 3))
    right_margin = random.randint(0, min(int(29 - c1), 3))
    return canvas[
        int(r0) - top_margin : int(r1) + bottom_margin + 1,
        int(c0) - left_margin : int(c1) + right_margin + 1,
    ].copy()


def _gen_attempt() -> np.ndarray | None:
    n_groups = random.choices([2, 3, 4], weights=[4, 4, 1])[0]
    hole_counts = _sample_hole_counts(n_groups)
    colors = random.sample(REGION_COLORS, n_groups)
    groups: list[np.ndarray] = []

    for color, target_holes in zip(colors, hole_counts):
        mask = _sample_region_mask(target_holes)
        sides = _flush_sides(mask)
        if not sides:
            return None
        rail_side = random.choice(sides)
        move_sign = random.choice([-1, 1])
        groups.append(_build_group(mask, color, rail_side, move_sign))

    grid = _pack_groups(groups)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if grid.ndim != 2 or out.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
