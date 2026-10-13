from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
INPUT_COLORS = (1, 2, 3, 6, 7, 8, 9)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_variants(mask: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return unique rotations/reflections of a tight boolean mask."""
    base = _trim_mask(mask)
    variants: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(base, k)
        for flipped in (False, True):
            m = np.fliplr(rot) if flipped else rot
            m = _trim_mask(m)
            key = (tuple(map(int, m.shape)), m.tobytes())
            if key in seen:
                continue
            seen.add(key)
            name = f"r{k}" + ("f" if flipped else "")
            variants.append((name, m.copy()))
    return variants


def _canvas_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = max(int(a.shape[0]), int(b.shape[0]))
    w = max(int(a.shape[1]), int(b.shape[1]))
    aa = np.zeros((h, w), dtype=bool)
    bb = np.zeros((h, w), dtype=bool)
    aa[: a.shape[0], : a.shape[1]] = a
    bb[: b.shape[0], : b.shape[1]] = b
    return aa, bb


def _best_difference(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, int, int, str]:
    """Align moving to reference by the dihedral transform with maximum overlap."""
    ref = _trim_mask(reference)
    best: tuple[int, int, int, str, np.ndarray] | None = None
    tie_count = 0

    for name, variant in _mask_variants(moving):
        a, b = _canvas_pair(ref, variant)
        overlap = int((a & b).sum())
        diff = int((a ^ b).sum())
        area = int(a.size)
        score = (overlap, -diff, -area)
        if best is None or score > best[:3]:
            best = (score[0], score[1], score[2], name, a ^ b)
            tie_count = 1
        elif best is not None and score == best[:3]:
            tie_count += 1

    assert best is not None
    diff_mask = best[4]
    return diff_mask, int(diff_mask.sum()), int(tie_count), best[3]


def _split_runs(size: int, separators: list[int]) -> list[tuple[int, int]]:
    runs = []
    start = 0
    for sep in separators:
        if start < sep:
            runs.append((start, sep))
        start = sep + 1
    if start < size:
        runs.append((start, size))
    return runs


def _panel_slices(g: np.ndarray) -> list[tuple[slice, slice]]:
    full_grey_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    full_grey_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    row_runs = _split_runs(int(g.shape[0]), full_grey_rows)
    col_runs = _split_runs(int(g.shape[1]), full_grey_cols)

    slices = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            panel = g[r0:r1, c0:c1]
            if np.any((panel != BLACK) & (panel != GREY)):
                slices.append((slice(r0, r1), slice(c0, c1)))
    return slices


def _object_masks(panel: np.ndarray) -> list[dict]:
    colors = []
    for r, c in np.argwhere((panel != BLACK) & (panel != GREY)):
        color = int(panel[int(r), int(c)])
        if color not in colors:
            colors.append(color)
    assert len(colors) == 2, "each panel must contain exactly two object colors"

    objects = []
    for color in colors:
        mask = panel == int(color)
        pts = np.argwhere(mask)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        objects.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
            }
        )
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return objects


def _solve_panel(panel: np.ndarray) -> np.ndarray:
    objects = _object_masks(panel)
    diff, _count, _ties, _name = _best_difference(objects[0]["mask"], objects[1]["mask"])
    out = np.full(diff.shape, BLACK, dtype=int)
    out[diff] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the shape differences between each pair of near-matching objects.

    Input structure:
    - Grey full rows or columns split the grid into black-background panels.
    - Each non-empty panel contains exactly two foreground objects of different colors.
    - The two objects are nearly the same binary shape after a rotation or reflection, with only a few pixels changed.

    Rule:
    - For every panel, crop each object to its tight bounding box and use the row-major earlier object as the reference orientation.
    - Try every rotation and horizontal reflection of the second object, top-left align the two normalized masks, and keep the transform with the largest overlap.
    - Emit a black crop for that panel with yellow pixels exactly where one aligned mask has a cell and the other does not.
    - Concatenate the panel crops in their top-to-bottom, left-to-right panel order, padding shorter crops with black at the bottom if needed.

    Key insight:
    - The objects are not merged; the shared shape is ignored after geometric alignment, leaving only the symmetric-difference pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panels = [_solve_panel(g[rs, cs]) for rs, cs in _panel_slices(g)]
    assert panels, "expected at least one non-empty panel"

    max_h = max(int(p.shape[0]) for p in panels)
    total_w = sum(int(p.shape[1]) for p in panels)
    out = np.full((max_h, total_w), BLACK, dtype=int)
    c = 0
    for panel_out in panels:
        h, w = map(int, panel_out.shape)
        out[:h, c : c + w] = panel_out
        c += w
    return out


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    seen = np.zeros(mask.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    sr, sc = map(int, pts[0])
    q.append((sr, sc))
    seen[sr, sc] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return int(seen.sum()) == int(mask.sum())


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _grow_connected_mask(h: int, w: int) -> np.ndarray:
    for _ in range(250):
        targets = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        random.shuffle(targets)
        mask = np.zeros((h, w), dtype=bool)
        r, c = targets[0]
        mask[r, c] = True
        for tr, tc in targets[1:]:
            while (r, c) != (tr, tc):
                steps = []
                if r < tr:
                    steps.append((1, 0))
                elif r > tr:
                    steps.append((-1, 0))
                if c < tc:
                    steps.append((0, 1))
                elif c > tc:
                    steps.append((0, -1))
                dr, dc = random.choice(steps)
                r += dr
                c += dc
                mask[r, c] = True

        max_area = h * w - 2
        if int(mask.sum()) > max_area:
            continue
        target_area = random.randint(int(mask.sum()), max_area)
        while int(mask.sum()) < target_area:
            frontier = []
            for rr, cc in np.argwhere(mask):
                rr, cc = int(rr), int(cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True

        if (
            4 <= int(mask.sum()) <= h * w - 2
            and _connected(mask)
            and _touches_all_edges(mask)
            and len(_mask_variants(mask)) >= 6
        ):
            return mask
    raise RuntimeError("failed to grow a connected mask")


def _mutate_mask(mask: np.ndarray) -> np.ndarray | None:
    h, w = map(int, mask.shape)
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(500):
        diff_count = random.randint(1, 3)
        mutated = mask.copy()
        for r, c in random.sample(cells, diff_count):
            mutated[r, c] = not bool(mutated[r, c])
        if np.array_equal(mutated, mask):
            continue
        if not (3 <= int(mutated.sum()) <= h * w - 1):
            continue
        if not _connected(mutated):
            continue
        if _trim_mask(mutated).shape != mask.shape:
            continue
        return mutated
    return None


def _make_pair() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(300):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        if h == 3 and w == 3:
            w = 4
        try:
            reference = _grow_connected_mask(h, w)
        except RuntimeError:
            continue

        variants = _mask_variants(reference)
        _name, clean_moving = random.choice(variants)
        moving = _mutate_mask(clean_moving)
        if moving is None:
            continue

        _diff, count, ties, _best_name = _best_difference(reference, moving)
        if 1 <= count <= 3 and ties == 1:
            return reference, moving
    return None


def _paint_mask(panel: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = map(int, mask.shape)
    region = panel[r0 : r0 + h, c0 : c0 + w]
    region[mask] = int(color)


def _make_panel(cell_h: int, cell_w: int) -> np.ndarray | None:
    pair = _make_pair()
    if pair is None:
        return None
    reference, moving = pair
    rh, rw = map(int, reference.shape)
    mh, mw = map(int, moving.shape)

    color_a, color_b = random.sample(INPUT_COLORS, 2)
    arrangements = ["horizontal", "vertical"]
    random.shuffle(arrangements)

    for arrangement in arrangements:
        if arrangement == "horizontal":
            min_gap = 1
            slack = cell_w - rw - mw - min_gap
            if slack < 0 or max(rh, mh) > cell_h:
                continue
            left_pad = random.randint(0, slack)
            gap = random.randint(min_gap, min_gap + slack - left_pad)
            c0 = left_pad
            c1 = c0 + rw + gap
            r0 = random.randint(0, cell_h - rh)
            r1 = random.randint(0, cell_h - mh)
        else:
            min_gap = 1
            slack = cell_h - rh - mh - min_gap
            if slack < 0 or max(rw, mw) > cell_w:
                continue
            top_pad = random.randint(0, slack)
            gap = random.randint(min_gap, min_gap + slack - top_pad)
            r0 = top_pad
            r1 = r0 + rh + gap
            c0 = random.randint(0, cell_w - rw)
            c1 = random.randint(0, cell_w - mw)

        panel = np.full((cell_h, cell_w), BLACK, dtype=int)
        _paint_mask(panel, reference, r0, c0, color_a)
        _paint_mask(panel, moving, r1, c1, color_b)
        return panel
    return None


def _layout_shape() -> tuple[int, int, int, int]:
    rows, cols = random.choice([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)])
    max_cell_h = (30 - (rows - 1)) // rows
    max_cell_w = (30 - (cols - 1)) // cols
    cell_h = random.randint(8, min(13, max_cell_h))
    cell_w = random.randint(8, min(13, max_cell_w))
    return rows, cols, cell_h, cell_w


def _gen_attempt() -> np.ndarray | None:
    rows, cols, cell_h, cell_w = _layout_shape()
    panels = []
    for _ in range(rows * cols):
        panel = _make_panel(cell_h, cell_w)
        if panel is None:
            return None
        panels.append(panel)

    height = rows * cell_h + (rows - 1)
    width = cols * cell_w + (cols - 1)
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    if rows > 1:
        for r in range(cell_h, height, cell_h + 1):
            grid[r, :] = GREY
    if cols > 1:
        for c in range(cell_w, width, cell_w + 1):
            grid[:, c] = GREY

    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            r0 = rr * (cell_h + 1)
            c0 = cc * (cell_w + 1)
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = panels[idx]
            idx += 1

    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if not np.any(out == YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(800):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shape-difference panel input")
