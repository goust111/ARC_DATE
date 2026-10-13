from __future__ import annotations

import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _rot90_orbit(side: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = side - 1
    return (
        (r, c),
        (c, last - r),
        (last - r, last - c),
        (last - c, r),
    )


def _rotation_closure(
    panel: np.ndarray,
    bg: int,
    side: int,
    r0: int,
    c0: int,
    coords: np.ndarray,
) -> dict[tuple[int, int], int] | None:
    painted: dict[tuple[int, int], int] = {}
    for r, c in coords:
        if not (r0 <= int(r) < r0 + side and c0 <= int(c) < c0 + side):
            return None
        color = int(panel[r, c])
        if color == bg:
            return None
        rr, cc = int(r - r0), int(c - c0)
        for pos in _rot90_orbit(side, rr, cc):
            old = painted.get(pos)
            if old is not None and old != color:
                return None
            painted[pos] = color
    return painted


def _find_rotation_square(panel: np.ndarray, bg: int) -> tuple[int, int, dict[tuple[int, int], int]]:
    coords = np.argwhere(panel != bg)
    assert coords.size > 0, "each panel must contain foreground pixels"
    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)
    min_side = max(int(rmax - rmin + 1), int(cmax - cmin + 1))
    h, w = panel.shape
    max_side = min(h, w)

    for side in range(min_side, max_side + 1):
        candidates: list[tuple[int, int, dict[tuple[int, int], int]]] = []
        r_lo = max(0, int(rmax - side + 1))
        r_hi = min(int(rmin), h - side)
        c_lo = max(0, int(cmax - side + 1))
        c_hi = min(int(cmin), w - side)
        for r0 in range(r_lo, r_hi + 1):
            for c0 in range(c_lo, c_hi + 1):
                painted = _rotation_closure(panel, bg, side, r0, c0, coords)
                if painted is not None:
                    candidates.append((r0, c0, painted))
        if candidates:
            assert len(candidates) == 1, "expected a unique smallest rotation square"
            return candidates[0]

    raise AssertionError("panel foreground is not consistent with a unique rotation square")


def _split_panels(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    h, w = grid.shape
    sep_mask = np.zeros(w, dtype=bool)
    sep_colors: list[int] = []
    for c in range(w):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            sep_mask[c] = True
            sep_colors.append(int(col[0]))

    assert sep_colors, "expected at least one separator column"
    assert len(set(sep_colors)) == 1, "all separator columns must share one color"

    panels: list[tuple[int, int]] = []
    c = 0
    while c < w:
        while c < w and sep_mask[c]:
            c += 1
        start = c
        while c < w and not sep_mask[c]:
            c += 1
        if start < c:
            panels.append((start, c))

    assert len(panels) >= 2, "expected multiple panels"
    widths = {c1 - c0 for c0, c1 in panels}
    assert len(widths) == 1, "panels must have equal width"
    return panels, sep_mask


def _complete_panel(panel: np.ndarray, bg: int) -> tuple[np.ndarray, int]:
    r0, c0, painted = _find_rotation_square(panel, bg)
    out = panel.copy()
    added = 0
    for (rr, cc), color in painted.items():
        ar, ac = r0 + rr, c0 + cc
        if int(out[ar, ac]) == bg:
            added += 1
        out[ar, ac] = color
    return out, added


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each panel's partial motif to four-way rotational symmetry, then reorder panels by completion cost.

    Input structure:
    - The grid is split by one-color full-height separator columns into equal-width vertical panels.
    - Each panel contains one partial foreground motif on the background, confined to a unique square region.
    - That square is the unique smallest region whose 90-degree rotation closure produces the intended motif.

    Rule:
    - For each panel, complete the motif by painting every cell in each 0/90/180/270-degree rotation orbit.
    - Count how many background cells were newly added during that completion.
    - Reorder the fully completed panel blocks from left to right by that added-pixel count in ascending order, while leaving separator columns unchanged.

    Key insight:
    - The visible motifs do not expose the sorting key directly; the correct key is the number of missing orbit cells revealed only after rotational completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    panels, _sep_mask = _split_panels(g, bg)

    completed_panels: list[np.ndarray] = []
    added_counts: list[int] = []
    for c0, c1 in panels:
        completed, added = _complete_panel(g[:, c0:c1], bg)
        completed_panels.append(completed)
        added_counts.append(int(added))

    assert len(set(added_counts)) == len(added_counts), "completion costs must be unique"

    order = np.argsort(np.array(added_counts, dtype=int))
    out = g.copy()
    for dst_idx, src_idx in enumerate(order):
        c0, c1 = panels[dst_idx]
        out[:, c0:c1] = completed_panels[int(src_idx)]
    return out


def _all_orbits(side: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(side):
        for c in range(side):
            cells = tuple(sorted(set(_rot90_orbit(side, r, c))))
            if cells not in seen:
                seen.add(cells)
                orbits.append(cells)
    return orbits


def _anchor_kept_cells(side: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    corners = ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1))
    options = [
        (corners[0], corners[3]),
        (corners[1], corners[2]),
        (corners[1], corners[2], corners[3]),
        (corners[0], corners[2], corners[3]),
        (corners[0], corners[1], corners[3]),
        (corners[0], corners[1], corners[2]),
    ]
    return tuple(rng.choice(options))


def _make_panel_candidate(
    panel_h: int,
    panel_w: int,
    bg: int,
    color: int,
    rng: random.Random,
) -> dict | None:
    max_side = min(panel_h, panel_w) - 2
    if max_side < 4:
        return None

    for _ in range(200):
        side = rng.randint(4, max_side)
        orbits = _all_orbits(side)
        corner_orbit = tuple(sorted(((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1))))
        extras = [orbit for orbit in orbits if orbit != corner_orbit]
        rng.shuffle(extras)

        selected: list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = []
        selected.append((corner_orbit, _anchor_kept_cells(side, rng)))

        extra_count = rng.randint(1, min(4, len(extras)))
        for orbit in extras[:extra_count]:
            if len(orbit) == 1:
                kept = orbit
            else:
                keep_n = rng.choices([1, 2, 3, 4], weights=[3, 4, 3, 1], k=1)[0]
                kept = tuple(rng.sample(list(orbit), keep_n))
            selected.append((orbit, kept))

        raw = sum(len(kept) for _orbit, kept in selected)
        total = sum(len(orbit) for orbit, _kept in selected)
        added = total - raw
        area = side * side

        if added <= 0:
            continue
        if raw < 4 or total < 8:
            continue
        if total > max(8, int(area * 0.7)):
            continue

        panel = np.full((panel_h, panel_w), bg, dtype=int)
        completed = np.full((panel_h, panel_w), bg, dtype=int)
        top = (panel_h - side) // 2
        left = (panel_w - side) // 2

        for orbit, kept in selected:
            for r, c in orbit:
                completed[top + r, left + c] = color
            for r, c in kept:
                panel[top + r, left + c] = color

        return {
            "panel": panel,
            "completed": completed,
            "raw": raw,
            "total": total,
            "added": added,
        }
    return None


def _metrics_are_good(candidates: list[dict]) -> bool:
    raw = [int(c["raw"]) for c in candidates]
    total = [int(c["total"]) for c in candidates]
    added = [int(c["added"]) for c in candidates]
    n = len(candidates)

    if len(set(raw)) != n or len(set(total)) != n or len(set(added)) != n:
        return False

    add_order = tuple(int(i) for i in np.argsort(np.array(added, dtype=int)))
    raw_order = tuple(int(i) for i in np.argsort(np.array(raw, dtype=int)))
    total_order = tuple(int(i) for i in np.argsort(np.array(total, dtype=int)))
    identity = tuple(range(n))

    return add_order != identity and add_order != raw_order and add_order != total_order


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        n_panels = rng.choice([3, 4])
        max_panel_w = (30 - (n_panels - 1)) // n_panels
        panel_w_hi = min(9, max_panel_w)
        if panel_w_hi < 6:
            continue
        panel_w = rng.randint(6, panel_w_hi)
        panel_h = rng.randint(8, 12)

        bg = 0
        sep_color = rng.randint(1, 9)
        palette = [c for c in range(1, 10) if c != sep_color]
        panel_colors = rng.sample(palette, n_panels)

        candidates: list[dict] = []
        ok = True
        for color in panel_colors:
            candidate = _make_panel_candidate(panel_h, panel_w, bg, color, rng)
            if candidate is None:
                ok = False
                break
            candidates.append(candidate)
        if not ok or not _metrics_are_good(candidates):
            continue

        width = n_panels * panel_w + (n_panels - 1)
        grid = np.full((panel_h, width), bg, dtype=int)
        for idx, candidate in enumerate(candidates):
            c0 = idx * (panel_w + 1)
            grid[:, c0 : c0 + panel_w] = candidate["panel"]
            if idx < n_panels - 1:
                grid[:, c0 + panel_w] = sep_color

        out = main(grid)
        assert out.shape == grid.shape
        return grid

    raise RuntimeError("failed to generate a valid panel-sorting rotational-symmetry input")
