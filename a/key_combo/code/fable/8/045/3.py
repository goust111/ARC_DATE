from __future__ import annotations

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _orbit(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    mr = h - 1 - r
    mc = w - 1 - c
    coords = [(r, c), (r, mc), (mr, c), (mr, mc)]
    out: list[tuple[int, int]] = []
    seen = set()
    for item in coords:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _orbit_id(shape: tuple[int, int], r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(_orbit(shape, r, c)))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _color_components(grid: np.ndarray, *, skip_mask: np.ndarray | None = None) -> list[tuple[int, np.ndarray]]:
    comps: list[tuple[int, np.ndarray]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != 0):
        mask = grid == color
        if skip_mask is not None:
            mask &= ~skip_mask
        for comp in _components(mask):
            comps.append((color, comp))
    return comps


def _is_intruder_component(grid: np.ndarray, comp: np.ndarray, damaged: np.ndarray) -> bool:
    comp_mask = np.zeros_like(damaged, dtype=bool)
    comp_mask[comp[:, 0], comp[:, 1]] = True
    color = int(grid[comp[0, 0], comp[0, 1]])

    for r, c in comp:
        refs = []
        for rr, cc in _orbit(grid.shape, int(r), int(c)):
            if (rr == int(r) and cc == int(c)) or comp_mask[rr, cc] or damaged[rr, cc]:
                continue
            refs.append(int(grid[rr, cc]))
        if not refs:
            return False
        if len(set(refs)) != 1:
            return False
        if refs[0] == color:
            return False
    return True


def _detect_damage(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    damaged = g == 0

    changed = True
    while changed:
        changed = False
        for _color, comp in _color_components(g, skip_mask=damaged):
            if _is_intruder_component(g, comp, damaged):
                damaged[comp[:, 0], comp[:, 1]] = True
                changed = True
    return damaged


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a biaxially symmetric mosaic that was damaged by black holes and monochrome paint blobs.

    Input structure:
    - The grid is a colorful mosaic that originally had both left-right and top-bottom mirror symmetry.
    - Some rectangular regions were replaced by black holes, and some connected monochrome blobs overwrote the mosaic with a legal color.
    - A damaged cell can have its horizontal mirror or vertical mirror damaged too, while the 180-degree counterpart stays intact.

    Rule:
    - Treat every black cell as damaged.
    - Detect each paint blob as a whole monochrome connected region whose cells disagree with their mirror orbit, and whose surviving orbit-mates give a unique replacement color for every covered cell.
    - For every black or blob-covered cell, copy the value from any intact horizontal, vertical, or 180-degree mirror mate; the intact references in one orbit always agree.

    Key insight:
    - Black holes are explicit, but paint damage is disguised as a valid color; the only reliable cue is that the wrong side is a whole same-color connected blob while the mirrored survivors remain consistent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    damaged = _detect_damage(g)
    out = g.copy()
    for r, c in np.argwhere(damaged):
        refs = {
            int(g[rr, cc])
            for rr, cc in _orbit(g.shape, int(r), int(c))
            if not damaged[rr, cc]
        }
        assert refs, "every damaged orbit needs at least one intact witness"
        assert len(refs) == 1, "intact orbit witnesses must agree"
        out[int(r), int(c)] = next(iter(refs))
    return out


_RNG = np.random.default_rng()
_MAX_ATTEMPTS = 600
_PALETTE = np.arange(1, 10, dtype=int)
_GRID_MIN = 10
_GRID_MAX = 18
_COLOR_COUNT_CHOICES = (4, 5, 6)
_MAX_CLEAN_COMPONENT = 6
_EXTRA_SMEAR_COUNTS = (0, 1, 2)
_EXTRA_SMEAR_WEIGHTS = np.array([5, 3, 2], dtype=float)
_EXTRA_BLACK_COUNTS = (0, 1, 2, 3)
_EXTRA_BLACK_WEIGHTS = np.array([2, 4, 3, 1], dtype=float)
_RECT_SIZES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
)
_RECT_WEIGHTS = np.array([7, 4, 4, 4, 2, 2, 1, 1], dtype=float)
_BASE_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
]


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[np.ndarray]:
    pts = np.array(shape, dtype=int)
    variants: list[np.ndarray] = []
    seen = set()
    for flip in (1, -1):
        cur = pts.copy()
        cur[:, 1] *= flip
        for _ in range(4):
            rows = cur[:, 0]
            cols = cur[:, 1]
            norm = np.stack([rows - rows.min(), cols - cols.min()], axis=1)
            key = tuple(sorted((int(r), int(c)) for r, c in norm))
            if key not in seen:
                seen.add(key)
                variants.append(np.array(key, dtype=int))
            cur = np.stack([cur[:, 1], -cur[:, 0]], axis=1)
    return variants


_SMEAR_VARIANTS = [variant for shape in _BASE_SHAPES for variant in _shape_variants(shape)]


def _quadrant_bounds(h: int, w: int, quadrant: int) -> tuple[int, int, int, int]:
    top_h = h // 2
    left_w = w // 2
    bottom_r0 = h - top_h
    right_c0 = w - left_w
    if quadrant == 0:
        return 0, top_h - 1, 0, left_w - 1
    if quadrant == 1:
        return 0, top_h - 1, right_c0, w - 1
    if quadrant == 2:
        return bottom_r0, h - 1, 0, left_w - 1
    return bottom_r0, h - 1, right_c0, w - 1


def _apply_orbit_color(grid: np.ndarray, r: int, c: int, color: int) -> None:
    for rr, cc in _orbit(grid.shape, r, c):
        grid[rr, cc] = int(color)


def _equal_neighbor_score(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> int:
    h, w = grid.shape
    score = 0
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == color:
                score += 1
    return score


def _max_component_size(grid: np.ndarray) -> int:
    sizes = [int(comp.shape[0]) for _color, comp in _color_components(grid)]
    return max(sizes, default=0)


def _is_biaxial_symmetric(grid: np.ndarray) -> bool:
    return bool(np.array_equal(grid, np.fliplr(grid)) and np.array_equal(grid, np.flipud(grid)))


def _generate_clean_grid() -> np.ndarray | None:
    h = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    w = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    color_count = int(_RNG.choice(_COLOR_COUNT_CHOICES))
    colors = _RNG.choice(_PALETTE, size=color_count, replace=False)

    anchors = [(r, c) for r in range((h + 1) // 2) for c in range((w + 1) // 2)]
    if len(anchors) < color_count:
        return None

    grid = np.full((h, w), -1, dtype=int)
    assigned = anchors.copy()
    _RNG.shuffle(assigned)
    seeded = assigned[:color_count]
    seed_map = {tuple(anchor): int(color) for anchor, color in zip(seeded, colors)}

    for r, c in anchors:
        orbit_cells = _orbit(grid.shape, r, c)
        if (r, c) in seed_map:
            picked = seed_map[(r, c)]
        else:
            scores = []
            for color in colors:
                score = _equal_neighbor_score(grid, orbit_cells, int(color))
                jitter = float(_RNG.random()) * 0.05
                scores.append((score + jitter, int(color)))
            min_score = min(item[0] for item in scores)
            best = [color for score, color in scores if abs(score - min_score) < 1e-9 or score == min_score]
            picked = int(_RNG.choice(best))
        _apply_orbit_color(grid, r, c, picked)

    if np.any(grid < 0):
        return None
    if len(np.unique(grid)) < 4:
        return None
    if _max_component_size(grid) > _MAX_CLEAN_COMPONENT:
        return None
    if not _is_biaxial_symmetric(grid):
        return None
    return grid


def _damage_recoverable(mask: np.ndarray) -> bool:
    for r, c in np.argwhere(mask):
        orbit = _orbit(mask.shape, int(r), int(c))
        if all(mask[rr, cc] for rr, cc in orbit):
            return False
    return True


def _paint_smear(observed: np.ndarray, coords: np.ndarray, color: int) -> None:
    observed[coords[:, 0], coords[:, 1]] = int(color)


def _candidate_smear_colors(clean: np.ndarray, observed: np.ndarray, coords: np.ndarray) -> list[int]:
    h, w = clean.shape
    cell_set = {tuple(map(int, cell)) for cell in coords.tolist()}
    ok: list[int] = []
    for color in _PALETTE:
        color = int(color)
        valid = True
        for r, c in cell_set:
            if int(clean[r, c]) == color:
                valid = False
                break
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set and int(observed[nr, nc]) == color:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            ok.append(color)
    return ok


def _choose_anchor_cell(coords: np.ndarray, damage_mask: np.ndarray) -> tuple[int, int] | None:
    for r, c in coords:
        hmate = (int(r), damage_mask.shape[1] - 1 - int(c))
        vmate = (damage_mask.shape[0] - 1 - int(r), int(c))
        pmate = (damage_mask.shape[0] - 1 - int(r), damage_mask.shape[1] - 1 - int(c))
        if not damage_mask[hmate] and not damage_mask[vmate] and not damage_mask[pmate]:
            return int(r), int(c)
    return None


def _place_smear(
    clean: np.ndarray,
    observed: np.ndarray,
    damage_mask: np.ndarray,
    used_orbits: set[tuple[tuple[int, int], ...]],
    *,
    require_anchor: bool,
) -> tuple[np.ndarray, int, tuple[int, int] | None] | None:
    h, w = clean.shape
    quadrants = [0, 1, 2, 3]
    _RNG.shuffle(quadrants)
    variants = list(range(len(_SMEAR_VARIANTS)))
    _RNG.shuffle(variants)

    for quadrant in quadrants:
        r0_min, r0_max, c0_min, c0_max = _quadrant_bounds(h, w, quadrant)
        if r0_min > r0_max or c0_min > c0_max:
            continue
        for variant_idx in variants:
            shape = _SMEAR_VARIANTS[variant_idx]
            sh = int(shape[:, 0].max()) + 1
            sw = int(shape[:, 1].max()) + 1
            if r0_min + sh - 1 > r0_max or c0_min + sw - 1 > c0_max:
                continue

            row_choices = list(range(r0_min, r0_max - sh + 2))
            col_choices = list(range(c0_min, c0_max - sw + 2))
            _RNG.shuffle(row_choices)
            _RNG.shuffle(col_choices)
            for sr in row_choices:
                for sc in col_choices:
                    coords = shape + np.array([sr, sc], dtype=int)
                    if np.any(damage_mask[coords[:, 0], coords[:, 1]]):
                        continue
                    orbit_ids = [_orbit_id(clean.shape, int(r), int(c)) for r, c in coords]
                    if len(set(orbit_ids)) != len(orbit_ids):
                        continue
                    if any(item in used_orbits for item in orbit_ids):
                        continue

                    proposal = damage_mask.copy()
                    proposal[coords[:, 0], coords[:, 1]] = True
                    if not _damage_recoverable(proposal):
                        continue

                    anchor = _choose_anchor_cell(coords, damage_mask) if require_anchor else None
                    if require_anchor and anchor is None:
                        continue

                    colors = _candidate_smear_colors(clean, observed, coords)
                    if not colors:
                        continue
                    color = int(_RNG.choice(colors))
                    return coords, color, anchor
    return None


def _place_black_rect(
    clean: np.ndarray,
    damage_mask: np.ndarray,
    preferred_cells: np.ndarray | None = None,
) -> np.ndarray | None:
    h, w = clean.shape
    sizes = _RNG.choice(len(_RECT_SIZES), size=len(_RECT_SIZES), replace=False, p=_RECT_WEIGHTS / _RECT_WEIGHTS.sum())
    quadrants = [0, 1, 2, 3]
    _RNG.shuffle(quadrants)

    if preferred_cells is not None:
        candidates = [preferred_cells]
    else:
        candidates = []

    for size_idx in sizes:
        rh, rw = _RECT_SIZES[int(size_idx)]
        if preferred_cells is None:
            for quadrant in quadrants:
                r0_min, r0_max, c0_min, c0_max = _quadrant_bounds(h, w, quadrant)
                if r0_min + rh - 1 > r0_max or c0_min + rw - 1 > c0_max:
                    continue
                rows = list(range(r0_min, r0_max - rh + 2))
                cols = list(range(c0_min, c0_max - rw + 2))
                _RNG.shuffle(rows)
                _RNG.shuffle(cols)
                for sr in rows:
                    for sc in cols:
                        rect = np.array(
                            [(sr + dr, sc + dc) for dr in range(rh) for dc in range(rw)],
                            dtype=int,
                        )
                        candidates.append(rect)

        for rect in candidates:
            if np.any(damage_mask[rect[:, 0], rect[:, 1]]):
                continue
            proposal = damage_mask.copy()
            proposal[rect[:, 0], rect[:, 1]] = True
            if _damage_recoverable(proposal):
                return rect
        if preferred_cells is not None:
            break
        candidates = []
    return None


def _point_fallback_exists(damage_mask: np.ndarray) -> bool:
    h, w = damage_mask.shape
    for r, c in np.argwhere(damage_mask):
        mr = h - 1 - int(r)
        mc = w - 1 - int(c)
        if damage_mask[int(r), mc] and damage_mask[mr, int(c)] and not damage_mask[mr, mc]:
            return True
    return False


def _validate_instance(
    clean: np.ndarray,
    observed: np.ndarray,
    damage_mask: np.ndarray,
    smear_components: list[np.ndarray],
) -> bool:
    if not _is_biaxial_symmetric(clean):
        return False
    if not _damage_recoverable(damage_mask):
        return False
    if not np.any(observed == 0):
        return False
    if not smear_components:
        return False
    if not _point_fallback_exists(damage_mask):
        return False

    diff = observed != clean
    if not np.array_equal(diff, damage_mask):
        return False

    for coords in smear_components:
        values = observed[coords[:, 0], coords[:, 1]]
        if len(set(int(v) for v in values)) != 1:
            return False
        if np.any(clean[coords[:, 0], coords[:, 1]] == values[0]):
            return False

    detected = _detect_damage(observed)
    if not np.array_equal(detected, damage_mask):
        return False

    try:
        repaired = main(observed)
    except Exception:
        return False
    return bool(np.array_equal(repaired, clean))


def _attempt() -> np.ndarray | None:
    clean = _generate_clean_grid()
    if clean is None:
        return None

    observed = clean.copy()
    damage_mask = np.zeros_like(clean, dtype=bool)
    used_orbits: set[tuple[tuple[int, int], ...]] = set()
    smear_components: list[np.ndarray] = []

    anchor_blob = _place_smear(clean, observed, damage_mask, used_orbits, require_anchor=True)
    if anchor_blob is None:
        return None
    anchor_coords, anchor_color, anchor_cell = anchor_blob
    _paint_smear(observed, anchor_coords, anchor_color)
    damage_mask[anchor_coords[:, 0], anchor_coords[:, 1]] = True
    smear_components.append(anchor_coords.copy())
    for r, c in anchor_coords:
        used_orbits.add(_orbit_id(clean.shape, int(r), int(c)))

    assert anchor_cell is not None
    ar, ac = anchor_cell
    hmate = np.array([[ar, clean.shape[1] - 1 - ac]], dtype=int)
    vmate = np.array([[clean.shape[0] - 1 - ar, ac]], dtype=int)
    for rect in (hmate, vmate):
        black = _place_black_rect(clean, damage_mask, preferred_cells=rect)
        if black is None:
            return None
        observed[black[:, 0], black[:, 1]] = 0
        damage_mask[black[:, 0], black[:, 1]] = True

    extra_smears = int(
        _RNG.choice(_EXTRA_SMEAR_COUNTS, p=_EXTRA_SMEAR_WEIGHTS / _EXTRA_SMEAR_WEIGHTS.sum())
    )
    for _ in range(extra_smears):
        blob = _place_smear(clean, observed, damage_mask, used_orbits, require_anchor=False)
        if blob is None:
            return None
        coords, color, _anchor = blob
        _paint_smear(observed, coords, color)
        damage_mask[coords[:, 0], coords[:, 1]] = True
        smear_components.append(coords.copy())
        for r, c in coords:
            used_orbits.add(_orbit_id(clean.shape, int(r), int(c)))

    extra_black = int(
        _RNG.choice(_EXTRA_BLACK_COUNTS, p=_EXTRA_BLACK_WEIGHTS / _EXTRA_BLACK_WEIGHTS.sum())
    )
    for _ in range(extra_black):
        rect = _place_black_rect(clean, damage_mask)
        if rect is None:
            return None
        observed[rect[:, 0], rect[:, 1]] = 0
        damage_mask[rect[:, 0], rect[:, 1]] = True

    if not _validate_instance(clean, observed, damage_mask, smear_components):
        return None
    return observed


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid double-damage mirror-repair task")
