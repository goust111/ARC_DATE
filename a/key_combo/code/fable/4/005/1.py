from __future__ import annotations

import random

import numpy as np


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for coords in _components(mask, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert coords.shape[0] == area, "expected rectangular missing regions"
        assert bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])), "expected solid rectangles"
        rects.append((r0, c0, r1, c1))
    rects.sort()
    return rects


def _extract_patches(zone: np.ndarray, bg: int) -> list[np.ndarray]:
    patches: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in _components(zone != bg, connectivity=8):
        r0, c0, r1, c1 = _bbox(coords)
        patch = zone[r0 : r1 + 1, c0 : c1 + 1].copy()
        area = patch.size
        assert coords.shape[0] == area, "each loose patch must be a filled rectangle"
        assert bool(np.all(patch != bg)), "patch rectangles cannot contain background"
        patches.append(((r0, c0, r1, c1), patch))
    patches.sort(key=lambda item: item[0])
    return [patch for _box, patch in patches]


def _orientations(a: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for base in (a, np.fliplr(a)):
        for k in range(4):
            t = np.rot90(base, k).copy()
            key = (tuple(map(int, t.shape)), t.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(t)
    return out


def _tile_fill(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    h, w = map(int, shape)
    th, tw = map(int, tile.shape)
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = tile[r % th, c % tw]
    return out


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h, _w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        color = int(grid[r, 0])
        if not bool(np.all(grid[r] == color)):
            continue
        above = grid[:r]
        below = grid[r + 1 :]
        if not bool(np.any(above != color)) or not bool(np.any(below != color)):
            continue
        if np.count_nonzero(below == color) <= below.size // 2:
            continue
        candidates.append((r, color))
    assert candidates, "expected a full-row separator"
    return candidates[0]


def _candidate_tile(
    upper: np.ndarray,
    bg: int,
    field_colors: set[int],
    ph: int,
    pw: int,
) -> np.ndarray | None:
    tile = np.full((ph, pw), -1, dtype=int)
    for r, c in np.argwhere(upper != bg):
        r, c = int(r), int(c)
        color = int(upper[r, c])
        if color not in field_colors:
            return None
        rr, cc = r % ph, c % pw
        if tile[rr, cc] == -1:
            tile[rr, cc] = color
        elif tile[rr, cc] != color:
            return None
    if bool(np.any(tile == -1)):
        return None
    if set(map(int, np.unique(tile))) != field_colors:
        return None
    return tile


def _infer_period(
    upper: np.ndarray,
    bg: int,
    field_colors: set[int],
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = upper.shape
    candidates = [(ph, pw) for ph in range(1, h + 1) for pw in range(1, w + 1)]
    candidates.sort(key=lambda x: (x[0] * x[1], x[0], x[1]))
    for ph, pw in candidates:
        tile = _candidate_tile(upper, bg, field_colors, ph, pw)
        if tile is not None:
            return tile, (ph, pw)
    raise AssertionError("could not infer a fully observed periodic motif")


def _matching_options(
    patch: np.ndarray,
    holes: list[tuple[int, int, int, int]],
    periodic: np.ndarray,
    field_colors: set[int],
    defect: int,
) -> list[tuple[int, np.ndarray]]:
    options: list[tuple[int, np.ndarray]] = []
    allowed = set(field_colors) | {int(defect)}
    assert set(map(int, np.unique(patch))) <= allowed, "patch has an unexpected color"
    assert bool(np.any(patch == defect)), "each patch must carry at least one defect"

    for hole_index, (r0, c0, r1, c1) in enumerate(holes):
        target = periodic[r0 : r1 + 1, c0 : c1 + 1]
        for t in _orientations(patch):
            if t.shape != target.shape:
                continue
            non_defect = t != defect
            if not bool(np.any(non_defect)):
                continue
            if not set(map(int, np.unique(t[non_defect]))) <= field_colors:
                continue
            if bool(np.all(t[non_defect] == target[non_defect])):
                options.append((hole_index, t))
    return options


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock transformed defective patches back into same-size periodic holes by phase.

    Input structure:
    - The upper part is a two-color periodic field with same-size rectangular holes in a third background color.
    - A full background separator row divides the field from a lower staging area.
    - The lower area contains one filled rectangular patch per hole; patches use the two field colors plus one defect color.

    Rule:
    - Infer the field period from the visible upper cells, using grid coordinates as the phase reference.
    - For each loose patch, try all rotations and mirror flips against every hole.
    - The only valid placement is the one whose non-defect cells match the inferred periodic phase, then the full patch is pasted into that hole.

    Key insight:
    - Periodic phase identifies the correct hole and orientation, while defect pixels override the ordinary period once the patch is docked.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    split, bg = _find_separator(g)
    upper = g[:split].copy()
    lower = g[split + 1 :].copy()

    field_colors = set(map(int, np.unique(upper))) - {int(bg)}
    assert len(field_colors) == 2, "expected exactly two periodic field colors"

    lower_colors = set(map(int, np.unique(lower))) - {int(bg)}
    defect_colors = lower_colors - field_colors
    assert len(defect_colors) == 1, "expected a single defect color"
    defect = next(iter(defect_colors))

    holes = _rectangles(upper == bg)
    assert holes, "expected at least one rectangular hole"
    hole_shapes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in holes}
    assert len(hole_shapes) == 1, "all holes must have the same size"

    patches = _extract_patches(lower, bg)
    assert len(patches) == len(holes), "expected one loose patch per hole"

    tile, _period = _infer_period(upper, bg, field_colors)
    periodic = _tile_fill(upper.shape, tile)
    out = upper.copy()
    used_holes: set[int] = set()

    for patch in patches:
        options = _matching_options(patch, holes, periodic, field_colors, defect)
        assert len(options) == 1, "each patch must have exactly one phase-aligned docking"
        hole_index, oriented = options[0]
        assert hole_index not in used_holes, "two patches cannot dock into the same hole"
        used_holes.add(hole_index)
        r0, c0, r1, c1 = holes[hole_index]
        out[r0 : r1 + 1, c0 : c1 + 1] = oriented

    assert len(used_holes) == len(holes), "every hole must be repaired"
    return out


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    candidates = [(ph, pw) for ph in range(1, h + 1) for pw in range(1, w + 1)]
    candidates.sort(key=lambda x: (x[0] * x[1], x[0], x[1]))
    for ph, pw in candidates:
        ok = True
        for r in range(h):
            for c in range(w):
                if tile[r, c] != tile[r % ph, c % pw]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return ph, pw
    return h, w


def _has_d4_symmetry(tile: np.ndarray) -> bool:
    for flipped, base in ((False, tile), (True, np.fliplr(tile))):
        for k in range(4):
            if not flipped and k == 0:
                continue
            t = np.rot90(base, k)
            if t.shape == tile.shape and bool(np.array_equal(t, tile)):
                return True
    return False


def _random_tile(ph: int, pw: int, c0: int, c1: int) -> np.ndarray | None:
    area = ph * pw
    for _ in range(400):
        bits = [random.randint(0, 1) for _ in range(area)]
        ones = sum(bits)
        if ones < 2 or ones > area - 2:
            continue
        tile = np.array([c1 if bit else c0 for bit in bits], dtype=int).reshape(ph, pw)
        if _minimal_period(tile) != (ph, pw):
            continue
        if _has_d4_symmetry(tile):
            continue
        return tile
    return None


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> bool:
    oh, ow = occupied.shape
    if r < 0 or c < 0 or r + h > oh or c + w > ow:
        return False
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(oh, r + h + pad)
    c1 = min(ow, c + w + pad)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _place_holes(
    upper_h: int,
    width: int,
    hole_h: int,
    hole_w: int,
    count: int,
    period: tuple[int, int],
) -> list[tuple[int, int, int, int]] | None:
    occupied = np.zeros((upper_h, width), dtype=bool)
    ph, pw = period
    phases: set[tuple[int, int]] = set()
    candidates = [
        (r, c)
        for r in range(1, upper_h - hole_h)
        for c in range(1, width - hole_w)
    ]
    random.shuffle(candidates)
    holes: list[tuple[int, int, int, int]] = []
    for r, c in candidates:
        if len(holes) == count:
            break
        phase = (r % ph, c % pw)
        if phase in phases:
            continue
        if not _can_place(occupied, r, c, hole_h, hole_w, pad=1):
            continue
        occupied[r : r + hole_h, c : c + hole_w] = True
        phases.add(phase)
        holes.append((r, c, r + hole_h - 1, c + hole_w - 1))
    if len(holes) != count:
        return None
    holes.sort()
    return holes


def _place_patches(
    patches: list[np.ndarray],
    lower_h: int,
    width: int,
    bg: int,
) -> np.ndarray | None:
    lower = np.full((lower_h, width), bg, dtype=int)
    occupied = np.zeros((lower_h, width), dtype=bool)
    ordered = patches[:]
    random.shuffle(ordered)
    for patch in ordered:
        h, w = patch.shape
        candidates = [
            (r, c)
            for r in range(0, lower_h - h + 1)
            for c in range(0, width - w + 1)
            if _can_place(occupied, r, c, h, w, pad=1)
        ]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        lower[r : r + h, c : c + w] = patch
        occupied[r : r + h, c : c + w] = True
    if np.count_nonzero(lower == bg) <= int(0.55 * lower.size):
        return None
    return lower


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    expected_period: tuple[int, int],
    expected_holes: int,
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    try:
        split, bg = _find_separator(inp)
        if split != expected.shape[0]:
            return False
        upper = inp[:split]
        lower = inp[split + 1 :]
        field_colors = set(map(int, np.unique(upper))) - {int(bg)}
        if len(field_colors) != 2:
            return False
        tile, period = _infer_period(upper, bg, field_colors)
        if period != expected_period:
            return False
        periodic = _tile_fill(expected.shape, tile)
        holes = _rectangles(upper == bg)
        if len(holes) != expected_holes:
            return False
        if not all(bool(np.any(expected[r0 : r1 + 1, c0 : c1 + 1] != periodic[r0 : r1 + 1, c0 : c1 + 1])) for r0, c0, r1, c1 in holes):
            return False
        patches = _extract_patches(lower, bg)
        if len(patches) != expected_holes:
            return False
        lower_colors = set(map(int, np.unique(lower))) - {int(bg)}
        defect_colors = lower_colors - field_colors
        if len(defect_colors) != 1:
            return False
        defect = next(iter(defect_colors))
        used: set[int] = set()
        for patch in patches:
            opts = _matching_options(patch, holes, periodic, field_colors, defect)
            if len(opts) != 1:
                return False
            used.add(opts[0][0])
        if len(used) != expected_holes:
            return False
        return bool(np.array_equal(main(inp), expected))
    except Exception:
        return False


def _attempt_generate() -> np.ndarray | None:
    colors = random.sample(range(10), 4)
    field_a, field_b, bg, defect = map(int, colors)

    period_choices = [(3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4), (3, 3), (4, 4)]
    ph, pw = random.choice(period_choices)
    tile = _random_tile(ph, pw, field_a, field_b)
    if tile is None:
        return None

    hole_sizes = [(3, 4), (4, 3), (3, 5), (5, 3), (4, 4), (5, 5)]
    hole_h, hole_w = random.choice(hole_sizes)
    upper_min_h = max(10, ph * 3, hole_h + 4)
    upper_max_h = 17
    if upper_min_h > upper_max_h:
        return None
    upper_h = random.randint(upper_min_h, upper_max_h)
    width_min = max(12, pw * 3, hole_w + 5)
    width = random.randint(width_min, 22)
    hole_count = random.randint(2, 4)

    holes = _place_holes(upper_h, width, hole_h, hole_w, hole_count, (ph, pw))
    if holes is None:
        return None

    periodic = _tile_fill((upper_h, width), tile)
    upper = periodic.copy()
    for r0, c0, r1, c1 in holes:
        upper[r0 : r1 + 1, c0 : c1 + 1] = bg
    if bool(np.any(np.all(upper == bg, axis=1))):
        return None

    expected = periodic.copy()
    target_patches: list[np.ndarray] = []
    for r0, c0, r1, c1 in holes:
        patch = periodic[r0 : r1 + 1, c0 : c1 + 1].copy()
        defect_max = min(3, patch.size - 2)
        defect_count = random.randint(1, defect_max)
        cells = [(r, c) for r in range(patch.shape[0]) for c in range(patch.shape[1])]
        for r, c in random.sample(cells, defect_count):
            patch[r, c] = defect
        if len(set(map(int, np.unique(patch))) - {defect}) < 2:
            return None
        expected[r0 : r1 + 1, c0 : c1 + 1] = patch
        target_patches.append(patch)

    input_patches: list[np.ndarray] = []
    for i, patch in enumerate(target_patches):
        variants = _orientations(patch)
        if i == 0:
            changed = [v for v in variants if v.shape != patch.shape or not bool(np.array_equal(v, patch))]
            input_patches.append(random.choice(changed or variants))
        else:
            input_patches.append(random.choice(variants))

    max_lower_h = 30 - upper_h - 1
    min_lower_h = max(7, max(p.shape[0] for p in input_patches) + 2)
    if min_lower_h > max_lower_h:
        return None
    lower_h = random.randint(min_lower_h, max_lower_h)
    lower = _place_patches(input_patches, lower_h, width, bg)
    if lower is None:
        return None

    separator = np.full((1, width), bg, dtype=int)
    inp = np.vstack([upper, separator, lower])
    if not _semantic_check(inp, expected, (ph, pw), hole_count):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid defective periodic patch input")
