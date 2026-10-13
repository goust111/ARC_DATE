from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
PALETTE = (BLUE, RED, YELLOW, ORANGE)

RNG = np.random.default_rng()
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(cells, dtype=int))
    return out


def _pattern_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components(grid != BLACK)
    assert comps, "expected a central non-black pattern"
    coords = max(comps, key=lambda arr: int(arr.shape[0]))
    return _bbox_from_coords(coords)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_inside(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _marker_direction(
    box: tuple[int, int, int, int], r: int, c: int
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if r0 <= r <= r1 and c < c0:
        return 0, 1
    if r0 <= r <= r1 and c > c1:
        return 0, -1
    if c0 <= c <= c1 and r < r0:
        return 1, 0
    if c0 <= c <= c1 and r > r1:
        return -1, 0
    raise AssertionError("marker must align with the pattern by row or column")


def _first_nonblack_on_ray(
    grid: np.ndarray, r: int, c: int, dr: int, dc: int
) -> tuple[int, int] | None:
    h, w = grid.shape
    rr = int(r)
    cc = int(c)
    while 0 <= rr < h and 0 <= cc < w:
        if int(grid[rr, cc]) != BLACK:
            return rr, cc
        rr += dr
        cc += dc
    return None


def _apply_markers(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = box
    marker_cells = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere(grid != BLACK)
        if not (r0 <= int(r) <= r1 and c0 <= int(c) <= c1)
    ]
    targets: dict[tuple[int, int], int] = {}
    for r, c, color in marker_cells:
        out[r, c] = BLACK
        dr, dc = _marker_direction(box, r, c)
        hit = _first_nonblack_on_ray(grid, r + dr, c + dc, dr, dc)
        if hit is None:
            continue
        hr, hc = hit
        if int(grid[hr, hc]) != GREEN:
            continue
        key = (hr, hc)
        prev = targets.get(key)
        assert prev is None or prev == color, "conflicting donors"
        targets[key] = color
    for (r, c), color in targets.items():
        out[r, c] = color
    return out


def _orbit_cells(
    box: tuple[int, int, int, int], r: int, c: int
) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = box
    rr = r - r0
    cc = c - c0
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    cells = {
        (r0 + rr, c0 + cc),
        (r0 + rr, c0 + (w - 1 - cc)),
        (r0 + (h - 1 - rr), c0 + cc),
        (r0 + (h - 1 - rr), c0 + (w - 1 - cc)),
    }
    return tuple(sorted(cells))


def _apply_symmetry(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = box
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            orbit = _orbit_cells(box, r, c)
            if orbit in seen:
                continue
            seen.add(orbit)
            values = [int(grid[rr, cc]) for rr, cc in orbit]
            greens = [(rr, cc) for rr, cc in orbit if int(grid[rr, cc]) == GREEN]
            if not greens:
                continue
            visible = {value for value in values if value not in (BLACK, GREEN)}
            assert len(visible) == 1, "every damaged orbit must have a unique visible color"
            fill = visible.pop()
            for rr, cc in greens:
                out[rr, cc] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide external color donors into the damaged pattern, then complete the remaining holes by two-axis reflection symmetry.

    Input structure:
    - The background is black.
    - The largest non-black connected component is a central colored pattern whose bounding box is symmetric under left-right and top-bottom reflection.
    - Some pattern cells are replaced by green placeholders, and extra single-pixel colored markers lie outside the pattern on the same row or column as the pattern.

    Rule:
    - For each external marker, look from the marker toward the pattern along its row or column and skip black cells.
    - If the first non-black cell on that ray is green, replace that green cell with the marker color and erase the marker; otherwise erase the marker without effect.
    - After all markers are processed, for every remaining green cell inside the pattern bounding box, fill it with the unique non-black non-green color from its horizontal/vertical reflection orbit.

    Key insight:
    - A donor marker only colors the first reachable damaged cell on its inward ray; the rest of the missing structure is recovered by propagating that donated color through the pattern's two-axis symmetry orbit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    box = _pattern_bbox(g)
    after_slide = _apply_markers(g, box)
    return _apply_symmetry(after_slide, box)


def _local_orbit(ph: int, pw: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    cells = {
        (r, c),
        (r, pw - 1 - c),
        (ph - 1 - r, c),
        (ph - 1 - r, pw - 1 - c),
    }
    return tuple(sorted(cells))


def _canonical_orbits(ph: int, pw: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(ph):
        for c in range(pw):
            orbit = _local_orbit(ph, pw, r, c)
            if orbit not in seen:
                seen.add(orbit)
                out.append(orbit)
    return out


def _build_output_pattern(ph: int, pw: int) -> np.ndarray:
    colors = RNG.choice(PALETTE, size=int(RNG.integers(3, 5)), replace=False)
    copy_prob = float(RNG.uniform(0.2, 0.5))
    qh = (ph + 1) // 2
    qw = (pw + 1) // 2
    seeds: dict[tuple[int, int], int] = {}
    for r in range(qh):
        for c in range(qw):
            neighbors = [
                seeds[key]
                for key in ((r - 1, c), (r, c - 1), (r - 1, c - 1))
                if key in seeds
            ]
            if neighbors and RNG.random() < copy_prob:
                color = int(RNG.choice(neighbors))
            else:
                color = int(RNG.choice(colors))
            seeds[(r, c)] = color

    forced = RNG.choice(qh * qw, size=min(len(colors), qh * qw), replace=False)
    shuffled = list(map(int, colors))
    random.shuffle(shuffled)
    for idx, color in zip(forced, shuffled):
        seeds[(int(idx) // qw, int(idx) % qw)] = int(color)

    patch = np.zeros((ph, pw), dtype=int)
    for (r, c), color in seeds.items():
        for rr, cc in _local_orbit(ph, pw, r, c):
            patch[rr, cc] = color

    used = sorted(set(map(int, patch.flatten())) - {BLACK})
    if len(used) < 3:
        return _build_output_pattern(ph, pw)
    row_transitions = [
        int(np.count_nonzero(patch[r, 1:] != patch[r, :-1]))
        for r in range(ph)
    ]
    col_transitions = [
        int(np.count_nonzero(patch[1:, c] != patch[:-1, c]))
        for c in range(pw)
    ]
    if max(row_transitions, default=0) < 2 or max(col_transitions, default=0) < 2:
        return _build_output_pattern(ph, pw)
    return patch


def _pick_bbox(h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    row_choices = [r for r in range(2, h - ph - 1) if abs((2 * r + ph) - h) <= 3]
    col_choices = [c for c in range(2, w - pw - 1) if abs((2 * c + pw) - w) <= 3]
    if not row_choices:
        row_choices = list(range(2, h - ph - 1))
    if not col_choices:
        col_choices = list(range(2, w - pw - 1))
    return int(RNG.choice(row_choices)), int(RNG.choice(col_choices))


def _orbit_sides(ph: int, pw: int, orbit: tuple[tuple[int, int], ...]) -> list[tuple[str, tuple[int, int]]]:
    sides: list[tuple[str, tuple[int, int]]] = []
    for cell in orbit:
        r, c = cell
        if c == 0:
            sides.append(("left", cell))
        if c == pw - 1:
            sides.append(("right", cell))
        if r == 0:
            sides.append(("top", cell))
        if r == ph - 1:
            sides.append(("bottom", cell))
    return sides


def _source_for_side(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    side: str,
    target: tuple[int, int],
) -> tuple[int, int]:
    h, w = shape
    r, c = target
    if side == "left":
        return r, 0
    if side == "right":
        return r, w - 1
    if side == "top":
        return 0, c
    assert side == "bottom"
    return h - 1, c


def _scan_result(
    grid: np.ndarray, box: tuple[int, int, int, int], source: tuple[int, int]
) -> tuple[int, int] | None:
    dr, dc = _marker_direction(box, source[0], source[1])
    return _first_nonblack_on_ray(grid, source[0] + dr, source[1] + dc, dr, dc)


def _direct_symmetry_only(inp: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    return _apply_symmetry(inp, box)


def _choose_damage(
    patch: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]] | None:
    ph, pw = patch.shape
    damaged = patch.copy()
    orbits = _canonical_orbits(ph, pw)
    donor_candidates = [orbit for orbit in orbits if len(orbit) >= 2 and _orbit_sides(ph, pw, orbit)]
    interior_candidates = [
        orbit
        for orbit in orbits
        if len(orbit) >= 2 and any(0 < r < ph - 1 and 0 < c < pw - 1 for r, c in orbit)
    ]
    if not donor_candidates or not interior_candidates:
        return None

    RNG.shuffle(donor_candidates)
    RNG.shuffle(interior_candidates)

    donor_count = int(RNG.integers(1, min(4, len(donor_candidates)) + 1))
    donor_orbits = donor_candidates[:donor_count]
    donor_set = {donor_orbits[0]}
    while len(donor_set) < donor_count:
        donor_set.add(donor_orbits[len(donor_set)])
    donor_orbits = list(donor_set)

    remaining = [orbit for orbit in orbits if orbit not in donor_set and len(orbit) >= 2]
    if not remaining:
        return None
    symmetry_count = int(RNG.integers(1, min(5, len(remaining)) + 1))
    symmetry_orbits = []
    guaranteed_interior = None
    for orbit in interior_candidates:
        if orbit in donor_set:
            continue
        guaranteed_interior = orbit
        break
    if guaranteed_interior is None:
        return None
    symmetry_orbits.append(guaranteed_interior)
    RNG.shuffle(remaining)
    for orbit in remaining:
        if orbit == guaranteed_interior:
            continue
        symmetry_orbits.append(orbit)
        if len(symmetry_orbits) >= symmetry_count:
            break

    donor_specs: list[dict[str, object]] = []
    for orbit in donor_orbits:
        color = int(patch[orbit[0]])
        for r, c in orbit:
            damaged[r, c] = GREEN
        donor_specs.append({"orbit": orbit, "color": color})

    symmetry_specs: list[dict[str, object]] = []
    for idx, orbit in enumerate(symmetry_orbits):
        cells = list(orbit)
        color = int(patch[cells[0]])
        max_green = len(cells) - 1
        if idx == 0:
            interior = [cell for cell in cells if 0 < cell[0] < ph - 1 and 0 < cell[1] < pw - 1]
            if not interior:
                return None
            chosen = {interior[int(RNG.integers(len(interior)))]}
            extra = [cell for cell in cells if cell not in chosen]
            RNG.shuffle(extra)
            for cell in extra[: int(RNG.integers(0, max_green))]:
                if len(chosen) >= max_green:
                    break
                chosen.add(cell)
        else:
            green_count = int(RNG.integers(1, max_green + 1))
            chosen = set(cells[int(i)] for i in RNG.choice(len(cells), size=green_count, replace=False))
        for r, c in chosen:
            damaged[r, c] = GREEN
        symmetry_specs.append({"orbit": orbit, "color": color, "greens": tuple(sorted(chosen))})
    return damaged, donor_specs, symmetry_specs


def _place_markers(
    inp: np.ndarray,
    patch_box: tuple[int, int, int, int],
    donor_specs: list[dict[str, object]],
    symmetry_specs: list[dict[str, object]],
) -> np.ndarray | None:
    h, w = inp.shape
    out = inp.copy()
    used_sources: set[tuple[int, int]] = set()

    for spec in donor_specs:
        orbit = spec["orbit"]
        assert isinstance(orbit, tuple)
        side_options = _orbit_sides(
            patch_box[2] - patch_box[0] + 1, patch_box[3] - patch_box[1] + 1, orbit
        )
        random.shuffle(side_options)
        placed = False
        for side, local_target in side_options:
            target = (patch_box[0] + local_target[0], patch_box[1] + local_target[1])
            source = _source_for_side(out.shape, patch_box, side, target)
            if source in used_sources or int(out[source]) != BLACK:
                continue
            hit = _scan_result(out, patch_box, source)
            if hit != target or int(out[target]) != GREEN:
                continue
            out[source] = int(spec["color"])
            used_sources.add(source)
            spec["source"] = source
            spec["target"] = target
            placed = True
            break
        if not placed:
            return None

    blocked_candidates: list[tuple[tuple[int, int], str, tuple[int, int], int]] = []
    for spec in symmetry_specs:
        color = int(spec["color"])
        greens = spec["greens"]
        assert isinstance(greens, tuple)
        for local_green in greens:
            global_green = (patch_box[0] + local_green[0], patch_box[1] + local_green[1])
            for side in ("left", "right", "top", "bottom"):
                if side == "left" and local_green[1] == 0:
                    continue
                if side == "right" and local_green[1] == patch_box[3] - patch_box[1]:
                    continue
                if side == "top" and local_green[0] == 0:
                    continue
                if side == "bottom" and local_green[0] == patch_box[2] - patch_box[0]:
                    continue
                source = _source_for_side(out.shape, patch_box, side, global_green)
                if source in used_sources or int(out[source]) != BLACK:
                    continue
                hit = _scan_result(out, patch_box, source)
                if hit is None or hit == global_green:
                    continue
                hr, hc = hit
                if int(out[hr, hc]) == GREEN:
                    continue
                blocked_candidates.append((source, side, global_green, color))

    if not blocked_candidates:
        return None

    random.shuffle(blocked_candidates)
    blocked_count = min(len(blocked_candidates), int(RNG.integers(1, 3)))
    blocked_added = 0
    for source, _side, _green, color in blocked_candidates:
        if source in used_sources:
            continue
        out[source] = color
        used_sources.add(source)
        blocked_added += 1
        if blocked_added >= blocked_count:
            break
    if blocked_added == 0:
        return None
    return out


def _make_instance() -> tuple[np.ndarray, np.ndarray] | None:
    h = int(RNG.integers(14, 17))
    w = int(RNG.integers(14, 17))
    ph = int(RNG.integers(8, min(11, h - 3)))
    pw = int(RNG.integers(8, min(11, w - 3)))
    if ph % 2 == 0 and pw % 2 == 0:
        if RNG.random() < 0.5:
            ph = min(10, ph + 1)
        else:
            pw = min(10, pw + 1)
        if ph > 10 or pw > 10:
            return None
    r0, c0 = _pick_bbox(h, w, ph, pw)
    box = (r0, c0, r0 + ph - 1, c0 + pw - 1)

    patch = _build_output_pattern(ph, pw)
    output = np.zeros((h, w), dtype=int)
    output[r0 : r0 + ph, c0 : c0 + pw] = patch

    damaged_info = _choose_damage(patch)
    if damaged_info is None:
        return None
    damaged_patch, donor_specs, symmetry_specs = damaged_info

    inp = np.zeros((h, w), dtype=int)
    inp[r0 : r0 + ph, c0 : c0 + pw] = damaged_patch
    inp = _place_markers(inp, box, donor_specs, symmetry_specs)
    if inp is None:
        return None
    if not _validate_instance(inp, output, donor_specs):
        return None
    return inp, output


def _validate_instance(
    inp: np.ndarray, output: np.ndarray, donor_specs: list[dict[str, object]]
) -> bool:
    try:
        box = _pattern_bbox(inp)
        if not np.array_equal(main(inp), output):
            return False
        slid = _apply_markers(inp, box)
        if np.array_equal(slid, output):
            return False
        if int(np.count_nonzero(slid == GREEN)) == 0:
            return False
        try:
            sym_only = _direct_symmetry_only(inp, box)
        except AssertionError:
            sym_only = None
        if sym_only is not None and np.array_equal(sym_only, output):
            return False
        marker_count = int(np.count_nonzero(inp != BLACK)) - int(np.count_nonzero(inp[box[0] : box[2] + 1, box[1] : box[3] + 1] != BLACK))
        if marker_count < len(donor_specs) + 1:
            return False
        for spec in donor_specs:
            source = spec.get("source")
            target = spec.get("target")
            color = int(spec["color"])
            if not isinstance(source, tuple) or not isinstance(target, tuple):
                return False
            hit = _scan_result(inp, box, source)
            if hit != target:
                return False
            if int(inp[target]) != GREEN or int(inp[source]) != color:
                return False
        if np.array_equal(inp, output):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        made = _make_instance()
        if made is None:
            continue
        inp, _out = made
        return inp
    raise RuntimeError("failed to generate a valid sliding-donor symmetry input")
