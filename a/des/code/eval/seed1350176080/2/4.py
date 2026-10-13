from __future__ import annotations

from functools import lru_cache

import numpy as np


HORIZONTAL = "horizontal"
VERTICAL = "vertical"
ROT90 = "rot90"
SYMMETRIES = (HORIZONTAL, VERTICAL, ROT90)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SEARCH_PAD = 1
MAX_ENUM_TIES = 10


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _component_count(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    coords = np.argwhere(mask)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        count += 1
        seen[rr, cc] = True
        stack = [(rr, cc)]
        while stack:
            cr, cc = stack.pop()
            for dr, dc in DIR4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _perimeter(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    per = int(mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    per += int(np.sum(mask[1:, :] != mask[:-1, :]))
    per += int(np.sum(mask[:, 1:] != mask[:, :-1]))
    return per


def _sym_exact(mask: np.ndarray, symmetry: str) -> bool:
    if symmetry == HORIZONTAL:
        return bool(np.array_equal(mask, np.flipud(mask)))
    if symmetry == VERTICAL:
        return bool(np.array_equal(mask, np.fliplr(mask)))
    assert symmetry == ROT90
    return mask.shape[0] == mask.shape[1] and bool(np.array_equal(mask, np.rot90(mask)))


def _transform_cell(r: int, c: int, shape: tuple[int, int], symmetry: str) -> tuple[int, int]:
    h, w = shape
    if symmetry == HORIZONTAL:
        return h - 1 - r, c
    if symmetry == VERTICAL:
        return r, w - 1 - c
    assert symmetry == ROT90
    return c, h - 1 - r


@lru_cache(maxsize=None)
def _orbits(shape: tuple[int, int], symmetry: str) -> tuple[tuple[tuple[int, int], ...], ...]:
    h, w = shape
    if symmetry == ROT90 and h != w:
        return tuple()

    visited = np.zeros((h, w), dtype=bool)
    result: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            orbit: list[tuple[int, int]] = []
            cr, cc = r, c
            while not visited[cr, cc]:
                visited[cr, cc] = True
                orbit.append((cr, cc))
                cr, cc = _transform_cell(cr, cc, shape, symmetry)
            result.append(tuple(orbit))
    return tuple(result)


@lru_cache(maxsize=None)
def _orbit_graph(shape: tuple[int, int], symmetry: str) -> tuple[
    tuple[tuple[tuple[int, int], ...], ...], tuple[frozenset[int], ...], tuple[frozenset[int], ...]
]:
    orbits = _orbits(shape, symmetry)
    h, w = shape
    orbit_of = -np.ones((h, w), dtype=int)
    for idx, orbit in enumerate(orbits):
        for r, c in orbit:
            orbit_of[r, c] = idx

    adjacency: list[set[int]] = [set() for _ in orbits]
    boundaries: list[set[int]] = [set() for _ in orbits]
    for idx, orbit in enumerate(orbits):
        for r, c in orbit:
            if r == 0:
                boundaries[idx].add(0)
            if r == h - 1:
                boundaries[idx].add(1)
            if c == 0:
                boundaries[idx].add(2)
            if c == w - 1:
                boundaries[idx].add(3)
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    other = int(orbit_of[nr, nc])
                    if other != idx:
                        adjacency[idx].add(other)

    return orbits, tuple(frozenset(a) for a in adjacency), tuple(frozenset(b) for b in boundaries)


def _evaluate_submask(
    submask: np.ndarray, symmetry: str, outside_count: int, require_tight: bool, max_cost: int
) -> tuple[tuple[int, int, int, int, int, int], np.ndarray] | None:
    orbits = _orbits(submask.shape, symmetry)
    if not orbits:
        return None

    cost = int(outside_count)
    if cost > max_cost:
        return None
    tie_orbits: list[tuple[tuple[int, int], ...]] = []
    base = np.zeros_like(submask, dtype=bool)
    current_count = int(submask.sum())

    for orbit in orbits:
        occ = sum(int(submask[r, c]) for r, c in orbit)
        size = len(orbit)
        cost += min(occ, size - occ)
        if cost > max_cost:
            return None
        if 2 * occ > size:
            for r, c in orbit:
                base[r, c] = True
        elif 2 * occ == size:
            tie_orbits.append(orbit)

    if len(tie_orbits) > MAX_ENUM_TIES:
        return None

    best_score: tuple[int, int, int, int, int, int] | None = None
    best_mask: np.ndarray | None = None
    for bits in range(1 << len(tie_orbits)):
        repaired = base.copy()
        for i, orbit in enumerate(tie_orbits):
            if (bits >> i) & 1:
                for r, c in orbit:
                    repaired[r, c] = True

        if not repaired.any():
            continue
        if require_tight and not _touches_all_edges(repaired):
            continue

        cropped = repaired if require_tight else _crop(repaired)
        area = int(cropped.sum())
        score = (
            cost,
            _component_count(cropped),
            _perimeter(cropped),
            abs(area - current_count),
            cropped.shape[0] * cropped.shape[1],
            -area,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_mask = cropped

    if best_score is None or best_mask is None:
        return None
    return best_score, best_mask


def _repair_object(mask: np.ndarray) -> tuple[str, np.ndarray]:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    h, w = mask.shape
    total = int(mask.sum())

    rr0 = max(0, r0 - SEARCH_PAD)
    cc0 = max(0, c0 - SEARCH_PAD)
    rr1 = min(h - 1, r1 + SEARCH_PAD)
    cc1 = min(w - 1, c1 + SEARCH_PAD)
    region = mask[rr0 : rr1 + 1, cc0 : cc1 + 1]

    br0 = r0 - rr0
    bc0 = c0 - cc0
    br1 = r1 - rr0
    bc1 = c1 - cc0

    best: tuple[tuple[int, int, int, int, int, int, int], str, np.ndarray] | None = None
    best_cost = 10**9
    for require_tight in (True, False):
        for symmetry in SYMMETRIES:
            rects: list[tuple[int, int, int, int]] = []
            r0_lo = max(0, br0 - SEARCH_PAD)
            r0_hi = min(region.shape[0] - 1, br0 + SEARCH_PAD)
            c0_lo = max(0, bc0 - SEARCH_PAD)
            c0_hi = min(region.shape[1] - 1, bc0 + SEARCH_PAD)
            for sr0 in range(r0_lo, r0_hi + 1):
                r1_lo = max(sr0, br1 - SEARCH_PAD)
                r1_hi = min(region.shape[0] - 1, br1 + SEARCH_PAD)
                for sr1 in range(r1_lo, r1_hi + 1):
                    for sc0 in range(c0_lo, c0_hi + 1):
                        c1_lo = max(sc0, bc1 - SEARCH_PAD)
                        c1_hi = min(region.shape[1] - 1, bc1 + SEARCH_PAD)
                        for sc1 in range(c1_lo, c1_hi + 1):
                            if symmetry == ROT90 and (sr1 - sr0) != (sc1 - sc0):
                                continue
                            rects.append((sr0, sr1, sc0, sc1))

            rects.sort(key=lambda item: abs(item[0] - br0) + abs(item[1] - br1) + abs(item[2] - bc0) + abs(item[3] - bc1))
            for sr0, sr1, sc0, sc1 in rects:
                sub = region[sr0 : sr1 + 1, sc0 : sc1 + 1]
                outside = total - int(sub.sum())
                if outside > best_cost:
                    continue
                candidate = _evaluate_submask(sub, symmetry, outside, require_tight, best_cost)
                if candidate is None:
                    continue
                score, repaired = candidate
                full_score = (*score, 0 if symmetry == HORIZONTAL else 1 if symmetry == VERTICAL else 2)
                if best is None or full_score < best[0]:
                    best = (full_score, symmetry, repaired)
                    best_cost = score[0]
        if best is not None:
            break

    assert best is not None
    return best[1], best[2]


def _render_output(
    shape: tuple[int, int],
    bg: int,
    marker_row: int,
    markers: list[tuple[int, int]],
    repaired_objects: list[dict[str, object]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for col, color in markers:
        out[marker_row, col] = color

    order = sorted(
        range(len(repaired_objects)),
        key=lambda i: (int(repaired_objects[i]["area"]), i),
    )
    for slot, obj_idx in enumerate(order):
        center_col = markers[slot][0]
        obj = repaired_objects[obj_idx]
        patch = np.asarray(obj["mask"], dtype=bool)
        color = int(obj["color"])
        ph, pw = patch.shape
        top = marker_row - ph
        left = center_col - pw // 2
        assert top >= 0 and 0 <= left and left + pw <= shape[1]
        out[top:marker_row, left : left + pw][patch] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair noisy symmetric shapes, then reorder them by repaired area above the marker row.

    Input structure:
    - The background is the most frequent color.
    - Each non-background color corresponds to one noisy monochrome object.
    - The bottom marker row contains one pixel of each object color in evenly spaced columns.

    Rule:
    - For each colored object, pick the symmetry type among horizontal reflection,
      vertical reflection, and 90-degree rotational symmetry that needs the fewest
      pixel edits to restore.
    - Repair the object to satisfy that symmetry exactly.
    - Sort repaired objects by area from small to large.
    - Place them above the marker slots from left to right, bottom-aligned to the
      row just above the markers, while keeping the marker row unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    non_bg_rows = np.where(np.any(g != bg, axis=1))[0]
    assert non_bg_rows.size > 0
    marker_row = int(non_bg_rows[-1])

    markers = [(int(c), int(g[marker_row, c])) for c in np.where(g[marker_row] != bg)[0]]
    assert markers
    marker_colors = [color for _, color in markers]
    assert len(marker_colors) == len(set(marker_colors))

    upper = g[:marker_row]
    repaired_objects: list[dict[str, object]] = []
    for color in marker_colors:
        color_mask = upper == color
        assert color_mask.any()
        symmetry, repaired = _repair_object(color_mask)
        repaired_objects.append(
            {
                "color": color,
                "symmetry": symmetry,
                "mask": repaired,
                "area": int(repaired.sum()),
            }
        )

    return _render_output(g.shape, bg, marker_row, markers, repaired_objects)


# --------------- generator ---------------

_RNG = np.random.default_rng()
PALETTE = tuple(range(10))
GRID_H_RANGE = (24, 32)
OBJ_COUNT_RANGE = (3, 5)
MARKER_STRIDE = 10
MARKER_MARGIN = 7
SIDE_CHOICES = (5, 7)
MAX_GENERATION_ATTEMPTS = 400


def _dilate(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    dilated[1:, 1:] |= mask[:-1, :-1]
    dilated[1:, :-1] |= mask[:-1, 1:]
    dilated[:-1, 1:] |= mask[1:, :-1]
    dilated[:-1, :-1] |= mask[1:, 1:]
    return dilated


def _start_orbits(shape: tuple[int, int], symmetry: str) -> list[int]:
    orbits = _orbits(shape, symmetry)
    h, w = shape
    if symmetry == HORIZONTAL:
        return [i for i, orbit in enumerate(orbits) if any(r == h // 2 for r, _ in orbit)]
    if symmetry == VERTICAL:
        return [i for i, orbit in enumerate(orbits) if any(c == w // 2 for _, c in orbit)]
    return [i for i, orbit in enumerate(orbits) if (h // 2, w // 2) in orbit]


def _mask_from_orbits(shape: tuple[int, int], selected: set[int], symmetry: str) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    orbits = _orbits(shape, symmetry)
    for idx in selected:
        for r, c in orbits[idx]:
            mask[r, c] = True
    return mask


def _generate_symmetric_mask(symmetry: str) -> np.ndarray | None:
    if symmetry == ROT90:
        side = int(_RNG.choice(SIDE_CHOICES))
        shape = (side, side)
    else:
        shape = (int(_RNG.choice(SIDE_CHOICES)), int(_RNG.choice(SIDE_CHOICES)))

    orbits, adjacency, boundaries = _orbit_graph(shape, symmetry)
    starts = _start_orbits(shape, symmetry)
    if not starts:
        return None

    min_area = max(7, (shape[0] * shape[1]) // 4)
    max_area = max(min_area + 1, (shape[0] * shape[1] * 3) // 5)

    for _ in range(80):
        start = int(_RNG.choice(starts))
        selected = {start}
        area = len(orbits[start])
        covered = set(boundaries[start])
        frontier = set(adjacency[start])
        target = int(_RNG.integers(min_area, max_area + 1))

        while frontier:
            uncovered = {0, 1, 2, 3} - covered
            if area >= target and not uncovered and len(selected) >= 4:
                break

            frontier_list = list(frontier)
            if uncovered:
                frontier_edges = [idx for idx in frontier_list if boundaries[idx] & uncovered]
                if frontier_edges and (_RNG.random() < 0.8 or area >= target):
                    pick = int(_RNG.choice(frontier_edges))
                else:
                    pick = int(_RNG.choice(frontier_list))
            else:
                pick = int(_RNG.choice(frontier_list))

            frontier.remove(pick)
            if pick in selected:
                continue
            selected.add(pick)
            area += len(orbits[pick])
            covered |= set(boundaries[pick])
            frontier |= set(adjacency[pick]) - selected

        mask = _mask_from_orbits(shape, selected, symmetry)
        if area < min_area or area > max_area:
            continue
        if not _touches_all_edges(mask):
            continue
        if _component_count(mask) != 1:
            continue
        if any(_sym_exact(mask, other) for other in SYMMETRIES if other != symmetry):
            continue
        return mask
    return None


def _noise_patch(base_mask: np.ndarray, symmetry: str) -> np.ndarray | None:
    h, w = base_mask.shape
    patch = np.zeros((h + 2, w + 2), dtype=bool)
    patch[1 : h + 1, 1 : w + 1] = base_mask

    branch_options = ["remove", "add"]
    if int(base_mask.sum()) >= 8:
        branch_options.append("mixed")
    branch = branch_options[int(_RNG.integers(0, len(branch_options)))]

    orbits = _orbits(base_mask.shape, symmetry)
    removable_orbits = [
        orbit
        for orbit in orbits
        if any(base_mask[r, c] for r, c in orbit) and len(orbit) > 1
    ]
    add_candidates = np.argwhere(_dilate(patch) & ~patch)

    if branch in ("remove", "mixed") and not removable_orbits:
        return None
    if branch in ("add", "mixed") and add_candidates.size == 0:
        return None

    if branch in ("remove", "mixed"):
        orbit = removable_orbits[int(_RNG.integers(0, len(removable_orbits)))]
        cell = orbit[int(_RNG.integers(0, len(orbit)))]
        patch[cell[0] + 1, cell[1] + 1] = False

    if branch in ("add", "mixed"):
        ar, ac = map(int, add_candidates[int(_RNG.integers(0, len(add_candidates)))])
        patch[ar, ac] = True

    return _crop(patch)


def _place_patches(
    canvas_shape: tuple[int, int], marker_row: int, patches: list[np.ndarray]
) -> list[tuple[int, int]] | None:
    h, w = canvas_shape
    occupied = np.zeros(canvas_shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        max_r0 = marker_row - ph - 1
        if max_r0 < 0 or pw > w:
            return None

        choices = [
            (r0, c0)
            for r0 in range(max_r0 + 1)
            for c0 in range(w - pw + 1)
            if not (_dilate(occupied)[r0 : r0 + ph, c0 : c0 + pw] & patch).any()
        ]
        if not choices:
            return None
        r0, c0 = choices[int(_RNG.integers(0, len(choices)))]
        placements[idx] = (r0, c0)
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch

    return [p for p in placements if p is not None]


def _build_expected(
    grid_shape: tuple[int, int],
    bg: int,
    marker_row: int,
    markers: list[tuple[int, int]],
    objects: list[dict[str, object]],
) -> np.ndarray:
    repaired_objects = [
        {"color": int(obj["color"]), "mask": np.asarray(obj["base"], dtype=bool), "area": int(np.asarray(obj["base"]).sum())}
        for obj in objects
    ]
    return _render_output(grid_shape, bg, marker_row, markers, repaired_objects)


def _gen_attempt() -> np.ndarray | None:
    n_obj = int(_RNG.integers(*OBJ_COUNT_RANGE))
    height = int(_RNG.integers(*GRID_H_RANGE))
    width = MARKER_STRIDE * (n_obj - 1) + 2 * MARKER_MARGIN + 1
    marker_row = height - 1

    bg = int(_RNG.integers(0, len(PALETTE)))
    usable_colors = [c for c in PALETTE if c != bg]
    colors = list(map(int, _RNG.choice(usable_colors, size=n_obj, replace=False)))

    symmetry_list = [HORIZONTAL, VERTICAL, ROT90]
    while len(symmetry_list) < n_obj:
        symmetry_list.append(SYMMETRIES[int(_RNG.integers(0, len(SYMMETRIES)))])
    _RNG.shuffle(symmetry_list)

    objects: list[dict[str, object]] = []
    areas: set[int] = set()
    for color, symmetry in zip(colors, symmetry_list):
        base = _generate_symmetric_mask(symmetry)
        if base is None:
            return None
        area = int(base.sum())
        if area in areas:
            return None
        noisy = _noise_patch(base, symmetry)
        if noisy is None:
            return None
        areas.add(area)
        objects.append({"color": color, "symmetry": symmetry, "base": base, "noisy": noisy})

    markers = [(MARKER_MARGIN + i * MARKER_STRIDE, color) for i, color in enumerate(colors)]
    patches = [np.asarray(obj["noisy"], dtype=bool) for obj in objects]
    placements = _place_patches((height, width), marker_row, patches)
    if placements is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for (r0, c0), obj in zip(placements, objects):
        noisy = np.asarray(obj["noisy"], dtype=bool)
        color = int(obj["color"])
        grid[r0 : r0 + noisy.shape[0], c0 : c0 + noisy.shape[1]][noisy] = color

    for col, color in markers:
        grid[marker_row, col] = color

    expected = _build_expected(grid.shape, bg, marker_row, markers, objects)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
