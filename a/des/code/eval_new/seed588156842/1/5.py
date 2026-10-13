from __future__ import annotations

import random

import numpy as np


SYMMETRIES = ("vertical", "horizontal", "rot90")
BRANCH_COUNTER = 0
GEN_ATTEMPTS = 500
MASK_ATTEMPTS = 1200
DAMAGE_ATTEMPTS = 300
COLOR_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _symmetry_flags(arr: np.ndarray) -> dict[str, bool]:
    return {
        "vertical": bool(np.array_equal(arr, np.fliplr(arr))),
        "horizontal": bool(np.array_equal(arr, np.flipud(arr))),
        "rot90": bool(arr.shape[0] == arr.shape[1] and np.array_equal(arr, np.rot90(arr))),
    }


def _detect_symmetry(sample: np.ndarray) -> str:
    flags = _symmetry_flags(sample)
    kinds = [name for name, ok in flags.items() if ok]
    assert len(kinds) == 1, "repair sample must reveal exactly one symmetry type"
    return kinds[0]


def _orbit(symmetry: str, r: int, c: int, h: int, w: int) -> tuple[tuple[int, int], ...]:
    if symmetry == "vertical":
        return tuple(sorted({(r, c), (r, w - 1 - c)}))
    if symmetry == "horizontal":
        return tuple(sorted({(r, c), (h - 1 - r, c)}))
    assert symmetry == "rot90"
    orbit = []
    rr, cc = r, c
    for _ in range(4):
        orbit.append((rr, cc))
        rr, cc = cc, w - 1 - rr
    return tuple(sorted(set(orbit)))


def _repair_target(crop: np.ndarray, bg: int, structure: int, repair: int, symmetry: str) -> np.ndarray:
    h, w = crop.shape
    out = crop.copy()
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r, c in np.argwhere(crop == structure):
        orbit = _orbit(symmetry, int(r), int(c), h, w)
        if orbit in seen:
            continue
        seen.add(orbit)
        values = [int(crop[rr, cc]) for rr, cc in orbit]
        assert all(v in (bg, structure) for v in values), "target may only contain bg and structure color"
        if not any(v == structure for v in values):
            continue
        for rr, cc in orbit:
            if out[rr, cc] == bg:
                out[rr, cc] = repair
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a damaged symmetric target using the symmetry type and repair color encoded by a small sample.

    Input structure:
    - The background is the most frequent color.
    - One small connected sample on the left is a perfect 2-color pattern whose exact colors obey one symmetry: vertical mirror, horizontal mirror, or 90-degree rotation.
    - One larger connected target on the right uses only the sample's structure color on the background, but some symmetric counterpart pixels have been erased to background.

    Rule:
    - Isolate the small sample and the large target from the two foreground components.
    - Read the symmetry type from the sample's exact 2-color arrangement, and identify the repair color as the sample color not used by the target.
    - For every target structure pixel, close its symmetry orbit under that same symmetry; any missing orbit position that is background becomes the repair color.
    - Existing target structure pixels stay unchanged, and the sample itself stays unchanged.

    Key insight:
    - The left pattern is not a shape template for copying; it is a legend that encodes which symmetry test to apply, while its second color tells you how to paint only the repaired pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    comps = _components(g != bg)
    assert len(comps) == 2, "expected exactly one sample component and one target component"

    info = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        info.append({"coords": coords, "bbox": (r0, c0, r1, c1), "shape": (h, w)})

    sample_candidates = [item for item in info if 3 <= item["shape"][0] <= 5 and 3 <= item["shape"][1] <= 5]
    target_candidates = [item for item in info if 8 <= item["shape"][0] <= 15 and 8 <= item["shape"][1] <= 15]
    assert len(sample_candidates) == 1 and len(target_candidates) == 1
    sample = sample_candidates[0]
    target = target_candidates[0]
    assert sample["bbox"][3] < target["bbox"][1], "sample must lie to the left of the target"

    sr0, sc0, sr1, sc1 = sample["bbox"]
    tr0, tc0, tr1, tc1 = target["bbox"]
    sample_crop = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    target_crop = g[tr0 : tr1 + 1, tc0 : tc1 + 1]

    sample_colors = {int(v) for v in np.unique(sample_crop) if int(v) != bg}
    target_colors = {int(v) for v in np.unique(target_crop) if int(v) != bg}
    assert len(sample_colors) == 2, "sample must use exactly two non-background colors"
    assert len(target_colors) == 1, "target input must use exactly one non-background color"

    structure = next(iter(target_colors))
    assert structure in sample_colors
    repair = next(iter(sample_colors - {structure}))
    symmetry = _detect_symmetry(sample_crop)

    out = g.copy()
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = _repair_target(target_crop, bg, structure, repair, symmetry)
    return out


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    path = [start]
    while (r, c) != end:
        steps = []
        if r < end[0]:
            steps.append((r + 1, c))
        elif r > end[0]:
            steps.append((r - 1, c))
        if c < end[1]:
            steps.append((r, c + 1))
        elif c > end[1]:
            steps.append((r, c - 1))
        r, c = random.choice(steps)
        path.append((r, c))
    return path


def _vertical_area(rep: set[tuple[int, int]], width: int) -> int:
    axis = width // 2
    return sum(1 if c == axis else 2 for _r, c in rep)


def _horizontal_area(rep: set[tuple[int, int]], height: int) -> int:
    axis = height // 2
    return sum(1 if r == axis else 2 for r, _c in rep)


def _rot90_area(rep: set[tuple[int, int]], size: int) -> int:
    center = size // 2
    return sum(1 if (r, c) == (center, center) else 4 for r, c in rep)


def _symmetrize_vertical(rep: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    full = np.zeros((h, w), dtype=bool)
    for r, c in rep:
        full[r, c] = True
        full[r, w - 1 - c] = True
    return full


def _symmetrize_horizontal(rep: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    full = np.zeros((h, w), dtype=bool)
    for r, c in rep:
        full[r, c] = True
        full[h - 1 - r, c] = True
    return full


def _symmetrize_rot90(rep: set[tuple[int, int]], size: int) -> np.ndarray:
    full = np.zeros((size, size), dtype=bool)
    for r, c in rep:
        for rr, cc in _orbit("rot90", r, c, size, size):
            full[rr, cc] = True
    return full


def _valid_unique_mask(mask: np.ndarray, symmetry: str) -> bool:
    flags = _symmetry_flags(mask)
    if symmetry == "vertical":
        return flags == {"vertical": True, "horizontal": False, "rot90": False}
    if symmetry == "horizontal":
        return flags == {"vertical": False, "horizontal": True, "rot90": False}
    return bool(flags["rot90"] and not flags["vertical"] and not flags["horizontal"])


def _generate_vertical_mask(h: int, w: int, frac_lo: float, frac_hi: float) -> np.ndarray | None:
    axis = w // 2
    for _ in range(MASK_ATTEMPTS):
        seed = (random.randrange(h), axis)
        anchors = [seed, (0, random.randrange(axis + 1)), (h - 1, random.randrange(axis + 1)), (random.randrange(h), 0)]
        rep: set[tuple[int, int]] = set()
        for anchor in anchors:
            rep.update(_random_path(seed, anchor))

        lo = max(_vertical_area(rep, w), int(round(h * w * frac_lo)))
        hi = max(lo, int(round(h * w * frac_hi)))
        target_area = random.randint(lo, hi)

        while _vertical_area(rep, w) < target_area:
            frontier = []
            for r, c in list(rep):
                for rr, cc in _neighbors4(r, c, h, axis + 1):
                    if (rr, cc) not in rep:
                        frontier.append((rr, cc))
            if not frontier:
                break
            rep.add(random.choice(frontier))

        full = _symmetrize_vertical(rep, h, w)
        if not _valid_unique_mask(full, "vertical"):
            continue
        if not _is_connected(full):
            continue
        if not (np.any(full[0]) and np.any(full[-1]) and np.any(full[:, 0]) and np.any(full[:, -1])):
            continue
        return full
    return None


def _generate_horizontal_mask(h: int, w: int, frac_lo: float, frac_hi: float) -> np.ndarray | None:
    axis = h // 2
    for _ in range(MASK_ATTEMPTS):
        seed = (axis, random.randrange(w))
        anchors = [seed, (0, random.randrange(w)), (random.randrange(axis + 1), 0), (random.randrange(axis + 1), w - 1)]
        rep: set[tuple[int, int]] = set()
        for anchor in anchors:
            rep.update(_random_path(seed, anchor))

        lo = max(_horizontal_area(rep, h), int(round(h * w * frac_lo)))
        hi = max(lo, int(round(h * w * frac_hi)))
        target_area = random.randint(lo, hi)

        while _horizontal_area(rep, h) < target_area:
            frontier = []
            for r, c in list(rep):
                for rr, cc in _neighbors4(r, c, axis + 1, w):
                    if (rr, cc) not in rep:
                        frontier.append((rr, cc))
            if not frontier:
                break
            rep.add(random.choice(frontier))

        full = _symmetrize_horizontal(rep, h, w)
        if not _valid_unique_mask(full, "horizontal"):
            continue
        if not _is_connected(full):
            continue
        if not (np.any(full[0]) and np.any(full[-1]) and np.any(full[:, 0]) and np.any(full[:, -1])):
            continue
        return full
    return None


def _generate_rot90_mask(size: int, frac_lo: float, frac_hi: float) -> np.ndarray | None:
    center = size // 2
    domain = {(center, center)}
    for r in range(center):
        for c in range(center, size):
            domain.add((r, c))

    for _ in range(MASK_ATTEMPTS * 3):
        seed = (center, center)
        anchors = [
            (0, random.randrange(center, size)),
            (random.randrange(center), size - 1),
            (random.randrange(center), random.randrange(center + 1, size)),
        ]
        rep: set[tuple[int, int]] = {seed}
        for anchor in anchors:
            for cell in _random_path(seed, anchor):
                if cell in domain:
                    rep.add(cell)
        rep.add((center - 1, center))

        lo = max(_rot90_area(rep, size), int(round(size * size * frac_lo)))
        hi = max(lo, int(round(size * size * frac_hi)))
        target_area = random.randint(lo, hi)

        while _rot90_area(rep, size) < target_area:
            frontier = []
            for r, c in list(rep):
                for rr, cc in _neighbors4(r, c, size, size):
                    if (rr, cc) in domain and (rr, cc) not in rep:
                        frontier.append((rr, cc))
            if not frontier:
                break
            rep.add(random.choice(frontier))

        full = _symmetrize_rot90(rep, size)
        if not _valid_unique_mask(full, "rot90"):
            continue
        if not _is_connected(full):
            continue
        if not (np.any(full[0]) and np.any(full[-1]) and np.any(full[:, 0]) and np.any(full[:, -1])):
            continue
        return full
    return None


def _generate_mask(symmetry: str, small: bool) -> np.ndarray | None:
    if symmetry == "vertical":
        h = random.randint(3, 5) if small else random.randint(8, 15)
        w = 5 if small else random.choice((9, 11, 13, 15))
        return _generate_vertical_mask(h, w, 0.35 if small else 0.28, 0.72 if small else 0.56)
    if symmetry == "horizontal":
        h = 5 if small else random.choice((9, 11, 13, 15))
        w = random.randint(3, 5) if small else random.randint(8, 15)
        return _generate_horizontal_mask(h, w, 0.35 if small else 0.28, 0.72 if small else 0.56)
    assert symmetry == "rot90"
    size = 5 if small else random.choice((9, 11, 13, 15))
    return _generate_rot90_mask(size, 0.28 if small else 0.2, 0.52 if small else 0.42)


def _mask_orbits(mask: np.ndarray, symmetry: str) -> list[tuple[tuple[int, int], ...]]:
    h, w = mask.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r, c in coords:
        if (r, c) in seen:
            continue
        orbit = _orbit(symmetry, r, c, h, w)
        assert all(mask[rr, cc] for rr, cc in orbit)
        seen.update(orbit)
        orbits.append(orbit)
    return orbits


def _color_sample(mask: np.ndarray, symmetry: str, bg: int, structure: int, repair: int) -> np.ndarray | None:
    orbits = _mask_orbits(mask, symmetry)
    if len(orbits) < 2:
        return None

    for _ in range(COLOR_ATTEMPTS):
        repair_count = random.randint(1, len(orbits) - 1)
        repair_orbits = set(random.sample(range(len(orbits)), repair_count))
        sample = np.full(mask.shape, bg, dtype=int)
        for idx, orbit in enumerate(orbits):
            color = repair if idx in repair_orbits else structure
            for r, c in orbit:
                sample[r, c] = color
        flags = _symmetry_flags(sample)
        if flags.get(symmetry, False) and sum(flags.values()) == 1:
            return sample
    return None


def _damage_target(mask: np.ndarray, symmetry: str) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    h, w = mask.shape
    orbits = _mask_orbits(mask, symmetry)
    candidates = [
        orbit
        for orbit in orbits
        if len(orbit) > 1 and all(0 < r < h - 1 and 0 < c < w - 1 for r, c in orbit)
    ]
    if not candidates:
        return None

    for _ in range(DAMAGE_ATTEMPTS):
        count = random.randint(1, min(len(candidates), max(2, len(candidates) // 2)))
        chosen = random.sample(candidates, count)
        removed: set[tuple[int, int]] = set()
        for orbit in chosen:
            if symmetry in {"vertical", "horizontal"}:
                removed.add(random.choice(orbit))
            else:
                remove_count = random.randint(1, len(orbit) - 1)
                removed.update(random.sample(list(orbit), remove_count))

        damaged = mask.copy()
        for r, c in removed:
            damaged[r, c] = False
        if _is_connected(damaged):
            return damaged, removed
    return None


def _render_target(mask: np.ndarray, damaged: np.ndarray, bg: int, structure: int, repair: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(mask.shape, bg, dtype=int)
    out = np.full(mask.shape, bg, dtype=int)
    inp[damaged] = structure
    out[damaged] = structure
    out[mask & ~damaged] = repair
    return inp, out


def _next_symmetry() -> str:
    global BRANCH_COUNTER
    symmetry = SYMMETRIES[BRANCH_COUNTER % len(SYMMETRIES)]
    BRANCH_COUNTER += 1
    return symmetry


def _generate_attempt(symmetry: str) -> tuple[np.ndarray, np.ndarray] | None:
    colors = random.sample(range(10), 3)
    bg, structure, repair = colors

    sample_mask = _generate_mask(symmetry, small=True)
    target_mask = _generate_mask(symmetry, small=False)
    if sample_mask is None or target_mask is None:
        return None

    sample = _color_sample(sample_mask, symmetry, bg, structure, repair)
    damaged = _damage_target(target_mask, symmetry)
    if sample is None or damaged is None:
        return None
    target_in, target_out = _render_target(target_mask, damaged[0], bg, structure, repair)

    sh, sw = sample.shape
    th, tw = target_in.shape
    min_h = max(sh, th, sh + th)
    min_w = max(sw + tw + 1, 2 * sw, 2 * tw)
    if min_h > 30 or min_w > 30:
        return None

    width_choices = [value for value in range(min_w, 31) if value % 2 == 0]
    if not width_choices:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        h = random.randint(min_h, 30)
        w = random.choice(width_choices)
        mid = w // 2
        if mid - sw < 0:
            continue
        if w - tw < mid:
            continue

        sr0 = random.randint(0, h - sh)
        tr0 = random.randint(0, h - th)
        sc0 = random.randint(0, mid - sw)
        tc0 = random.randint(mid, w - tw)

        if sc0 + sw == tc0:
            separated_rows = sr0 + sh <= tr0 or tr0 + th <= sr0
            if not separated_rows:
                continue

        inp = np.full((h, w), bg, dtype=int)
        out = np.full((h, w), bg, dtype=int)
        inp[sr0 : sr0 + sh, sc0 : sc0 + sw] = sample
        out[sr0 : sr0 + sh, sc0 : sc0 + sw] = sample
        inp[tr0 : tr0 + th, tc0 : tc0 + tw] = target_in
        out[tr0 : tr0 + th, tc0 : tc0 + tw] = target_out

        if _background_color(inp) != bg:
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp, out
    return None


def generate_input() -> np.ndarray:
    symmetry = _next_symmetry()
    for _ in range(GEN_ATTEMPTS):
        pair = _generate_attempt(symmetry)
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate valid symmetric-repair input")
