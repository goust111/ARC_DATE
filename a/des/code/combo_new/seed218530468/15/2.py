from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _extract_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rel = tuple(sorted((rr - r0, cc - c0) for rr, cc in coords))
            components.append(
                {
                    "coords": tuple(coords),
                    "rel": rel,
                    "area": len(coords),
                    "bbox": (r0, c0, r1, c1),
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                }
            )

    return components


def _marker_center(grid: np.ndarray) -> tuple[int, int]:
    comps = _extract_components(grid, MARKER)
    assert len(comps) == 1, "expected exactly one grey marker"
    marker = comps[0]
    assert marker["area"] == 3, "marker must have 3 pixels"

    coord_set = set(marker["coords"])
    degrees = []
    for cell in marker["coords"]:
        rr, cc = cell
        degree = sum((rr + dr, cc + dc) in coord_set for dr, dc in DIR4)
        degrees.append((degree, cell))

    best_degree = max(degree for degree, _cell in degrees)
    centers = [cell for degree, cell in degrees if degree == best_degree]
    assert best_degree == 2 and len(centers) == 1, "marker must have a unique center pixel"
    return centers[0]


def _can_tile_rectangle(fragments: list[dict], h: int, w: int) -> bool:
    placements: list[list[int]] = []
    for fragment in fragments:
        if fragment["h"] > h or fragment["w"] > w:
            return False

        masks: list[int] = []
        for r0 in range(h - fragment["h"] + 1):
            for c0 in range(w - fragment["w"] + 1):
                mask = 0
                for dr, dc in fragment["rel"]:
                    bit = (r0 + dr) * w + (c0 + dc)
                    mask |= 1 << bit
                masks.append(mask)
        if not masks:
            return False
        placements.append(masks)

    order = sorted(range(len(fragments)), key=lambda i: len(placements[i]))

    def search(k: int, used_mask: int) -> bool:
        if k == len(order):
            return True

        idx = order[k]
        for mask in placements[idx]:
            if mask & used_mask:
                continue
            if search(k + 1, used_mask | mask):
                return True
        return False

    return search(0, 0)


def _rectangle_candidates(fragments: list[dict]) -> list[tuple[int, int]]:
    area = sum(fragment["area"] for fragment in fragments)
    candidates: list[tuple[int, int]] = []
    for h in range(1, area + 1):
        if area % h != 0:
            continue
        w = area // h
        if _can_tile_rectangle(fragments, h, w):
            candidates.append((h, w))
    return candidates


def _solve_rectangle_dims(fragments: list[dict]) -> tuple[int, int]:
    candidates = _rectangle_candidates(fragments)
    assert len(candidates) == 1, f"expected a unique rectangle assembly, got {candidates}"
    return candidates[0]


def _draw_centered_rectangle(out: np.ndarray, center: tuple[int, int], h: int, w: int, color: int) -> None:
    r0 = center[0] - h // 2
    c0 = center[1] - w // 2
    assert 0 <= r0 <= out.shape[0] - h
    assert 0 <= c0 <= out.shape[1] - w
    out[r0 : r0 + h, c0 : c0 + w] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble each color's separated fragments into one rectangle, then stack the rectangles at the central marker.

    Input structure
    - The background is black.
    - Several non-grey foreground colors appear, and each color forms 2 or 3 separated 4-connected fragments.
    - A small grey 3-pixel marker indicates the common destination center.
    - For each foreground color, its fragments can be translated without rotation to tile one compact rectangle exactly.

    Rule
    - Ignore the grey marker after reading its center pixel.
    - For each foreground color, find the unique rectangle that its fragments can assemble into using translation only.
    - Redraw that solid rectangle centered on the marker location.
    - Stack all centered rectangles on the same canvas, painting them from largest pixel-count to smallest pixel-count so later smaller rectangles overwrite earlier larger ones.

    Key insight
    - The task has two linked steps: first infer each hidden solid rectangle from its scattered same-color puzzle pieces, then infer that the recovered rectangles are center-aligned and layered by total area rather than returned to their original locations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    center = _marker_center(g)
    colors = sorted({int(value) for value in g.ravel()} - {BG, MARKER})
    assert colors, "at least one foreground color is required"

    rectangles: list[dict] = []
    for color in colors:
        fragments = _extract_components(g, color)
        assert 2 <= len(fragments) <= 3, "each color must split into 2 or 3 fragments"
        rect_h, rect_w = _solve_rectangle_dims(fragments)
        area = sum(fragment["area"] for fragment in fragments)
        rectangles.append({"color": color, "h": rect_h, "w": rect_w, "area": area})

    areas = [record["area"] for record in rectangles]
    assert len(set(areas)) == len(areas), "stack order must be unambiguous"

    out = np.full_like(g, BG)
    for record in sorted(rectangles, key=lambda item: -item["area"]):
        _draw_centered_rectangle(out, center, record["h"], record["w"], record["color"])
    return out


def _random_composition(total: int, parts: int, minimum: int) -> list[int]:
    extra = total - parts * minimum
    assert extra >= 0
    cuts = sorted(random.sample(range(extra + parts - 1), parts - 1)) if parts > 1 else []
    values: list[int] = []
    prev = -1
    for cut in cuts + [extra + parts - 1]:
        values.append(cut - prev)
        prev = cut
    return [value + minimum - 1 for value in values]


def _grow_partition(h: int, w: int, parts: int) -> list[dict] | None:
    total = h * w
    minimum = 2 if total <= 12 else 3
    if total < parts * minimum:
        return None

    targets = _random_composition(total, parts, minimum)
    cells = [(r, c) for r in range(h) for c in range(w)]
    seeds = random.sample(cells, parts)

    labels = -np.ones((h, w), dtype=int)
    regions: list[set[tuple[int, int]]] = []
    frontiers: list[set[tuple[int, int]]] = []

    for label, (r, c) in enumerate(seeds):
        labels[r, c] = label
        regions.append({(r, c)})
        frontier: set[tuple[int, int]] = set()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                frontier.add((nr, nc))
        frontiers.append(frontier)

    while True:
        expandable = [
            label
            for label in range(parts)
            if len(regions[label]) < targets[label] and frontiers[label]
        ]
        if not expandable:
            break

        label = random.choice(expandable)
        cell = random.choice(tuple(frontiers[label]))
        rr, cc = cell
        labels[rr, cc] = label
        regions[label].add(cell)

        for other in range(parts):
            frontiers[other].discard(cell)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                frontiers[label].add((nr, nc))

    if np.any(labels == -1):
        return None
    if any(len(region) != target for region, target in zip(regions, targets, strict=False)):
        return None

    fragments: list[dict] = []
    for region in regions:
        rows = [r for r, _c in region]
        cols = [c for _r, c in region]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        rel = tuple(sorted((r - r0, c - c0) for r, c in region))
        fragments.append(
            {
                "coords": tuple(sorted(region)),
                "rel": rel,
                "area": len(region),
                "bbox": (r0, c0, r1, c1),
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
            }
        )

    return fragments


def _make_fragment_patch(fragment: dict, color: int) -> np.ndarray:
    patch = np.full((fragment["h"], fragment["w"]), BG, dtype=int)
    for dr, dc in fragment["rel"]:
        patch[dr, dc] = color
    return patch


def _reserve_marker_zone(occupied: np.ndarray, center: tuple[int, int]) -> None:
    h, w = occupied.shape
    for rr in range(max(0, center[0] - 1), min(h, center[0] + 2)):
        for cc in range(max(0, center[1] - 1), min(w, center[1] + 2)):
            occupied[rr, cc] = True


def _place_marker(grid: np.ndarray, center: tuple[int, int]) -> None:
    if random.random() < 0.5:
        offsets = [(-1, 0), (0, 0), (1, 0)]
    else:
        offsets = [(0, -1), (0, 0), (0, 1)]
    for dr, dc in offsets:
        grid[center[0] + dr, center[1] + dc] = MARKER


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    candidates: list[tuple[int, int]] = []

    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r0, c0))

    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    mask = patch != BG
    grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
    occupied[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True
    return True


def _expected_output(shape: tuple[int, int], center: tuple[int, int], rectangles: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for record in sorted(rectangles, key=lambda item: -item["area"]):
        _draw_centered_rectangle(out, center, record["h"], record["w"], record["color"])
    return out


def _sample_rectangles(grid_shape: tuple[int, int], color_count: int) -> list[dict] | None:
    gh, gw = grid_shape
    dims_pool = [
        (h, w)
        for h in (3, 5, 7)
        for w in (3, 5, 7)
        if h <= gh - 4 and w <= gw - 4
    ]
    random.shuffle(dims_pool)

    rectangles: list[dict] = []
    used_areas: set[int] = set()
    colors = random.sample(PALETTE, color_count)
    for color in colors:
        choices = [dims for dims in dims_pool if dims[0] * dims[1] not in used_areas]
        if not choices:
            return None
        h, w = random.choice(choices)
        used_areas.add(h * w)

        fragment_count = random.choice((2, 3))
        fragments = None
        for _ in range(80):
            maybe = _grow_partition(h, w, fragment_count)
            if maybe is None:
                continue
            if len(_rectangle_candidates(maybe)) != 1:
                continue
            fragments = maybe
            break
        if fragments is None:
            return None

        rectangles.append({"color": color, "h": h, "w": w, "area": h * w, "fragments": fragments})

    return rectangles


def _gen_attempt() -> np.ndarray | None:
    gh = random.randint(19, 29)
    gw = random.randint(19, 29)
    grid = np.full((gh, gw), BG, dtype=int)
    center = (gh // 2, gw // 2)

    color_count = random.choice((2, 3))
    rectangles = _sample_rectangles(grid.shape, color_count)
    if rectangles is None:
        return None

    occupied = np.zeros_like(grid, dtype=bool)
    _reserve_marker_zone(occupied, center)

    patches: list[np.ndarray] = []
    for record in rectangles:
        for fragment in record["fragments"]:
            patches.append(_make_fragment_patch(fragment, record["color"]))

    random.shuffle(patches)
    patches.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)

    for patch in patches:
        if not _place_patch(grid, occupied, patch):
            return None

    _place_marker(grid, center)

    expected = _expected_output(grid.shape, center, rectangles)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
