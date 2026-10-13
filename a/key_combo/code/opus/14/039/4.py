import numpy as np


BACKGROUND = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

ODD_SIZES = (3, 5, 7, 9, 11, 13, 15)
MIN_SIZE_FOR_COUNT = {2: 3, 3: 5, 4: 7, 5: 9, 6: 11, 7: 13}
OTHER_DIM_FLOOR = {2: 2, 3: 3, 4: 5, 5: 6, 6: 8, 7: 10}
RETRY_LIMIT = 400


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _analyze_grid(grid: np.ndarray) -> dict[int, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    info: dict[int, dict[str, object]] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BACKGROUND):
        mask = g == color
        coords = np.argwhere(mask)
        comps = _components_mask(mask)
        bbox = _bbox_from_coords(coords)
        h = bbox[2] - bbox[0] + 1
        w = bbox[3] - bbox[1] + 1
        info[color] = {
            "color": color,
            "coords": coords,
            "components": comps,
            "areas": [int(comp.shape[0]) for comp in comps],
            "count": len(comps),
            "bbox": bbox,
            "height": h,
            "width": w,
            "size": max(h, w),
            "pixels": int(coords.shape[0]),
        }
    return info


def _draw_ring(out: np.ndarray, size: int, color: int) -> None:
    n = out.shape[0]
    inset = (n - size) // 2
    lo = inset
    hi = inset + size - 1
    out[lo, lo : hi + 1] = color
    out[hi, lo : hi + 1] = color
    out[lo : hi + 1, lo] = color
    out[lo : hi + 1, hi] = color


def _build_output_from_layers(layers: list[tuple[int, int]]) -> np.ndarray:
    n = max(size for size, _color in layers)
    out = np.full((n, n), BACKGROUND, dtype=int)
    for size, color in sorted(layers, reverse=True):
        _draw_ring(out, int(size), int(color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild centered concentric square rings from the surviving fragment-spread colors.

    Input structure:
    - The input is a black-background grid with 4-5 non-black colors.
    - Each color appears as several separated monochromatic 4-connected fragments.
    - One color has the unique largest fragment count and is therefore the distractor.

    Rule:
    - Count the 4-connected components of every non-black color.
    - Drop the color with the largest component count.
    - For each remaining color, compute the smallest axis-aligned bbox covering all
      of its pixels and take `max(height, width)` as that color's spread size.
    - Sort the remaining colors by spread size from largest to smallest, then draw
      centered square rings with those side lengths inside a square output whose side
      equals the largest remaining spread size.

    Key insight:
    - Input positions are only used to measure each surviving color's global spatial
      spread; after filtering out the highest-frequency distractor, the fragments are
      discarded and replaced by concentric rings ordered by bbox-derived size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    info = _analyze_grid(g)
    assert 4 <= len(info) <= 5, "expected 4-5 foreground colors"

    counts = [int(data["count"]) for data in info.values()]
    assert len(set(counts)) == len(counts), "component counts must be unique"
    drop_color = max(info, key=lambda color: int(info[color]["count"]))

    layers: list[tuple[int, int]] = []
    for color, data in info.items():
        if color == drop_color:
            continue
        size = int(data["size"])
        layers.append((size, color))

    assert 3 <= len(layers) <= 4, "expected 3-4 retained colors"
    sizes = [size for size, _color in layers]
    assert all(size % 2 == 1 for size in sizes), "retained spread sizes must be odd"
    assert len(set(sizes)) == len(sizes), "retained spread sizes must be distinct"

    return _build_output_from_layers(layers)


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _random_blob(rng: np.random.Generator, area: int, max_side: int) -> tuple[tuple[int, int], ...]:
    for _ in range(120):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            pick = frontier[int(rng.integers(len(frontier)))]
            cells.add(pick)
        norm = _normalize_cells(cells)
        max_r = max(r for r, _c in norm)
        max_c = max(c for _r, c in norm)
        if max(max_r + 1, max_c + 1) <= max_side:
            return norm
    raise RuntimeError("failed to sample a compact blob")


def _can_place_cells(
    cells: tuple[tuple[int, int], ...],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for dr, dc in DIRS8:
            if (r + dr, c + dc) in occupied:
                return False
        if (r, c) in occupied:
            return False
    return True


def _translate_cells(
    cells: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], ...]:
    return tuple((r + dr, c + dc) for r, c in cells)


def _candidate_offsets_for_requirement(
    blob: tuple[tuple[int, int], ...],
    h: int,
    w: int,
    requirement: frozenset[str],
) -> list[tuple[int, int]]:
    max_r = max(r for r, _c in blob)
    max_c = max(c for _r, c in blob)

    if "top" in requirement:
        row_choices = [0]
    elif "bottom" in requirement:
        row_choices = [h - 1 - max_r]
    else:
        row_choices = list(range(h - max_r))

    if "left" in requirement:
        col_choices = [0]
    elif "right" in requirement:
        col_choices = [w - 1 - max_c]
    else:
        col_choices = list(range(w - max_c))

    out = []
    for r0 in row_choices:
        for c0 in col_choices:
            out.append((int(r0), int(c0)))
    return out


def _area_bounds_for_component(count: int, anchored: bool) -> tuple[int, int]:
    if anchored:
        return 2, 4 if count >= 4 else 5
    if count >= 6:
        return 2, 4
    if count >= 4:
        return 2, 5
    return 2, 6


def _make_local_patch(
    rng: np.random.Generator,
    height: int,
    width: int,
    count: int,
) -> np.ndarray:
    assert count >= 2
    diagonal_requirements = [
        [frozenset(("top", "left")), frozenset(("bottom", "right"))],
        [frozenset(("top", "right")), frozenset(("bottom", "left"))],
    ]

    for _ in range(240):
        occupied: set[tuple[int, int]] = set()
        parts: list[tuple[tuple[int, int], ...]] = []
        requirements = diagonal_requirements[int(rng.integers(2))] + [frozenset()] * (count - 2)
        ok = True

        for index, requirement in enumerate(requirements):
            anchored = index < 2
            area_lo, area_hi = _area_bounds_for_component(count, anchored)
            placed = False
            for _ in range(180):
                area = int(rng.integers(area_lo, area_hi + 1))
                max_side = 3 if count >= 5 else 4
                blob = _random_blob(rng, area, max_side=max_side)
                offsets = _candidate_offsets_for_requirement(blob, height, width, requirement)
                rng.shuffle(offsets)
                for r0, c0 in offsets:
                    shifted = _translate_cells(blob, r0, c0)
                    if _can_place_cells(shifted, occupied, height, width):
                        occupied.update(shifted)
                        parts.append(shifted)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        patch = np.zeros((height, width), dtype=int)
        for part in parts:
            for r, c in part:
                patch[r, c] = 1

        comps = _components_mask(patch == 1)
        if len(comps) != count:
            continue
        if any(comp.shape[0] < 2 or comp.shape[0] > 8 for comp in comps):
            continue
        bbox = _bbox_from_coords(np.argwhere(patch == 1))
        if bbox != (0, 0, height - 1, width - 1):
            continue
        return patch

    raise RuntimeError("failed to build a local patch")


def _sample_retained_sizes(rng: np.random.Generator, retained_count: int) -> list[int]:
    need_tight = bool(rng.integers(2))
    need_gap = bool(rng.integers(2))
    if not need_tight and not need_gap:
        need_tight = True
        need_gap = True

    for _ in range(200):
        chosen = rng.choice(np.array(ODD_SIZES, dtype=int), size=retained_count, replace=False)
        sizes = sorted((int(v) for v in chosen), reverse=True)
        diffs = [sizes[i] - sizes[i + 1] for i in range(len(sizes) - 1)]
        if need_tight and 2 not in diffs:
            continue
        if need_gap and not any(diff > 2 for diff in diffs):
            continue
        return sizes
    raise RuntimeError("failed to sample retained sizes")


def _assign_counts_to_sizes(
    rng: np.random.Generator,
    sizes: list[int],
    counts: tuple[int, ...],
) -> list[int]:
    for _ in range(200):
        perm = list(counts)
        rng.shuffle(perm)
        if all(size >= MIN_SIZE_FOR_COUNT[count] for size, count in zip(sizes, perm)):
            return perm
    raise RuntimeError("failed to assign counts to sizes")


def _choose_dims(
    rng: np.random.Generator,
    size: int,
    count: int,
) -> tuple[int, int]:
    other_floor = OTHER_DIM_FLOOR[count]
    options: list[tuple[int, int]] = []
    for other in range(other_floor, size + 1):
        options.append((size, other))
        if other != size:
            options.append((other, size))
    return options[int(rng.integers(len(options)))]


def _place_patches_on_canvas(
    rng: np.random.Generator,
    patches: list[dict[str, object]],
    side: int,
) -> np.ndarray | None:
    grid = np.full((side, side), BACKGROUND, dtype=int)
    occupied: set[tuple[int, int]] = set()

    order = sorted(
        patches,
        key=lambda patch: (
            int(patch["height"]) * int(patch["width"]),
            int(patch["count"]),
        ),
        reverse=True,
    )

    for patch in order:
        height = int(patch["height"])
        width = int(patch["width"])
        color = int(patch["color"])
        coords = tuple((int(r), int(c)) for r, c in patch["coords"])

        placements = [
            (r0, c0)
            for r0 in range(side - height + 1)
            for c0 in range(side - width + 1)
        ]
        rng.shuffle(placements)

        placed = False
        for r0, c0 in placements:
            shifted = _translate_cells(coords, r0, c0)
            if not _can_place_cells(shifted, occupied, side, side):
                continue
            for r, c in shifted:
                occupied.add((r, c))
                grid[r, c] = color
            placed = True
            break

        if not placed:
            return None

    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        info = _analyze_grid(grid)
        if len(info) not in (4, 5):
            return False

        counts = [int(data["count"]) for data in info.values()]
        if len(set(counts)) != len(counts):
            return False
        sorted_counts = sorted(counts)
        if sorted_counts[-1] - sorted_counts[-2] < 2:
            return False

        for data in info.values():
            if any(area < 2 or area > 8 for area in data["areas"]):
                return False

        drop_color = max(info, key=lambda color: int(info[color]["count"]))
        retained_items = [(color, data) for color, data in info.items() if color != drop_color]
        if len(retained_items) not in (3, 4):
            return False

        retained_sizes = [int(data["size"]) for _color, data in retained_items]
        if len(set(retained_sizes)) != len(retained_sizes):
            return False
        if any(size % 2 == 0 or size > 15 for size in retained_sizes):
            return False

        all_sizes = [int(data["size"]) for data in info.values()]
        if len(set(all_sizes)) != len(all_sizes):
            return False

        if not any(int(data["height"]) != int(data["width"]) for _color, data in retained_items):
            return False

        size_rank = [
            color
            for color, _data in sorted(
                retained_items,
                key=lambda item: int(item[1]["size"]),
                reverse=True,
            )
        ]
        count_rank = [
            color
            for color, _data in sorted(
                retained_items,
                key=lambda item: int(item[1]["count"]),
                reverse=True,
            )
        ]
        if size_rank == count_rank:
            return False

        has_pixel_inversion = False
        for i, (color_a, data_a) in enumerate(retained_items):
            for color_b, data_b in retained_items[i + 1 :]:
                size_a = int(data_a["size"])
                size_b = int(data_b["size"])
                pix_a = int(data_a["pixels"])
                pix_b = int(data_b["pixels"])
                if (size_a > size_b and pix_a < pix_b) or (size_b > size_a and pix_b < pix_a):
                    has_pixel_inversion = True
        if not has_pixel_inversion:
            return False

        expected = _build_output_from_layers(
            [(int(data["size"]), int(color)) for color, data in retained_items]
        )
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False

        unfiltered = _build_output_from_layers(
            [(int(data["size"]), int(color)) for color, data in info.items()]
        )
        if np.array_equal(unfiltered, actual):
            return False
    except Exception:
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    retained_count = 3 if float(rng.random()) < 0.55 else 4
    retained_sizes = _sample_retained_sizes(rng, retained_count)

    retained_counts = (2, 3, 4) if retained_count == 3 else (2, 3, 4, 5)
    distractor_count = 6 if retained_count == 3 else 7
    counts_by_size = _assign_counts_to_sizes(rng, retained_sizes, retained_counts)

    distractor_candidates = [
        size
        for size in ODD_SIZES
        if size not in retained_sizes and size >= MIN_SIZE_FOR_COUNT[distractor_count]
    ]
    if not distractor_candidates:
        return None
    distractor_size = int(rng.choice(np.array(distractor_candidates, dtype=int)))

    color_count = retained_count + 1
    colors = [int(v) for v in rng.choice(np.arange(1, 10), size=color_count, replace=False)]

    specs: list[dict[str, int]] = []
    for size, count in zip(retained_sizes, counts_by_size):
        height, width = _choose_dims(rng, int(size), int(count))
        specs.append(
            {
                "size": int(size),
                "count": int(count),
                "height": int(height),
                "width": int(width),
            }
        )

    d_height, d_width = _choose_dims(rng, distractor_size, distractor_count)
    specs.append(
        {
            "size": int(distractor_size),
            "count": int(distractor_count),
            "height": int(d_height),
            "width": int(d_width),
        }
    )

    retained_rectangles = sum(int(spec["height"]) != int(spec["width"]) for spec in specs[:-1])
    if retained_rectangles == 0:
        return None

    rng.shuffle(colors)
    for spec, color in zip(specs, colors):
        spec["color"] = int(color)

    patches: list[dict[str, object]] = []
    for spec in specs:
        try:
            local = _make_local_patch(
                rng,
                height=int(spec["height"]),
                width=int(spec["width"]),
                count=int(spec["count"]),
            )
        except RuntimeError:
            return None
        coords = np.argwhere(local == 1)
        patches.append(
            {
                "color": int(spec["color"]),
                "count": int(spec["count"]),
                "size": int(spec["size"]),
                "height": int(spec["height"]),
                "width": int(spec["width"]),
                "coords": tuple((int(r), int(c)) for r, c in coords),
            }
        )

    max_size = max(int(spec["size"]) for spec in specs)
    side = max(18, max_size + int(rng.integers(4, 8)) + (color_count - 4) * 2)
    side = min(side, 28)
    grid = _place_patches_on_canvas(rng, patches, side)
    if grid is None:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
