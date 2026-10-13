import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
TEMPLATE_COLOR = 9
GEN_RETRIES = 300
SHAPE_RETRIES = 80
PLACEMENT_RETRIES = 200


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    shifted = arr - arr.min(axis=0)
    return tuple(sorted(map(tuple, shifted.tolist())))


def _shape_to_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    arr = np.asarray(shape, dtype=int)
    mask = np.zeros((int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _shape_rotations(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    mask = _shape_to_mask(shape)
    rots: list[tuple[tuple[int, int], ...]] = []
    cur = mask
    for _ in range(4):
        rots.append(_normalize_coords(np.argwhere(cur)))
        cur = np.rot90(cur)
    return rots


def _canonical_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_shape_rotations(shape))


def _largest_monochrome_component(grid: np.ndarray, bg: int) -> dict:
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            area = int(len(coords))
            if best is None or area > best["area"]:
                best = {
                    "color": color,
                    "coords": coords,
                    "area": area,
                    "bbox": _bbox(coords),
                }
    assert best is not None, "expected at least one foreground component"
    return best


def _analyze_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    template = _largest_monochrome_component(g, bg)
    tr0, tc0, tr1, tc1 = template["bbox"]

    patch = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    notch_components = _components(patch != template["color"])
    assert notch_components, "template bbox must contain at least one notch"

    marker_cells: set[tuple[int, int]] = set()
    notches: list[dict] = []
    for rel_coords in notch_components:
        abs_coords = rel_coords + np.array([tr0, tc0], dtype=int)
        colored = [(int(r), int(c)) for r, c in abs_coords if int(g[r, c]) != bg]
        assert len(colored) == 1, "each notch must contain exactly one marker pixel"
        mr, mc = colored[0]
        marker_cells.add((mr, mc))
        notches.append(
            {
                "coords": abs_coords,
                "shape": _normalize_coords(abs_coords),
                "area": int(len(abs_coords)),
                "marker_color": int(g[mr, mc]),
            }
        )

    piece_mask = (g != bg) & (g != template["color"])
    for r, c in marker_cells:
        piece_mask[r, c] = False
    piece_components = _components(piece_mask)
    assert len(piece_components) == len(notches), "piece count must match notch count"

    pieces: list[dict] = []
    for coords in piece_components:
        shape = _normalize_coords(coords)
        pieces.append(
            {
                "coords": coords,
                "shape": shape,
                "area": int(len(coords)),
                "rotations": set(_shape_rotations(shape)),
            }
        )

    assignment: dict[int, int] = {}
    used_notches: set[int] = set()
    for p_idx, piece in enumerate(pieces):
        matches = [
            i
            for i, notch in enumerate(notches)
            if notch["area"] == piece["area"] and notch["shape"] in piece["rotations"]
        ]
        assert len(matches) == 1, "each piece must match exactly one notch up to rotation"
        notch_idx = matches[0]
        assert notch_idx not in used_notches, "shape matching must be one-to-one"
        used_notches.add(notch_idx)
        assignment[p_idx] = notch_idx

    assert len(used_notches) == len(notches), "all notches must be matched"
    return {
        "grid": g,
        "bg": bg,
        "template": template,
        "notches": notches,
        "pieces": pieces,
        "assignment": assignment,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill template notches by matching scattered pieces up to rotation,
    then recolor each inserted piece from the notch's hidden marker.
    """
    info = _analyze_instance(grid)
    out = info["grid"].copy()

    for piece in info["pieces"]:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = info["bg"]

    for piece_idx, notch_idx in info["assignment"].items():
        _ = piece_idx
        notch = info["notches"][notch_idx]
        coords = notch["coords"]
        out[coords[:, 0], coords[:, 1]] = notch["marker_color"]

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    empty = ~mask
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return bool(np.any(empty & ~seen))


def _random_base_notch_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 6))

        start_c = int(rng.integers(w))
        col = start_c
        cells = {(0, col)}
        for r in range(1, h):
            choices = [dc for dc in (-1, 0, 1) if 0 <= col + dc < w]
            col += int(rng.choice(choices))
            cells.add((r, col))

        max_extra = min(4, h * w - len(cells))
        target_area = len(cells) + int(rng.integers(max_extra + 1))
        while len(cells) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and nr < h - 1 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            frontier_list = sorted(frontier)
            cells.add(frontier_list[int(rng.integers(len(frontier_list)))])

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)

        if mask.shape[0] < 2 or int(mask.sum()) < 3:
            continue
        if int(mask[-1].sum()) != 1:
            continue
        if not bool(mask[0].any()):
            continue
        if _has_hole(mask):
            continue
        return mask

    raise RuntimeError("failed to generate a notch shape")


def _random_gap_partition(slack: int, count: int, rng: np.random.Generator) -> list[int]:
    gaps = [0] * count
    for _ in range(slack):
        gaps[int(rng.integers(count))] += 1
    return gaps


def _linear_requirement(lengths: list[int]) -> int:
    if not lengths:
        return 0
    return 2 + sum(lengths) + max(0, len(lengths) - 1)


def _build_notch_specs(rng: np.random.Generator) -> list[dict]:
    count = int(rng.integers(2, 5))
    specs: list[dict] = []
    seen_signatures: set[tuple[tuple[int, int], ...]] = set()
    sides = ("top", "bottom", "left", "right")
    side_to_k = {"top": 0, "left": 1, "bottom": 2, "right": 3}

    while len(specs) < count:
        base_mask = _random_base_notch_mask(rng)
        shape = _normalize_coords(np.argwhere(base_mask))
        signature = _canonical_shape(shape)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        marker_mask = np.zeros_like(base_mask, dtype=bool)
        deepest_col = int(np.flatnonzero(base_mask[-1])[0])
        marker_mask[-1, deepest_col] = True

        side = sides[int(rng.integers(4))]
        k = side_to_k[side]
        oriented_mask = np.rot90(base_mask, k)
        oriented_marker = np.rot90(marker_mask, k)

        specs.append(
            {
                "side": side,
                "mask": oriented_mask,
                "marker_mask": oriented_marker,
                "canonical": signature,
            }
        )

    return specs


def _assign_side_starts(lengths: list[int], total_length: int, rng: np.random.Generator) -> list[int]:
    if not lengths:
        return []
    base_total = sum(lengths) + len(lengths) - 1
    slack = total_length - 2 - base_total
    if slack < 0:
        raise ValueError("side too short for notch placement")
    extras = _random_gap_partition(slack, len(lengths) + 1, rng)
    starts: list[int] = []
    pos = 1 + extras[0]
    for i, length in enumerate(lengths):
        starts.append(pos)
        pos += length + 1 + extras[i + 1]
    return starts


def _relative_notch_cells(
    mask: np.ndarray,
    marker_mask: np.ndarray,
    side: str,
    start: int,
    template_h: int,
    template_w: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = mask.shape
    if side == "top":
        offset = np.array([0, start], dtype=int)
    elif side == "bottom":
        offset = np.array([template_h - h, start], dtype=int)
    elif side == "left":
        offset = np.array([start, 0], dtype=int)
    else:
        offset = np.array([start, template_w - w], dtype=int)

    coords = np.argwhere(mask) + offset
    marker = tuple((np.argwhere(marker_mask)[0] + offset).tolist())
    return coords, marker


def _non_touching(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    set_b = set(map(tuple, coords_b.tolist()))
    for r, c in coords_a.tolist():
        if (r, c) in set_b:
            return False
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in set_b:
                return False
    return True


def _build_template_layout(
    specs: list[dict], rng: np.random.Generator
) -> tuple[int, int, list[dict], np.ndarray]:
    by_side: dict[str, list[dict]] = {"top": [], "bottom": [], "left": [], "right": []}
    for spec in specs:
        by_side[spec["side"]].append(spec)

    top_lengths = [int(spec["mask"].shape[1]) for spec in by_side["top"]]
    bottom_lengths = [int(spec["mask"].shape[1]) for spec in by_side["bottom"]]
    left_lengths = [int(spec["mask"].shape[0]) for spec in by_side["left"]]
    right_lengths = [int(spec["mask"].shape[0]) for spec in by_side["right"]]

    top_depth = max((int(spec["mask"].shape[0]) for spec in by_side["top"]), default=0)
    bottom_depth = max((int(spec["mask"].shape[0]) for spec in by_side["bottom"]), default=0)
    left_depth = max((int(spec["mask"].shape[1]) for spec in by_side["left"]), default=0)
    right_depth = max((int(spec["mask"].shape[1]) for spec in by_side["right"]), default=0)

    min_h = max(
        8,
        _linear_requirement(left_lengths),
        _linear_requirement(right_lengths),
        top_depth + bottom_depth + 3,
    )
    min_w = max(
        8,
        _linear_requirement(top_lengths),
        _linear_requirement(bottom_lengths),
        left_depth + right_depth + 3,
    )

    template_h = min_h + int(rng.integers(0, 5))
    template_w = min_w + int(rng.integers(0, 5))

    layout: list[dict] = []
    for _ in range(PLACEMENT_RETRIES):
        current: list[dict] = []
        for side, axis_len in (
            ("top", template_w),
            ("bottom", template_w),
            ("left", template_h),
            ("right", template_h),
        ):
            side_specs = by_side[side][:]
            if not side_specs:
                continue
            order = rng.permutation(len(side_specs))
            ordered_specs = [side_specs[i] for i in order]
            lengths = [
                int(spec["mask"].shape[1] if side in ("top", "bottom") else spec["mask"].shape[0])
                for spec in ordered_specs
            ]
            starts = _assign_side_starts(lengths, axis_len, rng)
            for spec, start in zip(ordered_specs, starts):
                coords, marker = _relative_notch_cells(
                    spec["mask"], spec["marker_mask"], side, start, template_h, template_w
                )
                current.append({**spec, "coords": coords, "marker_rel": marker})

        ok = True
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                if not _non_touching(current[i]["coords"], current[j]["coords"]):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        template_mask = np.ones((template_h, template_w), dtype=bool)
        for spec in current:
            coords = spec["coords"]
            template_mask[coords[:, 0], coords[:, 1]] = False
        if len(_components(template_mask)) != 1:
            continue
        layout = current
        return template_h, template_w, layout, template_mask

    raise RuntimeError("failed to place notches on template")


def _can_place_piece(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords.tolist():
        if occupied[r, c]:
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _construct_expected(
    grid: np.ndarray,
    bg: int,
    piece_positions: list[np.ndarray],
    notch_positions: list[np.ndarray],
    marker_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for coords in piece_positions:
        out[coords[:, 0], coords[:, 1]] = bg
    for coords, color in zip(notch_positions, marker_colors):
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    specs = _build_notch_specs(rng)
    try:
        template_h, template_w, layout, template_mask = _build_template_layout(specs, rng)
    except (RuntimeError, ValueError):
        return None

    max_piece_h = max(int(spec["mask"].shape[0]) for spec in layout)
    max_piece_w = max(int(spec["mask"].shape[1]) for spec in layout)
    grid_h = max(18, template_h + max_piece_h + 7) + int(rng.integers(0, 5))
    grid_w = max(18, template_w + max_piece_w + 7) + int(rng.integers(0, 5))

    if grid_h - template_h < 4 or grid_w - template_w < 4:
        return None

    bg_choices = [c for c in range(10) if c != TEMPLATE_COLOR]
    bg = int(rng.choice(bg_choices))
    free_colors = [c for c in range(10) if c not in (bg, TEMPLATE_COLOR)]
    if len(free_colors) < len(layout) + 1:
        return None

    marker_colors = list(map(int, rng.choice(free_colors, size=len(layout), replace=False)))
    piece_color_pool = [c for c in free_colors if c not in marker_colors]
    piece_palette_size = 1 if len(piece_color_pool) == 1 or rng.random() < 0.5 else 2
    piece_palette = list(map(int, rng.choice(piece_color_pool, size=piece_palette_size, replace=False)))

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    top_margin = int(rng.integers(2, grid_h - template_h - 1))
    left_margin = int(rng.integers(2, grid_w - template_w - 1))

    template_coords = np.argwhere(template_mask) + np.array([top_margin, left_margin], dtype=int)
    grid[template_coords[:, 0], template_coords[:, 1]] = TEMPLATE_COLOR

    notch_positions: list[np.ndarray] = []
    marker_positions: list[tuple[int, int]] = []
    for spec, color in zip(layout, marker_colors):
        coords = spec["coords"] + np.array([top_margin, left_margin], dtype=int)
        mr, mc = spec["marker_rel"]
        marker_abs = (mr + top_margin, mc + left_margin)
        notch_positions.append(coords)
        marker_positions.append(marker_abs)
        grid[marker_abs] = color

    occupied = grid != bg
    piece_positions: list[np.ndarray] = []
    for spec in layout:
        rot_k = int(rng.integers(4))
        piece_mask = np.rot90(spec["mask"], rot_k)
        piece_coords_rel = np.argwhere(piece_mask)
        piece_color = piece_palette[int(rng.integers(len(piece_palette)))]

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            h, w = piece_mask.shape
            r0 = int(rng.integers(0, grid_h - h + 1))
            c0 = int(rng.integers(0, grid_w - w + 1))
            coords = piece_coords_rel + np.array([r0, c0], dtype=int)
            if not _can_place_piece(coords, occupied):
                continue
            grid[coords[:, 0], coords[:, 1]] = piece_color
            occupied[coords[:, 0], coords[:, 1]] = True
            piece_positions.append(coords)
            placed = True
            break
        if not placed:
            return None

    expected = _construct_expected(grid, bg, piece_positions, notch_positions, marker_colors)

    try:
        analyzed = _analyze_instance(grid)
    except AssertionError:
        return None

    if analyzed["template"]["color"] != TEMPLATE_COLOR:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if len(np.unique(grid)) < 4:
        return None
    if _background_color(grid) != bg:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
