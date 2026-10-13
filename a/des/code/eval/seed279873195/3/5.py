from __future__ import annotations

import numpy as np


BACKGROUND = 0
YELLOW = 4
GREY = 5
MAGENTA = 6
TEMPLATE_COLORS = (1, 2, 3, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRECTION_DELTAS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIRECTION_ROTATIONS = {
    "up": 0,
    "right": 3,
    "down": 2,
    "left": 1,
}
_RNG = np.random.default_rng()
GEN_MAX_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask[:0, :0]
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    cropped = _crop_mask(mask.astype(bool))
    return cropped.shape, tuple(cropped.astype(int).ravel())


def _rotation_signatures(mask: np.ndarray) -> tuple[tuple[tuple[int, int], tuple[int, ...]], ...]:
    return tuple(_mask_signature(np.rot90(mask, k)) for k in range(4))


def _canonical_rotation_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return min(_rotation_signatures(mask))


def _find_separator_col(grid: np.ndarray) -> int:
    cols = np.where(np.all(grid == GREY, axis=0))[0]
    if len(cols) == 0:
        raise ValueError("missing grey separator column")
    return int(cols[0])


def _extract_templates(template_region: np.ndarray) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for comp in _components(template_region != BACKGROUND):
        values = [int(template_region[r, c]) for r, c in comp]
        if values.count(MAGENTA) != 1:
            raise ValueError("each template must contain exactly one magenta anchor")
        body_colors = {value for value in values if value != MAGENTA}
        if len(body_colors) != 1:
            raise ValueError("each template must use one body color")
        color = next(iter(body_colors))
        r0, c0, r1, c1 = _bbox(comp)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in comp:
            patch[r - r0, c - c0] = int(template_region[r, c])
        templates[color] = patch
    if not templates:
        raise ValueError("template region is empty")
    return templates


def _extract_seeds(grid: np.ndarray, separator_col: int) -> list[tuple[int, int, int, str]]:
    h, w = grid.shape
    seeds: list[tuple[int, int, int, str]] = []
    for r in range(h):
        for c in range(separator_col + 1, w):
            color = int(grid[r, c])
            if color in (BACKGROUND, YELLOW, GREY, MAGENTA):
                continue
            directions = []
            for name, (dr, dc) in DIRECTION_DELTAS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == YELLOW:
                    directions.append(name)
            if len(directions) != 1:
                raise ValueError("each seed must have exactly one adjacent yellow marker")
            seeds.append((r, c, color, directions[0]))
    seeds.sort(key=lambda item: (item[0], item[1]))
    return seeds


def _rotated_patch(patch: np.ndarray, direction: str) -> np.ndarray:
    return np.rot90(patch, DIRECTION_ROTATIONS[direction])


def _stamp_patch(out: np.ndarray, patch: np.ndarray, anchor_row: int, anchor_col: int, color: int) -> None:
    anchor_pos = np.argwhere(patch == MAGENTA)
    if anchor_pos.shape != (1, 2):
        raise ValueError("rotated template must contain exactly one magenta anchor")
    ar, ac = map(int, anchor_pos[0])
    for rr, cc in np.argwhere(patch != BACKGROUND):
        tr = anchor_row + int(rr) - ar
        tc = anchor_col + int(cc) - ac
        if 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]:
            out[tr, tc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each seed pixel to its same-colored template, rotate that template from the
    yellow direction marker, and stamp it onto the canvas anchored at the seed.
    """
    g = np.asarray(grid, dtype=int)
    separator_col = _find_separator_col(g)
    templates = _extract_templates(g[:, :separator_col])
    seeds = _extract_seeds(g, separator_col)

    out = g.copy()
    out[out == YELLOW] = BACKGROUND
    for row, col, color, direction in seeds:
        patch = _rotated_patch(templates[color], direction)
        _stamp_patch(out, patch, row, col, color)
    return out


def _random_connected_mask(rng: np.random.Generator, height: int, width: int, target_area: int) -> np.ndarray:
    anchor = (height - 1, int(rng.integers(0, width)))
    cells = {anchor}
    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        frontier = list(dict.fromkeys(frontier))
        weights = []
        for nr, nc in frontier:
            weight = 1.0 + 2.5 * (height - 1 - nr) / max(1, height - 1)
            if nc != anchor[1]:
                weight += 0.5
            if nr == height - 1:
                weight *= 0.6
            touching = sum((nr + dr, nc + dc) in cells for dr, dc in NEIGHBORS4)
            if touching == 1:
                weight *= 1.25
            elif touching >= 3:
                weight *= 0.8
            weights.append(weight)
        probs = np.asarray(weights, dtype=float)
        probs /= probs.sum()
        choice = frontier[int(rng.choice(len(frontier), p=probs))]
        cells.add(choice)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _generate_template_shape(rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(200):
        height = int(rng.integers(3, 6))
        width = int(rng.integers(3, 6))
        area_low = max(5, (height * width) // 2 - 1)
        area_high = min(height * width - 1, area_low + 4)
        target_area = int(rng.integers(area_low, area_high + 1))
        mask = _random_connected_mask(rng, height, width, target_area)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if mask.sum() < 5:
            continue
        if len(set(_rotation_signatures(mask))) != 4:
            continue
        bottom_row = mask.shape[0] - 1
        bottom_cells = [(bottom_row, c) for c in range(mask.shape[1]) if mask[bottom_row, c]]
        if not bottom_cells or int(mask[:-1].sum()) < 3:
            continue
        candidates = []
        for r, c in bottom_cells:
            if any(
                0 <= r + dr < mask.shape[0]
                and 0 <= c + dc < mask.shape[1]
                and mask[r + dr, c + dc]
                and (dr != 0 or dc != 0)
                for dr, dc in NEIGHBORS4
            ):
                candidates.append((r, c))
        if not candidates:
            continue
        anchor = candidates[int(rng.integers(0, len(candidates)))]
        return mask, anchor
    raise RuntimeError("failed to generate a non-degenerate template shape")


def _build_template_patch(mask: np.ndarray, anchor: tuple[int, int], color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    patch[anchor] = MAGENTA
    return patch


def _template_variants(patch: np.ndarray) -> dict[str, dict[str, object]]:
    variants: dict[str, dict[str, object]] = {}
    for direction in DIRECTION_DELTAS:
        rotated = _rotated_patch(patch, direction)
        ar, ac = map(int, np.argwhere(rotated == MAGENTA)[0])
        rels = {(int(r) - ar, int(c) - ac) for r, c in np.argwhere(rotated != BACKGROUND)}
        variants[direction] = {"patch": rotated, "rels": rels}
    return variants


def _build_expected(
    grid: np.ndarray,
    templates: dict[int, np.ndarray],
    seeds: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    out[out == YELLOW] = BACKGROUND
    for seed in sorted(seeds, key=lambda item: (int(item["row"]), int(item["col"]))):
        patch = _rotated_patch(templates[int(seed["color"])], str(seed["direction"]))
        _stamp_patch(out, patch, int(seed["row"]), int(seed["col"]), int(seed["color"]))
    return out


def _find_seed_placement(
    rng: np.random.Generator,
    canvas_h: int,
    canvas_w: int,
    rels: set[tuple[int, int]],
    marker_delta: tuple[int, int],
    occupied_signal: set[tuple[int, int]],
    projected_union: set[tuple[int, int]],
    require_overlap: bool,
) -> tuple[int, int, int, int, set[tuple[int, int]]] | None:
    positions = [(r, c) for r in range(canvas_h) for c in range(canvas_w)]
    rng.shuffle(positions)
    for seed_r, seed_c in positions:
        marker_r = seed_r + marker_delta[0]
        marker_c = seed_c + marker_delta[1]
        if not (0 <= marker_r < canvas_h and 0 <= marker_c < canvas_w):
            continue
        if (seed_r, seed_c) in occupied_signal or (marker_r, marker_c) in occupied_signal:
            continue
        if any(abs(seed_r - r) + abs(seed_c - c) <= 1 for r, c in occupied_signal):
            continue
        if any(abs(marker_r - r) + abs(marker_c - c) <= 1 for r, c in occupied_signal):
            continue
        stamp_cells = {(seed_r + dr, seed_c + dc) for dr, dc in rels}
        if any(not (0 <= r < canvas_h and 0 <= c < canvas_w) for r, c in stamp_cells):
            continue
        overlap = len(stamp_cells & projected_union)
        if require_overlap:
            if overlap == 0 or overlap > max(2, len(stamp_cells) // 3):
                continue
        elif overlap != 0:
            continue
        return seed_r, seed_c, marker_r, marker_c, stamp_cells
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    template_count = int(rng.integers(2, 5))
    chosen_colors = list(rng.choice(TEMPLATE_COLORS, size=template_count, replace=False))

    templates: dict[int, np.ndarray] = {}
    variant_cache: dict[int, dict[str, dict[str, object]]] = {}
    used_shape_sigs: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for color in chosen_colors:
        for _ in range(120):
            mask, anchor = _generate_template_shape(rng)
            sig = _canonical_rotation_signature(mask)
            if sig in used_shape_sigs:
                continue
            used_shape_sigs.add(sig)
            patch = _build_template_patch(mask, anchor, color)
            templates[color] = patch
            variant_cache[color] = _template_variants(patch)
            break
        else:
            return None

    template_width = int(max(max(patch.shape[1] for patch in templates.values()) + 3, rng.integers(8, 12)))
    row = int(rng.integers(1, 3))
    template_positions: list[tuple[int, int, np.ndarray]] = []
    for color in chosen_colors:
        patch = templates[color]
        max_col = template_width - patch.shape[1] - 1
        if max_col < 1:
            return None
        col = int(rng.integers(1, max_col + 1))
        template_positions.append((row, col, patch))
        row += patch.shape[0] + int(rng.integers(2, 4))
    required_height = row + 1
    grid_h = int(max(required_height, rng.integers(20, 31)))
    canvas_w = int(rng.integers(15, 23))
    separator_col = template_width
    total_w = template_width + 1 + canvas_w

    grid = np.zeros((grid_h, total_w), dtype=int)
    grid[:, separator_col] = GREY
    for top, left, patch in template_positions:
        if top + patch.shape[0] > grid_h:
            return None
        grid[top : top + patch.shape[0], left : left + patch.shape[1]] = patch

    seed_count = int(rng.integers(max(4, template_count), 8))
    colors_for_seeds = chosen_colors.copy()
    while len(colors_for_seeds) < seed_count:
        colors_for_seeds.append(int(rng.choice(chosen_colors)))
    rng.shuffle(colors_for_seeds)

    directions = ["up", "right", "down", "left"]
    while len(directions) < seed_count:
        directions.append(str(rng.choice(("up", "right", "down", "left"))))
    rng.shuffle(directions)
    directions = directions[:seed_count]

    want_overlap = seed_count >= 5 and rng.random() < 0.35
    occupied_signal: set[tuple[int, int]] = set()
    projected_union: set[tuple[int, int]] = set()
    seeds: list[dict[str, object]] = []

    for idx, (color, direction) in enumerate(zip(colors_for_seeds, directions)):
        require_overlap = want_overlap and idx == seed_count - 1
        rels = set(variant_cache[int(color)][direction]["rels"])
        marker_delta = DIRECTION_DELTAS[direction]
        placement = _find_seed_placement(
            rng,
            grid_h,
            canvas_w,
            rels,
            marker_delta,
            occupied_signal,
            projected_union,
            require_overlap=require_overlap,
        )
        if placement is None:
            if require_overlap:
                placement = _find_seed_placement(
                    rng,
                    grid_h,
                    canvas_w,
                    rels,
                    marker_delta,
                    occupied_signal,
                    projected_union,
                    require_overlap=False,
                )
            if placement is None:
                return None
        seed_r, seed_c_local, marker_r, marker_c_local, stamp_cells = placement
        seed_c = separator_col + 1 + seed_c_local
        marker_c = separator_col + 1 + marker_c_local
        grid[seed_r, seed_c] = int(color)
        grid[marker_r, marker_c] = YELLOW
        occupied_signal.add((seed_r, seed_c_local))
        occupied_signal.add((marker_r, marker_c_local))
        projected_union |= stamp_cells
        seeds.append(
            {
                "row": seed_r,
                "col": seed_c,
                "color": int(color),
                "direction": direction,
            }
        )

    expected = _build_expected(grid, templates, seeds)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
