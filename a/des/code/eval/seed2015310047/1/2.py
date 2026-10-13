from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 22
GRID_MAX = 30
STRIPE_COUNT_RANGE = (2, 4)
TEMPLATE_COUNT_RANGE = (2, 4)
BAND_SIZE_RANGE = (5, 7)
GEN_ATTEMPTS = 3000
PLACE_ATTEMPTS = 120
SHAPE_ATTEMPTS = 200


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


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
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == w - 1)
    )


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches = True
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches:
                holes += 1
    return holes


def _canonical_rot_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append((int(rot.shape[0]), int(rot.shape[1]), rot.tobytes()))
    return min(variants)


def _match_rotation(mask: np.ndarray, target: np.ndarray) -> int | None:
    for k in range(4):
        rot = np.rot90(mask, k)
        if rot.shape == target.shape and np.array_equal(rot, target):
            return int(k)
    return None


def _detect_stripes(g: np.ndarray) -> tuple[dict[int, int], dict[int, int], np.ndarray]:
    h, w = g.shape
    row_stripes: dict[int, int] = {}
    col_stripes: dict[int, int] = {}
    for r in range(h):
        if np.all(g[r, :] == g[r, 0]):
            color = int(g[r, 0])
            if color not in (BLACK, GREY):
                row_stripes[int(r)] = color
    for c in range(w):
        if np.all(g[:, c] == g[0, c]):
            color = int(g[0, c])
            if color not in (BLACK, GREY):
                col_stripes[int(c)] = color
    stripe_mask = np.zeros_like(g, dtype=bool)
    if row_stripes:
        stripe_mask[list(row_stripes.keys()), :] = True
    if col_stripes:
        stripe_mask[:, list(col_stripes.keys())] = True
    return row_stripes, col_stripes, stripe_mask


def _extract_templates(g: np.ndarray, stripe_mask: np.ndarray) -> list[dict]:
    grey_mask = (g == GREY) & (~stripe_mask)
    templates: list[dict] = []
    for coords in _components(grey_mask):
        mask, (r0, c0, r1, c1) = _tight_mask(coords)
        h, w = mask.shape
        if h < 5 or w < 5:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]

        top = [c for c in range(1, w - 1) if int(sub[0, c]) == BLACK]
        bottom = [c for c in range(1, w - 1) if int(sub[h - 1, c]) == BLACK]
        left = [r for r in range(1, h - 1) if int(sub[r, 0]) == BLACK]
        right = [r for r in range(1, h - 1) if int(sub[r, w - 1]) == BLACK]
        counts = [len(top), len(bottom), len(left), len(right)]
        if counts.count(1) != 1 or counts.count(0) != 3:
            continue
        if top:
            direction = "up"
        elif bottom:
            direction = "down"
        elif left:
            direction = "left"
        else:
            direction = "right"

        black_comps = _components(sub == BLACK)
        holes = [comp for comp in black_comps if not _touches_border(comp, h, w)]
        if len(holes) != 1:
            continue

        hole_rel = holes[0]
        hole_mask, _ = _tight_mask(hole_rel)
        hole_abs = hole_rel + np.array([[r0, c0]])
        templates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "direction": direction,
                "hole_mask": hole_mask,
                "hole_abs": hole_abs,
            }
        )
    assert templates, "expected at least one grey template"
    return templates


def _extract_fragments(g: np.ndarray, stripe_mask: np.ndarray) -> list[dict]:
    frag_mask = (g != BLACK) & (g != GREY) & (~stripe_mask)
    fragments: list[dict] = []
    for coords in _components(frag_mask):
        mask, box = _tight_mask(coords)
        fragments.append({"coords": coords, "mask": mask, "bbox": box})
    assert fragments, "expected at least one colored fragment"
    return fragments


def _nearest_stripe_color(
    template: dict, row_stripes: dict[int, int], col_stripes: dict[int, int]
) -> int:
    r0, c0, r1, c1 = template["bbox"]
    direction = template["direction"]
    if direction == "up":
        rows = [r for r in row_stripes if r < r0]
        assert rows
        return int(row_stripes[max(rows)])
    if direction == "down":
        rows = [r for r in row_stripes if r > r1]
        assert rows
        return int(row_stripes[min(rows)])
    if direction == "left":
        cols = [c for c in col_stripes if c < c0]
        assert cols
        return int(col_stripes[max(cols)])
    cols = [c for c in col_stripes if c > c1]
    assert cols
    return int(col_stripes[min(cols)])


def _match_templates_to_fragments(templates: list[dict], fragments: list[dict]) -> dict[int, tuple[int, int]]:
    candidates: dict[int, list[tuple[int, int]]] = {}
    for ti, template in enumerate(templates):
        options: list[tuple[int, int]] = []
        for fi, fragment in enumerate(fragments):
            rot = _match_rotation(fragment["mask"], template["hole_mask"])
            if rot is not None:
                options.append((fi, rot))
        assert options, "every template must have a matching fragment"
        candidates[int(ti)] = options

    order = sorted(range(len(templates)), key=lambda i: len(candidates[i]))
    assignment: dict[int, tuple[int, int]] = {}
    used: set[int] = set()

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True
        ti = order[idx]
        for fi, rot in candidates[ti]:
            if fi in used:
                continue
            used.add(fi)
            assignment[ti] = (fi, rot)
            if backtrack(idx + 1):
                return True
            used.remove(fi)
            assignment.pop(ti, None)
        return False

    assert backtrack(0), "template-fragment matching must be unambiguous"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull scattered fragments back into grey templates and recolor them from the nearest stripe.
    """
    g = np.asarray(grid, dtype=int)
    row_stripes, col_stripes, stripe_mask = _detect_stripes(g)
    templates = _extract_templates(g, stripe_mask)
    fragments = _extract_fragments(g, stripe_mask)
    matches = _match_templates_to_fragments(templates, fragments)

    out = g.copy()
    for ti, (fi, _rot) in matches.items():
        frag = fragments[fi]
        out[frag["coords"][:, 0], frag["coords"][:, 1]] = BLACK
        fill = _nearest_stripe_color(templates[ti], row_stripes, col_stripes)
        hole_abs = templates[ti]["hole_abs"]
        out[hole_abs[:, 0], hole_abs[:, 1]] = fill
    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], h, w))
    while len(cells) < area:
        choices = [p for p in frontier if p not in cells]
        if not choices:
            return None
        rr, cc = random.choice(choices)
        cells.add((rr, cc))
        frontier.update(_neighbors4(rr, cc, h, w))
    coords = np.array(sorted(cells), dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _random_shape() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = random.randint(2, 4)
        box_w = random.randint(2, 4)
        max_area = min(box_h * box_w - 1, 7)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        mask = _random_connected_mask(box_h, box_w, area)
        if mask is None:
            continue
        coords = np.argwhere(mask)
        tight, _ = _tight_mask(coords)
        if tight.shape[0] < 2 or tight.shape[1] < 2:
            continue
        if bool(np.all(tight)):
            continue
        if _hole_count(tight) != 0:
            continue
        return tight
    return None


def _unique_shapes(n: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for _ in range(SHAPE_ATTEMPTS * 2):
        if len(shapes) == n:
            return shapes
        shape = _random_shape()
        if shape is None:
            continue
        key = _canonical_rot_key(shape)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(shape)
    return None


def _build_template(shape: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray]:
    hole_h, hole_w = shape.shape
    temp_h_lo = max(5, hole_h + 2)
    temp_h_hi = max(temp_h_lo + 1, hole_h + 5)
    temp_w_lo = max(5, hole_w + 2)
    temp_w_hi = max(temp_w_lo + 1, hole_w + 5)

    for _ in range(PLACE_ATTEMPTS):
        temp_h = random.randint(temp_h_lo, temp_h_hi)
        temp_w = random.randint(temp_w_lo, temp_w_hi)
        sprite = np.full((temp_h, temp_w), GREY, dtype=int)

        hole_r = random.randint(1, temp_h - hole_h - 1)
        hole_c = random.randint(1, temp_w - hole_w - 1)
        hole_mask = np.zeros((temp_h, temp_w), dtype=bool)
        hole_mask[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w] = shape

        if direction == "up":
            choices = [(0, c) for c in range(1, temp_w - 1) if not hole_mask[1, c]]
        elif direction == "down":
            choices = [(temp_h - 1, c) for c in range(1, temp_w - 1) if not hole_mask[temp_h - 2, c]]
        elif direction == "left":
            choices = [(r, 0) for r in range(1, temp_h - 1) if not hole_mask[r, 1]]
        else:
            choices = [(r, temp_w - 1) for r in range(1, temp_h - 1) if not hole_mask[r, temp_w - 2]]
        if not choices:
            continue

        gap_r, gap_c = random.choice(choices)
        sprite[hole_mask] = BLACK
        sprite[gap_r, gap_c] = BLACK
        return sprite, np.argwhere(hole_mask)
    raise AssertionError("failed to build a valid template")


def _sample_bands(count: int) -> tuple[list[int], int] | None:
    for _ in range(PLACE_ATTEMPTS):
        bands = [random.randint(*BAND_SIZE_RANGE) for _ in range(count)]
        total = sum(bands) + count - 1
        if GRID_MIN <= total <= GRID_MAX:
            return bands, int(total)
    return None


def _band_ranges(bands: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    ranges: list[tuple[int, int]] = []
    stripes: list[int] = []
    cursor = 0
    for i, size in enumerate(bands):
        ranges.append((cursor, cursor + size - 1))
        cursor += size
        if i + 1 < len(bands):
            stripes.append(cursor)
            cursor += 1
    return ranges, stripes


def _rect_is_free(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    h0, w0 = occupied.shape
    rr0 = max(0, r - pad)
    rr1 = min(h0, r + h + pad)
    cc0 = max(0, c - pad)
    cc1 = min(w0, c + w + pad)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _place_box(
    occupied: np.ndarray,
    box_h: int,
    box_w: int,
    family: str,
    bands: list[tuple[int, int]],
    allowed_bands: list[int],
) -> tuple[int, int, int] | None:
    h, w = occupied.shape
    band_ids = allowed_bands[:]
    random.shuffle(band_ids)
    for band_id in band_ids:
        start, end = bands[band_id]
        if family == "horizontal":
            top_lo = max(1, start)
            top_hi = min(h - 1 - box_h, end - box_h + 1)
            left_lo = 1
            left_hi = w - 1 - box_w
        else:
            top_lo = 1
            top_hi = h - 1 - box_h
            left_lo = max(1, start)
            left_hi = min(w - 1 - box_w, end - box_w + 1)
        if top_lo > top_hi or left_lo > left_hi:
            continue
        for _ in range(PLACE_ATTEMPTS):
            r = random.randint(top_lo, top_hi)
            c = random.randint(left_lo, left_hi)
            if _rect_is_free(occupied, r, c, box_h, box_w):
                return int(r), int(c), int(band_id)
    return None


def _balanced_directions(n: int, family: str) -> list[str]:
    if family == "horizontal":
        a, b = "up", "down"
    else:
        a, b = "left", "right"
    dirs = [a] * (n // 2) + [b] * (n // 2)
    if n % 2:
        dirs.append(random.choice([a, b]))
    random.shuffle(dirs)
    return dirs


def _expected_output(grid: np.ndarray, templates: list[dict], fragments: list[dict]) -> np.ndarray:
    out = grid.copy()
    for fragment in fragments:
        coords = fragment["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK
    for template in templates:
        hole_abs = template["hole_abs"]
        out[hole_abs[:, 0], hole_abs[:, 1]] = int(template["fill_color"])
    return out


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    template_count = random.randint(*TEMPLATE_COUNT_RANGE)
    stripe_count = random.randint(*STRIPE_COUNT_RANGE)
    family = random.choice(["horizontal", "vertical"])

    sampled = _sample_bands(stripe_count + 1)
    if sampled is None:
        return None
    band_sizes, band_axis_len = sampled
    band_ranges, stripe_positions = _band_ranges(band_sizes)

    if family == "horizontal":
        h = band_axis_len
        w = random.randint(max(GRID_MIN, 24), GRID_MAX)
    else:
        h = random.randint(max(GRID_MIN, 24), GRID_MAX)
        w = band_axis_len

    total_colors = stripe_count + template_count
    if total_colors > len(COLORS):
        return None
    chosen_colors = random.sample(COLORS, total_colors)
    stripe_colors = chosen_colors[:stripe_count]
    fragment_colors = chosen_colors[stripe_count:]
    shapes = _unique_shapes(template_count)
    if shapes is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    if family == "horizontal":
        row_stripes = dict(zip(stripe_positions, stripe_colors, strict=False))
        col_stripes: dict[int, int] = {}
        for row, color in row_stripes.items():
            grid[row, :] = int(color)
            occupied[row, :] = True
    else:
        row_stripes = {}
        col_stripes = dict(zip(stripe_positions, stripe_colors, strict=False))
        for col, color in col_stripes.items():
            grid[:, col] = int(color)
            occupied[:, col] = True

    directions = _balanced_directions(template_count, family)
    templates: list[dict] = []
    fragment_specs: list[dict] = []

    for idx, shape in enumerate(shapes):
        direction = directions[idx]
        sprite, hole_rel = _build_template(shape, direction)
        temp_h, temp_w = sprite.shape
        if family == "horizontal":
            if direction == "up":
                allowed = [i for i in range(1, len(band_ranges)) if band_ranges[i][1] - band_ranges[i][0] + 1 >= temp_h]
            else:
                allowed = [i for i in range(0, len(band_ranges) - 1) if band_ranges[i][1] - band_ranges[i][0] + 1 >= temp_h]
        else:
            if direction == "left":
                allowed = [i for i in range(1, len(band_ranges)) if band_ranges[i][1] - band_ranges[i][0] + 1 >= temp_w]
            else:
                allowed = [i for i in range(0, len(band_ranges) - 1) if band_ranges[i][1] - band_ranges[i][0] + 1 >= temp_w]
        placed = _place_box(occupied, temp_h, temp_w, family, band_ranges, allowed)
        if placed is None:
            return None
        top, left, band_id = placed
        grid[top : top + temp_h, left : left + temp_w] = sprite
        occupied[top : top + temp_h, left : left + temp_w] = True

        if direction == "up":
            fill_color = int(row_stripes[stripe_positions[band_id - 1]])
        elif direction == "down":
            fill_color = int(row_stripes[stripe_positions[band_id]])
        elif direction == "left":
            fill_color = int(col_stripes[stripe_positions[band_id - 1]])
        else:
            fill_color = int(col_stripes[stripe_positions[band_id]])

        hole_abs = hole_rel + np.array([[top, left]])
        templates.append(
            {
                "bbox": (top, left, top + temp_h - 1, left + temp_w - 1),
                "direction": direction,
                "hole_abs": hole_abs,
                "fill_color": fill_color,
            }
        )
        fragment_specs.append({"shape": shape, "color": fragment_colors[idx]})

    rotations = [random.randrange(4) for _ in range(template_count)]
    if template_count > 0 and all(rot == 0 for rot in rotations):
        rotations[random.randrange(template_count)] = random.choice([1, 2, 3])

    fragments: list[dict] = []
    for idx, spec in enumerate(fragment_specs):
        frag_mask = np.rot90(spec["shape"], rotations[idx])
        frag_h, frag_w = frag_mask.shape
        allowed = [
            i
            for i, (start, end) in enumerate(band_ranges)
            if (end - start + 1) >= (frag_h if family == "horizontal" else frag_w)
        ]
        placed = _place_box(occupied, frag_h, frag_w, family, band_ranges, allowed)
        if placed is None:
            return None
        top, left, _band_id = placed
        patch = np.zeros((frag_h, frag_w), dtype=int)
        patch[frag_mask] = int(spec["color"])
        grid[top : top + frag_h, left : left + frag_w] = patch
        occupied[top : top + frag_h, left : left + frag_w] = True
        coords = np.argwhere(frag_mask) + np.array([[top, left]])
        fragments.append({"coords": coords})

    expected = _expected_output(grid, templates, fragments)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_example()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
