from __future__ import annotations

import random

import numpy as np


BG_COLOR = 8
FRAME_COLOR = 5
COLOR_POOL = (0, 1, 2, 3, 4, 6, 7, 9)

TEMPLATE_SIZE_CHOICES = (3, 4, 5)
TEMPLATE_COUNT_RANGE = (3, 5)
SCALE_CHOICES = (2, 3)
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
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
    out: list[np.ndarray] = []
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
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rect_border_mask(h: int, w: int) -> np.ndarray:
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _find_frame(
    grid: np.ndarray,
) -> tuple[int, int, tuple[int, int, int, int]]:
    bg = _most_common_color(grid)
    candidates: list[tuple[int, int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            local = np.zeros((h, w), dtype=bool)
            local[comp[:, 0] - r0, comp[:, 1] - c0] = True
            if not np.array_equal(local, _rect_border_mask(h, w)):
                continue
            inside = grid[r0 + 1 : r1, c0 + 1 : c1]
            if not np.any(inside != bg):
                continue
            candidates.append((h * w, len(comp), color, (r0, c0, r1, c1)))
    assert candidates, "expected one rectangular frame"
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, frame_color, frame_bbox = candidates[-1]
    return bg, int(frame_color), frame_bbox


def _extract_template_mask(
    grid: np.ndarray,
    color: int,
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    fr0, fc0, fr1, fc1 = frame_bbox
    outside: list[np.ndarray] = []
    for comp in _components(grid == color):
        inside_frame = np.all(
            (comp[:, 0] > fr0)
            & (comp[:, 0] < fr1)
            & (comp[:, 1] > fc0)
            & (comp[:, 1] < fc1)
        )
        if inside_frame:
            continue
        outside.append(comp)
    assert len(outside) == 1, "expected one external template for the seed color"
    coords = outside[0]
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the frame with the uniquely matching rotated-and-scaled template."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg, _frame_color, (fr0, fc0, fr1, fc1) = _find_frame(g)
    inner = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    seed_colors = {int(v) for v in np.unique(inner) if int(v) != bg}
    assert len(seed_colors) == 1, "frame interior must contain one seed color"
    seed_color = next(iter(seed_colors))
    seed_mask = inner == seed_color
    assert seed_mask.any()

    template_mask = _extract_template_mask(g, seed_color, (fr0, fc0, fr1, fc1))
    th, tw = template_mask.shape
    ih, iw = inner.shape
    assert ih % th == 0 and iw % tw == 0
    scale_h = ih // th
    scale_w = iw // tw
    assert scale_h == scale_w and scale_h >= 1
    scale = int(scale_h)

    matches: list[np.ndarray] = []
    for rot in range(4):
        rotated = np.rot90(template_mask, rot)
        scaled = _scale_mask(rotated, scale)
        if scaled.shape != inner.shape:
            continue
        if np.all(scaled[seed_mask]):
            matches.append(scaled)
    assert len(matches) == 1, "seed fragments must imply a unique rotation"

    out = g.copy()
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = bg
    out_inner = out[fr0 + 1 : fr1, fc0 + 1 : fc1]
    out_inner[matches[0]] = seed_color
    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _rotations_distinct(mask: np.ndarray) -> bool:
    return len({np.rot90(mask, k).tobytes() for k in range(4)}) == 4


def _canonical_rotation_key(mask: np.ndarray) -> bytes:
    return min(np.rot90(mask, k).tobytes() for k in range(4))


def _random_shortest_path(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = start
    path: list[tuple[int, int]] = []
    while (r, c) != end:
        options = []
        if r < end[0]:
            options.append((r + 1, c))
        elif r > end[0]:
            options.append((r - 1, c))
        if c < end[1]:
            options.append((r, c + 1))
        elif c > end[1]:
            options.append((r, c - 1))
        r, c = random.choice(options)
        path.append((r, c))
    return path


def _build_template_mask(size: int) -> np.ndarray:
    low_area = max(size + 1, int(np.ceil(size * size * 0.4)))
    high_area = max(low_area, int(np.floor(size * size * 0.75)))
    for _ in range(300):
        anchors = [
            (0, random.randrange(size)),
            (size - 1, random.randrange(size)),
            (random.randrange(size), 0),
            (random.randrange(size), size - 1),
        ]
        random.shuffle(anchors)
        cells = {anchors[0]}
        for anchor in anchors[1:]:
            start = random.choice(tuple(cells))
            cells.update(_random_shortest_path(start, anchor))
        if len(cells) > high_area:
            continue
        target_area = random.randint(max(low_area, len(cells)), high_area)
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, size, size):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue
        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _touches_all_sides(mask):
            continue
        if not _rotations_distinct(mask):
            continue
        if np.array_equal(mask, _rect_border_mask(size, size)):
            continue
        return mask
    raise RuntimeError("failed to build a non-symmetric template")


def _choose_seed_mask(variants: list[np.ndarray], target_idx: int) -> np.ndarray:
    target = variants[target_idx]
    target_cells = [tuple(map(int, cell)) for cell in np.argwhere(target)]
    remaining = {idx for idx in range(4) if idx != target_idx}
    chosen: set[tuple[int, int]] = set()

    while remaining:
        best_cell = None
        best_cover: set[int] = set()
        for cell in target_cells:
            cover = {idx for idx in remaining if not bool(variants[idx][cell])}
            if len(cover) > len(best_cover):
                best_cell = cell
                best_cover = cover
        assert best_cell is not None and best_cover
        chosen.add(best_cell)
        remaining -= best_cover

    leftovers = [cell for cell in target_cells if cell not in chosen]
    random.shuffle(leftovers)
    if len(chosen) == 1 and leftovers:
        chosen.add(leftovers.pop())
    extra = random.randint(0, min(4, len(leftovers)))
    for cell in leftovers[:extra]:
        chosen.add(cell)

    seeds = np.zeros_like(target, dtype=bool)
    for r, c in chosen:
        seeds[r, c] = True

    assert sum(int(np.all(variant[seeds])) for variant in variants) == 1
    return seeds


def _place_box(
    occupied: np.ndarray, box_h: int, box_w: int, margin: int = 1
) -> tuple[int, int] | None:
    h, w = occupied.shape
    r_start = margin
    c_start = margin
    r_stop = h - box_h - margin
    c_stop = w - box_w - margin
    if r_stop < r_start or c_stop < c_start:
        return None
    candidates: list[tuple[int, int]] = []
    for r0 in range(r_start, r_stop + 1):
        for c0 in range(c_start, c_stop + 1):
            rr0 = r0 - margin
            cc0 = c0 - margin
            rr1 = r0 + box_h + margin
            cc1 = c0 + box_w + margin
            if not occupied[rr0:rr1, cc0:cc1].any():
                candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    occupied[r0 - margin : r0 + box_h + margin, c0 - margin : c0 + box_w + margin] = True
    return int(r0), int(c0)


def _draw_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    mask = patch != bg
    sub[mask] = patch[mask]


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    template_count = random.randint(*TEMPLATE_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, template_count)

    templates = []
    seen_keys: set[bytes] = set()
    for color in colors:
        for _ in range(200):
            size = random.choice(TEMPLATE_SIZE_CHOICES)
            mask = _build_template_mask(size)
            key = _canonical_rotation_key(mask)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            display_mask = np.rot90(mask, random.randrange(4))
            templates.append({"color": int(color), "mask": display_mask})
            break
        else:
            raise RuntimeError("failed to sample distinct templates")

    target_idx = random.randrange(template_count)
    target_template = templates[target_idx]
    target_color = int(target_template["color"])
    target_mask = np.asarray(target_template["mask"], dtype=bool)
    scale = random.choice(SCALE_CHOICES)
    rotation = random.randrange(4)

    variants = [_scale_mask(np.rot90(target_mask, rot), scale) for rot in range(4)]
    frame_shape = variants[rotation]
    seed_mask = _choose_seed_mask(variants, rotation)

    inner_h, inner_w = frame_shape.shape
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    largest = max(frame_h, frame_w, *(tpl["mask"].shape[0] for tpl in templates))
    area_est = frame_h * frame_w + sum(
        int(tpl["mask"].shape[0]) * int(tpl["mask"].shape[1]) for tpl in templates
    )
    base_dim = max(largest + 3, int(np.ceil(np.sqrt(area_est))) + 7)
    grid_h = random.randint(base_dim, base_dim + 8)
    grid_w = random.randint(base_dim, base_dim + 8)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: dict[str, tuple[int, int]] = {}

    frame_pos = _place_box(occupied, frame_h, frame_w, margin=1)
    if frame_pos is None:
        raise RuntimeError("failed to place frame")
    placements["frame"] = frame_pos

    template_order = sorted(
        range(template_count),
        key=lambda idx: int(np.prod(templates[idx]["mask"].shape)),
        reverse=True,
    )
    for idx in template_order:
        mask = np.asarray(templates[idx]["mask"], dtype=bool)
        pos = _place_box(occupied, mask.shape[0], mask.shape[1], margin=1)
        if pos is None:
            raise RuntimeError("failed to place template")
        placements[f"template_{idx}"] = pos

    inp = np.full((grid_h, grid_w), BG_COLOR, dtype=int)

    fr0, fc0 = placements["frame"]
    inp[fr0, fc0 : fc0 + frame_w] = FRAME_COLOR
    inp[fr0 + frame_h - 1, fc0 : fc0 + frame_w] = FRAME_COLOR
    inp[fr0 : fr0 + frame_h, fc0] = FRAME_COLOR
    inp[fr0 : fr0 + frame_h, fc0 + frame_w - 1] = FRAME_COLOR

    seed_rows, seed_cols = np.where(seed_mask)
    inp[fr0 + 1 + seed_rows, fc0 + 1 + seed_cols] = target_color

    for idx, tpl in enumerate(templates):
        mask = np.asarray(tpl["mask"], dtype=bool)
        patch = np.full(mask.shape, BG_COLOR, dtype=int)
        patch[mask] = int(tpl["color"])
        top, left = placements[f"template_{idx}"]
        _draw_patch(inp, patch, top, left, BG_COLOR)

    out = inp.copy()
    out[fr0 + 1 : fr0 + 1 + inner_h, fc0 + 1 : fc0 + 1 + inner_w] = BG_COLOR
    out_inner = out[fr0 + 1 : fr0 + 1 + inner_h, fc0 + 1 : fc0 + 1 + inner_w]
    out_inner[frame_shape] = target_color

    actual = main(inp)
    assert np.array_equal(actual, out)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, _ = _generate_case()
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
