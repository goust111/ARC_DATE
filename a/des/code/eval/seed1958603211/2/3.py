import random
from collections import deque

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_COLORS = tuple(range(10))
SEP_COLOR = 5
GEN_RETRIES = 400


def _mode_color(grid: np.ndarray) -> int:
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
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(r1), int(c0), int(c1)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    return _bbox_from_coords(coords)


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    out[r, c] = True
                    break
    return out


def _fill_outline(outline: np.ndarray) -> np.ndarray:
    h, w = outline.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not outline[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not outline[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not outline[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outline | (~outline & ~outside)


def _spans(size: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for idx in sorted(set(separators)) + [size]:
        if idx - prev > 1:
            spans.append((prev + 1, idx - 1))
        prev = idx
    return spans


def _find_library_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    candidates = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for coords in _components(grid == color):
            r0, r1, c0, c1 = _bbox_from_coords(coords)
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            full_rows = [r for r in range(sub.shape[0]) if np.all(sub[r, :] == color)]
            full_cols = [c for c in range(sub.shape[1]) if np.all(sub[:, c] == color)]
            row_spans = _spans(sub.shape[0], full_rows)
            col_spans = _spans(sub.shape[1], full_cols)
            if (
                full_rows
                and full_cols
                and full_rows[0] == 0
                and full_rows[-1] == sub.shape[0] - 1
                and full_cols[0] == 0
                and full_cols[-1] == sub.shape[1] - 1
                and len(row_spans) >= 2
                and len(col_spans) == 2
            ):
                candidates.append((c0, -len(coords), color, (r0, r1, c0, c1), row_spans, col_spans))
    assert candidates, "library frame not found"
    candidates.sort()
    _, _, color, bbox, row_spans, col_spans = candidates[0]
    return color, bbox, row_spans, col_spans


def _parse_templates(grid: np.ndarray, bg: int) -> tuple[int, int, list[dict[str, np.ndarray]]]:
    sep, (r0, r1, c0, c1), row_spans, col_spans = _find_library_frame(grid, bg)
    lib = grid[r0 : r1 + 1, c0 : c1 + 1]
    templates: list[dict[str, np.ndarray]] = []
    seen_shapes: list[np.ndarray] = []
    for rr0, rr1 in row_spans:
        left = lib[rr0 : rr1 + 1, col_spans[0][0] : col_spans[0][1] + 1]
        right = lib[rr0 : rr1 + 1, col_spans[1][0] : col_spans[1][1] + 1]
        right_mask = right != bg
        assert right_mask.any(), "template pattern missing"
        pr0, pr1, pc0, pc1 = _bbox_from_mask(right_mask)
        left_crop = left[pr0 : pr1 + 1, pc0 : pc1 + 1]
        right_crop = right[pr0 : pr1 + 1, pc0 : pc1 + 1]
        outline_mask = left_crop != bg
        assert outline_mask.any(), "template outline missing"
        shape = _fill_outline(outline_mask)
        if shape.shape != right_crop.shape or not np.array_equal(shape, right_crop != bg):
            shape = right_crop != bg
        scaled = np.kron(shape.astype(int), np.ones((2, 2), dtype=int)).astype(bool)
        assert np.any(scaled & ~_boundary_mask(scaled)), "template must admit interior after scaling"
        assert not any(np.array_equal(shape, prev) for prev in seen_shapes), "duplicate template shape"
        seen_shapes.append(shape.copy())
        templates.append({"shape": shape, "pattern": right_crop.copy()})
    assert templates
    return sep, c1, templates


def _match_template(shape_mask: np.ndarray, templates: list[dict[str, np.ndarray]]) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    hh, ww = shape_mask.shape
    for idx, tmpl in enumerate(templates):
        base = tmpl["shape"]
        h, w = base.shape
        if hh % h != 0 or ww % w != 0:
            continue
        scale = hh // h
        if scale < 1 or ww // w != scale:
            continue
        scaled = np.kron(base.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)
        if np.array_equal(scaled, shape_mask):
            matches.append((idx, scale))
    assert len(matches) == 1, "shape must match exactly one template"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    _sep, lib_c1, templates = _parse_templates(g, bg)

    out = g.copy()
    scene_mask = np.zeros_like(g, dtype=bool)
    scene_mask[:, lib_c1 + 1 :] = g[:, lib_c1 + 1 :] != bg

    for coords in _components(scene_mask):
        values = g[coords[:, 0], coords[:, 1]]
        colors, counts = np.unique(values, return_counts=True)
        base_color = int(colors[np.argmax(counts)])
        marker_coords = coords[values != base_color]
        if len(marker_coords) != 1:
            continue

        r0, r1, c0, c1 = _bbox_from_coords(coords)
        shape_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        shape_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        boundary = _boundary_mask(shape_mask)
        interior = shape_mask & ~boundary
        mr, mc = marker_coords[0]
        if not interior[mr - r0, mc - c0]:
            continue

        tmpl_idx, scale = _match_template(shape_mask, templates)
        scaled_pattern = np.kron(
            templates[tmpl_idx]["pattern"], np.ones((scale, scale), dtype=int)
        ).astype(int)
        patch = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[interior] = scaled_pattern[interior]
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _shape_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = {(r - min_r, c - min_c) for r, c in cells}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _random_shape(rng: random.Random, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(3, max_h)
        w = rng.randint(3, max_w)
        sr = rng.randint(0, h - 2)
        sc = rng.randint(0, w - 2)
        cells = {(sr, sc), (sr + 1, sc), (sr, sc + 1), (sr + 1, sc + 1)}
        target = rng.randint(5, min(h * w - 1, max(6, h * w - 2)))
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
            if len(cells) >= 6 and rng.random() < 0.28:
                break
        mask = _shape_from_cells(cells)
        for _ in range(rng.randint(0, 3)):
            mask = np.rot90(mask)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        if mask.shape[0] > max_h or mask.shape[1] > max_w:
            continue
        scaled = np.kron(mask.astype(int), np.ones((2, 2), dtype=int)).astype(bool)
        if np.any(scaled & ~_boundary_mask(scaled)):
            return mask
    raise RuntimeError("shape generation failed")


def _make_pattern(rng: random.Random, shape: np.ndarray, bg: int) -> tuple[np.ndarray, set[int]]:
    usable = [c for c in ALL_COLORS if c not in {bg, SEP_COLOR}]
    color_count = 2 if shape.sum() < 7 else rng.choice([2, 3])
    colors = rng.sample(usable, color_count)
    style = rng.choice(["checker", "hstripe", "vstripe", "diag", "bands"])
    pattern = np.full(shape.shape, bg, dtype=int)
    rows, cols = np.where(shape)
    for r, c in zip(rows, cols):
        if style == "checker":
            idx = (r + c) % color_count
        elif style == "hstripe":
            idx = r % color_count
        elif style == "vstripe":
            idx = c % color_count
        elif style == "diag":
            idx = (2 * r + c) % color_count
        else:
            idx = ((r >= shape.shape[0] // 2) + (c >= shape.shape[1] // 2)) % color_count
        pattern[r, c] = colors[idx]
    if len(np.unique(pattern[shape])) < 2:
        rr, cc = rows[0], cols[0]
        pattern[rr, cc] = colors[-1]
    return pattern, set(colors)


def _choose_color(rng: random.Random, forbidden: set[int]) -> int:
    choices = [c for c in ALL_COLORS if c not in forbidden]
    assert choices
    return rng.choice(choices)


def _stamp_template_slot(slot: np.ndarray, content: np.ndarray, top: int, left: int) -> None:
    h, w = content.shape
    slot[top : top + h, left : left + w] = content


def _place_objects(
    rng: random.Random, grid: np.ndarray, scene_c0: int, objects: list[np.ndarray], bg: int
) -> bool:
    h, w = grid.shape
    occupancy = np.zeros((h, w), dtype=bool)
    occupancy[:, :scene_c0] = True
    order = sorted(range(len(objects)), key=lambda i: objects[i].shape[0] * objects[i].shape[1], reverse=True)
    for idx in order:
        obj = objects[idx]
        oh, ow = obj.shape
        placed = False
        for _ in range(300):
            top = rng.randint(1, h - oh - 1)
            left = rng.randint(scene_c0, w - ow - 1)
            r0 = max(0, top - 1)
            r1 = min(h, top + oh + 1)
            c0 = max(0, left - 1)
            c1 = min(w, left + ow + 1)
            if occupancy[r0:r1, c0:c1].any():
                continue
            patch = obj != bg
            grid[top : top + oh, left : left + ow][patch] = obj[patch]
            occupancy[top : top + oh, left : left + ow][patch] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice([c for c in ALL_COLORS if c != SEP_COLOR])
    slot_h = rng.randint(4, 6)
    slot_w = rng.randint(4, 6)
    template_count = rng.randint(2, 4)

    templates = []
    for _ in range(template_count):
        for _ in range(200):
            shape = _random_shape(rng, slot_h, slot_w)
            if any(np.array_equal(shape, prev["shape"]) for prev in templates):
                continue
            pattern, pattern_colors = _make_pattern(rng, shape, bg)
            outline_color = _choose_color(rng, {bg, SEP_COLOR} | pattern_colors)
            outline = np.full(shape.shape, bg, dtype=int)
            outline[_boundary_mask(shape)] = outline_color
            templates.append(
                {
                    "shape": shape,
                    "pattern": pattern,
                    "pattern_colors": pattern_colors,
                    "outline": outline,
                }
            )
            break
        else:
            return None

    lib_h = template_count * slot_h + template_count + 1
    lib_w = 2 * slot_w + 3
    library = np.full((lib_h, lib_w), bg, dtype=int)
    library[:: slot_h + 1, :] = SEP_COLOR
    library[:, 0] = SEP_COLOR
    library[:, slot_w + 1] = SEP_COLOR
    library[:, -1] = SEP_COLOR
    library[-1, :] = SEP_COLOR

    for idx, tmpl in enumerate(templates):
        th, tw = tmpl["shape"].shape
        row0 = 1 + idx * (slot_h + 1)
        off_r = rng.randint(0, slot_h - th)
        off_c = rng.randint(0, slot_w - tw)
        _stamp_template_slot(library[row0 : row0 + slot_h, 1 : 1 + slot_w], tmpl["outline"], off_r, off_c)
        _stamp_template_slot(
            library[row0 : row0 + slot_h, slot_w + 2 : slot_w + 2 + slot_w],
            tmpl["pattern"],
            off_r,
            off_c,
        )

    marked_count = rng.randint(1, min(3, template_count + 1))
    unmarked_count = rng.randint(0, 2)
    object_specs = []
    for marked in [True] * marked_count + [False] * unmarked_count:
        tmpl_idx = rng.randrange(template_count)
        scale = rng.randint(2, 4)
        shape = np.kron(
            templates[tmpl_idx]["shape"].astype(int), np.ones((scale, scale), dtype=int)
        ).astype(bool)
        interior = shape & ~_boundary_mask(shape)
        if marked and not interior.any():
            return None
        forbidden = {bg, SEP_COLOR} | templates[tmpl_idx]["pattern_colors"]
        base_color = _choose_color(rng, forbidden)
        obj = np.full(shape.shape, bg, dtype=int)
        obj[shape] = base_color
        if marked:
            marker_color = _choose_color(rng, forbidden | {base_color})
            cells = np.argwhere(interior)
            rr, cc = cells[rng.randrange(len(cells))]
            obj[rr, cc] = marker_color
        object_specs.append(obj)

    gap = rng.randint(2, 4)
    scene_c0 = lib_w + gap
    max_obj_h = max(obj.shape[0] for obj in object_specs)
    max_obj_w = max(obj.shape[1] for obj in object_specs)
    grid_h = max(lib_h + rng.randint(0, 6), max_obj_h + 4)
    scene_w = max(max_obj_w + 4, 8 + sum(obj.shape[1] for obj in object_specs) // 2)
    grid_w = scene_c0 + scene_w + rng.randint(0, 6)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[:lib_h, :lib_w] = library
    if not _place_objects(rng, grid, scene_c0, object_specs, bg):
        return None

    if _mode_color(grid) != bg:
        return None
    counts = {int(c): int(np.count_nonzero(grid == c)) for c in np.unique(grid)}
    if counts[bg] <= max(v for c, v in counts.items() if c != bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
