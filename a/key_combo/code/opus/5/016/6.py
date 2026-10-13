from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _canonical_shape_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    best = None
    cur = np.asarray(mask, dtype=bool)
    for k in range(4):
        rot = np.rot90(cur, k)
        for variant in (rot, np.fliplr(rot)):
            key = (variant.tobytes(), tuple(map(int, variant.shape)))
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            mask, bbox = _tight_mask(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "mask": mask,
                    "shape_key": _canonical_shape_key(mask),
                    "area": len(coords),
                }
            )
    return comps


def _classify_components(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    comps = _find_components(g, bg)
    freq = Counter(comp["shape_key"] for comp in comps)
    canvases = [comp for comp in comps if freq[comp["shape_key"]] >= 2]
    templates = [comp for comp in comps if freq[comp["shape_key"]] == 1]
    canvases.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0], comp["color"]))
    templates.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0], comp["color"]))
    return bg, canvases, templates


def _choose_canvas_for_template(template: dict, canvases: list[dict]) -> int:
    _r0, tc0, _r1, tc1 = template["bbox"]
    t_center = (tc0 + tc1) / 2.0
    candidates: list[tuple[float, int, int]] = []
    for idx, canvas in enumerate(canvases):
        cc0 = int(canvas["bbox"][1])
        cc1 = cc0 + 4
        if max(tc0, cc0) <= min(tc1, cc1):
            center = cc0 + 2.0
            candidates.append((abs(t_center - center), cc0, idx))
    if not candidates:
        raise ValueError("every template must overlap at least one canvas column range")
    candidates.sort()
    return int(candidates[0][2])


def _carve_canvases(bg: int, canvases: list[dict], templates: list[dict]) -> list[np.ndarray]:
    tiles = [np.full((5, 5), int(canvas["color"]), dtype=int) for canvas in canvases]
    for template in templates:
        idx = _choose_canvas_for_template(template, canvases)
        mask = np.asarray(template["mask"], dtype=bool)
        th, tw = mask.shape
        if th > 5 or tw > 5:
            raise ValueError("template bbox must fit inside a 5x5 canvas")
        off_r = (5 - th) // 2
        off_c = (5 - tw) // 2
        window = tiles[idx][off_r : off_r + th, off_c : off_c + tw]
        window[mask] = int(bg)
    return tiles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort shape-frequency groups into canvases and templates, then carve the canvases.

    Input structure:
    - Black (0) is the background.
    - The grid contains several separated monochrome 4-connected objects, each with its own color.
    - Some objects share the same binary shape up to rotation/reflection, while other shapes appear only once.

    Rule:
    - Canonicalize each object's tight binary mask under the 8 dihedral transforms.
    - Objects whose canonical shape appears at least twice become the canvas group; each canvas keeps its original left column, is moved to the top, and is replaced by a solid 5x5 block of its own color.
    - Objects whose canonical shape appears once become the template group; each template keeps its original left column and is moved to the bottom.
    - Each template is matched to the overlapping top canvas block; if several overlap, choose the one with the nearest column center. Center the template mask inside that 5x5 block and punch those cells back to background.
    - Output only the carved canvas blocks, packed left-to-right in canvas-column order with 1-cell gaps and a 1-cell outer border.

    Key insight: Shape frequency determines function rather than direct placement: repeated shapes become 5x5 canvases, unique shapes become hole-punch templates, and column-preserving vertical sorting creates the pairing relation.
    """
    g = np.asarray(grid, dtype=int)
    bg, canvases, templates = _classify_components(g)
    if not canvases:
        raise ValueError("expected at least one repeated-shape canvas")
    tiles = _carve_canvases(bg, canvases, templates)
    out = np.full((7, 1 + len(tiles) * 5 + max(0, len(tiles) - 1) + 1), int(bg), dtype=int)
    for i, tile in enumerate(tiles):
        c0 = 1 + i * 6
        out[1:6, c0 : c0 + 5] = tile
    return out


_RNG = random.Random()
_COLORS = tuple(range(1, 10))
_GRID_H_RANGE = (20, 26)
_GRID_W_RANGE = (24, 29)
_MAX_RETRIES = 600


def _mask_signature(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    return mask.astype(np.uint8).tobytes(), tuple(map(int, mask.shape))


def _random_connected_mask(height: int, width: int, area: int, *, require_full_bbox: bool) -> np.ndarray | None:
    if not (1 <= area <= height * width):
        return None
    for _ in range(300):
        start = (_RNG.randrange(height), _RNG.randrange(width))
        cells = {start}
        frontier = set()
        for nr, nc in ((start[0] - 1, start[1]), (start[0] + 1, start[1]), (start[0], start[1] - 1), (start[0], start[1] + 1)):
            if 0 <= nr < height and 0 <= nc < width:
                frontier.add((nr, nc))
        while len(cells) < area and frontier:
            pick = _RNG.choice(tuple(frontier))
            frontier.remove(pick)
            cells.add(pick)
            pr, pc = pick
            for nr, nc in ((pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != area:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if require_full_bbox:
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
        tight, _ = _tight_mask(list(cells))
        if tight.shape != mask.shape:
            continue
        if height > 1 and width > 1 and (mask.all() or area <= max(height, width)):
            continue
        return mask
    return None


def _sample_shared_base_masks() -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    keys: set[tuple[bytes, tuple[int, int]]] = set()
    for _ in range(400):
        masks.clear()
        keys.clear()
        ok = True
        for _group in range(2):
            area = _RNG.randint(9, 11)
            mask = _random_connected_mask(5, 5, area, require_full_bbox=True)
            if mask is None:
                ok = False
                break
            key = _canonical_shape_key(mask)
            if key in keys:
                ok = False
                break
            keys.add(key)
            masks.append(mask)
        if ok:
            return [m.copy() for m in masks]
    return None


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, _RNG.randrange(4))
    if _RNG.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _sample_template_masks(forbidden: set[tuple[bytes, tuple[int, int]]], count: int) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    keys = set(forbidden)
    five_box_quota = 1 if _RNG.random() < 0.55 else 0
    for i in range(count):
        want_5x5 = i < five_box_quota
        found = None
        for _ in range(400):
            if want_5x5:
                height = width = 5
                area = _RNG.randint(9, 11)
            else:
                height = _RNG.randint(2, 5)
                width = _RNG.randint(2, 5)
                area = _RNG.randint(3, min(7, height * width))
            mask = _random_connected_mask(height, width, area, require_full_bbox=True)
            if mask is None:
                continue
            key = _canonical_shape_key(mask)
            if key in keys:
                continue
            keys.add(key)
            found = _random_transform(mask)
            break
        if found is None:
            return None
        masks.append(found)
    return masks


def _expanded_rect(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return r0 - 1, c0 - 1, r0 + h, c0 + w


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _choose_canvas_columns(width: int) -> list[int] | None:
    gaps = [_RNG.randint(1, 2) for _ in range(3)]
    span = 20 + sum(gaps)
    if span > width:
        return None
    start = _RNG.randint(0, width - span)
    cols = [start]
    for gap in gaps:
        cols.append(cols[-1] + 5 + gap)
    return cols


def _count_overlaps(c0: int, width: int, canvas_cols: list[int]) -> list[int]:
    c1 = c0 + width - 1
    out = []
    for idx, cc0 in enumerate(canvas_cols):
        if max(c0, cc0) <= min(c1, cc0 + 4):
            out.append(idx)
    return out


def _nearest_canvas_index(c0: int, width: int, canvas_cols: list[int]) -> int | None:
    overlaps = _count_overlaps(c0, width, canvas_cols)
    if not overlaps:
        return None
    center = c0 + (width - 1) / 2.0
    scored = sorted((abs(center - (canvas_cols[idx] + 2.0)), canvas_cols[idx], idx) for idx in overlaps)
    if len(scored) >= 2 and scored[0][0] == scored[1][0]:
        return None
    return int(scored[0][2])


def _single_overlap_cols(width: int, canvas_cols: list[int], target: int) -> list[int]:
    candidates = []
    max_c0 = canvas_cols[-1] + 4
    for c0 in range(max(0, canvas_cols[target] - width + 1), max_c0 + 1):
        overlaps = _count_overlaps(c0, width, canvas_cols)
        if overlaps == [target]:
            candidates.append(c0)
    return candidates


def _multi_overlap_cols(width: int, canvas_cols: list[int], target: int) -> list[int]:
    candidates = []
    max_c0 = canvas_cols[-1] + 4
    for c0 in range(0, max_c0 + 1):
        overlaps = _count_overlaps(c0, width, canvas_cols)
        if len(overlaps) < 2 or target not in overlaps:
            continue
        if _nearest_canvas_index(c0, width, canvas_cols) == target:
            candidates.append(c0)
    return candidates


def _assign_template_columns(template_masks: list[np.ndarray], width: int, canvas_cols: list[int]) -> list[tuple[int, int]] | None:
    choices_per_template: list[list[tuple[int, int, str]]] = []
    for mask in template_masks:
        tw = int(mask.shape[1])
        choices: list[tuple[int, int, str]] = []
        for target in range(4):
            for c0 in _single_overlap_cols(tw, canvas_cols, target):
                if c0 + tw <= width:
                    choices.append((c0, target, "single"))
            for c0 in _multi_overlap_cols(tw, canvas_cols, target):
                if c0 + tw <= width:
                    choices.append((c0, target, "multi"))
        if not choices:
            return None
        _RNG.shuffle(choices)
        choices_per_template.append(choices)

    can_multi = any(any(mode == "multi" for _c0, _target, mode in choices) for choices in choices_per_template)
    force_multi = can_multi and _RNG.random() < 0.45
    picked: list[tuple[int, int, str]] = [(-1, -1, "single")] * len(template_masks)

    def backtrack(i: int, used_targets: set[int], multi_done: bool) -> bool:
        if i == len(template_masks):
            if len(used_targets) < 2:
                return False
            if force_multi and not multi_done:
                return False
            return True
        choices = choices_per_template[i][:]
        if force_multi and not multi_done:
            choices.sort(key=lambda item: item[2] != "multi")
        for c0, target, mode in choices:
            picked[i] = (c0, target, mode)
            if backtrack(i + 1, used_targets | {target}, multi_done or mode == "multi"):
                return True
        return False

    if not backtrack(0, set(), False):
        return None
    return [(int(c0), int(target)) for c0, target, _mode in picked]


def _sample_rows(shapes: list[tuple[int, int, int]], grid_h: int) -> list[int] | None:
    placed: list[tuple[int, int, int, int]] = []
    rows: list[int] = []
    order = list(range(len(shapes)))
    _RNG.shuffle(order)
    order.sort(key=lambda i: shapes[i][1], reverse=True)
    by_index: dict[int, int] = {}
    for idx in order:
        c0, h, w = shapes[idx]
        candidates = list(range(0, grid_h - h + 1))
        _RNG.shuffle(candidates)
        pick = None
        for r0 in candidates:
            rect = _expanded_rect(r0, c0, h, w)
            if any(_rects_overlap(rect, other) for other in placed):
                continue
            pick = r0
            placed.append(rect)
            break
        if pick is None:
            return None
        by_index[idx] = pick
    for idx in range(len(shapes)):
        rows.append(by_index[idx])
    return rows


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(mask)
    for r, c in coords:
        grid[r0 + int(r), c0 + int(c)] = int(color)


def _render_instance(
    grid_h: int,
    grid_w: int,
    bg: int,
    canvas_masks: list[np.ndarray],
    canvas_cols: list[int],
    template_masks: list[np.ndarray],
    template_specs: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray | None:
    canvas_shapes = [(canvas_cols[i], 5, 5) for i in range(4)]
    template_shapes = [(template_specs[i][0], int(template_masks[i].shape[0]), int(template_masks[i].shape[1])) for i in range(len(template_masks))]
    rows = _sample_rows(canvas_shapes + template_shapes, grid_h)
    if rows is None:
        return None
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    for i, mask in enumerate(canvas_masks):
        _paint_mask(grid, mask, rows[i], canvas_cols[i], colors[i])
    offset = len(canvas_masks)
    for i, mask in enumerate(template_masks):
        c0, _target = template_specs[i]
        _paint_mask(grid, mask, rows[offset + i], c0, colors[offset + i])
    return grid


def _expected_from_blueprint(
    bg: int,
    canvas_colors: list[int],
    template_masks: list[np.ndarray],
    template_specs: list[tuple[int, int]],
) -> np.ndarray:
    tiles = [np.full((5, 5), int(color), dtype=int) for color in canvas_colors]
    for mask, (_c0, target) in zip(template_masks, template_specs):
        th, tw = mask.shape
        off_r = (5 - th) // 2
        off_c = (5 - tw) // 2
        tiles[target][off_r : off_r + th, off_c : off_c + tw][mask] = int(bg)
    out = np.full((7, 25), int(bg), dtype=int)
    for i, tile in enumerate(tiles):
        out[1:6, 1 + i * 6 : 1 + i * 6 + 5] = tile
    return out


def _validate_instance(
    grid: np.ndarray,
    bg: int,
    shared_keys: set[tuple[bytes, tuple[int, int]]],
    template_specs: list[tuple[int, int]],
    expected: np.ndarray,
) -> bool:
    try:
        parsed_bg, canvases, templates = _classify_components(grid)
    except Exception:
        return False
    if parsed_bg != bg or len(canvases) != 4 or len(templates) != len(template_specs):
        return False
    if {comp["shape_key"] for comp in canvases} != shared_keys:
        return False
    if len({comp["shape_key"] for comp in templates}) != len(templates):
        return False
    chosen_targets = [_choose_canvas_for_template(template, canvases) for template in templates]
    if sorted(chosen_targets) != sorted(target for _c0, target in template_specs):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = _RNG.randint(*_GRID_H_RANGE)
    grid_w = _RNG.randint(*_GRID_W_RANGE)
    bg = 0
    canvas_cols = _choose_canvas_columns(grid_w)
    if canvas_cols is None:
        return None

    shared_bases = _sample_shared_base_masks()
    if shared_bases is None:
        return None
    shared_keys = {_canonical_shape_key(mask) for mask in shared_bases}
    pair_order = [0, 0, 1, 1]
    _RNG.shuffle(pair_order)
    canvas_masks = [_random_transform(shared_bases[pair_order[i]]) for i in range(4)]

    n_templates = _RNG.randint(2, 4)
    template_masks = _sample_template_masks(shared_keys, n_templates)
    if template_masks is None:
        return None
    template_specs = _assign_template_columns(template_masks, grid_w, canvas_cols)
    if template_specs is None:
        return None

    all_colors = list(_COLORS)
    _RNG.shuffle(all_colors)
    colors = all_colors[: 4 + n_templates]
    grid = _render_instance(grid_h, grid_w, bg, canvas_masks, canvas_cols, template_masks, template_specs, colors)
    if grid is None:
        return None

    expected = _expected_from_blueprint(bg, colors[:4], template_masks, template_specs)
    if not _validate_instance(grid, bg, shared_keys, template_specs, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")
