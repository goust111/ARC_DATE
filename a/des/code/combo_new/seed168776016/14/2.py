from __future__ import annotations

import random

import numpy as np


BG = 8
SEP = 5
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 9]

GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PATTERN_ATTEMPTS = 80
PLACEMENT_PAD = 1


def _array_key(arr: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in arr.astype(bool))


def _apply_transform(arr: np.ndarray, op: int) -> np.ndarray:
    out = np.asarray(arr)
    if op >= 4:
        out = np.fliplr(out)
    return np.rot90(out, op % 4).copy()


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return False
    return len(_components(mask)) == 1


def _border_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    border = np.zeros_like(mask, dtype=bool)
    for r, c in zip(*np.where(mask), strict=False):
        r = int(r)
        c = int(c)
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                border[r, c] = True
                break
    return border


def _interior_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask & ~_border_mask(mask)


def _distinct_transform_keys(mask: np.ndarray) -> set[tuple[str, ...]]:
    return {_array_key(_apply_transform(mask, op)) for op in range(8)}


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    return min(_distinct_transform_keys(mask))


def _separator_span(grid: np.ndarray) -> tuple[int, int]:
    grey_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    assert grey_rows, "expected a full-width grey separator band"
    runs: list[tuple[int, int]] = []
    start = prev = grey_rows[0]
    for r in grey_rows[1:]:
        if r == prev + 1:
            prev = r
            continue
        runs.append((start, prev))
        start = prev = r
    runs.append((start, prev))
    candidates = [(r0, r1) for r0, r1 in runs if r0 > 0 and r1 < grid.shape[0] - 1]
    assert len(candidates) == 1, "separator band must be unique and split template/work areas"
    return candidates[0]


def _template_variants(template_region: np.ndarray) -> dict[tuple[str, ...], tuple[np.ndarray, int]]:
    fg = template_region != BG
    comps = _components(fg)
    assert comps, "expected at least one template object"

    variants: dict[tuple[str, ...], tuple[np.ndarray, int]] = {}
    for coords in comps:
        bbox = _bbox(coords)
        mask = _mask_from_coords(coords, bbox)
        r0, c0, r1, c1 = bbox
        sub = template_region[r0 : r1 + 1, c0 : c1 + 1]

        border = _border_mask(mask)
        border_colors = {int(x) for x in sub[border]}
        assert len(border_colors) == 1, "template border color must be unique"
        outline_color = next(iter(border_colors))

        fill_colors = {int(x) for x in sub[mask]} - {outline_color}
        assert len(fill_colors) == 1, "template must use exactly one fill color"
        fill_color = next(iter(fill_colors))

        fill_mask = (sub == fill_color) & _interior_mask(mask)
        assert fill_mask.any(), "template pattern must occupy interior cells"

        for op in range(8):
            oriented_mask = _apply_transform(mask, op)
            oriented_fill = _apply_transform(fill_mask, op)
            key = _array_key(oriented_mask)
            assert key not in variants, "template shapes must be unambiguous under D4 matching"
            variants[key] = (oriented_fill.astype(bool), int(fill_color))
    return variants


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match workspace shapes to top templates under rotation/reflection and repaint only their interiors with the template texture.

    Input structure:
    - The background is azure.
    - One or two full-width grey rows split the grid into an upper template area and a lower workspace area.
    - The template area contains several disconnected two-color objects: one border color plus one interior pattern color.
    - The workspace area contains disconnected monochrome objects whose tight masks match template masks up to rotation or reflection.

    Rule:
    - Remove the separator band and output only the workspace area.
    - For each template object, record every rotated/flipped version of its mask together with the correspondingly transformed interior pattern cells.
    - For each workspace object, find the unique template orientation with the same tight mask.
    - Recolor exactly those matched interior pattern cells with the template fill color, while leaving all remaining object pixels in their original workspace color.

    Key insight:
    - The top area is a shape-to-texture lookup table: the solver must first identify D4-equivalent shapes, then transfer the template's oriented interior pattern back onto the lower monochrome copy.
    """
    g = np.asarray(grid, dtype=int)
    sep0, sep1 = _separator_span(g)
    template_region = g[:sep0, :]
    workspace = g[sep1 + 1 :, :].copy()

    variants = _template_variants(template_region)
    out = workspace.copy()
    for coords in _components(workspace != BG):
        bbox = _bbox(coords)
        mask = _mask_from_coords(coords, bbox)
        key = _array_key(mask)
        assert key in variants, "workspace object does not match any template shape"
        fill_mask, fill_color = variants[key]
        fill_mask = fill_mask & _interior_mask(mask)
        r0, c0, r1, c1 = bbox
        out[r0 : r1 + 1, c0 : c1 + 1][fill_mask] = int(fill_color)

    return out


def _random_shape() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(5, 7)
        w = random.randint(5, 7)
        mask = np.ones((h, w), dtype=bool)
        boundary = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
        removed = random.randint(2, min(5, len(boundary) - 1))
        for r, c in random.sample(boundary, removed):
            mask[r, c] = False
        if not _is_connected(mask):
            continue
        interior = _interior_mask(mask)
        if int(interior.sum()) < 4:
            continue
        if len(_distinct_transform_keys(mask)) != 8:
            continue
        return mask
    return None


def _random_pattern(mask: np.ndarray) -> np.ndarray | None:
    interior = _interior_mask(mask)
    total = int(interior.sum())
    if total < 4:
        return None

    h, w = mask.shape
    rr, cc = np.indices((h, w))
    candidates: list[np.ndarray] = []
    for phase in (0, 1):
        candidates.append(((rr + cc + phase) % 2) == 0)
        candidates.append(((rr + phase) % 2) == 0)
        candidates.append(((cc + phase) % 2) == 0)
        candidates.append(((rr - cc + phase) % 2) == 0)
    center_sum = (h + w - 2) // 2
    for delta in (-1, 0, 1):
        candidates.append((rr - cc) == delta)
        candidates.append((rr + cc) == center_sum + delta)
    candidates.append(rr < cc)
    candidates.append(rr > cc)
    candidates.append((rr >= h // 2) != (cc >= w // 2))
    candidates.append(rr == h // 2)
    candidates.append(cc == w // 2)

    random.shuffle(candidates)
    for raw in candidates[:PATTERN_ATTEMPTS]:
        fill = interior & raw
        count = int(fill.sum())
        if 2 <= count <= total - 2:
            return fill
    return None


def _place_boxes(area_h: int, area_w: int, boxes: list[tuple[int, int]], pad: int) -> list[tuple[int, int]] | None:
    reserved = np.zeros((area_h, area_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(boxes)
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0] * boxes[i][1], reverse=True)

    for idx in order:
        h, w = boxes[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(area_h - h + 1):
            for c0 in range(area_w - w + 1):
                rs = max(0, r0 - pad)
                re = min(area_h, r0 + h + pad)
                cs = max(0, c0 - pad)
                ce = min(area_w, c0 + w + pad)
                if not reserved[rs:re, cs:ce].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        reserved[r0 : r0 + h, c0 : c0 + w] = True

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _build_template_records(count: int) -> list[dict] | None:
    records: list[dict] = []
    signatures: set[tuple[str, ...]] = set()
    outline_colors = random.sample(COLOR_POOL, count)

    for outline_color in outline_colors:
        mask = _random_shape()
        if mask is None:
            return None
        signature = _shape_signature(mask)
        if signature in signatures:
            return None
        pattern = _random_pattern(mask)
        if pattern is None:
            return None
        signatures.add(signature)

        fill_choices = [color for color in COLOR_POOL if color != outline_color]
        fill_color = random.choice(fill_choices)
        template_op = random.randrange(8)
        records.append(
            {
                "mask": _apply_transform(mask, template_op),
                "fill_mask": _apply_transform(pattern, template_op),
                "outline_color": int(outline_color),
                "fill_color": int(fill_color),
            }
        )

    return records


def _choose_work_ops(count: int) -> list[int]:
    ops = [random.randrange(4), 4 + random.randrange(4)]
    while len(ops) < count:
        ops.append(random.randrange(8))
    random.shuffle(ops)
    return ops[:count]


def _render_objects(grid: np.ndarray, placements: list[tuple[int, int]], records: list[dict], solid: bool) -> None:
    for (r0, c0), rec in zip(placements, records, strict=False):
        mask = rec["mask"]
        h, w = mask.shape
        sub = grid[r0 : r0 + h, c0 : c0 + w]
        if solid:
            sub[mask] = int(rec["color"])
        else:
            sub[mask] = int(rec["outline_color"])
            sub[rec["fill_mask"]] = int(rec["fill_color"])


def _gen_attempt() -> np.ndarray | None:
    template_count = random.randint(2, 4)
    templates = _build_template_records(template_count)
    if templates is None:
        return None

    work_count = template_count + random.randint(0, 2)
    template_indices = list(range(template_count))
    template_indices.extend(random.randrange(template_count) for _ in range(work_count - template_count))
    random.shuffle(template_indices)
    work_ops = _choose_work_ops(work_count)

    work_records: list[dict] = []
    for i, template_idx in enumerate(template_indices):
        tpl = templates[template_idx]
        op = work_ops[i]
        work_color_choices = [color for color in COLOR_POOL if color != tpl["fill_color"]]
        work_records.append(
            {
                "mask": _apply_transform(tpl["mask"], op),
                "fill_mask": _apply_transform(tpl["fill_mask"], op),
                "color": int(random.choice(work_color_choices)),
                "fill_color": int(tpl["fill_color"]),
            }
        )

    width_min = max(
        18,
        max(rec["mask"].shape[1] for rec in templates) + 6,
        max(rec["mask"].shape[1] for rec in work_records) + 6,
    )
    width = random.randint(width_min, 28)

    sep_h = random.choice((1, 2))
    template_h = random.randint(max(6, max(rec["mask"].shape[0] for rec in templates) + 1), 8)
    work_h_min = max(10, max(rec["mask"].shape[0] for rec in work_records) + 1)
    work_h_max = 30 - sep_h - template_h
    if work_h_min > work_h_max:
        return None
    work_h = random.randint(work_h_min, min(work_h_max, 18))

    template_boxes = [rec["mask"].shape for rec in templates]
    work_boxes = [rec["mask"].shape for rec in work_records]
    template_pos = _place_boxes(template_h, width, template_boxes, PLACEMENT_PAD)
    work_pos = _place_boxes(work_h, width, work_boxes, PLACEMENT_PAD)
    if template_pos is None or work_pos is None:
        return None

    template_grid = np.full((template_h, width), BG, dtype=int)
    workspace_in = np.full((work_h, width), BG, dtype=int)
    workspace_out = np.full((work_h, width), BG, dtype=int)

    _render_objects(template_grid, template_pos, templates, solid=False)
    _render_objects(workspace_in, work_pos, work_records, solid=True)
    _render_objects(workspace_out, work_pos, work_records, solid=True)
    for (r0, c0), rec in zip(work_pos, work_records, strict=False):
        mask = rec["fill_mask"]
        h, w = mask.shape
        workspace_out[r0 : r0 + h, c0 : c0 + w][mask] = int(rec["fill_color"])

    separator = np.full((sep_h, width), SEP, dtype=int)
    grid = np.vstack([template_grid, separator, workspace_in])
    if not np.array_equal(main(grid), workspace_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
