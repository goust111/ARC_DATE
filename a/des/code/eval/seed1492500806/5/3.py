from __future__ import annotations

import numpy as np


BG = 0
SEED_ON = 8
LEGEND_ON = 7
BORDER_COLORS = (1, 2, 3, 4, 5, 6, 9)

SEED_BOX = 5
SEED_INNER = 3
LEGEND_MAX_WIDTH = 4
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _seed_ring() -> np.ndarray:
    ring = np.zeros((SEED_BOX, SEED_BOX), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    return ring


def _find_seed_blocks(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    ring = _seed_ring()
    seeds: list[dict[str, object]] = []
    for color in map(int, np.unique(g)):
        if color in (BG, SEED_ON, LEGEND_ON):
            continue
        for comp in _components(g == color):
            if comp.shape[0] != int(ring.sum()):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 != SEED_BOX or c1 - c0 + 1 != SEED_BOX:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            if not np.array_equal(sub == color, ring):
                continue
            inner = sub[1:-1, 1:-1]
            if not np.all(np.isin(inner, (BG, SEED_ON))):
                continue
            seed_mask = inner == SEED_ON
            if not np.any(seed_mask):
                continue
            seeds.append(
                {
                    "color": color,
                    "r": r0,
                    "c": c0,
                    "seed_mask": seed_mask.copy(),
                }
            )
    assert seeds, "expected at least one framed 3x3 seed"
    colors = [int(seed["color"]) for seed in seeds]
    assert len(colors) == len(set(colors)), "seed border colors must be unique"
    seeds.sort(key=lambda item: (int(item["r"]), int(item["c"])))
    return seeds


def _find_legend_mapping(grid: np.ndarray) -> dict[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping: dict[int, int] = {}
    for comp in _components(g == LEGEND_ON):
        r0, c0, r1, c1 = _bbox(comp)
        width = int(c1 - c0 + 1)
        if r0 != r1 or comp.shape[0] != width or not (1 <= width <= 3):
            continue
        if c0 == 0:
            continue
        key_c = c0 - 1
        key_color = int(g[r0, key_c])
        if key_color in (BG, SEED_ON, LEGEND_ON):
            continue
        if key_c - 1 >= 0 and int(g[r0, key_c - 1]) != BG:
            continue
        if c1 + 1 < w and int(g[r0, c1 + 1]) != BG:
            continue
        seg0 = key_c
        seg1 = c1
        if r0 > 0 and np.any(g[r0 - 1, seg0 : seg1 + 1] != BG):
            continue
        if r0 + 1 < h and np.any(g[r0 + 1, seg0 : seg1 + 1] != BG):
            continue
        mapping[key_color] = width
    assert mapping, "expected a color-to-depth legend"
    return mapping


def _expand_seed(seed_mask: np.ndarray, depth: int) -> np.ndarray:
    assert 1 <= depth <= 3
    out = np.asarray(seed_mask, dtype=bool).copy()
    for _ in range(depth - 1):
        out = np.kron(out.astype(int), seed_mask.astype(int)).astype(bool)
    return out


def _render_layout(items: list[dict[str, object]]) -> np.ndarray:
    rows = sorted({int(item["r"]) for item in items})
    cols = sorted({int(item["c"]) for item in items})
    row_heights = {
        row: max(int(item["patch"].shape[0]) for item in items if int(item["r"]) == row)
        for row in rows
    }
    col_widths = {
        col: max(int(item["patch"].shape[1]) for item in items if int(item["c"]) == col)
        for col in cols
    }

    row_starts: dict[int, int] = {}
    col_starts: dict[int, int] = {}
    cursor = 0
    for row in rows:
        row_starts[row] = cursor
        cursor += row_heights[row] + 1
    out_h = cursor - 1

    cursor = 0
    for col in cols:
        col_starts[col] = cursor
        cursor += col_widths[col] + 1
    out_w = cursor - 1

    out = np.full((out_h, out_w), BG, dtype=int)
    for item in items:
        patch = np.asarray(item["patch"], dtype=int)
        r0 = row_starts[int(item["r"])]
        c0 = col_starts[int(item["c"])]
        h, w = patch.shape
        target = out[r0 : r0 + h, c0 : c0 + w]
        assert not np.any((target != BG) & (patch != BG)), "expanded patterns must stay disjoint"
        target[patch != BG] = patch[patch != BG]
    return out


def _expected_from_specs(specs: list[dict[str, object]]) -> np.ndarray:
    items: list[dict[str, object]] = []
    for spec in specs:
        color = int(spec["color"])
        seed_mask = np.asarray(spec["seed_mask"], dtype=bool)
        depth = int(spec["depth"])
        fractal = _expand_seed(seed_mask, depth)
        patch = np.full(fractal.shape, BG, dtype=int)
        patch[fractal] = color
        items.append({"r": int(spec["r"]), "c": int(spec["c"]), "patch": patch})
    return _render_layout(items)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each framed 3x3 seed by the depth encoded in the color legend, then
    arrange the grown fractals by the seeds' relative row/column positions.
    """
    g = np.asarray(grid, dtype=int)
    seeds = _find_seed_blocks(g)
    mapping = _find_legend_mapping(g)

    items: list[dict[str, object]] = []
    for seed in seeds:
        color = int(seed["color"])
        assert color in mapping, "every seed color must appear in the legend"
        fractal = _expand_seed(np.asarray(seed["seed_mask"], dtype=bool), int(mapping[color]))
        patch = np.full(fractal.shape, BG, dtype=int)
        patch[fractal] = color
        items.append({"r": int(seed["r"]), "c": int(seed["c"]), "patch": patch})

    return _render_layout(items)


def _valid_seed_masks() -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    for bits in range(1, 1 << (SEED_INNER * SEED_INNER)):
        flat = np.array([(bits >> i) & 1 for i in range(SEED_INNER * SEED_INNER)], dtype=bool)
        mask = flat.reshape(SEED_INNER, SEED_INNER)
        count = int(mask.sum())
        if not (3 <= count <= 6):
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        masks.append(mask)
    return tuple(masks)


SEED_LIBRARY = _valid_seed_masks()


def _seed_patch(color: int, seed_mask: np.ndarray) -> np.ndarray:
    patch = np.full((SEED_BOX, SEED_BOX), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1:-1, 1:-1][seed_mask] = SEED_ON
    return patch


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _candidate_seed_slots(h: int, w: int) -> list[tuple[int, int]]:
    mid_r = h // 2 - 2
    mid_c = w // 2 - 2
    slots = [
        (1, 1),
        (1, mid_c),
        (1, w - SEED_BOX - 1),
        (mid_r, 1),
        (mid_r, w - SEED_BOX - 1),
        (h - SEED_BOX - 1, 1),
        (h - SEED_BOX - 1, mid_c),
        (h - SEED_BOX - 1, w - SEED_BOX - 1),
    ]
    dedup: list[tuple[int, int]] = []
    for slot in slots:
        if slot not in dedup:
            dedup.append(slot)
    return dedup


def _candidate_legend_anchors(h: int, w: int, legend_h: int) -> list[tuple[int, int]]:
    anchors = [
        (1, 1),
        (1, w - LEGEND_MAX_WIDTH - 1),
        (h - legend_h - 1, 1),
        (h - legend_h - 1, w - LEGEND_MAX_WIDTH - 1),
        (max(1, h // 2 - legend_h // 2), 1),
        (max(1, h // 2 - legend_h // 2), w - LEGEND_MAX_WIDTH - 1),
    ]
    out: list[tuple[int, int]] = []
    for r, c in anchors:
        if 0 <= r and r + legend_h <= h and 0 <= c and c + LEGEND_MAX_WIDTH <= w:
            if (r, c) not in out:
                out.append((r, c))
    return out


def _draw_legend(grid: np.ndarray, rows: list[tuple[int, int]], anchor: tuple[int, int]) -> None:
    r0, c0 = anchor
    for i, (color, depth) in enumerate(rows):
        rr = r0 + 2 * i
        grid[rr, c0] = int(color)
        grid[rr, c0 + 1 : c0 + 1 + int(depth)] = LEGEND_ON


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(22, 31))
    w = int(rng.integers(24, 35))
    seed_count = int(rng.integers(3, 6))

    slots = _candidate_seed_slots(h, w)
    if len(slots) < seed_count:
        return None
    slot_idx = rng.choice(len(slots), size=seed_count, replace=False)
    chosen_slots = [slots[int(i)] for i in slot_idx]
    if len({r for r, _c in chosen_slots}) < 2 or len({c for _r, c in chosen_slots}) < 2:
        return None

    colors = [int(c) for c in rng.choice(BORDER_COLORS, size=seed_count, replace=False)]
    mask_idx = rng.choice(len(SEED_LIBRARY), size=seed_count, replace=False)
    masks = [SEED_LIBRARY[int(i)] for i in mask_idx]

    depths = [1, 2, 3] + [int(x) for x in rng.integers(1, 4, size=seed_count - 3)]
    rng.shuffle(depths)

    specs: list[dict[str, object]] = []
    seed_boxes: list[tuple[int, int, int, int]] = []
    for (r, c), color, mask, depth in zip(chosen_slots, colors, masks, depths, strict=False):
        specs.append({"r": r, "c": c, "color": color, "seed_mask": mask.copy(), "depth": depth})
        seed_boxes.append((r, c, r + SEED_BOX - 1, c + SEED_BOX - 1))

    legend_h = 2 * seed_count - 1
    legend_bbox_choices = []
    for anchor in _candidate_legend_anchors(h, w, legend_h):
        bbox = (anchor[0], anchor[1], anchor[0] + legend_h - 1, anchor[1] + LEGEND_MAX_WIDTH - 1)
        if any(_rects_overlap(bbox, seed_bbox, pad=1) for seed_bbox in seed_boxes):
            continue
        legend_bbox_choices.append(anchor)
    if not legend_bbox_choices:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        r = int(spec["r"])
        c = int(spec["c"])
        patch = _seed_patch(int(spec["color"]), np.asarray(spec["seed_mask"], dtype=bool))
        grid[r : r + SEED_BOX, c : c + SEED_BOX] = patch

    legend_rows = [(int(spec["color"]), int(spec["depth"])) for spec in specs]
    rng.shuffle(legend_rows)
    anchor = legend_bbox_choices[int(rng.integers(len(legend_bbox_choices)))]
    _draw_legend(grid, legend_rows, anchor)

    expected = _expected_from_specs(specs)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
