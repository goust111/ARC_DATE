from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GEN_ATTEMPTS = 400
RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _local_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    outside = ~mask
    reached = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if outside[r, c] and not reached[r, c]:
                reached[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if outside[r, c] and not reached[r, c]:
                reached[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and outside[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                q.append((nr, nc))
    holes = outside & ~reached
    return mask | holes


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants: list[tuple[tuple[int, int], bytes]] = []
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            cur = np.rot90(base, k)
            variants.append((tuple(map(int, cur.shape)), cur.astype(np.uint8).tobytes()))
    return min(variants)


def _mode_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(grid != int(bg))
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_rect_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    frame = np.ones((h, w), dtype=bool)
    frame[1:-1, 1:-1] = False
    return bool(np.array_equal(mask, frame))


def _find_legend_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], np.ndarray]:
    candidates: list[tuple[int, int, int, int, tuple[int, int, int, int], np.ndarray]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != int(bg)):
        for coords in _components(grid == int(color)):
            bb = _bbox(coords)
            local = _local_mask(coords, bb)
            if not _is_rect_frame(local):
                continue
            r0, c0, r1, c1 = bb
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue
            clue_mask = (inner != int(bg)) & (inner != int(color))
            if not clue_mask.any():
                continue
            clue_colors = len(np.unique(inner[clue_mask]))
            area = int(coords.shape[0])
            candidates.append((c0, -clue_colors, -area, color, bb, coords))
    assert candidates, "legend frame not found"
    _, _, _, color, bb, coords = min(candidates, key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    return int(color), bb, coords


def _extract_legend_map(grid: np.ndarray, legend_bbox: tuple[int, int, int, int], outline_color: int, bg: int) -> dict[tuple[tuple[int, int], bytes], int]:
    r0, c0, r1, c1 = legend_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != int(bg)) & (inner != int(outline_color))
    mapping: dict[tuple[tuple[int, int], bytes], int] = {}
    for coords in _components(mask):
        bb = _bbox(coords)
        local = _local_mask(coords, bb)
        color = _mode_color(inner[coords[:, 0], coords[:, 1]])
        key = _canonical_key(_fill_holes(local))
        assert key not in mapping, "ambiguous legend mapping"
        mapping[key] = int(color)
    assert mapping
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(np.bincount(g.ravel()).argmax())

    outline_color, legend_bbox, legend_coords = _find_legend_frame(g, bg)
    legend_map = _extract_legend_map(g, legend_bbox, outline_color, bg)

    outline_mask = g == int(outline_color)
    outline_mask[legend_coords[:, 0], legend_coords[:, 1]] = False

    out = np.full(g.shape, int(bg), dtype=int)
    for coords in _components(outline_mask):
        bb = _bbox(coords)
        local = _local_mask(coords, bb)
        solid = _fill_holes(local)
        key = _canonical_key(solid)
        color = legend_map[key]
        patch = np.full(local.shape, int(bg), dtype=int)
        patch[local] = int(color)
        if int((solid & ~local).sum()) % 2 == 1:
            patch = np.fliplr(patch)
        r0, c0, r1, c1 = bb
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return _tight_crop(out, bg)


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _transform_mask(mask: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    out = np.rot90(mask, int(rot))
    return np.fliplr(out) if flip else out


def _hole_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                out.append((r, c))
    return out


def _carve_holes(mask: np.ndarray, parity: int | None, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    candidates = _hole_candidates(mask)
    limit = min(4, len(candidates))
    if parity is None:
        counts = list(range(0, limit + 1))
    else:
        counts = [n for n in range(1, limit + 1) if n % 2 == int(parity)]
        if parity == 0:
            counts = [n for n in counts if n > 0] or [0]
    if counts:
        counts = [counts[i] for i in rng.permutation(len(counts))]
    if parity is None and 0 not in counts:
        counts.append(0)

    for count in counts:
        if count == 0:
            return mask.copy(), 0
        for _ in range(80):
            idxs = rng.choice(len(candidates), size=int(count), replace=False)
            carved = mask.copy()
            for idx in np.atleast_1d(idxs):
                r, c = candidates[int(idx)]
                carved[r, c] = False
            if len(_components(carved)) != 1:
                continue
            holes = _fill_holes(carved) & ~carved
            if int(holes.sum()) == int(count):
                return carved, int(count)
    return mask.copy(), 0


def _render_mask(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.full(mask.shape, BLACK, dtype=int)
    out[mask] = int(color)
    return out


def _make_object_instance(template: np.ndarray, color: int, parity: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(80):
        rot = int(rng.integers(0, 4))
        flip = bool(rng.integers(0, 2))
        oriented = _transform_mask(template, rot, flip)
        carved, hole_count = _carve_holes(oriented, parity, rng)
        if hole_count % 2 != parity:
            continue
        input_patch = _render_mask(carved, GRAY)
        output_patch = _render_mask(carved, color)
        if parity == 1:
            flipped = np.fliplr(output_patch)
            if np.array_equal(flipped, output_patch):
                continue
            output_patch = flipped
        return input_patch, output_patch
    raise RuntimeError("failed to create main object instance")


def _make_legend_instance(template: np.ndarray, color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(80):
        rot = int(rng.integers(0, 4))
        flip = bool(rng.integers(0, 2))
        oriented = _transform_mask(template, rot, flip)
        carved, _ = _carve_holes(oriented, None, rng)
        return _render_mask(carved, color)
    raise RuntimeError("failed to create legend instance")


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    canvas[r0 : r0 + h, c0 : c0 + w] = patch


def _pack_legend(patches: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    ordered = [patches[i] for i in rng.permutation(len(patches))]
    ncols = 1 if len(ordered) <= 2 else 2
    cols = [ordered[i::ncols] for i in range(ncols)]
    row_gap = 1
    col_gap = int(rng.integers(1, 3))
    pad = int(rng.integers(1, 3))
    col_widths = [max((int(p.shape[1]) for p in col), default=0) for col in cols]
    col_heights = [sum(int(p.shape[0]) for p in col) + row_gap * max(0, len(col) - 1) for col in cols]
    inner_h = max(col_heights) + 2 * pad
    inner_w = sum(col_widths) + col_gap * max(0, ncols - 1) + 2 * pad
    frame = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    frame[0, :] = GRAY
    frame[-1, :] = GRAY
    frame[:, 0] = GRAY
    frame[:, -1] = GRAY

    x = 1 + pad
    for col, width, height in zip(cols, col_widths, col_heights):
        slack = inner_h - 2 * pad - height
        y = 1 + pad + int(rng.integers(0, slack + 1))
        for patch in col:
            xoff = int(rng.integers(0, width - int(patch.shape[1]) + 1))
            _paste(frame, patch, y, x + xoff)
            y += int(patch.shape[0]) + row_gap
        x += width + col_gap
    return frame


def _place_objects(
    shape: tuple[int, int],
    patches: list[np.ndarray],
    min_col: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    h, w = shape
    used = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda i: int(np.prod(patches[i].shape)), reverse=True)
    for idx in order:
        ph, pw = map(int, patches[idx].shape)
        if h - ph - 1 <= 1 or w - pw - 1 <= min_col:
            return None
        placed = False
        for _ in range(250):
            r0 = int(rng.integers(1, h - ph))
            c0 = int(rng.integers(min_col, w - pw))
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + pw + 1)
            if used[rr0:rr1, cc0:cc1].any():
                continue
            used[rr0:rr1, cc0:cc1] = True
            positions[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


_RAW_TEMPLATES = [
    ["11111", "11111", "11111", "11111", "11111"],
    ["111111", "111111", "111111", "111111", "111111"],
    ["11111", "11111", "01110", "01110", "01110"],
    ["01110", "11111", "11111", "11111", "01110"],
    ["11110", "11110", "11110", "11111", "11111"],
    ["11011", "11111", "11111", "11111", "11011"],
    ["11110", "11110", "11111", "01111", "01111"],
    ["11111", "11111", "01110", "01110", "01110", "01110"],
    ["111111", "111111", "011110", "111111", "111111"],
    ["111100", "111100", "111111", "001111", "001111", "001111"],
]


def _build_template_library() -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for rows in _RAW_TEMPLATES:
        mask = _mask_from_rows(rows)
        if len(_components(mask)) != 1:
            continue
        if len(_hole_candidates(mask)) < 4:
            continue
        key = _canonical_key(mask)
        if key in seen:
            continue
        seen.add(key)
        out.append(mask)
    assert len(out) >= 6
    return out


TEMPLATES = _build_template_library()


def _gen_attempt() -> np.ndarray | None:
    rng = RNG
    num_templates = int(rng.integers(3, min(6, len(TEMPLATES) + 1)))
    template_ids = list(rng.choice(len(TEMPLATES), size=num_templates, replace=False))
    colors = list(rng.choice(PALETTE, size=num_templates, replace=False))

    legend_patches = [_make_legend_instance(TEMPLATES[idx], color, rng) for idx, color in zip(template_ids, colors)]
    legend = _pack_legend(legend_patches, rng)
    legend_h, legend_w = map(int, legend.shape)

    num_objects = num_templates + int(rng.integers(1, 3))
    object_templates = template_ids.copy()
    while len(object_templates) < num_objects:
        object_templates.append(int(rng.choice(template_ids)))
    rng.shuffle(object_templates)

    parities = [i % 2 for i in range(num_objects)]
    rng.shuffle(parities)
    color_by_template = {tid: color for tid, color in zip(template_ids, colors)}
    object_inputs: list[np.ndarray] = []
    object_outputs: list[np.ndarray] = []
    for tid, parity in zip(object_templates, parities):
        input_patch, output_patch = _make_object_instance(TEMPLATES[tid], color_by_template[tid], parity, rng)
        object_inputs.append(input_patch)
        object_outputs.append(output_patch)

    min_h = max(legend_h + 3, 18)
    min_w = max(legend_w + 12, 24)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(80):
        grid_h = int(rng.integers(min_h, 31))
        grid_w = int(rng.integers(min_w, 31))
        if grid_h <= legend_h + 2 or grid_w <= legend_w + 8:
            continue
        legend_r0 = int(rng.integers(1, grid_h - legend_h))
        legend_c0 = int(rng.integers(1, 3))
        gap = int(rng.integers(2, 5))
        main_min_col = legend_c0 + legend_w + gap
        positions = _place_objects((grid_h, grid_w), object_inputs, main_min_col, rng)
        if positions is None:
            continue

        inp = np.full((grid_h, grid_w), BLACK, dtype=int)
        _paste(inp, legend, legend_r0, legend_c0)
        for patch, (r0, c0) in zip(object_inputs, positions):
            _paste(inp, patch, r0, c0)

        expected_canvas = np.full((grid_h, grid_w), BLACK, dtype=int)
        for patch, (r0, c0) in zip(object_outputs, positions):
            _paste(expected_canvas, patch, r0, c0)
        expected = _tight_crop(expected_canvas, BLACK)

        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
