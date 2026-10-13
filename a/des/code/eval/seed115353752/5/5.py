import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
COLOR_POOL = (2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 400
SHAPE_ATTEMPTS = 200
SPARSIFY_ATTEMPTS = 200
MAX_TEMPLATE_SIDE = 4


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords) -> np.ndarray:
    pts = np.asarray(list(coords), dtype=int)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(pts)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[pts[:, 0] - r0, pts[:, 1] - c0] = True
    return out


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return tuple()
    r0, c0, r1, c1 = _bbox(pts)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    return tuple("".join("1" if cell else "0" for cell in row) for row in sub)


def _canonical_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = np.asarray(mask, dtype=bool)
    variants = []
    for k in range(4):
        rot = np.rot90(base, k)
        variants.append(_mask_key(rot))
        variants.append(_mask_key(np.fliplr(rot)))
    return min(variants)


def _compress_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    rows = {int(r): i for i, r in enumerate(sorted(set(map(int, arr[:, 0]))))}
    cols = {int(c): i for i, c in enumerate(sorted(set(map(int, arr[:, 1]))))}
    return tuple(sorted((rows[int(r)], cols[int(c)]) for r, c in arr))


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.rot90(out, random.randrange(4)).copy()


def _random_connected_mask(area: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    pos = (r + dr, c + dc)
                    if pos not in cells:
                        frontier.add(pos)
            cells.add(random.choice(tuple(frontier)))
        mask = _coords_to_mask(cells)
        if mask.shape[0] <= MAX_TEMPLATE_SIDE and mask.shape[1] <= MAX_TEMPLATE_SIDE:
            return mask
    return None


def _sample_template_masks(count: int) -> list[np.ndarray] | None:
    out: list[np.ndarray] = []
    seen: set[tuple[str, ...]] = set()
    for _ in range(8 * SHAPE_ATTEMPTS):
        if len(out) >= count:
            return out
        area = random.randint(3, 8)
        mask = _random_connected_mask(area)
        if mask is None:
            continue
        sig = _canonical_signature(mask)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(_random_dihedral(mask))
    return None


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords):
        canvas[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)
    return coords + np.array([r0, c0], dtype=int)


def _sparsify_mask(mask: np.ndarray, force_fragmented: bool) -> np.ndarray | None:
    base = np.asarray(mask, dtype=bool)
    pts = np.argwhere(base)
    h, w = map(int, base.shape)
    axis_choices = [(0, i) for i in range(max(0, h - 1))] + [(1, i) for i in range(max(0, w - 1))]
    for _ in range(SPARSIFY_ATTEMPTS):
        row_gaps = [random.choice((1, 2)) for _ in range(max(0, h - 1))]
        col_gaps = [random.choice((1, 2)) for _ in range(max(0, w - 1))]
        if force_fragmented and axis_choices:
            axis, idx = random.choice(axis_choices)
            if axis == 0:
                row_gaps[idx] = 2
            else:
                col_gaps[idx] = 2

        row_pos = [0]
        for gap in row_gaps:
            row_pos.append(row_pos[-1] + gap)
        col_pos = [0]
        for gap in col_gaps:
            col_pos.append(col_pos[-1] + gap)

        sparse = np.zeros((row_pos[-1] + 1, col_pos[-1] + 1), dtype=bool)
        for r, c in pts:
            sparse[row_pos[int(r)], col_pos[int(c)]] = True

        comps = _components(sparse)
        if not comps:
            continue
        if max(len(comp) for comp in comps) > 4:
            continue
        if force_fragmented and len(comps) < 2:
            continue
        return np.argwhere(sparse)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-color fragments below the separator, compress away empty rows/cols
    inside each color-group layout, then match the recovered shape to grey
    templates above the separator up to rotation/reflection.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = map(int, g.shape)
    sep_rows = [r for r in range(1, h - 1) if np.all(g[r, :] == int(BLUE))]
    assert len(sep_rows) == 1, "expected a unique blue separator row"
    split = sep_rows[0]

    top = g[:split, :].copy()
    bottom = g[split + 1 :, :]

    template_by_sig: dict[tuple[str, ...], list[np.ndarray]] = {}
    for comp in _components(top == int(GREY)):
        local = comp - comp.min(axis=0)
        sig = _canonical_signature(_coords_to_mask(local))
        template_by_sig.setdefault(sig, []).append(comp)

    out = top.copy()
    colors = [int(c) for c in np.unique(bottom) if int(c) not in (BLACK, BLUE, GREY)]
    for color in sorted(colors):
        pts = np.argwhere(bottom == int(color))
        if len(pts) == 0:
            continue
        compressed = _compress_coords(pts)
        sig = _canonical_signature(_coords_to_mask(compressed))
        matches = template_by_sig.get(sig, [])
        if not matches:
            continue
        target = matches.pop(0)
        out[target[:, 0], target[:, 1]] = int(color)
    return out


def _expected_output(top: np.ndarray, template_pixels: list[np.ndarray], recolors: dict[int, int]) -> np.ndarray:
    out = top.copy()
    for idx, color in recolors.items():
        pts = template_pixels[idx]
        out[pts[:, 0], pts[:, 1]] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    template_count = random.randint(3, 4)
    templates = _sample_template_masks(template_count)
    if templates is None:
        return None

    widths = [int(mask.shape[1]) for mask in templates]
    heights = [int(mask.shape[0]) for mask in templates]
    top_gaps = [random.randint(2, 4) for _ in range(template_count - 1)]
    top_span = sum(widths) + sum(top_gaps)
    top_h = max(heights) + random.randint(2, 4)

    matched_count = min(template_count, random.choice((1, 2, 2, 3)))
    matched_indices = random.sample(range(template_count), matched_count)
    fragment_heavy = random.random() < 0.75

    colors = random.sample(COLOR_POOL, matched_count)
    groups: list[dict[str, object]] = []
    fragmented_seen = False
    for rank, (template_idx, color) in enumerate(zip(matched_indices, colors)):
        oriented = _random_dihedral(templates[template_idx])
        force_fragmented = bool(oriented.sum() > 4) or (fragment_heavy and rank == 0) or (random.random() < 0.45)
        sparse = _sparsify_mask(oriented, force_fragmented)
        if sparse is None:
            return None
        sparse = np.asarray(sparse, dtype=int)
        fragmented_seen = fragmented_seen or len(_components(_coords_to_mask(sparse))) > 1
        groups.append(
            {
                "template_idx": int(template_idx),
                "color": int(color),
                "coords": sparse,
                "h": int(sparse[:, 0].max()) + 1,
                "w": int(sparse[:, 1].max()) + 1,
            }
        )
    if fragment_heavy and not fragmented_seen:
        return None

    random.shuffle(groups)
    bottom_gaps = [random.randint(2, 4) for _ in range(max(0, len(groups) - 1))]
    bottom_span = sum(int(group["w"]) for group in groups) + sum(bottom_gaps)
    bottom_h = max(int(group["h"]) for group in groups) + random.randint(2, 5)

    grid_w = max(top_span, bottom_span) + random.randint(3, 7)
    if grid_w <= top_span or grid_w <= bottom_span:
        return None

    top = np.zeros((top_h, grid_w), dtype=int)
    bottom = np.zeros((bottom_h, grid_w), dtype=int)

    top_left = random.randint(1, grid_w - top_span - 1)
    template_pixels: list[np.ndarray] = []
    cursor = top_left
    for idx, mask in enumerate(templates):
        h0, w0 = map(int, mask.shape)
        r0 = random.randint(0, top_h - h0)
        template_pixels.append(_paint_mask(top, mask, r0, cursor, GREY))
        if idx < len(top_gaps):
            cursor += w0 + top_gaps[idx]
        else:
            cursor += w0

    bottom_left = random.randint(1, grid_w - bottom_span - 1)
    cursor = bottom_left
    recolors: dict[int, int] = {}
    for idx, group in enumerate(groups):
        coords = np.asarray(group["coords"], dtype=int)
        gh = int(group["h"])
        gw = int(group["w"])
        r0 = random.randint(0, bottom_h - gh)
        color = int(group["color"])
        bottom[r0 + coords[:, 0], cursor + coords[:, 1]] = color
        recolors[int(group["template_idx"])] = color
        if idx < len(bottom_gaps):
            cursor += gw + bottom_gaps[idx]
        else:
            cursor += gw

    grid = np.vstack([top, np.full((1, grid_w), BLUE, dtype=int), bottom])
    expected = _expected_output(top, template_pixels, recolors)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
