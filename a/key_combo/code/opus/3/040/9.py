from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

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
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def component_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = bbox_coords(coords)
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, box


def shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "shape must be non-empty"
    coords = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def mask_from_signature(sig: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _ in sig]
    cols = [c for _, c in sig]
    mask = np.zeros((max(rows) + 1, max(cols) + 1), dtype=bool)
    for r, c in sig:
        mask[r, c] = True
    return mask


def canonical_shape(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(shape_signature(rotated))
        variants.append(shape_signature(np.fliplr(rotated)))
    return min(variants)


def unique_variants(sig: tuple[tuple[int, int], ...]) -> list[np.ndarray]:
    base = mask_from_signature(sig)
    seen: set[tuple[tuple[int, int], ...]] = set()
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(base, k)
        for arr in (rotated, np.fliplr(rotated)):
            cur = canonical_shape(arr)
            raw = shape_signature(arr)
            if raw not in seen:
                seen.add(raw)
                variants.append(mask_from_signature(raw))
            assert cur == sig
    return variants


def dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            dst_r0 = max(0, dr)
            src_c0 = max(0, -dc)
            dst_c0 = max(0, dc)
            rh = h - abs(dr)
            cw = w - abs(dc)
            if rh <= 0 or cw <= 0:
                continue
            out[dst_r0 : dst_r0 + rh, dst_c0 : dst_c0 + cw] |= mask[
                src_r0 : src_r0 + rh, src_c0 : src_c0 + cw
            ]
    return out


def row_runs(row: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start = 0
    cur = int(row[0])
    for idx in range(1, len(row) + 1):
        if idx == len(row) or int(row[idx]) != cur:
            runs.append((start, idx - 1, cur))
            if idx < len(row):
                start = idx
                cur = int(row[idx])
    return runs


def place_mask(
    grid: np.ndarray,
    mask: np.ndarray,
    color: int,
    forbidden: np.ndarray,
) -> np.ndarray | None:
    h, w = grid.shape
    mh, mw = mask.shape
    fg = np.argwhere(mask)
    blocked = dilate8(grid != BLACK)
    candidates: list[tuple[int, int]] = []

    for r0 in range(h - mh + 1):
        for c0 in range(w - mw + 1):
            rows = r0 + fg[:, 0]
            cols = c0 + fg[:, 1]
            if np.any(forbidden[rows, cols] | blocked[rows, cols]):
                continue
            candidates.append((r0, c0))

    if not candidates:
        return None

    r0, c0 = random.choice(candidates)
    rows = r0 + fg[:, 0]
    cols = c0 + fg[:, 1]
    grid[rows, cols] = int(color)
    return np.column_stack((rows, cols)).astype(int)


RAW_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
]
SHAPE_LIBRARY: list[tuple[tuple[int, int], ...]] = []
for raw in RAW_SHAPES:
    sig = canonical_shape(raw)
    if sig not in SHAPE_LIBRARY:
        SHAPE_LIBRARY.append(sig)


def _find_divider_and_silhouettes(
    grid: np.ndarray,
) -> tuple[np.ndarray, int, int, list[np.ndarray]]:
    h, w = grid.shape
    grey_comps = components_mask(grid == GREY, connectivity=4)
    assert grey_comps, "expected grey divider and silhouettes"

    divider_idx = None
    divider = None
    for idx, comp in enumerate(grey_comps):
        rows = comp[:, 0]
        cols = comp[:, 1]
        if np.any(rows == 0) and np.any(cols == w - 1):
            if divider is None or comp.shape[0] > divider.shape[0]:
                divider_idx = idx
                divider = comp

    assert divider_idx is not None and divider is not None, "expected a grey L divider"
    top_cols = sorted({int(c) for r, c in divider if r == 0})
    right_rows = sorted({int(r) for r, c in divider if c == w - 1})
    assert len(top_cols) == 1 and len(right_rows) == 1
    split_c = top_cols[0]
    split_r = right_rows[0]
    assert all(
        (int(c) == split_c and int(r) <= split_r) or (int(r) == split_r and int(c) >= split_c)
        for r, c in divider
    ), "grey border component must be an L"

    silhouettes = [comp for idx, comp in enumerate(grey_comps) if idx != divider_idx]
    assert silhouettes, "expected at least one grey silhouette"
    return divider, split_r, split_c, silhouettes


def _template_specs(grid: np.ndarray, split_r: int, split_c: int) -> list[dict]:
    legend = grid[:split_r, :split_c]
    comps = components_mask((legend != BLACK) & (legend != GREY), connectivity=4)
    specs = []
    for comp in comps:
        colors = {int(legend[r, c]) for r, c in comp}
        assert len(colors) == 1, "legend template must be monochromatic"
        color = next(iter(colors))
        mask, box = component_mask(comp)
        specs.append(
            {
                "color": color,
                "bbox": box,
                "sig": canonical_shape(mask),
            }
        )
    assert len(specs) == 3, "expected exactly three legend templates"
    specs.sort(key=lambda spec: (spec["bbox"][0], spec["bbox"][1]))
    assert len({spec["sig"] for spec in specs}) == 3, "legend shapes must be distinct"
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """Match grey silhouettes to legend templates, elect the majority shape, and erase the minority.

    Input structure:
    - Black background with a grey L-shaped divider touching the top edge and right edge.
    - The top-left legend region contains three small colored template sprites with distinct shapes.
    - Outside the legend are several grey 4-connected silhouettes matching those templates up to rotation or reflection.
    - The bottom row is a three-segment color bar whose left-to-right segment colors follow the legend templates from top to bottom.

    Rule:
    - Extract the three legend templates and compare each grey silhouette by shape only, allowing all rotations and reflections.
    - Count how many silhouettes belong to each template and find the unique majority template.
    - Recolor only the majority silhouettes to that template's color, erase all minority silhouettes to black, and keep only the matching bottom color-bar segment.
    - Leave the legend templates and the grey divider unchanged.

    Key insight:
    - Shape matching only produces class labels; those labels are then tallied in a vote, and the vote winner determines both which silhouettes survive and which color-bar segment remains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5

    _divider, split_r, split_c, silhouettes = _find_divider_and_silhouettes(g)
    templates = _template_specs(g, split_r, split_c)
    template_sigs = [template["sig"] for template in templates]
    template_colors = [template["color"] for template in templates]

    runs = row_runs(g[-1])
    assert len(runs) == 3, "bottom bar must have three segments"
    seg_w = runs[0][1] - runs[0][0] + 1
    assert all((c1 - c0 + 1) == seg_w for c0, c1, _color in runs), "segments must be equal width"
    assert [color for _c0, _c1, color in runs] == template_colors, "bar order must match legend order"

    counts = [0, 0, 0]
    labels: list[tuple[np.ndarray, int]] = []
    for comp in silhouettes:
        mask, _box = component_mask(comp)
        sig = canonical_shape(mask)
        matches = [idx for idx, template_sig in enumerate(template_sigs) if sig == template_sig]
        assert len(matches) == 1, "every silhouette must match exactly one template"
        label = matches[0]
        counts[label] += 1
        labels.append((comp, label))

    max_count = max(counts)
    assert counts.count(max_count) == 1, "majority template must be unique"
    majority_idx = counts.index(max_count)

    out = g.copy()
    for comp, label in labels:
        rows = comp[:, 0]
        cols = comp[:, 1]
        if label == majority_idx:
            out[rows, cols] = template_colors[label]
        else:
            out[rows, cols] = BLACK

    for idx, (c0, c1, _color) in enumerate(runs):
        if idx != majority_idx:
            out[-1, c0 : c1 + 1] = BLACK

    return out


def _sample_counts(total: int, majority_idx: int) -> tuple[int, int, int]:
    options = []
    for a in range(1, total - 1):
        for b in range(1, total - a):
            c = total - a - b
            counts = (a, b, c)
            best = max(counts)
            if counts.count(best) == 1 and counts[majority_idx] == best:
                options.append(counts)
    assert options
    return random.choice(options)


def _build_expected(
    inp: np.ndarray,
    silhouette_specs: list[tuple[int, np.ndarray]],
    template_colors: list[int],
    seg_w: int,
    majority_idx: int,
) -> np.ndarray:
    out = inp.copy()
    for label, coords in silhouette_specs:
        rows = coords[:, 0]
        cols = coords[:, 1]
        if label == majority_idx:
            out[rows, cols] = template_colors[label]
        else:
            out[rows, cols] = BLACK
    for idx in range(3):
        if idx != majority_idx:
            out[-1, idx * seg_w : (idx + 1) * seg_w] = BLACK
    return out


def _gen_attempt() -> np.ndarray | None:
    template_sigs = random.sample(SHAPE_LIBRARY, 3)
    random.shuffle(template_sigs)
    legend_masks = [random.choice(unique_variants(sig)) for sig in template_sigs]
    template_colors = random.sample(PALETTE, 3)

    y = random.randint(0, 1)
    tops: list[int] = []
    for idx, mask in enumerate(legend_masks):
        tops.append(y)
        y += mask.shape[0]
        if idx < len(legend_masks) - 1:
            y += random.randint(1, 2)
    split_r = y + random.randint(1, 2)
    max_w = max(mask.shape[1] for mask in legend_masks)
    split_c = max_w + random.randint(2, 4)
    lefts = [random.randint(0, split_c - mask.shape[1] - 1) for mask in legend_masks]

    seg_w = random.randint(max(5, split_c - 1), 8)
    width = 3 * seg_w
    height = split_r + random.randint(6, 9)
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[: split_r + 1, split_c] = GREY
    grid[split_r, split_c:] = GREY

    for color, mask, top, left in zip(template_colors, legend_masks, tops, lefts):
        fg = np.argwhere(mask)
        grid[top + fg[:, 0], left + fg[:, 1]] = color

    for idx, color in enumerate(template_colors):
        grid[-1, idx * seg_w : (idx + 1) * seg_w] = color

    silhouette_forbidden = np.zeros_like(grid, dtype=bool)
    silhouette_forbidden[:split_r, :split_c] = True
    silhouette_forbidden[0, :] = True
    silhouette_forbidden[-1, :] = True
    silhouette_forbidden[:, 0] = True
    silhouette_forbidden[:, -1] = True

    total_silhouettes = random.randint(5, 7)
    majority_idx = random.randrange(3)
    counts = _sample_counts(total_silhouettes, majority_idx)

    silhouette_jobs: list[tuple[int, np.ndarray]] = []
    for label, count in enumerate(counts):
        for _ in range(count):
            silhouette_jobs.append((label, random.choice(unique_variants(template_sigs[label]))))
    random.shuffle(silhouette_jobs)
    silhouette_jobs.sort(key=lambda item: (-int(item[1].sum()), -(item[1].shape[0] * item[1].shape[1])))

    placed_silhouettes: list[tuple[int, np.ndarray]] = []
    for label, mask in silhouette_jobs:
        coords = place_mask(grid, mask, GREY, silhouette_forbidden)
        if coords is None:
            return None
        placed_silhouettes.append((label, coords))

    expected = _build_expected(grid, placed_silhouettes, template_colors, seg_w, majority_idx)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
