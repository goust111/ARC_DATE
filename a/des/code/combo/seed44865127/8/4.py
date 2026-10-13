import random
from collections import Counter, deque

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 400
MIN_LAYERS = 2
MAX_LAYERS = 4
MIN_OUTER_SIZE = 7
MAX_OUTER_SIZE = 18
INNER_MIN_SIZE = 2
FRAME_THICKNESSES = (1, 2)
FRAME_GAPS = (0, 1, 2)
MARGIN_MIN = 1
MARGIN_MAX = 5


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    rows = pts[:, 0]
    cols = pts[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        mask = grid == color
        seen = np.zeros((h, w), dtype=bool)
        for r0, c0 in np.argwhere(mask):
            r = int(r0)
            c = int(c0)
            if seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _direct_children(comps: list[dict]) -> list[list[int]]:
    n = len(comps)
    parent = [-1] * n
    children = [[] for _ in range(n)]
    for i in range(n):
        containers = [j for j in range(n) if i != j and _strictly_contains(comps[j]["bbox"], comps[i]["bbox"])]
        if containers:
            parent[i] = min(containers, key=lambda j: (_bbox_area(comps[j]["bbox"]), j))
            children[parent[i]].append(i)
    return children


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the innermost object's color outward through nested foreground layers.

    Input structure:
    - The dominant color is the background.
    - Foreground objects form one or more nested chains by bounding-box containment.
    - Outer and middle layers are frames; the innermost layer is a solid object.

    Rule:
    - Build the nesting hierarchy from component bounding boxes.
    - Keep the innermost object unchanged.
    - Recolor every outer layer with the color of its direct inner child.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    comps = _find_components(g, bg)
    assert len(comps) >= 2, "expected at least two nested foreground objects"

    children = _direct_children(comps)

    out = g.copy()
    for i, child_ids in enumerate(children):
        if not child_ids:
            continue
        assert len(child_ids) == 1, "expected a single direct child per nested layer"
        child_color = int(comps[child_ids[0]]["color"])
        rows, cols = zip(*comps[i]["coords"])
        out[rows, cols] = child_color
    return out


def _frame_mask(height: int, width: int, thickness: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    mask[thickness:height - thickness, thickness:width - thickness] = False
    return mask


def _diamond_mask(height: int, width: int) -> np.ndarray:
    rr, cc = np.indices((height, width))
    cr = (height - 1) / 2.0
    cc0 = (width - 1) / 2.0
    dr = np.abs(rr - cr) / max(cr, 0.5)
    dc = np.abs(cc - cc0) / max(cc0, 0.5)
    return dr + dc <= 1.0 + 1e-9


def _inner_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    kinds = ["rect"]
    if height >= 3 and width >= 3:
        kinds.append("diamond")
        if height >= 4 and width >= 4:
            kinds.append("cut_corners")
    kind = rng.choice(kinds)
    if kind == "rect":
        return np.ones((height, width), dtype=bool)
    if kind == "cut_corners":
        mask = np.ones((height, width), dtype=bool)
        mask[0, 0] = False
        mask[0, -1] = False
        mask[-1, 0] = False
        mask[-1, -1] = False
        return mask
    mask = _diamond_mask(height, width)
    bb = _mask_bbox(mask)
    if bb is None or bb != (0, 0, height - 1, width - 1):
        return np.ones((height, width), dtype=bool)
    return mask


def _shrink_bbox(bbox: tuple[int, int, int, int], amount: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + amount, c0 + amount, r1 - amount, c1 - amount


def _bbox_shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_layers = rng.randint(MIN_LAYERS, MAX_LAYERS)

    outer_min = max(MIN_OUTER_SIZE, INNER_MIN_SIZE + 2 * (n_layers - 1))
    outer_h = rng.randint(outer_min, MAX_OUTER_SIZE)
    outer_w = rng.randint(outer_min, MAX_OUTER_SIZE)
    current_bbox = (0, 0, outer_h - 1, outer_w - 1)

    frame_specs: list[dict] = []
    for frame_index in range(n_layers - 1):
        remaining_frames = (n_layers - 1) - frame_index - 1
        min_next_h = INNER_MIN_SIZE + 2 * remaining_frames
        min_next_w = INNER_MIN_SIZE + 2 * remaining_frames

        feasible: list[tuple[int, int]] = []
        for thickness in FRAME_THICKNESSES:
            for gap in FRAME_GAPS:
                shrink = thickness + gap
                next_h = _bbox_shape(current_bbox)[0] - 2 * shrink
                next_w = _bbox_shape(current_bbox)[1] - 2 * shrink
                if next_h >= min_next_h and next_w >= min_next_w:
                    feasible.append((thickness, gap))
        if not feasible:
            return None

        thickness, gap = rng.choice(feasible)
        frame_specs.append({"bbox": current_bbox, "thickness": thickness})
        current_bbox = _shrink_bbox(current_bbox, thickness + gap)

    inner_h, inner_w = _bbox_shape(current_bbox)
    inner_spec = {"bbox": current_bbox, "mask": _inner_mask(inner_h, inner_w, rng)}

    top_margin = rng.randint(MARGIN_MIN, MARGIN_MAX)
    bottom_margin = rng.randint(MARGIN_MIN, MARGIN_MAX)
    left_margin = rng.randint(MARGIN_MIN, MARGIN_MAX)
    right_margin = rng.randint(MARGIN_MIN, MARGIN_MAX)

    grid_h = outer_h + top_margin + bottom_margin
    grid_w = outer_w + left_margin + right_margin

    palette = list(range(10))
    bg = rng.choice(palette)
    fg_palette = [c for c in palette if c != bg]
    layer_colors = rng.sample(fg_palette, n_layers)

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    out = inp.copy()
    window_in = inp[top_margin:top_margin + outer_h, left_margin:left_margin + outer_w]
    window_out = out[top_margin:top_margin + outer_h, left_margin:left_margin + outer_w]

    layers_outer_to_inner: list[tuple[np.ndarray, int]] = []
    for idx, spec in enumerate(frame_specs):
        r0, c0, r1, c1 = spec["bbox"]
        mask = np.zeros((outer_h, outer_w), dtype=bool)
        local_h, local_w = _bbox_shape(spec["bbox"])
        mask[r0:r1 + 1, c0:c1 + 1] = _frame_mask(local_h, local_w, spec["thickness"])
        layers_outer_to_inner.append((mask, layer_colors[idx]))

    r0, c0, r1, c1 = inner_spec["bbox"]
    inner_mask_global = np.zeros((outer_h, outer_w), dtype=bool)
    inner_mask_global[r0:r1 + 1, c0:c1 + 1] = inner_spec["mask"]
    layers_outer_to_inner.append((inner_mask_global, layer_colors[-1]))

    for i, (mask, color) in enumerate(layers_outer_to_inner):
        window_in[mask] = color
        replacement = color if i == len(layers_outer_to_inner) - 1 else layers_outer_to_inner[i + 1][1]
        window_out[mask] = replacement

    values, counts = np.unique(inp, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(values, counts)}
    if freq[bg] <= max(freq[color] for color in freq if color != bg):
        return None

    comps = _find_components(inp, bg)
    if len(comps) != n_layers:
        return None
    if any(len(child_ids) > 1 for child_ids in _direct_children(comps)):
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        case = _build_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid nested-color-propagation input")
