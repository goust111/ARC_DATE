from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def mask_from_coords(coords, shape: tuple[int, int]) -> np.ndarray:
    """Local helper: build a boolean mask from bbox-relative component coordinates."""
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        assert 0 <= rr < H and 0 <= cc < W
        m[rr, cc] = True
    return m


def canonical_shape_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    """Canonicalize a boolean mask up to rotation and horizontal reflection."""
    canon = np.asarray(mask, dtype=bool)
    best = (canon.tobytes(), tuple(map(int, canon.shape)))
    for _ in range(4):
        canon = np.rot90(canon)
        for v in (canon, np.fliplr(canon)):
            key = (v.tobytes(), tuple(map(int, v.shape)))
            if key < best:
                best = key
    return best

def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort objects to top or bottom edge by whether their shape (up to rotation/flip) is shared.

    Input structure:
    - bg = most common color.
    - Several multicolor connected components (8-connected) scattered across the grid.

    Rule:
    - Compute each object's canonical shape (binary mask, invariant to rotation and flip).
    - Objects whose shape appears in two or more components → translate to TOP edge (row 0).
    - Objects whose shape is unique → translate to BOTTOM edge (row H−h).
    - Horizontal position (left column) is preserved; only vertical translation changes.

    Key insight: The sorting criterion is shape frequency (rotation-invariant), not area — duplicated shapes go up, unique shapes go down.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert not (not comps)

    extracted = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))

        # Isolate this component exactly (bbox rectangle may include background).
        rel_coords = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = mask_from_coords(rel_coords, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]

        extracted.append(
            {
                "c0": int(c0),
                "pat": pat,
                "shape_key": canonical_shape_key(m),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(H - h)
        dest_c0 = int(d["c0"])
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))

    return out

# --------------- generator (att2) ---------------

_RNG = np.random.default_rng()

# ── CFG: canvas ──────────────────────────────────────────────────────
GRID_H_RANGE = (19, 31)          # H uniform in [19, 30]
GRID_W_RANGE = (19, 31)          # W uniform in [19, 30]
NUM_COLORS = 10                   # palette 0-9

# ── CFG: component shapes ───────────────────────────────────────────
BBOX_MAX = 5                      # max h or w of a component bbox
AREA_RANGE = (3, 13)              # component area in [3, 12]
MIN_FILL_RATIO_X10 = 4            # area * 10 >= 4 * h * w
IRREGULAR_PROB = 0.85             # prefer non-rectangular bboxes

# ── CFG: layout ─────────────────────────────────────────────────────
DUP_LAYOUTS = [(2,), (3,), (2, 2)]  # possible duplicate-group structures
MIN_COL_GAP = 1                   # minimum column gap between components
MIN_VERTICAL_GAP = 5              # min gap between top-group bottom and bottom-group top

# ── CFG: generation ─────────────────────────────────────────────────
MAX_RETRIES = 300                 # outer generate_input retry budget

_NEIGHBORS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1)]


def _feasible_dims(area: int) -> list[tuple[int, int]]:
    """Return all (h, w) bboxes where area fits with fill_ratio >= MIN_FILL_RATIO_X10/10."""
    dims = []
    for h in range(1, BBOX_MAX + 1):
        for w in range(1, BBOX_MAX + 1):
            if area > h * w:
                continue
            if area * 10 < MIN_FILL_RATIO_X10 * h * w:
                continue
            if h == 1 and area != w:
                continue
            if w == 1 and area != h:
                continue
            if h > 1 and w > 1 and area < max(h, w):
                continue
            dims.append((h, w))
    return dims


def _line_mask(h: int, w: int) -> np.ndarray:
    """Create a seed mask along a diagonal line spanning the bbox, randomly flipped."""
    mask = np.zeros((h, w), dtype=bool)
    if h == 1:
        mask[0, :] = True
        return mask
    if w == 1:
        mask[:, 0] = True
        return mask
    steps = max(h, w)
    for i in range(steps):
        r = int(round(i * (h - 1) / (steps - 1)))
        c = int(round(i * (w - 1) / (steps - 1)))
        mask[r, c] = True
    if bool(_RNG.integers(0, 2)):
        mask = np.flipud(mask)
    if bool(_RNG.integers(0, 2)):
        mask = np.fliplr(mask)
    return mask


def _make_mask(h: int, w: int, area: int) -> np.ndarray:
    """Grow an 8-connected mask from a line seed to the target area."""
    mask = _line_mask(h, w)
    while int(mask.sum()) < area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in _NEIGHBORS8:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        pick = list(frontier)
        nr, nc = pick[int(_RNG.integers(0, len(pick)))]
        mask[nr, nc] = True
    return mask


def _mask_to_pat(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    """Paint a boolean mask into a colored patch on a bg-filled array."""
    pat = np.full(mask.shape, bg, dtype=int)
    pat[mask] = color
    return pat


def _make_component(area: int, color: int, bg: int) -> dict:
    """Generate a random 8-connected component with the given area and color."""
    dims = _feasible_dims(area)
    irregular_dims = [d for d in dims if d[0] * d[1] > area]
    if irregular_dims and _RNG.random() < IRREGULAR_PROB:
        dims = irregular_dims
    h, w = dims[int(_RNG.integers(0, len(dims)))]
    mask = _make_mask(h, w, area)
    return {"area": area, "color": color, "h": h, "w": w,
            "mask": mask, "pat": _mask_to_pat(mask, color, bg),
            "shape_key": canonical_shape_key(mask)}


def _random_transform(mask: np.ndarray) -> np.ndarray:
    """Apply a random rotation (0-3) and optional horizontal flip to a mask."""
    mask = np.rot90(mask, int(_RNG.integers(0, 4)))
    if bool(_RNG.integers(0, 2)):
        mask = np.fliplr(mask)
    return mask


def _clone_component(template: dict, color: int, bg: int) -> dict:
    """Create a duplicate of template's shape with a new color and random transform."""
    mask = _random_transform(template["mask"].copy())
    h, w = mask.shape
    return {"area": template["area"], "color": color, "h": h, "w": w,
            "mask": mask, "pat": _mask_to_pat(mask, color, bg),
            "shape_key": canonical_shape_key(mask)}


def _assign_columns(widths: list[int], W: int) -> list[int] | None:
    """Distribute components horizontally with random spacing (min gap = MIN_COL_GAP)."""
    n = len(widths)
    total = sum(widths) + MIN_COL_GAP * (n - 1)
    if total > W:
        return None
    slack = W - total
    extras = _RNG.multinomial(slack, np.ones(n + 1) / (n + 1))
    cols = []
    c = int(extras[0])
    for i, w in enumerate(widths):
        cols.append(c)
        c += w
        if i < n - 1:
            c += MIN_COL_GAP + int(extras[i + 1])
    return cols


def _dilate_occupied(occupied: np.ndarray) -> np.ndarray:
    """Return a mask that is True wherever occupied or any 8-neighbor is occupied."""
    dilated = occupied.copy()
    dilated[1:, :] |= occupied[:-1, :]
    dilated[:-1, :] |= occupied[1:, :]
    dilated[:, 1:] |= occupied[:, :-1]
    dilated[:, :-1] |= occupied[:, 1:]
    dilated[1:, 1:] |= occupied[:-1, :-1]
    dilated[1:, :-1] |= occupied[:-1, 1:]
    dilated[:-1, 1:] |= occupied[1:, :-1]
    dilated[:-1, :-1] |= occupied[1:, 1:]
    return dilated


def _can_place(dilated: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    """Check if mask placed at (r0, c0) overlaps any dilated-occupied cell."""
    h, w = mask.shape
    return not bool((dilated[r0:r0 + h, c0:c0 + w] & mask).any())


def _build_output(comps: list[dict], H: int, W: int, bg: int) -> np.ndarray:
    """Paint all components at their output positions."""
    out = np.full((H, W), bg, dtype=int)
    for comp in comps:
        paste(out, comp["pat"], comp["out_r0"], comp["c0"], transparent=bg)
    return out


def _derive_input(comps: list[dict], H: int, W: int, bg: int) -> np.ndarray | None:
    """Place components at random non-output rows without 8-neighbor overlap."""
    inp = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    order = list(range(len(comps)))
    _RNG.shuffle(order)
    order.sort(key=lambda i: comps[i]["h"] * comps[i]["w"], reverse=True)

    for i in order:
        comp = comps[i]
        h, c0 = comp["h"], comp["c0"]
        rows = [r for r in range(H - h + 1) if r != comp["out_r0"]]
        if not rows:
            rows = list(range(H - h + 1))
        _RNG.shuffle(rows)

        dilated = _dilate_occupied(occupied)
        valid = [r for r in rows if _can_place(dilated, comp["mask"], r, c0)]
        if not valid:
            valid = [r for r in range(H - h + 1)
                     if _can_place(dilated, comp["mask"], r, c0)]
        if not valid:
            return None

        r0 = valid[int(_RNG.integers(0, len(valid)))]
        comp["in_r0"] = r0
        paste(inp, comp["pat"], r0, c0, transparent=bg)
        mh, mw = comp["mask"].shape
        occupied[r0:r0 + mh, c0:c0 + mw] |= comp["mask"]
    return inp


def _build_groups(bg: int) -> tuple[list[dict], list[dict]]:
    """Create duplicate-shape (top) and unique-shape (bottom) component groups."""
    dup_layout = DUP_LAYOUTS[int(_RNG.integers(0, len(DUP_LAYOUTS)))]
    n_singletons = 2 if sum(dup_layout) == 4 else int(_RNG.integers(2, 4))

    colors = [c for c in range(NUM_COLORS) if c != bg]
    _RNG.shuffle(colors)

    used_shape_keys: set[tuple[bytes, tuple[int, int]]] = set()
    top, ci = [], 0
    for count in dup_layout:
        for _ in range(200):
            area = int(_RNG.integers(*AREA_RANGE))
            template = _make_component(area, colors[ci], bg)
            if template["shape_key"] not in used_shape_keys:
                used_shape_keys.add(template["shape_key"])
                break
        else:
            raise RuntimeError("failed to create a new duplicate template shape")
        top.append(template)
        ci += 1
        for _ in range(count - 1):
            top.append(_clone_component(template, colors[ci], bg))
            ci += 1

    bottom = []
    while len(bottom) < n_singletons:
        area = int(_RNG.integers(*AREA_RANGE))
        comp = _make_component(area, colors[ci], bg)
        if comp["shape_key"] in used_shape_keys:
            continue
        used_shape_keys.add(comp["shape_key"])
        bottom.append(comp)
        ci += 1
    return top, bottom


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build output-first, derive input, validate with oracle."""
    H = int(_RNG.integers(*GRID_H_RANGE))
    W = int(_RNG.integers(*GRID_W_RANGE))
    bg = int(_RNG.integers(0, NUM_COLORS))

    top, bottom = _build_groups(bg)
    _RNG.shuffle(top)
    _RNG.shuffle(bottom)

    if H - max(c["h"] for c in top) - max(c["h"] for c in bottom) < MIN_VERTICAL_GAP:
        return None

    top_cols = _assign_columns([c["w"] for c in top], W)
    bottom_cols = _assign_columns([c["w"] for c in bottom], W)
    if top_cols is None or bottom_cols is None:
        return None

    for comp, c0 in zip(top, top_cols):
        comp["c0"], comp["out_r0"] = c0, 0
    for comp, c0 in zip(bottom, bottom_cols):
        comp["c0"], comp["out_r0"] = c0, H - comp["h"]

    comps = top + bottom
    expected = _build_output(comps, H, W, bg)
    inp = _derive_input(comps, H, W, bg)
    if inp is None:
        return None
    if int(get_bg(inp)) != bg:
        return None

    found = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(found) != len(comps):
        return None
    if Counter(len(c["coords"]) for c in found) != Counter(c["area"] for c in comps):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 62593bfd sample")
