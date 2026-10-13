import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
WORK_SIZE = 30
GEN_RETRIES = 400

_BASE_SHAPES = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if len(points) != expected:
        return False
    cells = set(points)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in cells) != on_border:
                return False
    return True


def _template_offsets(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    grey = np.argwhere(interior == GREY)
    assert grey.size > 0, "each template frame must contain grey pixels"
    gr0, gc0 = grey.min(axis=0)
    gr1, gc1 = grey.max(axis=0)
    assert (gr1 - gr0) % 2 == 0 and (gc1 - gc0) % 2 == 0, "template bbox must have a center"
    anchor_r = (int(gr0) + int(gr1)) // 2
    anchor_c = (int(gc0) + int(gc1)) // 2
    return sorted((int(r) - anchor_r, int(c) - anchor_c) for r, c in grey)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each colored marker as a key to select a same-colored template frame and stamp that template at the marker location.

    Input structure:
    - The background is black.
    - Several one-pixel-thick colored rectangular frames sit near the top; each frame contains a grey template shape on black.
    - Lower in the grid, scattered single-pixel colored markers use the same colors as the template-frame borders.

    Rule:
    - Detect each colored rectangular frame and extract the grey template inside it.
    - For every single-pixel marker, find the template whose border color matches the marker color.
    - Clear the original marker and repaint the template shape in the marker color, centered on the marker position.
    - Leave the template frames themselves unchanged.

    Key insight:
    - The marker color selects which stored grey shape to use, while the marker position tells where to stamp it.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) not in (BG, GREY)]
    components_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    templates: dict[int, list[tuple[int, int]]] = {}

    for color in colors:
        comps = _components(g == color)
        components_by_color[color] = comps
        for comp in comps:
            if _is_rect_frame(comp):
                assert color not in templates, "each template color must have exactly one frame"
                templates[color] = _template_offsets(g, _bbox(comp))

    assert templates, "expected at least one template frame"
    assert set(templates) == set(colors), "every non-grey foreground color should correspond to one template frame"

    markers: list[tuple[int, int, int]] = []
    for color, comps in components_by_color.items():
        for comp in comps:
            if _is_rect_frame(comp):
                continue
            assert len(comp) == 1, "non-frame objects must be single-pixel markers"
            r, c = comp[0]
            markers.append((color, r, c))

    for color, r, c in sorted(markers):
        out[r, c] = BG
        for dr, dc in templates[color]:
            nr = r + dr
            nc = c + dc
            assert 0 <= nr < h and 0 <= nc < w, "stamp must stay inside the grid"
            out[nr, nc] = color

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            transformed = _trim_mask(np.rot90(base, k))
            key = (transformed.tobytes(), transformed.shape)
            if key not in seen:
                seen.add(key)
                variants.append(transformed)
    return variants


_SHAPE_CATALOG = [variant for shape in _BASE_SHAPES for variant in _unique_transforms(shape)]


def _shape_offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    assert h % 2 == 1 and w % 2 == 1, "template mask bbox must be odd-sized"
    anchor_r = h // 2
    anchor_c = w // 2
    assert mask[anchor_r, anchor_c], "template center must be filled"
    pts = np.argwhere(mask)
    return sorted((int(r) - anchor_r, int(c) - anchor_c) for r, c in pts)


def _build_frame_patch(color: int, shape: np.ndarray, margin_r: int, margin_c: int) -> np.ndarray:
    sh, sw = shape.shape
    interior_h = sh + 2 * margin_r
    interior_w = sw + 2 * margin_c
    patch = np.full((interior_h + 2, interior_w + 2), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    rr = 1 + margin_r
    cc = 1 + margin_c
    interior = patch[rr : rr + sh, cc : cc + sw]
    interior[shape] = GREY
    return patch


def _free_patch_positions(
    occupied: np.ndarray,
    patch_h: int,
    patch_w: int,
    r_min: int,
    r_max: int,
    c_min: int,
    c_max: int,
    pad: int,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    top_min = max(0, r_min)
    top_max = min(h - patch_h, r_max)
    left_min = max(0, c_min)
    left_max = min(w - patch_w, c_max)
    for r0 in range(top_min, top_max + 1):
        for c0 in range(left_min, left_max + 1):
            rr0 = max(0, r0 - pad)
            rr1 = min(h, r0 + patch_h + pad)
            cc0 = max(0, c0 - pad)
            cc1 = min(w, c0 + patch_w + pad)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    region = patch != BG
    target = grid[top : top + ph, left : left + pw]
    target[region] = patch[region]
    occupied[top : top + ph, left : left + pw] = True


def _bbox_from_offsets(offsets: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    drs = [dr for dr, _ in offsets]
    dcs = [dc for _, dc in offsets]
    return min(drs), min(dcs), max(drs), max(dcs)


def _marker_positions(
    occupied: np.ndarray,
    offsets: list[tuple[int, int]],
    anchor_r_min: int,
    pad: int,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    dr0, dc0, dr1, dc1 = _bbox_from_offsets(offsets)
    positions: list[tuple[int, int]] = []
    for r in range(max(anchor_r_min, -dr0), h - dr1):
        for c in range(-dc0, w - dc1):
            rr0 = max(0, r + dr0 - pad)
            rr1 = min(h, r + dr1 + pad + 1)
            cc0 = max(0, c + dc0 - pad)
            cc1 = min(w, c + dc1 + pad + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r, c))
    return positions


def _used_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_with_margin(
    grid: np.ndarray, keep_mask: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = _used_bbox(keep_mask)
    top = rng.randint(0, min(2, r0))
    left = rng.randint(0, min(2, c0))
    bottom = rng.randint(0, min(2, h - 1 - r1))
    right = rng.randint(0, min(2, w - 1 - c1))
    crop = (r0 - top, c0 - left, r1 + bottom, c1 + right)
    cr0, cc0, cr1, cc1 = crop
    return grid[cr0 : cr1 + 1, cc0 : cc1 + 1].copy(), crop


def _apply_expected_stamps(
    grid: np.ndarray, markers: list[tuple[int, int, int, list[tuple[int, int]]]]
) -> np.ndarray:
    out = grid.copy()
    for color, r, c, offsets in markers:
        out[r, c] = BG
        for dr, dc in offsets:
            out[r + dr, c + dc] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.full((WORK_SIZE, WORK_SIZE), BG, dtype=int)
    occupied = np.zeros((WORK_SIZE, WORK_SIZE), dtype=bool)

    template_count = rng.randint(2, 4)
    colors = rng.sample(list(FRAME_COLORS), template_count)
    shape_indices = rng.sample(range(len(_SHAPE_CATALOG)), template_count)

    frame_specs = []
    for color, idx in zip(colors, shape_indices):
        shape = _SHAPE_CATALOG[idx]
        margin_r = rng.randint(1, 2)
        margin_c = rng.randint(1, 2)
        patch = _build_frame_patch(color, shape, margin_r, margin_c)
        frame_specs.append((color, shape, _shape_offsets(shape), patch))

    rng.shuffle(frame_specs)
    frame_bottom = -1
    top_band_limit = 12
    for color, shape, offsets, patch in frame_specs:
        ph, pw = patch.shape
        positions = _free_patch_positions(
            occupied=occupied,
            patch_h=ph,
            patch_w=pw,
            r_min=1,
            r_max=max(1, top_band_limit - ph),
            c_min=1,
            c_max=WORK_SIZE - 2 - pw,
            pad=1,
        )
        if not positions:
            return None
        top, left = rng.choice(positions)
        _place_patch(grid, occupied, patch, top, left)
        frame_bottom = max(frame_bottom, top + ph - 1)

    marker_row_min = frame_bottom + rng.randint(3, 5)
    if marker_row_min >= WORK_SIZE - 2:
        return None

    markers: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    per_color_counts = [rng.randint(1, 3) for _ in frame_specs]
    for (color, _shape, offsets, _patch), count in zip(frame_specs, per_color_counts):
        for _ in range(count):
            positions = _marker_positions(
                occupied=occupied,
                offsets=offsets,
                anchor_r_min=marker_row_min,
                pad=1,
            )
            if not positions:
                return None
            r, c = rng.choice(positions)
            grid[r, c] = color
            dr0, dc0, dr1, dc1 = _bbox_from_offsets(offsets)
            occupied[r + dr0 : r + dr1 + 1, c + dc0 : c + dc1 + 1] = True
            markers.append((color, r, c, offsets))

    expected = _apply_expected_stamps(grid, markers)
    keep_mask = (grid != BG) | (expected != BG)
    cropped_inp, crop = _crop_with_margin(grid, keep_mask, rng)
    cr0, cc0, cr1, cc1 = crop
    cropped_expected = expected[cr0 : cr1 + 1, cc0 : cc1 + 1].copy()

    if cropped_inp.shape[0] > 30 or cropped_inp.shape[1] > 30:
        return None
    solved = main(cropped_inp)
    if not np.array_equal(solved, cropped_expected):
        return None
    if np.array_equal(cropped_inp, cropped_expected):
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
