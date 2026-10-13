import math
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    vals = np.bincount(grid.ravel())
    return int(np.argmax(vals))


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity == 4:
        dirs = DIR4
    else:
        dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            coord_arr = np.array(coords, dtype=int)
            r0 = int(coord_arr[:, 0].min())
            c0 = int(coord_arr[:, 1].min())
            r1 = int(coord_arr[:, 0].max())
            c1 = int(coord_arr[:, 1].max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            rel = coord_arr - np.array([r0, c0], dtype=int)
            mask[rel[:, 0], rel[:, 1]] = True
            comps.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_mask(mask.astype(bool))
    return cropped.shape, cropped.astype(np.uint8).tobytes()


def _quadrant_ccw_turns(coords: np.ndarray, shape: tuple[int, int]) -> int:
    h, w = shape
    row_mid = h / 2.0
    col_mid = w / 2.0
    rows = coords[:, 0]
    cols = coords[:, 1]
    top = int(rows.max()) < row_mid
    bottom = int(rows.min()) >= row_mid
    left = int(cols.max()) < col_mid
    right = int(cols.min()) >= col_mid
    assert top ^ bottom, "piece must lie entirely above or below the grid midpoint"
    assert left ^ right, "piece must lie entirely left or right of the grid midpoint"
    if top and left:
        return 0
    if top and right:
        return 3
    if bottom and right:
        return 2
    return 1


def _template_and_pieces(grid: np.ndarray, bg: int) -> tuple[dict, list[dict]]:
    comps = _monochrome_components(grid, bg)
    assert comps, "expected at least one foreground component"
    max_area = max(comp["area"] for comp in comps)
    template_candidates = [comp for comp in comps if comp["area"] == max_area]
    assert len(template_candidates) == 1, "template must be the unique largest monochrome component"
    template = template_candidates[0]
    pieces = [comp for comp in comps if comp is not template]
    return template, pieces


def _template_holes(grid: np.ndarray, bg: int, template: dict) -> list[dict]:
    r0, c0, r1, c1 = template["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes = []
    for coords in _components_mask(sub == bg, connectivity=4):
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == sub.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == sub.shape[1] - 1)
        ):
            continue
        hr0 = int(coords[:, 0].min())
        hc0 = int(coords[:, 1].min())
        hr1 = int(coords[:, 0].max())
        hc1 = int(coords[:, 1].max())
        mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        rel = coords - np.array([hr0, hc0], dtype=int)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append(
            {
                "top": r0 + hr0,
                "left": c0 + hc0,
                "mask": mask,
                "key": _mask_key(mask),
            }
        )
    assert holes, "template must contain internal slots"
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble scattered colored pieces into the grey template slots, using each piece's quadrant to choose its rotation.

    Input structure:
    - The background is black and dominates the grid.
    - One large grey monochrome template sits near the center-right and contains several enclosed background slots.
    - Outside the template are monochrome colored fragments, each lying fully inside one grid quadrant.

    Rule:
    - Detect the template as the unique largest monochrome non-background component and extract its enclosed background slots.
    - For each colored fragment, determine its quadrant relative to the grid midpoint and rotate it accordingly: upper-left 0, upper-right 90 clockwise, lower-right 180, lower-left 270 clockwise.
    - Match the rotated fragment's normalized shape to the unique slot with the same shape and paint that slot with the fragment's color, while leaving the original scattered fragments, template outline, and background unchanged.

    Key insight:
    - Shape alone is not enough because the same fragment can fit different slot orientations; the fragment's quadrant supplies the missing rotation before matching.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    template, pieces = _template_and_pieces(g, bg)
    holes = _template_holes(g, bg, template)
    assert len(pieces) == len(holes), "piece count must match slot count"

    unused = {idx: hole for idx, hole in enumerate(holes)}
    out = g.copy()

    for piece in pieces:
        turns = _quadrant_ccw_turns(piece["coords"], g.shape)
        rotated = np.rot90(piece["mask"], turns)
        key = _mask_key(rotated)
        matches = [idx for idx, hole in unused.items() if hole["key"] == key]
        assert len(matches) == 1, "each rotated piece must match exactly one remaining slot"
        hole = unused.pop(matches[0])
        patch = out[
            hole["top"] : hole["top"] + hole["mask"].shape[0],
            hole["left"] : hole["left"] + hole["mask"].shape[1],
        ]
        patch[hole["mask"]] = int(piece["color"])

    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()
_BG = 0
_TEMPLATE = 5
_PIECE_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
_GEN_RETRIES = 300


def _rotate(mask: np.ndarray, turns: int) -> np.ndarray:
    return _crop_mask(np.rot90(mask, turns))


def _random_base_mask() -> np.ndarray:
    dims = [(3, 3), (3, 4), (4, 3)]
    for _ in range(200):
        h, w = dims[int(_RNG.integers(0, len(dims)))]
        area = int(_RNG.integers(4, min(6, h * w - 1) + 1))
        start = (int(_RNG.integers(0, h)), int(_RNG.integers(0, w)))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(_RNG.integers(0, len(frontier)))]
            cells.add(pick)
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if min(mask.shape) < 2:
            continue
        if mask.sum() == mask.size:
            continue
        rotations = [_rotate(mask, k) for k in range(4)]
        if len({_mask_key(rot) for rot in rotations}) != 4:
            continue
        return mask
    raise RuntimeError("failed to build an asymmetric base fragment")


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = int(color)


def _sample_piece_position(
    grid_shape: tuple[int, int],
    template_bbox: tuple[int, int, int, int],
    quadrant: str,
    piece_shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = grid_shape
    ph, pw = piece_shape
    _tr0, tc0, _tr1, tc1 = template_bbox
    row_mid = h / 2.0
    col_mid = w / 2.0

    if quadrant[0] == "u":
        r_lo = 1
        r_hi = int(math.floor(row_mid - ph))
    else:
        r_lo = int(math.ceil(row_mid))
        r_hi = h - ph - 1

    if quadrant[1] == "l":
        c_lo = 1
        c_hi = min(int(math.floor(col_mid - pw)), tc0 - pw - 1)
    else:
        c_lo = max(int(math.ceil(col_mid)), tc1 + 2)
        c_hi = w - pw - 1

    if r_lo > r_hi or c_lo > c_hi:
        return None
    top = int(_RNG.integers(r_lo, r_hi + 1))
    left = int(_RNG.integers(c_lo, c_hi + 1))
    return top, left


def _gen_attempt() -> np.ndarray | None:
    base = _random_base_mask()
    slot_masks = [_rotate(base, k) for k in range(4)]
    max_h = max(mask.shape[0] for mask in slot_masks)
    max_w = max(mask.shape[1] for mask in slot_masks)

    chamber_h = max_h + 1
    chamber_w = max_w + 1
    border = 2
    row_gap = int(_RNG.integers(2, 4))
    col_gap = int(_RNG.integers(2, 4))
    template_h = 2 * chamber_h + row_gap + 2 * border
    template_w = 2 * chamber_w + col_gap + 2 * border

    top_strip = max_h + 2 + int(_RNG.integers(0, 2))
    bottom_strip = max_h + 2 + int(_RNG.integers(0, 2))
    left_strip = max_w + 4 + int(_RNG.integers(0, 4))
    right_strip = max_w + 2 + int(_RNG.integers(0, 2))
    outer = 1

    grid_h = top_strip + template_h + bottom_strip + 2 * outer
    grid_w = left_strip + template_w + right_strip + 2 * outer
    if grid_h > 30 or grid_w > 30:
        return None

    template_top = outer + top_strip
    template_left = outer + left_strip
    template_bbox = (
        template_top,
        template_left,
        template_top + template_h - 1,
        template_left + template_w - 1,
    )

    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    grid[template_top : template_top + template_h, template_left : template_left + template_w] = _TEMPLATE
    expected = grid.copy()

    chamber_bases = [
        (border, border),
        (border, border + chamber_w + col_gap),
        (border + chamber_h + row_gap, border),
        (border + chamber_h + row_gap, border + chamber_w + col_gap),
    ]
    rotation_order = _RNG.permutation(4)
    slots = []
    for chamber_idx, rot_idx in enumerate(rotation_order):
        mask = slot_masks[int(rot_idx)]
        base_r, base_c = chamber_bases[chamber_idx]
        slack_r = chamber_h - mask.shape[0]
        slack_c = chamber_w - mask.shape[1]
        top = base_r + int(_RNG.integers(0, slack_r + 1))
        left = base_c + int(_RNG.integers(0, slack_c + 1))
        slots.append({"top": top, "left": left, "mask": mask})
        _paint_mask(grid, template_top + top, template_left + left, mask, _BG)

    expected[:] = grid

    quadrant_specs = [("ul", 0), ("ur", 3), ("ll", 1), ("lr", 2)]
    slot_for_quadrant = _RNG.permutation(4)
    colors = _RNG.choice(_PIECE_COLORS, size=4, replace=False)

    for idx, (quadrant, turns) in enumerate(quadrant_specs):
        slot = slots[int(slot_for_quadrant[idx])]
        piece_mask = _rotate(slot["mask"], -turns)
        pos = _sample_piece_position(grid.shape, template_bbox, quadrant, piece_mask.shape)
        if pos is None:
            return None
        piece_top, piece_left = pos
        color = int(colors[idx])
        _paint_mask(grid, piece_top, piece_left, piece_mask, color)
        _paint_mask(expected, piece_top, piece_left, piece_mask, color)
        _paint_mask(
            expected,
            template_top + slot["top"],
            template_left + slot["left"],
            slot["mask"],
            color,
        )

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quadrant-rotation template puzzle")
