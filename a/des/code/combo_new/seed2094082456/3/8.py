from __future__ import annotations

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each quadrant's single colored object using the center compass encoding.

    Input structure:
    - The grid has odd height and width and is dominated by a black background.
    - The exact center contains a 3x3 compass: a grey center, four colored arms in
      the cardinal directions, and background on the four diagonal cells.
    - The center row and center column split the canvas into four quadrants.
    - Each quadrant contains exactly one small monochrome 4-connected object whose
      color is different from the background and all compass colors.

    Rule:
    - Read the compass arms: top encodes 0 degrees, right encodes 90 degrees
      clockwise, bottom encodes 180 degrees, and left encodes 270 degrees.
    - The quadrant-to-arm choice is not the nearest-arm pair average: upper-left
      uses the top arm, upper-right uses the right arm, lower-right uses the bottom
      arm, and lower-left uses the left arm.
    - Rotate each quadrant object around the center of its tight bounding box by the
      chosen angle, keeping the compass and background unchanged.

    Key insight:
    - The center motif is a compass that first defines a color-to-rotation mapping
      and then reuses one specific arm color per quadrant to decide that quadrant's
      in-place rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    assert H % 2 == 1 and W % 2 == 1, "expected odd grid dimensions"
    assert H >= 7 and W >= 7, "expected room for the compass and four quadrants"

    bg = _get_bg(g)
    cr, cc = H // 2, W // 2

    center_color = int(g[cr, cc])
    arms = {
        "up": int(g[cr - 1, cc]),
        "right": int(g[cr, cc + 1]),
        "down": int(g[cr + 1, cc]),
        "left": int(g[cr, cc - 1]),
    }
    assert len(set(arms.values())) == 4, "compass arm colors must be distinct"
    assert int(g[cr - 1, cc - 1]) == bg
    assert int(g[cr - 1, cc + 1]) == bg
    assert int(g[cr + 1, cc - 1]) == bg
    assert int(g[cr + 1, cc + 1]) == bg

    color_to_turns = {
        arms["up"]: 0,
        arms["right"]: 1,
        arms["down"]: 2,
        arms["left"]: 3,
    }
    excluded = np.array([bg, center_color, *arms.values()], dtype=int)

    out = g.copy()
    object_mask = ~np.isin(g, excluded)
    out[object_mask] = bg

    quadrant_specs = [
        ("UL", slice(0, cr), slice(0, cc), "up"),
        ("UR", slice(0, cr), slice(cc + 1, W), "right"),
        ("LR", slice(cr + 1, H), slice(cc + 1, W), "down"),
        ("LL", slice(cr + 1, H), slice(0, cc), "left"),
    ]

    for _name, row_slice, col_slice, arm_name in quadrant_specs:
        sub = g[row_slice, col_slice]
        comps = _find_components(~np.isin(sub, excluded))
        assert len(comps) == 1, "expected one object in each quadrant"
        comp = comps[0]
        patch = _extract_patch(sub, comp, bg)
        rotated, dr, dc = _rotate_patch_about_center(patch, color_to_turns[arms[arm_name]])

        r0, c0, _, _ = comp["bbox"]
        local_r0 = int(r0 + dr)
        local_c0 = int(c0 + dc)
        assert 0 <= local_r0 <= sub.shape[0] - rotated.shape[0]
        assert 0 <= local_c0 <= sub.shape[1] - rotated.shape[1]

        abs_r0 = int((row_slice.start or 0) + local_r0)
        abs_c0 = int((col_slice.start or 0) + local_c0)
        _paste(out, rotated, abs_r0, abs_c0, bg)

    return out


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []

        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in _NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        comps.append(
            {
                "coords": coords,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
            }
        )

    return comps


def _extract_patch(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in comp["coords"]:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch


def _rotate_patch_about_center(patch: np.ndarray, turns_cw: int) -> tuple[np.ndarray, int, int]:
    turns_cw %= 4
    if turns_cw == 0:
        return patch.copy(), 0, 0

    rotated = np.rot90(patch, -turns_cw)
    h, w = patch.shape
    rh, rw = rotated.shape
    assert (h - rh) % 2 == 0 and (w - rw) % 2 == 0, (
        "90-degree center rotation needs bbox dimensions with matching parity"
    )
    return rotated, (h - rh) // 2, (w - rw) // 2


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    region[mask] = patch[mask]


def _mask_from_lines(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)


def _mask_to_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _valid_positions(qh: int, qw: int, h: int, w: int, quadrant: str) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r0 in range(qh - h + 1):
        for c0 in range(qw - w + 1):
            if quadrant in ("UL", "UR") and r0 + h > qh - 1:
                continue
            if quadrant in ("LL", "LR") and r0 < 1:
                continue
            if quadrant in ("UL", "LL") and c0 + w > qw - 1:
                continue
            if quadrant in ("UR", "LR") and c0 < 1:
                continue
            positions.append((r0, c0))
    return positions


def _quadrant_origin(quadrant: str, cr: int, cc: int) -> tuple[int, int]:
    if quadrant == "UL":
        return 0, 0
    if quadrant == "UR":
        return 0, cc + 1
    if quadrant == "LR":
        return cr + 1, cc + 1
    return cr + 1, 0


def _gen_attempt() -> np.ndarray | None:
    H = int(_RNG.choice(_ODD_SIZES))
    W = int(_RNG.choice(_ODD_SIZES))
    cr, cc = H // 2, W // 2
    qh, qw = cr, cc

    grid = np.full((H, W), _BG, dtype=int)
    arm_colors = list(map(int, _RNG.permutation(_ARM_COLORS)))
    grid[cr, cc] = _CENTER
    grid[cr - 1, cc] = arm_colors[0]
    grid[cr, cc + 1] = arm_colors[1]
    grid[cr + 1, cc] = arm_colors[2]
    grid[cr, cc - 1] = arm_colors[3]

    shape_ids = list(map(int, _RNG.choice(len(_SHAPES), size=4, replace=False)))
    object_colors = list(map(int, _RNG.permutation(_OBJECT_COLORS)))
    changed = 0

    for quadrant, shape_id, color in zip(_QUADRANT_ORDER, shape_ids, object_colors):
        mask = np.rot90(_SHAPES[shape_id], -int(_RNG.integers(0, 4)))
        h, w = mask.shape
        positions = _valid_positions(qh, qw, h, w, quadrant)
        if not positions:
            return None
        r0, c0 = positions[int(_RNG.integers(0, len(positions)))]
        patch = _mask_to_patch(mask, color, _BG)
        abs_r0, abs_c0 = _quadrant_origin(quadrant, cr, cc)
        _paste(grid, patch, abs_r0 + r0, abs_c0 + c0, _BG)

        turns = _QUADRANT_TURNS[quadrant]
        rotated, _, _ = _rotate_patch_about_center(patch, turns)
        if turns != 0 and not np.array_equal(rotated, patch):
            changed += 1

    if changed < 2:
        return None

    try:
        solved = main(grid)
    except Exception:
        return None

    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compass-rotation sample")


_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_BG = 0
_ARM_COLORS = (1, 2, 3, 4)
_CENTER = 5
_OBJECT_COLORS = (6, 7, 8, 9)
_ODD_SIZES = np.arange(11, 30, 2)
_MAX_ATTEMPTS = 300

_SHAPES = [
    _mask_from_lines(("#.", "##")),
    _mask_from_lines(("#..", "#..", "###")),
    _mask_from_lines(("###", ".#.", ".#.")),
    _mask_from_lines(("##.", ".##", "..#")),
    _mask_from_lines((".##", "##.", ".#.")),
    _mask_from_lines(("##.", "#..", "###")),
    _mask_from_lines(("#..", "###", ".##")),
    _mask_from_lines((".#.", "###", "#..")),
]

_QUADRANT_ORDER = ("UL", "UR", "LR", "LL")
_QUADRANT_TURNS = {"UL": 0, "UR": 1, "LR": 2, "LL": 3}
_RNG = np.random.default_rng()
