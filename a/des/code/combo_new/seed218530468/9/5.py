import random
from collections import deque

import numpy as np


BG = 0
ARC_MAX = 30
MAX_GENERATION_ATTEMPTS = 300
GEN_FRAME_COUNT = 2  # Three or more valid frame layers cannot unfold within 30 columns.


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _area(bbox: tuple[int, int, int, int]) -> int:
    h, w = _shape(bbox)
    return h * w


def _component_mask(
    coords: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _is_solid_rectangle(comp: dict) -> bool:
    bbox = comp["bbox"]
    return comp["area"] == _area(bbox)


def _is_frame_rectangle(comp: dict) -> bool:
    bbox = comp["bbox"]
    h, w = _shape(bbox)
    if h < 3 or w < 3:
        return False
    if comp["area"] != 2 * h + 2 * w - 4:
        return False
    mask = _component_mask(comp["coords"], bbox)
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _center_key(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r0 + r1, c0 + c1


def _extract_layers(grid: np.ndarray) -> tuple[list[dict], dict]:
    comps = _find_components(grid, bg=BG)
    assert len(comps) >= 2, "expected nested non-background components"

    frames: list[dict] = []
    solids: list[dict] = []
    for comp in comps:
        if _is_solid_rectangle(comp):
            solids.append(comp)
        else:
            assert _is_frame_rectangle(comp), "every non-core component must be a rectangular frame"
            frames.append(comp)

    assert len(solids) == 1, "expected a unique solid core rectangle"
    assert frames, "expected at least one frame"

    core = solids[0]
    key = _center_key(core["bbox"])
    for frame in frames:
        assert _center_key(frame["bbox"]) == key, "all layers must be concentric"

    frames = sorted(frames, key=lambda comp: _area(comp["bbox"]), reverse=True)
    nested = frames + [core]
    for outer, inner in zip(nested, nested[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 + 2 <= ir0 <= ir1 <= or1 - 2, "inner layer must leave a black gap inside outer layer"
        assert oc0 + 2 <= ic0 <= ic1 <= oc1 - 2, "inner layer must leave a black gap inside outer layer"

    return frames, core


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unwrap nested concentric frames into a horizontal row of filled standalone frames.

    Input structure:
    - Black background.
    - One nested family of axis-aligned monochrome rectangular frames, each drawn as a 1-pixel border with at least one black cell separating adjacent layers.
    - The innermost object is a solid monochrome rectangle whose color differs from every frame color.

    Rule:
    - Detect the layers from the outermost frame inward to the solid core.
    - Repaint each frame as a standalone rectangle of the same size and border color, but fill its entire interior with the core color.
    - Place those standalone frames left-to-right in nesting order with one black separator column, then place the unchanged solid core rectangle at the far right.
    - Use the outermost frame height as the output height and vertically center every emitted element within that canvas.

    Key insight:
    - The nested rectangles encode an ordered size hierarchy; the visible fill color for every unfolded frame comes from the innermost solid core rather than from the frame itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames, core = _extract_layers(g)
    outer_h, _outer_w = _shape(frames[0]["bbox"])
    core_color = int(core["color"])

    widths = [_shape(frame["bbox"])[1] for frame in frames]
    core_h, core_w = _shape(core["bbox"])
    out_w = sum(widths) + core_w + len(frames)
    out = np.full((outer_h, out_w), BG, dtype=int)

    cursor = 0
    for frame in frames:
        color = int(frame["color"])
        frame_h, frame_w = _shape(frame["bbox"])
        top = (outer_h - frame_h) // 2
        out[top : top + frame_h, cursor : cursor + frame_w] = core_color
        out[top, cursor : cursor + frame_w] = color
        out[top + frame_h - 1, cursor : cursor + frame_w] = color
        out[top : top + frame_h, cursor] = color
        out[top : top + frame_h, cursor + frame_w - 1] = color
        cursor += frame_w + 1

    top = (outer_h - core_h) // 2
    out[top : top + core_h, cursor : cursor + core_w] = core_color
    return out


def _draw_frame(
    grid: np.ndarray, top: int, left: int, height: int, width: int, color: int
) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color


def _sample_core_dims() -> tuple[int, int]:
    mode = random.randrange(3)
    if mode == 0:
        return random.randint(1, 5), random.randint(1, 5)
    if mode == 1:
        return random.randint(4, 14), random.randint(1, 3)
    return random.randint(1, 4), random.randint(3, 6)


def _gen_attempt() -> np.ndarray | None:
    core_h, core_w = _sample_core_dims()
    inner_gap_y = random.randint(1, 4)
    inner_gap_x = random.randint(1, 4)
    outer_gap_y = random.randint(1, 4)
    outer_gap_x = random.randint(1, 4)

    inner_h = core_h + 2 * (inner_gap_y + 1)
    inner_w = core_w + 2 * (inner_gap_x + 1)
    outer_h = inner_h + 2 * (outer_gap_y + 1)
    outer_w = inner_w + 2 * (outer_gap_x + 1)

    out_w = outer_w + inner_w + core_w + GEN_FRAME_COUNT
    if outer_h > ARC_MAX or outer_w > ARC_MAX or out_w > ARC_MAX:
        return None

    grid_h = random.randint(outer_h, ARC_MAX)
    grid_w = random.randint(outer_w, ARC_MAX)
    top = random.randint(0, grid_h - outer_h)
    left = random.randint(0, grid_w - outer_w)

    outer_color, inner_color, core_color = random.sample(range(1, 10), 3)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    _draw_frame(grid, top, left, outer_h, outer_w, outer_color)

    inner_top = top + outer_gap_y + 1
    inner_left = left + outer_gap_x + 1
    _draw_frame(grid, inner_top, inner_left, inner_h, inner_w, inner_color)

    core_top = inner_top + inner_gap_y + 1
    core_left = inner_left + inner_gap_x + 1
    grid[core_top : core_top + core_h, core_left : core_left + core_w] = core_color

    try:
        frames, core = _extract_layers(grid)
        if len(frames) != GEN_FRAME_COUNT:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if max(grid.shape) > ARC_MAX or max(out.shape) > ARC_MAX:
        return None
    if int(core["color"]) != core_color:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(
        f"failed to generate a valid nested-frame input after {MAX_GENERATION_ATTEMPTS} attempts"
    )
