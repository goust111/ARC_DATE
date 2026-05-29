# Combo variant of bf45cf4b (index 6)
# combo_spec: D6/0+D13/2
# Name: 奇偶镜像精灵色填充

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste

"""
Design analysis for the fused variant.

Concrete behavior trace
- Original task:
  A filled control pixel always stamps the same tile verbatim; a hole leaves a bg-sized gap.
  Example with ctrl color 6 and tile
    1 0 2
    3 4 5
    2 1 3
  would place that exact tile at every filled ctrl pixel, so the internal 0 stays a hole.
- Fused task:
  First, tile bg cells are replaced by the control color, so the tile above becomes
    1 6 2
    3 4 5
    2 1 3
  Then filled ctrl cells alternate by bbox parity:
  `(r+c)%2==0` -> use the filled tile as-is.
  `(r+c)%2==1` -> use its horizontal mirror.
  Control holes still expand to pure bg rectangles.
- Difference:
  D13/2 changes tile-internal holes into informative control-colored pixels.
  D6/0 changes repetition into a checkerboard of original/mirrored filled tiles.
  The two changes interact: the control color becomes part of both tile variants, not an
  independent post-process.

Complexity calibration
- If the tile were horizontally symmetric, mirroring would be invisible; the generator rejects that.
- If the tile had no bg hole, the control-color fill would be invisible; the generator rejects that.
- If all filled ctrl pixels landed on one parity only, the alternation would be invisible; the
  generator rejects that.
- The parity rule is one binary pattern, not a large family of transforms, so it remains inferable
  from 2-4 ARC examples.

Implicit-assumption audit against the original main()
- `len(comps)==2`: still valid; generator keeps exactly one ctrl component and one tile component.
- `block` identifiable by color richness and squareness: still valid; ctrl stays monochrome and tile
  stays square with >=2 non-bg colors.
- `mask = ctrl == ctrl_col`: still valid; ctrl is still single-color plus bg holes.
- `paste(out, block, ...)`: no longer valid as written.
  The solver must instead:
  1. build `filled_block = np.where(block == bg, ctrl_col, block)`;
  2. choose original vs mirrored filled tile by parity.
- Output shape assumption is unchanged because every ctrl cell still expands to one tile-sized region.

Placement legality and observability
- The generator uses a bg gap between the two input objects and verifies exactly two 8-connected
  foreground components.
- Tile holes are interior-only, so the tile remains a single component.
- Control masks are connected, contain real holes, span multiple rows and cols, and include both
  parities, which gives repeated visual evidence for the alternation rule.

Quantified non-degeneracy target
- For generated samples, `main(inp)` must differ from the original-task composition.
- We measure:
  `hole_pixels`: number of tile bg cells that become ctrl-colored in output.
  `mirror_diff`: number of cells where the filled tile differs from its horizontal mirror.
  `changed_vs_original`: output cells changed relative to the original task behavior.
  All three must be positive for accepted samples.
"""


GRID_MIN = 12
GRID_MAX = 18
TILE_SIZE_MIN = 3
TILE_SIZE_MAX = 5
CTRL_SIZE_MIN = 3
CTRL_SIZE_MAX = 6


def _identify_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(comps) == 2

    infos = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        infos.append(
            {
                "patch": patch,
                "cols": cols,
                "square": patch.shape[0] == patch.shape[1],
            }
        )

    infos.sort(
        key=lambda d: (
            len(d["cols"]) >= 2,
            len(d["cols"]),
            int(d["square"]),
            d["patch"].shape[0] * d["patch"].shape[1],
        ),
        reverse=True,
    )
    block = infos[0]["patch"]
    ctrl = infos[1]["patch"]

    ctrl_cols = colors_present(ctrl, exclude={bg})
    block_cols = colors_present(block, exclude={bg})
    assert len(ctrl_cols) == 1
    assert len(block_cols) >= 2
    assert block.shape[0] == block.shape[1]
    return block, ctrl


def _filled_block(block: np.ndarray, ctrl_color: int, bg: int) -> np.ndarray:
    return np.where(block == int(bg), int(ctrl_color), block)


def _compose_output(block: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    ctrl_cols = colors_present(ctrl, exclude={bg})
    if not ctrl_cols:
        return np.full((0, 0), int(bg), dtype=int)
    ctrl_color = int(ctrl_cols[0])
    filled = _filled_block(block, ctrl_color, bg)
    mirrored = np.fliplr(filled)

    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) == int(bg):
                continue
            patch = filled if (r + c) % 2 == 0 else mirrored
            paste(out, patch, r * bh, c * bw)
    return out


def _compose_original_output(block: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    bh, bw = map(int, block.shape)
    ch, cw = map(int, ctrl.shape)
    out = np.full((ch * bh, cw * bw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
                paste(out, block, r * bh, c * bw)
    return out


def fusion_metrics_from_input(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    block, ctrl = _identify_objects(g, bg)
    ctrl_color = int(colors_present(ctrl, exclude={bg})[0])
    filled = _filled_block(block, ctrl_color, bg)
    fused = _compose_output(block, ctrl, bg)
    original = _compose_original_output(block, ctrl, bg)
    parity_counts = [0, 0]
    for r in range(ctrl.shape[0]):
        for c in range(ctrl.shape[1]):
            if int(ctrl[r, c]) != int(bg):
                parity_counts[(r + c) % 2] += 1
    return {
        "hole_pixels": int(np.sum(block == int(bg))),
        "mirror_diff": int(np.sum(filled != np.fliplr(filled))),
        "changed_vs_original": int(np.sum(fused != original)),
        "filled_even": int(parity_counts[0]),
        "filled_odd": int(parity_counts[1]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    block, ctrl = _identify_objects(g, bg)
    return _compose_output(block, ctrl, bg)


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _has_adjacent_pair(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            if r + 1 < h and bool(mask[r + 1, c]):
                return True
            if c + 1 < w and bool(mask[r, c + 1]):
                return True
    return False


def _make_ctrl(bg: int, color: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        h = int(rng.integers(CTRL_SIZE_MIN, CTRL_SIZE_MAX + 1))
        w = int(rng.integers(CTRL_SIZE_MIN, CTRL_SIZE_MAX + 1))
        area = h * w
        min_fill = max(5, int(np.ceil(0.45 * area)))
        max_fill = max(min_fill, int(np.floor(0.72 * area)))

        ar = int(rng.integers(0, h))
        ac = int(rng.integers(0, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, ac] = True
        mask[ar, :] = True

        base_fill = int(mask.sum())
        target = int(rng.integers(max(base_fill, min_fill), max_fill + 1))
        while int(mask.sum()) < target:
            frontier = []
            for r in range(h):
                for c in range(w):
                    if bool(mask[r, c]):
                        continue
                    if any(bool(mask[rr, cc]) for rr, cc in _neighbors8(h, w, r, c)):
                        frontier.append((r, c))
            if not frontier:
                break
            rr, cc = frontier[int(rng.integers(len(frontier)))]
            mask[rr, cc] = True

        parity_counts = [0, 0]
        for r in range(h):
            for c in range(w):
                if bool(mask[r, c]):
                    parity_counts[(r + c) % 2] += 1
        rows = {int(r) for r, _c in np.argwhere(mask)}
        cols = {int(c) for _r, c in np.argwhere(mask)}

        if int(mask.sum()) >= area:
            continue
        if parity_counts[0] == 0 or parity_counts[1] == 0:
            continue
        if len(rows) < 2 or len(cols) < 2:
            continue
        if not _has_adjacent_pair(mask):
            continue

        ctrl = np.full((h, w), int(bg), dtype=int)
        ctrl[mask] = int(color)
        return ctrl
    raise RuntimeError("failed to build control sprite")


def _make_tile(bg: int, tile_colors: list[int], rng: np.random.Generator, ctrl_color: int) -> np.ndarray:
    for _ in range(300):
        size = int(rng.integers(TILE_SIZE_MIN, TILE_SIZE_MAX + 1))
        n_colors = int(rng.integers(2, min(3, len(tile_colors)) + 1))
        colors = [int(x) for x in rng.choice(tile_colors, size=n_colors, replace=False)]

        tile = np.full((size, size), int(colors[0]), dtype=int)
        tile[0, :] = int(colors[1 % n_colors])
        tile[-1, :] = int(colors[0])
        tile[:, 0] = int(colors[(2 % n_colors) if n_colors > 2 else 0])
        tile[:, -1] = int(colors[1 % n_colors])

        for r in range(1, size - 1):
            for c in range(1, size - 1):
                tile[r, c] = int(colors[(r + 2 * c + int(rng.integers(n_colors))) % n_colors])

        # Force each non-bg tile color to appear at least once in visible locations.
        for i, col in enumerate(colors):
            rr = 1 + (i % max(1, size - 2))
            cc = 1 + ((2 * i + 1) % max(1, size - 2))
            tile[rr, cc] = int(col)

        interior = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
        hole_count = 1 if len(interior) == 1 else int(rng.integers(1, min(3, len(interior)) + 1))
        holes = rng.choice(len(interior), size=hole_count, replace=False)
        for idx in np.atleast_1d(holes):
            rr, cc = interior[int(idx)]
            tile[rr, cc] = int(bg)

        if len(colors_present(tile, exclude={bg})) < 2:
            continue
        if not np.any(tile == int(bg)):
            continue

        filled = _filled_block(tile, ctrl_color, bg)
        if np.array_equal(filled, np.fliplr(filled)):
            continue
        return tile
    raise RuntimeError("failed to build tile")


def _make_input(tile: np.ndarray, ctrl: np.ndarray, bg: int, rng: np.random.Generator) -> np.ndarray:
    gap = 1
    horizontal = bool(rng.integers(0, 2))

    if horizontal:
        if bool(rng.integers(0, 2)):
            left_obj, right_obj = tile, ctrl
            left_is_tile = True
        else:
            left_obj, right_obj = ctrl, tile
            left_is_tile = False
        lh, lw = left_obj.shape
        rh, rw = right_obj.shape
        H = int(rng.integers(max(GRID_MIN, max(lh, rh)), GRID_MAX + 1))
        W = int(rng.integers(max(GRID_MIN, lw + rw + gap), GRID_MAX + 1))
        slack = W - (lw + rw + gap)
        c_left = int(rng.integers(0, slack + 1))
        c_right = int(rng.integers(c_left + lw + gap, W - rw + 1))
        r_left = int(rng.integers(0, H - lh + 1))
        r_right = int(rng.integers(0, H - rh + 1))
        if left_is_tile:
            tile_pos = (r_left, c_left)
            ctrl_pos = (r_right, c_right)
        else:
            ctrl_pos = (r_left, c_left)
            tile_pos = (r_right, c_right)
    else:
        if bool(rng.integers(0, 2)):
            top_obj, bot_obj = tile, ctrl
            top_is_tile = True
        else:
            top_obj, bot_obj = ctrl, tile
            top_is_tile = False
        th, tw = top_obj.shape
        bh, bw = bot_obj.shape
        H = int(rng.integers(max(GRID_MIN, th + bh + gap), GRID_MAX + 1))
        W = int(rng.integers(max(GRID_MIN, max(tw, bw)), GRID_MAX + 1))
        slack = H - (th + bh + gap)
        r_top = int(rng.integers(0, slack + 1))
        r_bot = int(rng.integers(r_top + th + gap, H - bh + 1))
        c_top = int(rng.integers(0, W - tw + 1))
        c_bot = int(rng.integers(0, W - bw + 1))
        if top_is_tile:
            tile_pos = (r_top, c_top)
            ctrl_pos = (r_bot, c_bot)
        else:
            ctrl_pos = (r_top, c_top)
            tile_pos = (r_bot, c_bot)

    grid = np.full((H, W), int(bg), dtype=int)
    paste(grid, tile, *tile_pos)
    paste(grid, ctrl, *ctrl_pos, transparent=bg)
    return grid


def _valid_sample(inp: np.ndarray, expected: np.ndarray) -> bool:
    bg = int(get_bg(inp))
    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return False
    try:
        block, ctrl = _identify_objects(inp, bg)
    except AssertionError:
        return False

    if not np.array_equal(main(inp), expected):
        return False

    metrics = fusion_metrics_from_input(inp)
    if metrics["hole_pixels"] <= 0:
        return False
    if metrics["mirror_diff"] <= 0:
        return False
    if metrics["changed_vs_original"] <= 0:
        return False
    if metrics["filled_even"] <= 0 or metrics["filled_odd"] <= 0:
        return False

    # Guard the original disambiguation assumption explicitly.
    if len(colors_present(ctrl, exclude={bg})) != 1:
        return False
    if len(colors_present(block, exclude={bg})) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=5, replace=False)]
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    tile_colors = [int(x) for x in palette[2:]]

    try:
        ctrl = _make_ctrl(bg, ctrl_color, rng)
        tile = _make_tile(bg, tile_colors, rng, ctrl_color)
        inp = _make_input(tile, ctrl, bg, rng)
        expected = _compose_output(tile, ctrl, bg)
    except RuntimeError:
        return None

    return inp if _valid_sample(inp, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(500):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
