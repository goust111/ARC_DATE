from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label


BLACK = 0
GREY = 5
BLOCK_SIZE = 5
BLOCK_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    structure = STRUCT4 if int(connectivity) == 4 else np.ones((3, 3), dtype=int)
    lab, n = label(mask.astype(bool), structure=structure)
    return [np.argwhere(lab == k) for k in range(1, int(n) + 1)]


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    colors: tuple[int, ...] | list[int] | None = None,
) -> list[dict]:
    if colors is None:
        colors_iter = sorted(set(map(int, grid.ravel())) - {int(bg)})
    else:
        colors_iter = [int(c) for c in colors if int(c) != int(bg)]

    comps: list[dict] = []
    for color in colors_iter:
        for coords in components_mask(grid == color, connectivity=connectivity):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


MOTIF_LIBRARY: dict[str, np.ndarray] = {
    "center": np.array(
        [
            [1],
            [1],
            [1],
        ],
        dtype=bool,
    ),
    "pair": np.array(
        [
            [1, 0],
            [1, 1],
        ],
        dtype=bool,
    ),
    "triple": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
}


def _center_offsets(mask: np.ndarray) -> tuple[int, int]:
    h, w = map(int, mask.shape)
    return (BLOCK_SIZE - h) // 2, (BLOCK_SIZE - w) // 2


def _channel_columns(mask: np.ndarray) -> tuple[int, ...]:
    off_r, off_c = _center_offsets(mask)
    del off_r
    cols = sorted({int(off_c + c) for _r, c in np.argwhere(mask)})
    return tuple(cols)


def _parse_objects(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4)

    blocks = []
    motifs_by_color: dict[int, np.ndarray] = {}
    motif_coords_by_color: dict[int, list[tuple[int, int]]] = {}
    weights: list[tuple[int, int]] = []

    for comp in comps:
        color = int(comp["color"])
        area = int(comp["area"])
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        if color == GREY:
            assert area == 1
            weights.append((int(r0), int(c0)))
            continue

        if area == 25 and h == 5 and w == 5 and np.all(crop_bbox(g, comp["bbox"]) == color):
            blocks.append(
                {
                    "color": color,
                    "bbox": comp["bbox"],
                    "r0": int(r0),
                    "c0": int(c0),
                }
            )
            continue

        assert 3 <= area <= 5
        assert h <= 3 and w <= 3
        mask = crop_bbox(g, comp["bbox"]) == color
        assert int(mask.sum()) == area
        assert color not in motifs_by_color
        motifs_by_color[color] = mask
        motif_coords_by_color[color] = [(int(r), int(c)) for r, c in comp["coords"]]

    assert len(blocks) in (2, 3)
    assert len(weights) in (2, 3)
    blocks.sort(key=lambda b: (b["r0"], b["c0"]))
    weights.sort()

    block_colors = {int(block["color"]) for block in blocks}
    assert block_colors == set(motifs_by_color)
    assert len(block_colors) == len(blocks)

    for wr, wc in weights:
        carriers = [block for block in blocks if block["c0"] <= wc < block["c0"] + BLOCK_SIZE]
        assert len(carriers) == 1
        assert wr < carriers[0]["r0"]

    return {
        "bg": bg,
        "blocks": blocks,
        "motifs_by_color": motifs_by_color,
        "motif_coords_by_color": motif_coords_by_color,
        "weights": weights,
    }


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    parsed = _parse_objects(grid)
    bg = int(parsed["bg"])
    out = np.asarray(grid, dtype=int).copy()

    for coords in parsed["motif_coords_by_color"].values():
        for r, c in coords:
            out[r, c] = bg
    for r, c in parsed["weights"]:
        out[r, c] = bg

    block_channels: dict[int, tuple[int, ...]] = {}
    for block in parsed["blocks"]:
        color = int(block["color"])
        r0 = int(block["r0"])
        c0 = int(block["c0"])
        motif = parsed["motifs_by_color"][color]
        off_r, off_c = _center_offsets(motif)
        channels = _channel_columns(motif)
        block_channels[color] = channels

        for rr, cc in np.argwhere(motif):
            out[r0 + off_r + int(rr), c0 + off_c + int(cc)] = bg

        # Drill every motif-selected column through the full block height so the
        # carved motif becomes a usable vertical channel for gravity.
        for local_c in channels:
            out[r0 : r0 + BLOCK_SIZE, c0 + int(local_c)] = bg

    final_weights: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for wr, wc in parsed["weights"]:
        r = int(wr)
        c = int(wc)
        while r + 1 < out.shape[0] and int(out[r + 1, c]) == bg:
            r += 1
        out[r, c] = GREY
        final_weights.append(((int(wr), int(wc)), (int(r), int(c))))

    if not return_meta:
        return out

    return out, {
        **parsed,
        "block_channels": block_channels,
        "final_weights": final_weights,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored motif to its same-colored block, carve gravity channels, then drop the grey weights.

    Input structure:
    - Black background with 2-3 solid 5x5 monochrome blocks in distinct non-grey colors.
    - For each block color there is one small same-colored motif elsewhere on the canvas.
    - There are 2-3 grey single-pixel weights above the blocks, each sharing a column with some block.

    Rule:
    - Pair every motif with the solid 5x5 block of the same color.
    - Center the motif inside that block; the covered motif cells are removed, and every block column touched by
      the centered motif is drilled into a black vertical channel through the block.
    - Erase the original motif locations.
    - Erase the original grey weights, then process them from top to bottom and left to right: each weight falls
      straight down through black cells, including drilled channels, and stops just above the first non-black cell
      or on the bottom row. Settled weights block later ones.

    Key insight:
    - The motif is both a same-color matching key and a channel blueprint: it tells which block to modify and
      which columns become passable under gravity.
    """
    return _solve(grid, return_meta=False)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    for rr, cc in np.argwhere(mask):
        grid[int(r0) + int(rr), int(c0) + int(cc)] = int(color)


def _block_columns(c0: int) -> list[int]:
    return list(range(int(c0), int(c0) + BLOCK_SIZE))


def _validate_semantics(grid: np.ndarray) -> None:
    out, meta = _solve(grid, return_meta=True)

    assert grid.ndim == 2 and out.ndim == 2
    assert max(grid.shape) <= 30 and max(out.shape) <= 30
    assert not np.array_equal(grid, out)

    blocks = meta["blocks"]
    weights = meta["weights"]
    final_weights = meta["final_weights"]
    pass_count = 0
    blocked_count = 0

    for (wr, wc), (fr, fc) in final_weights:
        assert wc == fc
        carrier = [block for block in blocks if block["c0"] <= wc < block["c0"] + BLOCK_SIZE][0]
        local_c = int(wc - carrier["c0"])
        if local_c in meta["block_channels"][carrier["color"]]:
            assert fr >= carrier["r0"] + BLOCK_SIZE
            pass_count += 1
        else:
            assert fr == carrier["r0"] - 1 or int(out[fr + 1, fc]) == GREY
            blocked_count += 1
        assert fr >= wr

    assert pass_count >= 1
    assert blocked_count >= 1

    column_counts = Counter(c for _r, c in weights)
    if max(column_counts.values()) > 1:
        for col, count in column_counts.items():
            if count <= 1:
                continue
            finals = [fr for (_src_r, src_c), (fr, _fc) in final_weights if src_c == col]
            assert finals == sorted(finals, reverse=True)
            assert len(set(finals)) == count


def _candidate_masks() -> list[tuple[str, np.ndarray]]:
    return [(name, MOTIF_LIBRARY[name].copy()) for name in ("center", "pair", "triple")]


def _sample_layout() -> tuple[int, int, list[int], int]:
    num_blocks = random.choice((2, 3))
    gaps = [random.randint(2, 4) for _ in range(num_blocks - 1)]
    block_span = BLOCK_SIZE * num_blocks + sum(gaps)
    width = random.randint(max(20, block_span + 4), min(28, block_span + 10))
    left = random.randint(1, width - block_span - 1)
    cols = [left]
    for gap in gaps:
        cols.append(cols[-1] + BLOCK_SIZE + gap)
    height = random.randint(20, 24)
    block_r0 = random.randint(7, 10)
    return height, width, cols, block_r0


def _weight_rows(count: int, top_limit: int, *, separate: bool) -> list[int]:
    choices = list(range(0, top_limit))
    for _ in range(100):
        rows = sorted(random.sample(choices, count))
        if not separate or all(rows[i + 1] - rows[i] >= 2 for i in range(count - 1)):
            return rows
    raise RuntimeError("failed to choose valid weight rows")


def _build_instance() -> np.ndarray:
    height, width, block_cols, block_r0 = _sample_layout()
    num_blocks = len(block_cols)
    grid = np.zeros((height, width), dtype=int)

    colors = random.sample(BLOCK_COLORS, num_blocks)
    motifs = random.sample(_candidate_masks(), num_blocks)

    block_info = []
    for c0, color, (name, mask) in zip(block_cols, colors, motifs):
        grid[block_r0 : block_r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = int(color)
        motif_r = block_r0 + 6 + random.randint(0, 1)
        motif_c = c0 + random.randint(0, BLOCK_SIZE - mask.shape[1])
        _paint_mask(grid, mask, motif_r, motif_c, color)
        block_info.append(
            {
                "color": int(color),
                "c0": int(c0),
                "mask": mask,
                "channels": _channel_columns(mask),
                "name": name,
            }
        )

    pass_block = random.choice(block_info)
    pass_col = int(pass_block["c0"] + random.choice(pass_block["channels"]))

    solid_blocks = [b for b in block_info if len(set(range(BLOCK_SIZE)) - set(b["channels"])) >= 1]
    block_blocker = random.choice(solid_blocks)
    blocker_local_choices = sorted(set(range(BLOCK_SIZE)) - set(block_blocker["channels"]))
    block_col = int(block_blocker["c0"] + random.choice(blocker_local_choices))

    num_weights = random.choice((2, 3))
    duplicated_column = False
    if num_weights == 2:
        weight_cols = [pass_col, block_col]
    else:
        if random.random() < 0.45:
            weight_cols = [pass_col, block_col, pass_col]
            duplicated_column = True
        else:
            extra_block = random.choice(block_info)
            extra_local = random.choice(_block_columns(extra_block["c0"]))
            weight_cols = [pass_col, block_col, int(extra_local)]

    rows = _weight_rows(len(weight_cols), block_r0 - 1, separate=duplicated_column)
    for r, c in zip(rows, weight_cols):
        grid[int(r), int(c)] = GREY

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_instance()
        try:
            _validate_semantics(grid)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
