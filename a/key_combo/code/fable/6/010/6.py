from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(ARC_ROOT) not in sys.path:
    sys.path.append(str(ARC_ROOT))

from solvers.arc_utils import (  # noqa: E402
    bbox_h,
    bbox_w,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)


PALETTE = list(range(10))
BLOCK_SIDE_RANGE = (4, 7)
GRID_SIDE_RANGE = (16, 24)
PLACEMENT_PADDING = 2
GEN_RETRIES = 400


def _component_records(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    records: list[dict] = []
    for comp in comps:
        box = comp["bbox"]
        patch = crop_bbox(g, box)
        fg = patch != bg
        colors = sorted(int(v) for v in np.unique(patch[fg]))
        records.append(
            {
                "bbox": box,
                "patch": patch,
                "fg": fg,
                "colors": colors,
                "h": int(bbox_h(box)),
                "w": int(bbox_w(box)),
                "area": int(fg.sum()),
            }
        )
    return bg, records


def _identify_objects(grid: np.ndarray) -> dict:
    bg, records = _component_records(grid)
    assert len(records) == 3, "expected exactly three foreground objects"

    block_candidates = [
        rec
        for rec in records
        if rec["h"] == rec["w"] and bool(rec["fg"].all()) and len(rec["colors"]) == 1
    ]
    assert block_candidates, "missing solid monochrome square block"
    block = max(block_candidates, key=lambda rec: (rec["area"], -rec["bbox"][0], -rec["bbox"][1]))
    assert sum(rec["area"] == block["area"] for rec in block_candidates) == 1, "block must be uniquely largest"

    block_color = int(block["colors"][0])
    block_side = int(block["h"])

    motif_candidates = [
        rec
        for rec in records
        if rec is not block
        and len(rec["colors"]) == 1
        and int(rec["colors"][0]) == block_color
        and rec["h"] < block_side
        and rec["w"] < block_side
    ]
    assert len(motif_candidates) == 1, "expected exactly one same-color motif"
    motif = motif_candidates[0]

    tile_candidates = [
        rec
        for rec in records
        if rec is not block
        and rec is not motif
        and rec["h"] == rec["w"]
        and bool(rec["fg"].all())
        and len(rec["colors"]) >= 2
    ]
    assert len(tile_candidates) == 1, "expected exactly one multicolor square tile"
    tile = tile_candidates[0]

    motif_mask = motif["patch"] == block_color
    assert motif_mask.any(), "motif must be nonempty"
    assert (block_side - motif_mask.shape[0]) % 2 == 0, "motif height must center exactly"
    assert (block_side - motif_mask.shape[1]) % 2 == 0, "motif width must center exactly"

    return {
        "bg": bg,
        "block_side": block_side,
        "block_color": block_color,
        "motif_mask": motif_mask,
        "tile": np.array(tile["patch"], dtype=int),
        "records": records,
    }


def _carved_mask(block_side: int, motif_mask: np.ndarray) -> np.ndarray:
    side = int(block_side)
    mh, mw = map(int, motif_mask.shape)
    assert mh < side and mw < side, "motif must be strictly smaller than the block"

    carved = np.ones((side, side), dtype=bool)
    off_r = (side - mh) // 2
    off_c = (side - mw) // 2
    window = carved[off_r : off_r + mh, off_c : off_c + mw]
    window[motif_mask] = False
    assert carved.any() and (~carved).any(), "carving must leave both solid and empty cells"
    return carved


def _render_output(mask: np.ndarray, tile: np.ndarray, bg: int) -> np.ndarray:
    side = int(mask.shape[0])
    tile_side = int(tile.shape[0])
    out = np.full((side * tile_side, side * tile_side), int(bg), dtype=int)
    for r in range(side):
        for c in range(side):
            if not bool(mask[r, c]):
                continue
            paste(out, tile, r * tile_side, c * tile_side)
    return out


def _oracle(block_side: int, motif_mask: np.ndarray, tile: np.ndarray, bg: int) -> np.ndarray:
    return _render_output(_carved_mask(block_side, motif_mask), tile, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve a hidden same-color stencil out of a solid square, then enlarge the surviving mask with a multicolor tile.

    Input structure:
    - The background contains exactly three disconnected objects: one solid monochrome square block,
      one smaller monochrome motif of the same color, and one fully filled multicolor square tile.
    - The motif is not shown inside the block; it sits elsewhere as a separate stencil.

    Rule:
    - Match the motif to the solid block by color and center the motif inside the block.
    - Carve every motif cell out to background, producing a square mask that keeps the block cells not covered by the motif.
    - Replace each remaining mask cell by a full copy of the multicolor tile, and each carved cell by a same-sized background patch.

    Key insight:
    - The controlling mask never appears directly in the input: it must be constructed by carving first, and only then used for pixel-wise enlargement.
    """
    spec = _identify_objects(grid)
    return _oracle(spec["block_side"], spec["motif_mask"], spec["tile"], spec["bg"])


def _choose_block_and_tile_size() -> tuple[int, int]:
    block_side = random.randint(*BLOCK_SIDE_RANGE)
    tile_max = min(5, 30 // block_side)
    tile_side = random.randint(2, tile_max)
    return int(block_side), int(tile_side)


def _choose_motif_dims(block_side: int) -> tuple[int, int]:
    dims = [d for d in range(1, block_side) if d % 2 == block_side % 2]
    assert dims, "block side must allow a smaller centered motif"
    return int(random.choice(dims)), int(random.choice(dims))


def _motif_mask_for_dims(mh: int, mw: int) -> np.ndarray:
    mask = np.zeros((mh, mw), dtype=bool)
    kinds = ["full", "l_tl", "l_br", "tee_top", "tee_bottom"]
    if mh % 2 == 1 and mw % 2 == 1 and mh > 1 and mw > 1:
        kinds.append("cross")
    if mh > 2 and mw > 2:
        kinds.append("border")
    kind = random.choice(kinds)

    if kind == "full":
        mask[:, :] = True
    elif kind == "l_tl":
        mask[0, :] = True
        mask[:, 0] = True
    elif kind == "l_br":
        mask[-1, :] = True
        mask[:, -1] = True
    elif kind == "tee_top":
        mask[0, :] = True
        mask[:, mw // 2] = True
    elif kind == "tee_bottom":
        mask[-1, :] = True
        mask[:, mw // 2] = True
    elif kind == "cross":
        mask[mh // 2, :] = True
        mask[:, mw // 2] = True
    elif kind == "border":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
    else:
        raise AssertionError("unknown motif kind")

    assert mask.any()
    return mask


def _sample_motif(block_side: int, color: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(50):
        mh, mw = _choose_motif_dims(block_side)
        if mh == 1 and mw == 1:
            mask = np.ones((1, 1), dtype=bool)
        else:
            mask = _motif_mask_for_dims(mh, mw)
        if mask.all() and (mh > 1 or mw > 1) and random.random() < 0.6:
            continue
        patch = np.full((mh, mw), int(bg), dtype=int)
        patch[mask] = int(color)
        return patch, mask
    raise RuntimeError("failed to sample motif")


def _sample_tile(side: int, colors: list[int]) -> np.ndarray:
    for _ in range(100):
        mode = random.choice(["random", "hstripe", "vstripe", "quad"])
        tile = np.empty((side, side), dtype=int)

        if mode == "random":
            for r in range(side):
                for c in range(side):
                    tile[r, c] = random.choice(colors)
        elif mode == "hstripe":
            for r in range(side):
                tile[r, :] = random.choice(colors)
        elif mode == "vstripe":
            for c in range(side):
                tile[:, c] = random.choice(colors)
        else:
            split_r = random.randint(1, side - 1)
            split_c = random.randint(1, side - 1)
            quad_colors = [random.choice(colors) for _ in range(4)]
            tile[:split_r, :split_c] = quad_colors[0]
            tile[:split_r, split_c:] = quad_colors[1]
            tile[split_r:, :split_c] = quad_colors[2]
            tile[split_r:, split_c:] = quad_colors[3]

        if len(np.unique(tile)) < 2:
            continue
        if len({tuple(row.tolist()) for row in tile}) < 2 and len({tuple(tile[:, c].tolist()) for c in range(side)}) < 2:
            continue
        return tile
    raise RuntimeError("failed to sample tile")


def _place_patches(patches: list[tuple[str, np.ndarray]], bg: int) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    for _ in range(150):
        h = random.randint(*GRID_SIDE_RANGE)
        w = random.randint(*GRID_SIDE_RANGE)
        grid = np.full((h, w), int(bg), dtype=int)
        occupancy = np.zeros((h, w), dtype=int)
        positions: dict[str, tuple[int, int]] = {}
        order = patches[:]
        random.shuffle(order)
        ok = True

        for name, patch in order:
            box = np.ones(patch.shape, dtype=int)
            try:
                r0, c0 = random_free_location(
                    occupancy,
                    box,
                    transparent=0,
                    border=1,
                    padding=PLACEMENT_PADDING,
                )
            except ValueError:
                ok = False
                break
            paste(occupancy, box, r0, c0)
            paste(grid, patch, r0, c0, transparent=bg)
            positions[name] = (int(r0), int(c0))

        if ok:
            return grid, positions
    raise RuntimeError("failed to place patches")


def _semantic_checks(grid: np.ndarray, spec: dict) -> None:
    detected = _identify_objects(grid)
    assert detected["bg"] == spec["bg"]
    assert detected["block_side"] == spec["block_side"]
    assert np.array_equal(detected["motif_mask"], spec["motif_mask"])
    assert np.array_equal(detected["tile"], spec["tile"])

    expected = _oracle(spec["block_side"], spec["motif_mask"], spec["tile"], spec["bg"])
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver disagrees with oracle"

    block_side = spec["block_side"]
    motif_mask = spec["motif_mask"]
    tile = spec["tile"]
    bg = spec["bg"]
    mh, mw = map(int, motif_mask.shape)
    off_r = (block_side - mh) // 2
    off_c = (block_side - mw) // 2
    assert off_r > 0 or off_c > 0, "motif must require a nontrivial centering step"

    full_block = np.ones((block_side, block_side), dtype=bool)
    assert not np.array_equal(_render_output(full_block, tile, bg), expected)

    motif_only = np.zeros((block_side, block_side), dtype=bool)
    motif_window = motif_only[off_r : off_r + mh, off_c : off_c + mw]
    motif_window[motif_mask] = True
    assert not np.array_equal(_render_output(motif_only, tile, bg), expected)

    top_left_carved = np.ones((block_side, block_side), dtype=bool)
    wrong_window = top_left_carved[:mh, :mw]
    wrong_window[motif_mask] = False
    assert not np.array_equal(_render_output(top_left_carved, tile, bg), expected)

    assert max(grid.shape) <= 30
    assert max(expected.shape) <= 30


def _gen_attempt() -> np.ndarray:
    block_side, tile_side = _choose_block_and_tile_size()
    bg, block_color = random.sample(PALETTE, 2)

    remaining = [c for c in PALETTE if c not in {bg, block_color}]
    tile_color_count = random.randint(2, min(4, len(remaining)))
    tile_colors = random.sample(remaining, tile_color_count)

    block_patch = np.full((block_side, block_side), int(block_color), dtype=int)
    motif_patch, motif_mask = _sample_motif(block_side, block_color, bg)
    tile_patch = _sample_tile(tile_side, tile_colors)

    grid, _positions = _place_patches(
        [
            ("block", block_patch),
            ("motif", motif_patch),
            ("tile", tile_patch),
        ],
        bg,
    )

    spec = {
        "bg": int(bg),
        "block_side": int(block_side),
        "motif_mask": motif_mask,
        "tile": tile_patch,
    }
    _semantic_checks(grid, spec)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _gen_attempt()
        except (AssertionError, RuntimeError, ValueError):
            continue
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
