from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste


def _motif_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    return (grid != int(bg)) & (grid != Color.GREY)


def _locate_task(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    bars = find_components(g, bg=bg, colors=[Color.GREY], connectivity=4)
    assert len(bars) == 1, "expected one grey ruler bar"
    bar = bars[0]
    r0, c0, r1, c1 = bar["bbox"]

    motif_components = components_mask(_motif_mask(g, bg), connectivity=4)
    assert len(motif_components) == 1, "expected one attached colored motif"
    motif_coords = motif_components[0]
    mr0, mc0, mr1, mc1 = bbox_coords(motif_coords)
    motif = crop_bbox(g, (mr0, mc0, mr1, mc1))
    assert motif_coords.shape[0] == motif.size, "motif must be a filled rectangle"

    if r0 == r1 and c0 < c1:
        orientation = "h"
        bar_len = c1 - c0 + 1
        tile_len = motif.shape[1]
        assert mr0 == r0 + 1 or mr1 == r0 - 1, "horizontal motif must sit above or below the bar"
        side = 1 if mr0 == r0 + 1 else -1
        assert mc0 == c0 or mc1 == c1, "horizontal motif must attach at the bar's left or right end"
        forward = 1 if mc0 == c0 else -1
        assert bar_len % tile_len == 0 and tile_len < bar_len
        strip = g[mr0 : mr1 + 1, c0 : c1 + 1]
        strip_mask = np.zeros_like(strip, dtype=bool)
        rel_c0 = 0 if forward == 1 else bar_len - tile_len
        strip_mask[:, rel_c0 : rel_c0 + tile_len] = True
        assert np.all(strip[strip_mask] != bg)
        assert np.all(strip[strip_mask] != Color.GREY)
        assert np.all(strip[~strip_mask] == bg)
    elif c0 == c1 and r0 < r1:
        orientation = "v"
        bar_len = r1 - r0 + 1
        tile_len = motif.shape[0]
        assert mc0 == c0 + 1 or mc1 == c0 - 1, "vertical motif must sit left or right of the bar"
        side = 1 if mc0 == c0 + 1 else -1
        assert mr0 == r0 or mr1 == r1, "vertical motif must attach at the bar's top or bottom end"
        forward = 1 if mr0 == r0 else -1
        assert bar_len % tile_len == 0 and tile_len < bar_len
        strip = g[r0 : r1 + 1, mc0 : mc1 + 1]
        strip_mask = np.zeros_like(strip, dtype=bool)
        rel_r0 = 0 if forward == 1 else bar_len - tile_len
        strip_mask[rel_r0 : rel_r0 + tile_len, :] = True
        assert np.all(strip[strip_mask] != bg)
        assert np.all(strip[strip_mask] != Color.GREY)
        assert np.all(strip[~strip_mask] == bg)
    else:
        raise AssertionError("grey component must be a straight 1-cell-thick bar")

    return {
        "bg": int(bg),
        "bar_bbox": (r0, c0, r1, c1),
        "motif_bbox": (mr0, mc0, mr1, mc1),
        "motif": motif,
        "orientation": orientation,
        "forward": int(forward),
        "side": int(side),
    }


def _paint_rolled_copies(out: np.ndarray, spec: dict) -> np.ndarray:
    motif = spec["motif"]
    r0, c0, r1, c1 = spec["bar_bbox"]
    mr0, mc0, mr1, mc1 = spec["motif_bbox"]
    orientation = spec["orientation"]
    forward = spec["forward"]

    if orientation == "h":
        bar_len = c1 - c0 + 1
        tile_w = motif.shape[1]
        repeats = bar_len // tile_w
        for idx in range(repeats):
            phase = np.roll(motif, shift=forward * idx, axis=1)
            if forward == 1:
                tile_c = c0 + idx * tile_w
            else:
                tile_c = c1 - (idx + 1) * tile_w + 1
            paste(out, phase, mr0, tile_c)
    else:
        bar_len = r1 - r0 + 1
        tile_h = motif.shape[0]
        repeats = bar_len // tile_h
        for idx in range(repeats):
            phase = np.roll(motif, shift=forward * idx, axis=0)
            if forward == 1:
                tile_r = r0 + idx * tile_h
            else:
                tile_r = r1 - (idx + 1) * tile_h + 1
            paste(out, phase, tile_r, mc0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Tile the attached motif beside the grey ruler, rolling it one cell farther on each successive copy.

    Input structure:
    - A black background contains one straight horizontal or vertical grey ruler bar.
    - One filled rectangular colored motif touches one side of the ruler at exactly one end.
    - The motif's length along the ruler direction divides the ruler length, and the rest of that side strip is empty background.

    Rule:
    - Keep the ruler and the original motif in place.
    - Fill the full strip beside the ruler with consecutive copies of the motif, starting from the attached end and covering the ruler length exactly.
    - Each later copy is the previous copy cyclically shifted by one cell along the tiling axis: horizontal rulers roll motif columns, vertical rulers roll motif rows, with the shift direction pointing away from the attached end.

    Key insight:
    - The grey bar is a ruler that fixes how many motif-sized slots exist, while the motif's attached end anchors both the copy order and the direction of the cumulative cyclic roll.
    """
    g = np.asarray(grid, dtype=int)
    spec = _locate_task(g)
    return _paint_rolled_copies(g.copy(), spec)


_GEN_RNG = np.random.default_rng()
_GEN_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
_GEN_ATTEMPTS = 400


def _make_tile(height: int, width: int, roll_axis: int) -> np.ndarray:
    for _ in range(200):
        n_colors = int(_GEN_RNG.integers(2, min(len(_GEN_COLORS), height * width) + 1))
        palette = _GEN_RNG.choice(_GEN_COLORS, size=n_colors, replace=False)
        tile = _GEN_RNG.choice(palette, size=(height, width), replace=True).astype(int)
        if len(np.unique(tile)) < 2:
            continue
        if roll_axis == 1:
            cols = [tuple(int(v) for v in tile[:, c]) for c in range(width)]
            if len(set(cols)) != width:
                continue
            if any(np.array_equal(tile, np.roll(tile, s, axis=1)) for s in range(1, width)):
                continue
        else:
            rows = [tuple(int(v) for v in tile[r, :]) for r in range(height)]
            if len(set(rows)) != height:
                continue
            if any(np.array_equal(tile, np.roll(tile, s, axis=0)) for s in range(1, height)):
                continue
        return tile
    raise RuntimeError("failed to sample a non-degenerate motif")


def _render_case(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    H = spec["shape"][0]
    W = spec["shape"][1]
    bg = Color.BLACK
    grid = np.full((H, W), bg, dtype=int)
    out = grid.copy()

    orientation = spec["orientation"]
    side = spec["side"]
    forward = spec["forward"]
    tile = spec["tile"]
    box_r = spec["box_r"]
    box_c = spec["box_c"]
    repeats = spec["repeats"]

    if orientation == "h":
        tile_h, tile_w = tile.shape
        bar_len = repeats * tile_w
        bar_r = box_r if side == 1 else box_r + tile_h
        bar_c = box_c
        grid[bar_r, bar_c : bar_c + bar_len] = Color.GREY
        out[bar_r, bar_c : bar_c + bar_len] = Color.GREY

        tile_r = bar_r + 1 if side == 1 else bar_r - tile_h
        tile_c = bar_c if forward == 1 else bar_c + bar_len - tile_w
        paste(grid, tile, tile_r, tile_c)
        out[:] = _paint_rolled_copies(
            out,
            {
                "motif": tile,
                "bar_bbox": (bar_r, bar_c, bar_r, bar_c + bar_len - 1),
                "motif_bbox": (tile_r, tile_c, tile_r + tile_h - 1, tile_c + tile_w - 1),
                "orientation": orientation,
                "forward": forward,
            },
        )
    else:
        tile_h, tile_w = tile.shape
        bar_len = repeats * tile_h
        bar_r = box_r
        bar_c = box_c if side == 1 else box_c + tile_w
        grid[bar_r : bar_r + bar_len, bar_c] = Color.GREY
        out[bar_r : bar_r + bar_len, bar_c] = Color.GREY

        tile_r = bar_r if forward == 1 else bar_r + bar_len - tile_h
        tile_c = bar_c + 1 if side == 1 else bar_c - tile_w
        paste(grid, tile, tile_r, tile_c)
        out[:] = _paint_rolled_copies(
            out,
            {
                "motif": tile,
                "bar_bbox": (bar_r, bar_c, bar_r + bar_len - 1, bar_c),
                "motif_bbox": (tile_r, tile_c, tile_r + tile_h - 1, tile_c + tile_w - 1),
                "orientation": orientation,
                "forward": forward,
            },
        )

    return grid, out


def _attempt_case() -> np.ndarray | None:
    orientation = str(_GEN_RNG.choice(np.array(["h", "v"])))
    side = int(_GEN_RNG.choice(np.array([-1, 1], dtype=int)))
    forward = int(_GEN_RNG.choice(np.array([-1, 1], dtype=int)))
    repeats = int(_GEN_RNG.integers(3, 6))

    if orientation == "h":
        tile_h = int(_GEN_RNG.integers(2, 6))
        tile_w = int(_GEN_RNG.integers(2, 6))
        box_h = tile_h + 1
        box_w = repeats * tile_w
        roll_axis = 1
    else:
        tile_h = int(_GEN_RNG.integers(2, 6))
        tile_w = int(_GEN_RNG.integers(2, 6))
        box_h = repeats * tile_h
        box_w = tile_w + 1
        roll_axis = 0

    max_h = min(30, box_h + 10)
    max_w = min(30, box_w + 10)
    if box_h + 2 > max_h or box_w + 2 > max_w:
        return None

    H = int(_GEN_RNG.integers(box_h + 2, max_h + 1))
    W = int(_GEN_RNG.integers(box_w + 2, max_w + 1))
    box_r = int(_GEN_RNG.integers(1, H - box_h + 1))
    box_c = int(_GEN_RNG.integers(1, W - box_w + 1))
    tile = _make_tile(tile_h, tile_w, roll_axis=roll_axis)

    inp, expected = _render_case(
        {
            "shape": (H, W),
            "orientation": orientation,
            "side": side,
            "forward": forward,
            "repeats": repeats,
            "box_r": box_r,
            "box_c": box_c,
            "tile": tile,
        }
    )

    if get_bg(inp) != Color.BLACK:
        return None
    if np.array_equal(inp, expected):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _attempt_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rolled-ruler tiling input")
