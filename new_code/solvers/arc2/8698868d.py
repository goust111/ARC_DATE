from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import bbox_coords, bbox_h, bbox_w, crop_bbox, find_components, label_mask

def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, then nest each sprite inside its frame.

    Input structure:
    - bg = azure (8). Large hollow squares (frames) arranged in a grid; smaller filled squares (sprites) placed elsewhere.
    - Each frame and sprite has a distinct color. Sprites may contain holes (non-sprite-color pixels).

    Rule:
    - Frames form a regular grid; some may be bg-colored (invisible).
    - Count connected hole regions inside each frame's interior and each sprite.
    - Pair each frame with the sprite sharing the same hole count.
    - Place the matched sprite inside the frame border; recolor sprite holes to the frame's color.

    Key insight: The matching key is the number of connected hole components, not hole area or color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    BG = 8  # Azure background for this puzzle family

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    # Find square regions (lower fill threshold to catch sprites with holes).
    MIN_FILL_RATIO = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * MIN_FILL_RATIO):
            square_regions.append((int(h), comp))

    assert not (not square_regions)

    # Group by size - frames are larger, sprites are smaller.
    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(int(frame_size), [])
    sprites_raw = sizes.get(int(sprite_size), [])
    assert not (not frames)

    # Determine frame grid structure from frame positions.
    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
    min_r = min(r for r, c in frame_positions)
    min_c = min(c for r, c in frame_positions)
    max_r = max(r for r, c in frame_positions)
    max_c = max(c for r, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    # Build frame grid (including potential background-colored frames).
    frame_grid = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])

            inner = region[1:-1, 1:-1]
            hole_mask = inner != int(frame_color)
            _lab, num_holes = label_mask(hole_mask, connectivity=4)

            frame_grid.append({"color": int(frame_color), "out_r": int(ri * frame_size), "out_c": int(ci * frame_size), "holes": int(num_holes)})

    # Process sprites - count holes in each.
    sprites = []
    for comp in sprites_raw:
        b = comp["bbox"]
        sprite_region = crop_bbox(g, b)
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != int(sprite_color)
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append({"color": int(sprite_color), "holes": int(num_holes), "content": sprite_region.copy(), "h": int(sprite_region.shape[0]), "w": int(sprite_region.shape[1])})

    # Match frames to sprites by hole count.
    matches: dict[tuple[int, int], tuple[dict, dict]] = {}
    used_sprites: set[int] = set()
    for f in frame_grid:
        for i, s in enumerate(sprites):
            if int(i) not in used_sprites and int(f["holes"]) == int(s["holes"]):
                matches[(int(f["out_r"]), int(f["out_c"]))] = (f, s)
                used_sprites.add(int(i))
                break

    # Build output grid.
    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for f in frame_grid:
        fr, fc = int(f["out_r"]), int(f["out_c"])
        frame_color = int(f["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)

        if (fr, fc) in matches:
            _f, sprite = matches[(fr, fc)]
            content = np.array(sprite["content"], dtype=int).copy()
            sprite_color = int(sprite["color"])
            content[content != int(sprite_color)] = int(frame_color)

            sh, sw = int(sprite["h"]), int(sprite["w"])
            out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out

