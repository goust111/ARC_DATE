# Combo variant of dbff022c (index 27)
# combo_spec: D9/2+D13/2
# Name: 按距离排序处理 + 最近邻决定填充色

from __future__ import annotations

import random
import sys

import numpy as np

_SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import components_mask


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The explicit edge palette is removed entirely. Instead, the grid contains one small
#      reference-point marker: a single colored corner cell that is not itself a hollow sprite.
#    - Hollow sprites are still the main objects. Their holes are background in the input and
#      become filled in the output.
#    - Fusion of the two dimensions:
#      1. First sort sprites by distance to the visible reference point.
#      2. Then process them in that order.
#      3. The anchor-nearest sprite uses its nearest peer overall as a seed donor.
#      4. Every later sprite uses the border color of the nearest already-processed sprite.
#    - Example trace with anchor in the top-left corner:
#      A is closest to the anchor, B is next, C is farthest. A fills from its nearest peer,
#      say B, so A's hole becomes B's border color. When B is processed, it is not allowed
#      to look ahead to C; it must choose the nearest earlier sprite, namely A. C then chooses
#      whichever processed sprite (A or B) is spatially closest.
#    - Difference from the original task:
#      old: each sprite used a local border->fill lookup from an edge legend; order did not
#      matter.
#      new: there is no legend, and the candidate donor set is constrained by a spatial order
#      induced by the reference point.
# b) Complexity calibration:
#    - Keep 3-4 sprites so the ordering relation is visible and at least one nontrivial
#      "can't use the still-unprocessed closer future sprite" situation can be generated.
#    - Sprite border colors remain unique, making donor attribution readable by eye.
#    - The anchor is only one corner cell, so it is salient but does not clutter the scene.
# c) Original assumptions that no longer hold:
#    - Any code that searched for a 2xN/Nx2 edge palette is invalid because the palette is
#      removed.
#    - Filling no longer depends only on the enclosing sprite's border color; it depends on
#      global inter-sprite distances and on the processing frontier.
#    - The original "all enclosed holes can be handled independently" assumption breaks because
#      later sprites are restricted to earlier donors.
# d) Placement legality:
#    - Sprites and the anchor use the same padded occupancy rule, so nothing overlaps and every
#      hole remains strictly internal.
#    - The generator enforces strictly increasing anchor distances, which guarantees a readable
#      processing order.
#    - The generator also enforces at least one sprite whose sequential donor differs from its
#      nearest overall donor, preventing regression to the plain D13 variant.
# e) Visual observability:
#    - The order cue is fully visible: the colored corner marker defines the reference point.
#    - The donor cue is also visible: every fill color is copied from another sprite border that
#      already exists in the same grid.
#    - Because one mid-order sprite is forced to be closer to a later sprite than to any earlier
#      sprite, the examples exhibit a real "ordering beats raw nearest-overall" effect.
# f) Visual cleanliness:
#    - Only two visual element types remain: hollow sprites and a one-cell anchor.
#    - One-cell padding between objects keeps borders and filled interiors easy to parse.
# g) Color-change inferability:
#    - The number of active non-bg colors stays low: one anchor color plus 3-4 sprite colors.
#    - Fill colors are never arbitrary; each one can be traced to a visible donor border.
# h) New solver assumptions audited against the generator:
#    - Solver treats any hole-bearing monochrome component as a sprite; generator always gives
#      every sprite at least one enclosed hole.
#    - Solver chooses the smallest hole-free component as the reference marker; generator creates
#      exactly one such component and makes it a single corner cell.
#    - Solver uses 8-connectivity for colored components so the optional 3x3 diamond remains one
#      sprite; generator may emit that shape.
#    - Solver tie-breaks by bbox for arbitrary external inputs, but generator rejects equal anchor
#      distances and nearest-donor distance ties so training examples stay visually unambiguous.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _coords_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Minimum Manhattan distance between two coordinate sets."""
    dr = np.abs(a[:, None, 0] - b[None, :, 0])
    dc = np.abs(a[:, None, 1] - b[None, :, 1])
    return int(np.min(dr + dc))


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return local bg components that stay strictly inside the sprite bbox."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _extract_scene(g: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    """Extract hole-bearing sprites and hole-free marker candidates."""
    sprites: list[dict] = []
    markers: list[dict] = []
    for color in sorted(set(map(int, g.flatten())) - {bg}):
        for comp in components_mask(g == color, connectivity=8):
            coords = np.asarray(comp, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in coords:
                mask[int(r) - r0, int(c) - c0] = True
            local_holes = _get_enclosed_holes(mask)
            item = {
                "color": int(color),
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "area": int(len(coords)),
            }
            if local_holes:
                item["holes"] = [
                    np.asarray([(r0 + int(r), c0 + int(c)) for r, c in pts], dtype=int)
                    for pts in local_holes
                ]
                sprites.append(item)
            else:
                markers.append(item)
    return sprites, markers


def _choose_anchor(markers: list[dict]) -> dict | None:
    if not markers:
        return None
    return min(
        markers,
        key=lambda item: (
            int(item["area"]),
            int(item["bbox"][0]),
            int(item["bbox"][1]),
            int(item["bbox"][2]),
            int(item["bbox"][3]),
        ),
    )


def _anchor_distance(sprite: dict, anchor: dict) -> int:
    return _coords_distance(sprite["coords"], anchor["coords"])


def _processing_order(sprites: list[dict], anchor: dict) -> list[int]:
    ranked = []
    for idx, sprite in enumerate(sprites):
        r0, c0, r1, c1 = sprite["bbox"]
        ranked.append((_anchor_distance(sprite, anchor), r0, c0, r1, c1, idx))
    ranked.sort()
    return [int(idx) for *_rest, idx in ranked]


def _donor_choice(src: dict, candidates: list[int], sprites: list[dict]) -> int:
    best_key = None
    best_idx = None
    for idx in candidates:
        other = sprites[idx]
        r0, c0, r1, c1 = other["bbox"]
        key = (_coords_distance(src["coords"], other["coords"]), r0, c0, r1, c1)
        if best_key is None or key < best_key:
            best_key = key
            best_idx = idx
    assert best_idx is not None, "_donor_choice requires at least one candidate"
    return int(best_idx)


def _overall_donors(sprites: list[dict]) -> list[int]:
    donors = []
    for i, src in enumerate(sprites):
        donors.append(_donor_choice(src, [j for j in range(len(sprites)) if j != i], sprites))
    return donors


def _sequential_donors(sprites: list[dict], anchor: dict) -> tuple[list[int], list[int]]:
    order = _processing_order(sprites, anchor)
    donors = [-1] * len(sprites)
    for pos, idx in enumerate(order):
        if pos == 0:
            candidates = [j for j in range(len(sprites)) if j != idx]
        else:
            candidates = order[:pos]
        donors[idx] = _donor_choice(sprites[idx], candidates, sprites)
    return donors, order


def _build_output(inp: np.ndarray, sprites: list[dict], anchor: dict) -> np.ndarray:
    donors, order = _sequential_donors(sprites, anchor)
    out = inp.copy()
    for idx in order:
        fill = int(sprites[donors[idx]]["color"])
        for hole in sprites[idx]["holes"]:
            for r, c in hole:
                out[int(r), int(c)] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites by walking outward from a visible reference point and copying colors from
    nearby earlier sprites.

    Input structure:
    - The border is mostly background.
    - One isolated non-hole marker cell sits in a corner and acts as the reference point.
    - Several separated hollow single-color sprites appear elsewhere, with their enclosed holes
      still background-colored in the input.

    Rule:
    - Detect the corner marker and sort all hollow sprites by increasing Manhattan distance to it.
    - Process sprites in that order.
    - The anchor-nearest sprite takes the border color of its nearest other sprite overall.
    - Every later sprite takes the border color of the nearest sprite among those already
      processed, and all of its enclosed holes are filled with that color.

    Key insight: The palette lookup is gone; both the processing order and the donor color come
    from visible spatial relationships. The solver must first establish an anchor-based order and
    then restrict each sprite's nearest-neighbor search to the earlier part of that order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    sprites, markers = _extract_scene(g, bg)
    anchor = _choose_anchor(markers)
    if anchor is None or len(sprites) < 2:
        return g.copy()
    return _build_output(g, sprites, anchor)


# --------------- generator ---------------

# Grid dimensions
_GRID_H_MIN, _GRID_H_MAX = 10, 13
_GRID_W_MIN, _GRID_W_MAX = 13, 15
_TINY_H_THRESHOLD = 5

# Sprite geometry
_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16

# Sprite mutation probabilities
_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

# Generation
# This generator uses rejection sampling over a fairly constrained 3-sprite layout.
# The average successful sample appears well before this cap, but a larger ceiling
# avoids occasional false negatives during batch validation.
_MAX_RETRIES = 2000


def _make_diamond() -> np.ndarray:
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = s[1, 0] = s[1, 2] = s[2, 1] = True
    return s


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    sprite = np.ones((h, w), dtype=bool)
    if h >= 3 and w >= 3:
        sprite[1:-1, 1:-1] = False

    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        n = rng.randint(1, w - 2)
        s = rng.randint(1, w - n - 1)
        sprite[br, s : s + n] = True

    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        n = rng.randint(1, h - 2)
        s = rng.randint(1, h - n - 1)
        sprite[s : s + n, bc] = True

    if rng.random() < _P_BORDER_REMOVAL:
        bcells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(bcells)
        for cr, cc in bcells[: rng.randint(1, min(4, len(bcells)))]:
            test = sprite.copy()
            test[cr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test

    if not _get_enclosed_holes(sprite) or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _choose_sprite_count(H: int, W: int, rng: random.Random) -> int:
    del H, W, rng
    return 3


def _find_valid_positions(
    H: int, W: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    sh, sw = mask.shape
    positions = []
    for r0 in range(H - sh + 1):
        for c0 in range(W - sw + 1):
            if occ[max(0, r0 - 1) : min(H, r0 + sh + 1), max(0, c0 - 1) : min(W, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == H - 1 or c0 + hc == W - 1
                for hpts in holes
                for hr, hc in hpts
            ):
                continue
            positions.append((r0, c0))
    return positions


def _mask_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask)


def _global_coords(local_coords: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.asarray([(r0 + int(r), c0 + int(c)) for r, c in local_coords], dtype=int)


def _global_holes(r0: int, c0: int, holes: list[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray([(r0 + int(r), c0 + int(c)) for r, c in pts], dtype=int) for pts in holes]


def _anchor_cell(H: int, W: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, W - 1
    if corner == "bl":
        return H - 1, 0
    return H - 1, W - 1


def _candidate_sprite(mask: np.ndarray, r0: int, c0: int, color: int, holes: list[np.ndarray]) -> dict:
    coords = _global_coords(_mask_coords(mask), r0, c0)
    sh, sw = mask.shape
    return {
        "color": int(color),
        "coords": coords,
        "bbox": (int(r0), int(c0), int(r0 + sh - 1), int(c0 + sw - 1)),
        "holes": _global_holes(r0, c0, holes),
    }


def _place_sprite(grid: np.ndarray, occ: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    sh, sw = mask.shape
    grid[r0 : r0 + sh, c0 : c0 + sw][mask] = int(color)
    occ[max(0, r0 - 1) : min(grid.shape[0], r0 + sh + 1), max(0, c0 - 1) : min(grid.shape[1], c0 + sw + 1)] = True


def _strictly_increasing(values: list[int]) -> bool:
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def _nearest_gap(src: dict, candidates: list[dict]) -> int:
    dists = sorted(_coords_distance(src["coords"], item["coords"]) for item in candidates)
    if not dists:
        return -1
    if len(dists) == 1:
        return 999
    return int(dists[1] - dists[0])


def _fusion_delta(sprites: list[dict], anchor: dict) -> int:
    seq_donors, _ = _sequential_donors(sprites, anchor)
    overall_donors = _overall_donors(sprites)
    return int(sum(int(a != b) for a, b in zip(seq_donors, overall_donors, strict=False)))


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    H = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    W = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    total_sprites = _choose_sprite_count(H, W, rng)

    max_sh = min(5, H - 2)
    max_sw = min(5, W - 2)
    if H <= _TINY_H_THRESHOLD:
        max_sh = 3
    dim_sum_cap = 8

    shapes = []
    for _ in range(total_sprites):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > dim_sum_cap:
            sw = max(3, dim_sum_cap - sh)
        if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
            mask = _make_diamond()
        else:
            mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    all_colors = rng.sample(list(range(1, 10)), total_sprites + 1)
    sprite_colors = all_colors[:total_sprites]
    anchor_color = int(all_colors[-1])

    grid = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=bool)

    corner = rng.choice(["tl", "tr", "bl", "br"])
    ar, ac = _anchor_cell(H, W, corner)
    grid[ar, ac] = anchor_color
    occ[max(0, ar - 1) : min(H, ar + 2), max(0, ac - 1) : min(W, ac + 2)] = True
    anchor = {
        "color": anchor_color,
        "coords": np.asarray([(ar, ac)], dtype=int),
        "bbox": (ar, ac, ar, ac),
        "area": 1,
    }

    sprite_specs = list(zip(shapes, sprite_colors, strict=False))
    rng.shuffle(sprite_specs)
    sprite_specs.sort(key=lambda item: int(item[0].sum()))

    placed: list[dict] = []

    # Place A near the anchor.
    mask_a, color_a = sprite_specs[0]
    holes_a = _get_enclosed_holes(mask_a)
    pos_a = _find_valid_positions(H, W, occ, mask_a, holes_a)
    if not pos_a:
        return None
    ranked_a = []
    for r0, c0 in pos_a:
        cand = _candidate_sprite(mask_a, r0, c0, int(color_a), holes_a)
        ranked_a.append((_anchor_distance(cand, anchor), rng.random(), r0, c0))
    ranked_a.sort()
    _, _, rA, cA = ranked_a[min(len(ranked_a) - 1, rng.randint(0, max(0, len(ranked_a) // 4)))]
    sprite_a = _candidate_sprite(mask_a, rA, cA, int(color_a), holes_a)
    _place_sprite(grid, occ, mask_a, int(color_a), rA, cA)
    placed.append(sprite_a)

    # Place B farther from the anchor than A.
    mask_b, color_b = sprite_specs[1]
    holes_b = _get_enclosed_holes(mask_b)
    pos_b = _find_valid_positions(H, W, occ, mask_b, holes_b)
    if not pos_b:
        return None
    dist_a = _anchor_distance(sprite_a, anchor)
    ranked_b = []
    for r0, c0 in pos_b:
        cand = _candidate_sprite(mask_b, r0, c0, int(color_b), holes_b)
        da = _anchor_distance(cand, anchor)
        if da <= dist_a + 1:
            continue
        ranked_b.append((da, rng.random(), r0, c0))
    if not ranked_b:
        return None
    ranked_b.sort(reverse=True)
    _, _, rB, cB = ranked_b[rng.randint(0, min(len(ranked_b) - 1, max(0, len(ranked_b) // 3)))]
    sprite_b = _candidate_sprite(mask_b, rB, cB, int(color_b), holes_b)
    _place_sprite(grid, occ, mask_b, int(color_b), rB, cB)
    placed.append(sprite_b)

    # Place C even farther, but spatially close to B so B's nearest overall donor is C.
    mask_c, color_c = sprite_specs[2]
    holes_c = _get_enclosed_holes(mask_c)
    pos_c = _find_valid_positions(H, W, occ, mask_c, holes_c)
    if not pos_c:
        return None
    dist_b = _anchor_distance(sprite_b, anchor)
    dist_ab = _coords_distance(sprite_a["coords"], sprite_b["coords"])
    ranked_c = []
    for r0, c0 in pos_c:
        cand = _candidate_sprite(mask_c, r0, c0, int(color_c), holes_c)
        dc = _anchor_distance(cand, anchor)
        dbc = _coords_distance(sprite_b["coords"], cand["coords"])
        if dc <= dist_b + 1:
            continue
        if dbc >= dist_ab:
            continue
        ranked_c.append((dbc, dc, rng.random(), r0, c0))
    if not ranked_c:
        return None
    ranked_c.sort()
    _, _, _, rC, cC = ranked_c[min(len(ranked_c) - 1, rng.randint(0, max(0, len(ranked_c) // 3)))]
    sprite_c = _candidate_sprite(mask_c, rC, cC, int(color_c), holes_c)
    _place_sprite(grid, occ, mask_c, int(color_c), rC, cC)
    placed.append(sprite_c)

    order = _processing_order(placed, anchor)
    anchor_dists = [_anchor_distance(placed[idx], anchor) for idx in order]
    if not _strictly_increasing(anchor_dists):
        return None

    seq_donors, _ = _sequential_donors(placed, anchor)
    overall_donors = _overall_donors(placed)
    if seq_donors == overall_donors:
        return None
    if _fusion_delta(placed, anchor) < 1:
        return None

    for idx, sprite in enumerate(placed):
        if idx == order[0]:
            others = [placed[j] for j in range(len(placed)) if j != idx]
        else:
            pos = order.index(idx)
            others = [placed[j] for j in order[:pos]]
        if _nearest_gap(sprite, others) == 0:
            return None

    border = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    output = _build_output(grid, placed, anchor)
    if np.array_equal(output, grid):
        return None
    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        r = _gen_attempt()
        if r is not None:
            return r
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
