# Combo variant of bf45cf4b (index 19)
# combo_spec: D8/0+D12/1
# Name: 三分量对称性消歧

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste, random_free_location

"""
Design analysis for the combo variant

a) Concrete behavior and trace
   Original bf45cf4b:
   - input contains 2 disconnected foreground objects
   - one monochrome holey rectangle is the control sprite
   - one multicolor square is the tile
   - output stamps the tile at every filled control pixel
   Combo behavior:
   - input contains 3 disconnected foreground components
   - exactly 2 of them are multicolor square candidate tiles
   - the unique component with LR symmetry + UD symmetry + 90-degree rotational symmetry
     is the true tile
   - the unique least-symmetric component is the control sprite
   - the remaining multicolor square is a distractor and is ignored
   Small trace:
   - object A: 4x4 multicolor square, symmetry flags = (LR, UD, R90) = (1,1,1)
   - object B: 4x4 multicolor square, symmetry flags = (1,0,0)
   - object C: 5x4 monochrome sprite with holes, symmetry flags = (0,0,0)
   Output uses A as the repeated block and C as the mask. B is present only to force the
   symmetry decision; it is not transformed.

b) Complexity calibration
   The D8 source variant introduces 3 components with 2 candidate tiles. To keep that extra
   object meaningful, the distractor is also a multicolor square tile rather than random noise
   or a second sprite. Generator additionally forces the two tile candidates to have:
   - the same size
   - the same number of colors
   - the same broad type (multicolor square)
   This blocks fallback heuristics like "pick the more colorful square" or "pick the larger one".
   The only stable cue left between the 2 tile candidates is symmetry.

c) Implicit assumptions broken from the original main()
   Original assumptions that no longer hold:
   - `len(comps) == 2`
   - "the unique multicolor object is the tile"
   - "the other object is the control"
   - color-count ranking is enough
   New solver assumptions:
   - exactly 3 foreground components exist
   - exactly 2 are multicolor squares
   - exactly 1 is monochrome and fully asymmetric
   - among the 2 tile candidates, exactly 1 has symmetry score 3
   The solver therefore must classify all three objects explicitly and rank symmetry before
   applying the original stamping rule.

d) Placement legality
   Generator uses `random_free_location(..., border=1, padding=1, padding_connectivity=8)` for
   all 3 objects, so placements stay inside the 12-18 grid, do not overlap, and do not touch
   under 8-connectivity. Transparent pasting preserves bg holes inside the control sprite.

e) Visual observability
   Across examples, a human sees 2 plausible decorative tiles and 1 monochrome sprite. The
   repeated pattern in the output always matches the fully symmetric tile, while the output
   silhouette always matches the least-symmetric sprite. Because the distractor tile shares size
   and color-count with the true tile, symmetry is the visible reason one tile is chosen.

f) Visual cleanliness
   The three objects are separated by background margin, with no overlays or markers. Candidate
   tiles stay compact squares; the control stays a single holey rectangle. The scene remains easy
   to parse even though role assignment now needs an extra comparison.

g) Change inferability
   No hidden color remapping is introduced. Output colors are copied verbatim from the chosen
   tile. The only changed latent variable is role selection, and generator concentrates the cue
   into a small, repeated set of symmetry regimes: full D4 symmetry for the true tile, partial
   mirror symmetry for the distractor, and no symmetry for the control.

8) Degeneracy metric used after implementation
   To verify the combo behavior is active rather than incidental, the useful metric is:
   - distractor-vs-true output divergence rate:
     `compose(true_tile, ctrl) != compose(distractor_tile, ctrl)`
   Generator is accepted only if this stays high in aggregate; otherwise the symmetry-based
   choice would not affect the produced output. We also monitor that the two tile candidates
   keep identical size and identical foreground color-count, so the distractor cannot be
   ignored by a simpler non-combo heuristic.

9) New solver assumptions and generator guarantees
   Solver assumptions:
   - there are exactly 3 foreground components
   - exactly 2 are multicolor squares
   - exactly 1 is monochrome and has symmetry score 0
   - exactly 1 multicolor square has symmetry score 3
   - the other multicolor square has symmetry score 1 or 2
   Generator guarantees each of these, and also rejects any sample where:
   - the distractor tile would produce the same expanded output as the true tile
   - the candidate tiles differ in size or color-count, which would weaken the intended cue
"""

TILE_SIZES = (3, 4, 5)
CTRL_DIM_RANGE = range(3, 9)
DISTRACTOR_MODES = ("lr", "ud", "lrud")


def _symmetry_flags(patch: np.ndarray) -> tuple[bool, bool, bool]:
    lr = bool(np.array_equal(patch, patch[:, ::-1]))
    ud = bool(np.array_equal(patch, patch[::-1, :]))
    rot = bool(patch.shape[0] == patch.shape[1] and np.array_equal(patch, np.rot90(patch)))
    return lr, ud, rot


def _component_infos(grid: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    infos = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        sym_flags = _symmetry_flags(patch)
        infos.append(
            {
                "bbox": comp["bbox"],
                "patch": patch,
                "cols": cols,
                "shape": tuple(int(x) for x in patch.shape),
                "sym_flags": sym_flags,
                "sym_score": sum(int(flag) for flag in sym_flags),
            }
        )
    return infos


def _classify_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    infos = _component_infos(grid, bg)
    assert len(infos) == 3

    tile_candidates = [
        info
        for info in infos
        if len(info["cols"]) >= 2 and info["shape"][0] == info["shape"][1]
    ]
    ctrl_candidates = [info for info in infos if len(info["cols"]) == 1]
    assert len(tile_candidates) == 2
    assert len(ctrl_candidates) == 1

    tile_scores = [info["sym_score"] for info in tile_candidates]
    assert sorted(tile_scores) in ([1, 3], [2, 3])

    full_sym_tiles = [info for info in tile_candidates if info["sym_score"] == 3]
    assert len(full_sym_tiles) == 1
    tile_info = full_sym_tiles[0]

    ctrl_info = ctrl_candidates[0]
    assert ctrl_info["sym_score"] == min(info["sym_score"] for info in infos) == 0

    distractor_info = next(info for info in tile_candidates if info is not tile_info)
    assert len(tile_info["cols"]) == len(distractor_info["cols"])
    assert tile_info["shape"] == distractor_info["shape"]

    return ctrl_info["patch"], tile_info["patch"], distractor_info["patch"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Input contains three disconnected foreground components.
    - Unique fully symmetric multicolor square -> tile
    - Unique least-symmetric monochrome sprite -> control
    - Remaining multicolor square -> distractor, ignored
    Output is the original block-stamping expansion, tightly cropped.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrl, tile, _distractor = _classify_objects(g, bg)
    th, tw = map(int, tile.shape)
    ch, cw = map(int, ctrl.shape)

    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != bg:
                paste(out, tile, r * th, c * tw)
    return out


def _dihedral_orbits(size: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
    s = int(size)
    for r in range(s):
        for c in range(s):
            if (r, c) in seen:
                continue
            orbit = {
                (r, c),
                (c, s - 1 - r),
                (s - 1 - r, s - 1 - c),
                (s - 1 - c, r),
                (r, s - 1 - c),
                (s - 1 - r, c),
                (c, r),
                (s - 1 - c, s - 1 - r),
            }
            cells = sorted(orbit)
            seen.update(cells)
            out.append(cells)
    return out


def _mirror_orbits(size: int, mode: str) -> list[list[tuple[int, int]]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[list[tuple[int, int]]] = []
    s = int(size)
    for r in range(s):
        for c in range(s):
            cells = {(r, c)}
            if mode in {"lr", "lrud"}:
                cells.add((r, s - 1 - c))
            if mode in {"ud", "lrud"}:
                cells.add((s - 1 - r, c))
            if mode == "lrud":
                cells.add((s - 1 - r, s - 1 - c))
            orbit = tuple(sorted(cells))
            if orbit in seen:
                continue
            seen.add(orbit)
            out.append(list(orbit))
    return out


def _fill_orbits(
    size: int,
    orbits: list[list[tuple[int, int]]],
    colors: list[int],
    expected_flags: tuple[bool, bool, bool],
    rng: np.random.Generator,
) -> np.ndarray:
    assert len(orbits) >= len(colors)
    for _ in range(400):
        tile = np.full((size, size), int(colors[0]), dtype=int)
        assignments: dict[int, int] = {}
        orbit_ids = list(range(len(orbits)))
        rng.shuffle(orbit_ids)
        for color, idx in zip(colors, orbit_ids):
            assignments[idx] = int(color)
        for idx in range(len(orbits)):
            if idx not in assignments:
                assignments[idx] = int(rng.choice(colors))
        for idx, orbit in enumerate(orbits):
            for r, c in orbit:
                tile[r, c] = assignments[idx]
        if colors_present(tile) != sorted(set(int(color) for color in colors)):
            continue
        if _symmetry_flags(tile) == expected_flags:
            return tile
    raise RuntimeError(f"failed to synthesize tile with symmetry {expected_flags}")


def _make_true_tile(size: int, tile_colors: list[int], rng: np.random.Generator) -> np.ndarray:
    return _fill_orbits(
        size,
        _dihedral_orbits(size),
        tile_colors,
        (True, True, True),
        rng,
    )


def _make_distractor_tile(
    size: int,
    tile_colors: list[int],
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    expected = {
        "lr": (True, False, False),
        "ud": (False, True, False),
        "lrud": (True, True, False),
    }[mode]
    return _fill_orbits(
        size,
        _mirror_orbits(size, mode),
        tile_colors,
        expected,
        rng,
    )


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _connected(mask: np.ndarray) -> bool:
    comps = find_components(
        mask.astype(int),
        bg=0,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    return len(comps) == 1


def _touches_bbox_edges(mask: np.ndarray) -> bool:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return bool(rows[0] and rows[-1] and cols[0] and cols[-1])


def _make_control_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    area = h * w
    min_fill = max(h + w - 1, int(np.ceil(0.28 * area)))
    max_fill = min(area - 1, int(np.floor(0.72 * area)))
    assert min_fill <= max_fill

    for _ in range(250):
        target = int(rng.integers(min_fill, max_fill + 1))
        ar = int(rng.integers(0, h))
        ac = int(rng.integers(0, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, ac] = True
        mask[ar, :] = True

        while int(mask.sum()) < target:
            frontier = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    if any(mask[rr, cc] for rr, cc in _neighbors8(h, w, r, c)):
                        frontier.append((r, c))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            mask[pick] = True

        if int(mask.sum()) != target:
            continue
        if not _connected(mask):
            continue
        if not _touches_bbox_edges(mask):
            continue
        if _symmetry_flags(mask.astype(int)) != (False, False, False):
            continue
        return mask
    raise RuntimeError("failed to build asymmetric control mask")


def _mask_to_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _compose_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != bg:
                paste(out, tile, r * th, c * tw)
    return out


def _make_input(
    tile: np.ndarray,
    ctrl: np.ndarray,
    distractor: np.ndarray,
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    objects = [tile, ctrl, distractor]
    max_h = max(int(obj.shape[0]) for obj in objects)
    max_w = max(int(obj.shape[1]) for obj in objects)
    h_min = max(12, max_h + 2)
    w_min = max(12, max_w + 2)

    for _ in range(150):
        H = int(rng.integers(h_min, 19))
        W = int(rng.integers(w_min, 19))
        grid = np.full((H, W), int(bg), dtype=int)
        order = [0, 1, 2]
        rng.shuffle(order)
        ok = True
        for idx in order:
            patch = objects[idx]
            try:
                pos = random_free_location(
                    grid,
                    patch,
                    transparent=bg,
                    border=1,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                ok = False
                break
            paste(grid, patch, pos[0], pos[1], transparent=bg)
        if ok:
            return grid
    raise RuntimeError("placement failed")


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=5, replace=False)]
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    tile_palette = [int(x) for x in palette[2:]]

    size = int(rng.choice(TILE_SIZES))
    n_tile_colors = int(rng.integers(2, min(4, len(tile_palette)) + 1))
    tile_colors = [int(x) for x in rng.choice(tile_palette, size=n_tile_colors, replace=False)]

    tile = _make_true_tile(size, tile_colors, rng)
    distractor = _make_distractor_tile(size, tile_colors, str(rng.choice(DISTRACTOR_MODES)), rng)

    ch = int(rng.choice(tuple(CTRL_DIM_RANGE)))
    cw = int(rng.choice(tuple(CTRL_DIM_RANGE)))
    ctrl = _mask_to_patch(_make_control_mask(ch, cw, rng), ctrl_color, bg)

    inp = _make_input(tile, ctrl, distractor, bg, rng)
    expected = _compose_output(tile, ctrl, bg)
    wrong_tile_output = _compose_output(distractor, ctrl, bg)

    infos = _component_infos(inp, bg)
    if len(infos) != 3:
        return None

    tiles = [info for info in infos if len(info["cols"]) >= 2 and info["shape"][0] == info["shape"][1]]
    ctrls = [info for info in infos if len(info["cols"]) == 1]
    if len(tiles) != 2 or len(ctrls) != 1:
        return None
    if tiles[0]["shape"] != tiles[1]["shape"]:
        return None
    if len(tiles[0]["cols"]) != len(tiles[1]["cols"]):
        return None
    if sorted(info["sym_score"] for info in tiles) not in ([1, 3], [2, 3]):
        return None
    if ctrls[0]["sym_score"] != 0:
        return None

    if np.array_equal(expected, wrong_tile_output):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
