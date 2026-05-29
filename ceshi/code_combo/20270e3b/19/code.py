# Combo variant of 20270e3b (index 19)
# combo_spec: D1/0+D8/1+D10/1
# Name: 多色连接器三精灵链
#
# Design analysis
# - a) Concrete behavior and trace:
#   Original 20270e3b has exactly 2 sprites, one connector on each sprite, one fixed connector
#   color, and the pair is matched by identical horizontal-bar shape. The plug is shifted so its
#   connector sits one row above the base connector; then all connector cells are recolored to
#   body color.
#   This combo changes three things together:
#   1. There are now 3 sprites A-B-C, not 2. A is still anchored at the input origin.
#   2. Matching is by connector color, not by equal shape. We deliberately allow unequal lengths.
#   3. B is the middle piece and therefore has two connectors: an upper connector sharing a color
#      with A, and a lower connector sharing a different color with C. The two shared colors define
#      both the pairing and the chain order.
#   Concrete trace:
#   - Suppose A has a red downward connector of length 3.
#   - B has a red upward connector of length 1 and a green downward connector of length 2.
#   - C has a green upward connector of length 3.
#   - Old logic would fail immediately because it expects exactly 2 sprites and equal connector
#     shapes. New logic first identifies the unique origin sprite A, the only 2-connector sprite B,
#     and the remaining terminal sprite C. Then it matches red for A-B and green for B-C, aligns
#     both seams by left endpoints, assembles A+B and then C onto shifted B, and recolors all
#     connector colors to body color.
# - b) Complexity calibration:
#   The scene stays readable because only B has two connectors and only two connector colors are
#   shared across sprites. We keep body/background colors fixed and use only 2 shared connector
#   colors per sample. To make color matching genuinely necessary, the generator enforces
#   `len(A.red) != len(B.red)` and `len(B.green) != len(C.green)` in every sample; shape cannot be
#   the key.
# - c) Hidden assumptions broken from the original solver:
#   `len(best_comps) == 2` is invalid; we now require exactly 3 components.
#   `ctrl = 7` is invalid; connector colors are any non-{bg,obj} colors.
#   "one connector per sprite" is invalid; B has exactly 2 connectors.
#   `shpA == shpB` is invalid as a matching rule; we align by left endpoint after color matching.
#   A single translation is insufficient; we need one shift for B relative to A and one for C
#   relative to shifted B.
#   Overlap checks can no longer reference one fixed ctrl color; any non-{bg,obj} connector color
#   may participate in a seam.
# - d) Placement legality:
#   The generator constructs A first, then B in docked coordinates against A, then C in docked
#   coordinates against the partial A-B chain. This lets it reject illegal overlap before scattering
#   the sprites onto the input canvas. Input placement uses padding so the 3 sprites remain separate
#   8-connected components.
# - e) Visual observability:
#   From a few examples, a human can see:
#   * one piece has two differently colored connectors, so it must be the middle link,
#   * A shares exactly one connector color with B, and C shares the other connector color with B,
#   * matched bars often have different lengths, so color rather than length decides pairing.
# - f) Visual cleanliness:
#   No distractor connectors are added beyond the chain connectors themselves. The overlap in the
#   assembled state is limited to the two docking seams, and all connector rows remain single clean
#   horizontal runs.
# - g) Inferability of the color change:
#   Only background/body plus two shared connector colors appear. All non-body connector colors
#   disappear into the body after assembly. Because the same connector color repeats exactly on the
#   intended pair, the recoloring rule is structurally visible rather than arbitrary.
# - h) Main/generator contract:
#   `main()` assumes exactly one origin-anchored sprite with one downward connector, exactly one
#   middle sprite with one upward and one downward connector of different colors, and exactly one
#   terminal sprite with one upward connector. `generate_input()` explicitly constructs samples that
#   satisfy those assumptions and also enforces unequal matched lengths to prevent degeneration back
#   to the original equal-shape task.

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


def _crop_non_bg(arr: np.ndarray, bg: int) -> tuple[np.ndarray, int, int]:
    nz = np.argwhere(arr != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int), 0, 0
    r0, c0, r1, c1 = bbox_coords(nz)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy(), int(r0), int(c0)


def _extract_components(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    lab, n = label_mask(g != int(bg), connectivity=8)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(k))
        r0, c0, r1, c1 = bbox_coords(pts)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        sprite = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for r, c in pts:
            sprite[int(r - r0), int(c - c0)] = int(g[int(r), int(c)])
        comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "sprite": sprite,
                "top_left_sum": int(r0 + c0),
            }
        )
    comps.sort(key=lambda item: int(item["top_left_sum"]))
    return comps


def _connector_runs(sprite: np.ndarray, bg: int, obj: int) -> list[dict[str, object]]:
    h, w = sprite.shape
    runs: list[dict[str, object]] = []
    for r in range(int(h)):
        c = 0
        while c < int(w):
            col = int(sprite[int(r), int(c)])
            if col in (int(bg), int(obj)):
                c += 1
                continue
            c0 = int(c)
            while c + 1 < int(w) and int(sprite[int(r), int(c + 1)]) == int(col):
                c += 1
            c1 = int(c)
            pts = {(int(r), int(cc)) for cc in range(int(c0), int(c1) + 1)}
            runs.append(
                {
                    "color": int(col),
                    "row": int(r),
                    "c0": int(c0),
                    "c1": int(c1),
                    "len": int(c1 - c0 + 1),
                    "pts": pts,
                }
            )
            c += 1
    return runs


def _classify_run(sprite: np.ndarray, run: dict[str, object], bg: int, obj: int) -> dict[str, object]:
    row = int(run["row"])
    cols = list(range(int(run["c0"]), int(run["c1"]) + 1))

    up = row + 1 < sprite.shape[0] and all(int(sprite[int(row + 1), int(c)]) == int(obj) for c in cols)
    up = up and all(int(row) == 0 or int(sprite[int(row - 1), int(c)]) == int(bg) for c in cols)

    down = row > 0 and all(int(sprite[int(row - 1), int(c)]) == int(obj) for c in cols)
    down = down and all(int(row + 1) >= sprite.shape[0] or int(sprite[int(row + 1), int(c)]) == int(bg) for c in cols)

    assert up ^ down
    desc = dict(run)
    desc["orientation"] = "up" if up else "down"
    return desc


def _sprite_desc(comp: dict[str, object], bg: int, obj: int) -> dict[str, object]:
    sprite = np.array(comp["sprite"], dtype=int)
    runs = [_classify_run(sprite, run, bg, obj) for run in _connector_runs(sprite, bg, obj)]
    by_color = {int(run["color"]): run for run in runs}
    assert len(by_color) == len(runs)
    desc = dict(comp)
    desc["connectors"] = runs
    desc["by_color"] = by_color
    desc["colors"] = set(int(run["color"]) for run in runs)
    return desc


def _stamp_shifted(out: np.ndarray, patch: np.ndarray, top: int, left: int, *, bg: int, obj: int) -> None:
    h, w = patch.shape
    for r in range(int(h)):
        for c in range(int(w)):
            val = int(patch[int(r), int(c)])
            if val == int(bg):
                continue
            rr = int(top + r)
            cc = int(left + c)
            cur = int(out[int(rr), int(cc)])
            if cur != int(bg):
                assert cur != int(obj) or val != int(obj)
                assert cur not in (int(bg), int(obj)) or val not in (int(bg), int(obj))
            out[int(rr), int(cc)] = int(val)


def _assemble_chain(
    base_sprite: np.ndarray,
    middle_sprite: np.ndarray,
    end_sprite: np.ndarray,
    *,
    bg: int,
    obj: int,
    base_ab: dict[str, object],
    middle_ab: dict[str, object],
    middle_bc: dict[str, object],
    end_bc: dict[str, object],
) -> np.ndarray:
    tr_mid = int(int(base_ab["row"]) - 1 - int(middle_ab["row"]))
    tc_mid = int(int(base_ab["c0"]) - int(middle_ab["c0"]))
    tr_end = int(int(middle_bc["row"]) + tr_mid - 1 - int(end_bc["row"]))
    tc_end = int(int(middle_bc["c0"]) + tc_mid - int(end_bc["c0"]))

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(bg))}
    mid_pts = {(int(r + tr_mid), int(c + tc_mid)) for r, c in np.argwhere(middle_sprite != int(bg))}
    end_pts = {(int(r + tr_end), int(c + tc_end)) for r, c in np.argwhere(end_sprite != int(bg))}
    coords_all = np.array(list(base_pts | mid_pts | end_pts), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))
    assert rr0 == 0 and cc0 == 0

    out = np.full((int(rr1 + 1), int(cc1 + 1)), int(bg), dtype=int)
    _stamp_shifted(out, base_sprite, 0, 0, bg=bg, obj=obj)
    _stamp_shifted(out, middle_sprite, tr_mid, tc_mid, bg=bg, obj=obj)
    _stamp_shifted(out, end_sprite, tr_end, tc_end, bg=bg, obj=obj)
    out[(out != int(bg)) & (out != int(obj))] = int(obj)
    cropped, _r0, _c0 = _crop_non_bg(out, bg)
    return cropped


def _variant_metrics_from_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg, obj = 1, 4
    comps = _extract_components(g, bg)
    assert len(comps) == 3
    sprites = [_sprite_desc(comp, bg, obj) for comp in comps]

    base_candidates = [spr for spr in sprites if int(spr["top_left_sum"]) == 0]
    assert len(base_candidates) == 1
    base = base_candidates[0]
    others = [spr for spr in sprites if spr is not base]

    middle = None
    end = None
    for spr in others:
        orientations = sorted(str(run["orientation"]) for run in spr["connectors"])
        if orientations == ["down", "up"]:
            middle = spr
        elif orientations == ["up"]:
            end = spr
    assert middle is not None and end is not None

    shared_ab = set(base["colors"]) & set(middle["colors"])
    shared_bc = set(middle["colors"]) & set(end["colors"])
    assert len(shared_ab) == 1 and len(shared_bc) == 1
    color_ab = int(next(iter(shared_ab)))
    color_bc = int(next(iter(shared_bc)))
    base_ab = base["by_color"][color_ab]
    middle_ab = middle["by_color"][color_ab]
    middle_bc = middle["by_color"][color_bc]
    end_bc = end["by_color"][color_bc]
    return {
        "ab_color": int(color_ab),
        "bc_color": int(color_bc),
        "ab_lengths": (int(base_ab["len"]), int(middle_ab["len"])),
        "bc_lengths": (int(middle_bc["len"]), int(end_bc["len"])),
        "middle_colors": sorted(int(x) for x in middle["colors"]),
        "base_end_shared": sorted(int(x) for x in (set(base["colors"]) & set(end["colors"]))),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, obj = 1, 4

    comps = _extract_components(g, bg)
    assert len(comps) == 3
    sprites = [_sprite_desc(comp, bg, obj) for comp in comps]

    base_candidates = [spr for spr in sprites if int(spr["top_left_sum"]) == 0]
    assert len(base_candidates) == 1
    base = base_candidates[0]
    others = [spr for spr in sprites if spr is not base]

    assert len(base["connectors"]) == 1
    assert str(base["connectors"][0]["orientation"]) == "down"

    middle_candidates = []
    end_candidates = []
    for spr in others:
        orientations = sorted(str(run["orientation"]) for run in spr["connectors"])
        if orientations == ["down", "up"] and len(spr["connectors"]) == 2:
            middle_candidates.append(spr)
        elif orientations == ["up"] and len(spr["connectors"]) == 1:
            end_candidates.append(spr)
    assert len(middle_candidates) == 1 and len(end_candidates) == 1
    middle = middle_candidates[0]
    end = end_candidates[0]

    shared_ab = set(base["colors"]) & set(middle["colors"])
    shared_bc = set(middle["colors"]) & set(end["colors"])
    shared_ac = set(base["colors"]) & set(end["colors"])
    assert len(shared_ab) == 1
    assert len(shared_bc) == 1
    assert len(shared_ac) == 0

    color_ab = int(next(iter(shared_ab)))
    color_bc = int(next(iter(shared_bc)))
    assert color_ab != color_bc

    base_ab = dict(base["by_color"][color_ab])
    middle_ab = dict(middle["by_color"][color_ab])
    middle_bc = dict(middle["by_color"][color_bc])
    end_bc = dict(end["by_color"][color_bc])

    assert str(base_ab["orientation"]) == "down"
    assert str(middle_ab["orientation"]) == "up"
    assert str(middle_bc["orientation"]) == "down"
    assert str(end_bc["orientation"]) == "up"

    return _assemble_chain(
        np.array(base["sprite"], dtype=int),
        np.array(middle["sprite"], dtype=int),
        np.array(end["sprite"], dtype=int),
        bg=bg,
        obj=obj,
        base_ab=base_ab,
        middle_ab=middle_ab,
        middle_bc=middle_bc,
        end_bc=end_bc,
    )


def _grow_shape(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allow=None,
) -> None:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (int(r + dr), int(c + dc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= int(r + dr) < h
            and 0 <= int(c + dc) < w
            and not bool(mask[int(r + dr), int(c + dc)])
            and (allow is None or bool(allow(int(r + dr), int(c + dc))))
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[int(nr), int(nc)] = True
        frontier.append((int(nr), int(nc)))
        count += 1


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    h, w = mask.shape
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        r = int(r)
        c = int(c)
        if (int(r), int(c)) in keep:
            continue
        adj_bg = any(
            not (0 <= int(r + dr) < h and 0 <= int(c + dc) < w) or not bool(mask[int(r + dr), int(c + dc)])
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[int(r), int(c)] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[int(r), int(c)] = True


def _down_connector_candidates(mask: np.ndarray, conn_len: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r in range(1, int(h)):
        for c0 in range(0, int(w - conn_len + 1)):
            cols = range(int(c0), int(c0 + conn_len))
            if not all(bool(mask[int(r), int(c)]) for c in cols):
                continue
            if not all(bool(mask[int(r - 1), int(c)]) for c in cols):
                continue
            if any(int(r + 1) < h and bool(mask[int(r + 1), int(c)]) for c in cols):
                continue
            out.append((int(r), int(c0)))
    return out


def _build_base(conn_len: int, color_ab: int) -> tuple[np.ndarray, dict[str, int]]:
    bg, obj = 1, 4
    base_h = _rng.randint(3, 7)
    base_w = _rng.randint(max(3, int(conn_len) + 1), 9)
    base_target = _rng.randint(max(6, int(conn_len) + 3), min(24, int(base_h * base_w)))
    mask = np.zeros((int(base_h), int(base_w)), dtype=bool)
    mask[0, 0] = True
    _grow_shape(mask, int(base_target), [(0, 0)])
    if not is_contiguous(mask, background=0, connectivity=8):
        raise ValueError("base-not-contiguous")
    _thin(mask)

    candidates = _down_connector_candidates(mask, int(conn_len))
    if not candidates:
        raise ValueError("base-no-connector")
    row, c0 = _rng.choice(candidates)
    grid = np.full(mask.shape, int(bg), dtype=int)
    grid[mask] = int(obj)
    grid[int(row), int(c0) : int(c0 + conn_len)] = int(color_ab)
    if not is_contiguous(grid != int(bg), background=0, connectivity=8):
        raise ValueError("base-grid-not-contiguous")
    cropped, _r0, c_shift = _crop_non_bg(grid, bg)
    return cropped, {"row": int(row), "c0": int(c0 - c_shift), "len": int(conn_len), "color": int(color_ab)}


def _build_middle(
    len_top: int,
    len_bot: int,
    color_ab: int,
    color_bc: int,
    base_grid: np.ndarray,
    base_conn: dict[str, int],
) -> tuple[np.ndarray, dict[str, dict[str, int]], tuple[int, int]]:
    bg, obj = 1, 4
    base_h, base_w = base_grid.shape
    mid_h = _rng.randint(4, 8)
    mid_w = _rng.randint(max(3, int(max(len_top, len_bot)) + 1), 9)

    max_top_c0 = min(int(mid_w - len_top), int(base_conn["c0"]))
    if max_top_c0 < 0:
        raise ValueError("middle-left-overflow")
    top_c0 = _rng.randint(0, int(max_top_c0))
    bot_c0 = _rng.randint(0, int(mid_w - len_bot))
    dock_r0 = int(base_conn["row"] - 1)
    dock_c0 = int(base_conn["c0"] - top_c0)

    top_set = set(range(int(top_c0), int(top_c0 + len_top)))
    bot_set = set(range(int(bot_c0), int(bot_c0 + len_bot)))
    top_center = int(top_c0 + len_top // 2)
    bot_center = int(bot_c0 + len_bot // 2)

    mask = np.zeros((int(mid_h), int(mid_w)), dtype=bool)
    for c in top_set:
        mask[0, int(c)] = True
        mask[1, int(c)] = True
    for c in bot_set:
        mask[int(mid_h - 1), int(c)] = True
        mask[int(mid_h - 2), int(c)] = True
    for r in range(1, int(mid_h - 1)):
        mask[int(r), int(top_center)] = True
    for c in range(min(int(top_center), int(bot_center)), max(int(top_center), int(bot_center)) + 1):
        mask[int(mid_h - 2), int(c)] = True

    keep = {
        (0, int(c)) for c in top_set
    } | {
        (1, int(c)) for c in top_set
    } | {
        (int(mid_h - 1), int(c)) for c in bot_set
    } | {
        (int(mid_h - 2), int(c)) for c in bot_set
    }

    seed_count = int(mask.sum())
    target = _rng.randint(max(8, int(seed_count)), min(24, int(mid_h * mid_w)))

    def allow(r: int, c: int) -> bool:
        if int(r) == 0:
            return int(c) in top_set
        if int(r) == int(mid_h - 1):
            return int(c) in bot_set
        rr = int(dock_r0 + r)
        cc = int(dock_c0 + c)
        if 0 <= rr < int(base_h) and 0 <= cc < int(base_w):
            cur = int(base_grid[int(rr), int(cc)])
            if cur == int(obj):
                return int(r) == 0 and int(c) in top_set
            if cur != int(bg):
                return int(r) == 1 and int(c) in top_set
        return True

    _grow_shape(mask, int(target), [(int(r), int(c)) for r, c in np.argwhere(mask)], allow=allow)
    if not is_contiguous(mask, background=0, connectivity=8):
        raise ValueError("middle-not-contiguous")
    _thin(mask, keep=keep)
    if not is_contiguous(mask, background=0, connectivity=8):
        raise ValueError("middle-not-contiguous-after-thin")

    if sorted(int(c) for c in np.where(mask[0])[0]) != list(range(int(top_c0), int(top_c0 + len_top))):
        raise ValueError("middle-top-row-dirty")
    if sorted(int(c) for c in np.where(mask[int(mid_h - 1)])[0]) != list(range(int(bot_c0), int(bot_c0 + len_bot))):
        raise ValueError("middle-bottom-row-dirty")
    if not all(bool(mask[1, int(c)]) for c in top_set):
        raise ValueError("middle-top-support-broken")
    if not all(bool(mask[int(mid_h - 2), int(c)]) for c in bot_set):
        raise ValueError("middle-bottom-support-broken")

    for r, c in np.argwhere(mask):
        rr = int(dock_r0 + r)
        cc = int(dock_c0 + c)
        if 0 <= rr < int(base_h) and 0 <= cc < int(base_w):
            cur = int(base_grid[int(rr), int(cc)])
            if cur == int(obj):
                if not (int(r) == 0 and int(c) in top_set):
                    raise ValueError("middle-overlaps-base-body")
            elif cur != int(bg):
                if not (int(r) == 1 and int(c) in top_set):
                    raise ValueError("middle-overlaps-base-connector")

    grid = np.full(mask.shape, int(bg), dtype=int)
    grid[mask] = int(obj)
    grid[0, int(top_c0) : int(top_c0 + len_top)] = int(color_ab)
    grid[int(mid_h - 1), int(bot_c0) : int(bot_c0 + len_bot)] = int(color_bc)
    if not is_contiguous(grid != int(bg), background=0, connectivity=8):
        raise ValueError("middle-grid-not-contiguous")
    cropped, _r_shift, c_shift = _crop_non_bg(grid, bg)
    return (
        cropped,
        {
            "ab": {"row": 0, "c0": int(top_c0 - c_shift), "len": int(len_top), "color": int(color_ab)},
            "bc": {
                "row": int(cropped.shape[0] - 1),
                "c0": int(bot_c0 - c_shift),
                "len": int(len_bot),
                "color": int(color_bc),
            },
        },
        (int(dock_r0), int(dock_c0 + c_shift)),
    )


def _build_end(
    conn_len: int,
    color_bc: int,
    chain_grid: np.ndarray,
    chain_conn: dict[str, int],
) -> tuple[np.ndarray, dict[str, int]]:
    bg, obj = 1, 4
    chain_h, chain_w = chain_grid.shape
    end_h = _rng.randint(3, 7)
    end_w = _rng.randint(max(2, int(conn_len) + 1), 9)
    max_c0 = min(int(end_w - conn_len), int(chain_conn["c0"]))
    if max_c0 < 0:
        raise ValueError("end-left-overflow")
    conn_c0 = _rng.randint(0, int(max_c0))
    dock_r0 = int(chain_conn["row"] - 1)
    dock_c0 = int(chain_conn["c0"] - conn_c0)
    conn_set = set(range(int(conn_c0), int(conn_c0 + conn_len)))

    mask = np.zeros((int(end_h), int(end_w)), dtype=bool)
    for c in conn_set:
        mask[0, int(c)] = True
        mask[1, int(c)] = True
        if end_h > 2:
            mask[2, int(c)] = True
    keep = {(0, int(c)) for c in conn_set} | {(1, int(c)) for c in conn_set}
    if end_h > 2:
        keep |= {(2, int(c)) for c in conn_set}

    target = _rng.randint(max(7, int(mask.sum())), min(22, int(end_h * end_w)))

    def allow(r: int, c: int) -> bool:
        if int(r) == 0:
            return int(c) in conn_set
        rr = int(dock_r0 + r)
        cc = int(dock_c0 + c)
        if 0 <= rr < int(chain_h) and 0 <= cc < int(chain_w):
            cur = int(chain_grid[int(rr), int(cc)])
            if cur == int(obj):
                return int(r) == 0 and int(c) in conn_set
            if cur != int(bg):
                return int(r) == 1 and int(c) in conn_set
        return True

    _grow_shape(mask, int(target), [(int(r), int(c)) for r, c in np.argwhere(mask)], allow=allow)
    if not is_contiguous(mask, background=0, connectivity=8):
        raise ValueError("end-not-contiguous")
    _thin(mask, keep=keep)
    if not is_contiguous(mask, background=0, connectivity=8):
        raise ValueError("end-not-contiguous-after-thin")

    if sorted(int(c) for c in np.where(mask[0])[0]) != list(range(int(conn_c0), int(conn_c0 + conn_len))):
        raise ValueError("end-top-row-dirty")
    if not all(bool(mask[1, int(c)]) for c in conn_set):
        raise ValueError("end-support-broken")

    for r, c in np.argwhere(mask):
        rr = int(dock_r0 + r)
        cc = int(dock_c0 + c)
        if 0 <= rr < int(chain_h) and 0 <= cc < int(chain_w):
            cur = int(chain_grid[int(rr), int(cc)])
            if cur == int(obj):
                if not (int(r) == 0 and int(c) in conn_set):
                    raise ValueError("end-overlaps-chain-body")
            elif cur != int(bg):
                if not (int(r) == 1 and int(c) in conn_set):
                    raise ValueError("end-overlaps-chain-connector")

    grid = np.full(mask.shape, int(bg), dtype=int)
    grid[mask] = int(obj)
    grid[0, int(conn_c0) : int(conn_c0 + conn_len)] = int(color_bc)
    if not is_contiguous(grid != int(bg), background=0, connectivity=8):
        raise ValueError("end-grid-not-contiguous")
    cropped, _r_shift, c_shift = _crop_non_bg(grid, bg)
    return cropped, {"row": 0, "c0": int(conn_c0 - c_shift), "len": int(conn_len), "color": int(color_bc)}


def _gen_attempt() -> np.ndarray | None:
    bg, obj = 1, 4
    palette = [2, 3, 5, 6, 7, 8, 9]
    color_ab, color_bc = _rng.sample(palette, 2)

    len_a = _rng.randint(1, 3)
    len_b_top = _rng.randint(1, 3)
    len_b_bot = _rng.randint(1, 3)
    len_c = _rng.randint(1, 3)
    if int(len_a) == int(len_b_top):
        return None
    if int(len_b_bot) == int(len_c):
        return None
    if int(len_b_top) == int(len_b_bot):
        return None

    try:
        base_grid, base_conn = _build_base(int(len_a), int(color_ab))
        middle_grid, middle_conns, mid_shift = _build_middle(
            int(len_b_top),
            int(len_b_bot),
            int(color_ab),
            int(color_bc),
            base_grid,
            base_conn,
        )

        chain_ab = np.full(
            (
                max(int(base_grid.shape[0]), int(mid_shift[0] + middle_grid.shape[0])),
                max(int(base_grid.shape[1]), int(mid_shift[1] + middle_grid.shape[1])),
            ),
            int(bg),
            dtype=int,
        )
        _stamp_shifted(chain_ab, base_grid, 0, 0, bg=bg, obj=obj)
        _stamp_shifted(chain_ab, middle_grid, int(mid_shift[0]), int(mid_shift[1]), bg=bg, obj=obj)
        chain_bc_conn = {
            "row": int(mid_shift[0] + middle_conns["bc"]["row"]),
            "c0": int(mid_shift[1] + middle_conns["bc"]["c0"]),
            "len": int(middle_conns["bc"]["len"]),
            "color": int(color_bc),
        }
        end_grid, end_conn = _build_end(int(len_c), int(color_bc), chain_ab, chain_bc_conn)
    except (AssertionError, ValueError):
        return None

    expected = _assemble_chain(
        base_grid,
        middle_grid,
        end_grid,
        bg=bg,
        obj=obj,
        base_ab=base_conn,
        middle_ab=middle_conns["ab"],
        middle_bc=middle_conns["bc"],
        end_bc=end_conn,
    )
    if expected.size == 0 or not np.any(expected != int(bg)):
        return None
    if expected.shape[0] > 13 or expected.shape[1] > 15:
        return None

    if int(np.sum(base_grid != int(bg))) < 6:
        return None
    if int(np.sum(middle_grid != int(bg))) < 8:
        return None
    if int(np.sum(end_grid != int(bg))) < 6:
        return None

    bh, bw = base_grid.shape
    mh, mw = middle_grid.shape
    eh, ew = end_grid.shape
    canvas_h_min = max(10, int(max(bh, mh, eh)) + 2)
    canvas_w_min = max(11, int(bw + mw + ew + 1))
    canvas_h = _rng.randint(int(canvas_h_min), max(int(canvas_h_min), 15))
    canvas_w = _rng.randint(int(canvas_w_min), max(int(canvas_w_min), 19))
    canvas = np.full((int(canvas_h), int(canvas_w)), int(bg), dtype=int)
    paste(canvas, base_grid, 0, 0, transparent=bg)
    try:
        mr, mc = random_free_location(canvas, middle_grid, transparent=bg, padding=1)
        paste(canvas, middle_grid, int(mr), int(mc), transparent=bg)
        er, ec = random_free_location(canvas, end_grid, transparent=bg, padding=1)
        paste(canvas, end_grid, int(er), int(ec), transparent=bg)
    except ValueError:
        return None

    if int(mr + mc) == 0 or int(er + ec) == 0:
        return None
    if np.sum(canvas == int(bg)) <= canvas.size // 2:
        return None

    try:
        actual = main(canvas)
        metrics = _variant_metrics_from_grid(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if int(metrics["ab_lengths"][0]) == int(metrics["ab_lengths"][1]):
        return None
    if int(metrics["bc_lengths"][0]) == int(metrics["bc_lengths"][1]):
        return None
    if len(metrics["base_end_shared"]) != 0:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(1200):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("generate_input: failed after 1200 attempts")
