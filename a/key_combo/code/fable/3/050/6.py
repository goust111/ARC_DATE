from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 800
SHAPE_ATTEMPTS = 500

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

_GENERATED_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[Coord] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    assert coords.size > 0, "empty mask"
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([r0, c0])
    out[local[:, 0], local[:, 1]] = True
    return out


def _mask_from_cells(cells: set[Coord]) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    return _mask_from_coords(coords)


def _mask_sig(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        r = _trim(np.rot90(mask, k))
        sig = _mask_sig(r)
        if sig not in seen:
            seen.add(sig)
            out.append(r)
    return out


def _canonical_rotation_sig(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_sig(r) for r in _rotations(mask))


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1 and int(comps[0].shape[0]) == int(np.count_nonzero(mask))


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _components(grid == int(color)):
            box = _bbox_coords(coords)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": box,
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _find_bars(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    min_width = max(9, int(round(w * 0.58)))
    bars: list[dict] = []
    for comp in _color_components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bh < 4 or bw < min_width or r1 < h // 3:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        color = int(comp["color"])
        if not np.all((sub == color) | (sub == int(bg))):
            continue
        rows = np.count_nonzero(sub == color, axis=1)
        if np.min(rows) < max(4, int(round(bw * 0.45))):
            continue
        density = float(comp["area"]) / float(bh * bw)
        if density < 0.52:
            continue
        bars.append(comp)

    bars.sort(key=lambda b: (b["bbox"][0], b["bbox"][1]))
    assert bars, "expected horizontal thick bars"
    return bars


def _holes_for_bar(grid: np.ndarray, bar: dict, bg: int) -> list[dict]:
    r0, c0, r1, c1 = bar["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[dict] = []
    for local in _components(sub == int(bg)):
        lr0, lc0, lr1, lc1 = _bbox_coords(local)
        if lr0 == 0 or lc0 == 0 or lr1 == sub.shape[0] - 1 or lc1 == sub.shape[1] - 1:
            continue
        if local.shape[0] < 4:
            continue
        coords = local + np.array([r0, c0])
        holes.append(
            {
                "bar_color": int(bar["color"]),
                "coords": coords,
                "bbox": _bbox_coords(coords),
                "mask": _mask_from_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return holes


def _extract_sprites(grid: np.ndarray, bg: int, bars: list[dict]) -> list[dict]:
    top_limit = min(int(bar["bbox"][0]) for bar in bars)
    sprites: list[dict] = []
    for comp in _color_components(grid, bg):
        r0, _c0, r1, _c1 = comp["bbox"]
        if r1 >= top_limit or comp["area"] < 3:
            continue
        mask = _mask_from_coords(comp["coords"])
        sprites.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "mask": mask,
                "area": int(comp["area"]),
            }
        )
    sprites.sort(key=lambda s: (s["bbox"][0], s["bbox"][1], s["bbox"][2], s["bbox"][3]))
    return sprites


def _placements(piece: np.ndarray, target: np.ndarray) -> list[dict]:
    target = np.asarray(target, dtype=bool)
    th, tw = target.shape
    out: list[dict] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for rot in _rotations(piece):
        ph, pw = rot.shape
        if ph > th or pw > tw:
            continue
        rs, cs = np.nonzero(rot)
        for r0 in range(th - ph + 1):
            for c0 in range(tw - pw + 1):
                canvas = np.zeros_like(target, dtype=bool)
                canvas[r0 + rs, c0 + cs] = True
                if np.any(canvas & ~target):
                    continue
                sig = _mask_sig(canvas)
                if sig in seen:
                    continue
                seen.add(sig)
                out.append({"canvas": canvas, "piece": rot, "r0": int(r0), "c0": int(c0)})
    return out


def _tile_solutions(target: np.ndarray, a: np.ndarray, b: np.ndarray, limit: int | None = None) -> list[dict]:
    target = np.asarray(target, dtype=bool)
    placements_a = _placements(a, target)
    placements_b = _placements(b, target)
    sols: list[dict] = []
    b_by_sig = {_mask_sig(p["canvas"]): p for p in placements_b}
    for pa in placements_a:
        ca = pa["canvas"]
        rest = target & ~ca
        if int(np.count_nonzero(rest)) != int(np.count_nonzero(b)):
            continue
        pb = b_by_sig.get(_mask_sig(rest))
        if pb is None:
            continue
        assert not np.any(ca & pb["canvas"])
        sols.append({"a": pa, "b": pb})
        if limit is not None and len(sols) >= int(limit):
            return sols
    return sols


def _analyze(grid: np.ndarray) -> tuple[int, list[dict], list[dict], list[dict], list[tuple[int, int, dict]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background(g)
    bars = _find_bars(g, bg)

    holes: list[dict] = []
    for bar in bars:
        local_holes = _holes_for_bar(g, bar, bg)
        assert len(local_holes) == 1, "each thick bar must contain exactly one internal hole"
        holes.extend(local_holes)

    sprites = _extract_sprites(g, bg, bars)
    assert len(sprites) == 2 * len(holes), "expected exactly two sprites per hole"

    matches: list[tuple[int, int, dict]] = []
    for hole in holes:
        sols: list[tuple[int, int, dict]] = []
        target = hole["mask"]
        for i, j in combinations(range(len(sprites)), 2):
            if sprites[i]["area"] + sprites[j]["area"] != hole["area"]:
                continue
            tilings = _tile_solutions(target, sprites[i]["mask"], sprites[j]["mask"], limit=2)
            for tiling in tilings:
                sols.append((int(i), int(j), tiling))
                if len(sols) > 1:
                    break
            if len(sols) > 1:
                break
        assert len(sols) == 1, "each hole must have one unique two-sprite rotated tiling"
        matches.append(sols[0])

    used = {idx for i, j, _tiling in matches for idx in (i, j)}
    assert used == set(range(len(sprites))), "all sprites must be used exactly once"
    return int(bg), bars, holes, sprites, matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each notched bar hole by the unique pair of rotated sprites, recolored to the bar color.

    Input structure:
    - The dominant background surrounds an upper sprite field and a lower stack
      of wide horizontal one-color thick bars.
    - Each thick bar contains exactly one internal background-colored irregular
      hole. The upper field contains exactly twice as many solid monochrome
      sprites as there are holes, with sprite colors acting only as decoys.

    Rule:
    - For every hole, find the unique pair of sprites whose shapes, after
      independent 0/90/180/270 degree rotations and translations within the
      hole bbox, tile that hole with no overlap and no uncovered cell.
    - Erase all upper sprites.
    - Paint every cell of each uniquely tiled hole with its own bar's color, so
      the inserted two pieces become a single solid continuation of the bar.

    Key insight:
    - Matching is a two-piece rotated tiling problem. Sprite colors do not select
      bars; after docking, both pieces are recolored to the destination bar color.
    """
    g = np.asarray(grid, dtype=int)
    bg, _bars, holes, sprites, _matches = _analyze(g)
    out = g.copy()

    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for hole in holes:
        coords = hole["coords"]
        out[coords[:, 0], coords[:, 1]] = int(hole["bar_color"])

    return out


def _random_connected_mask(area: int) -> np.ndarray | None:
    area = int(area)
    for _ in range(120):
        cells: set[Coord] = {(0, 0)}
        while len(cells) < area:
            frontier: list[Coord] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))

        mask = _trim(_mask_from_cells(cells))
        h, w = mask.shape
        if h < 2 or w < 2 or h > 4 or w > 4:
            continue
        if int(np.count_nonzero(mask)) != area or not _is_connected(mask):
            continue
        if int(np.count_nonzero(mask)) == h * w:
            continue
        return mask
    return None


def _adjacent(a: np.ndarray, b: np.ndarray) -> bool:
    ar, ac = np.nonzero(a)
    bset = set(zip(*np.nonzero(b)))
    for r, c in zip(ar, ac):
        for nr, nc in _neighbors4(int(r), int(c)):
            if (nr, nc) in bset:
                return True
    return False


def _compose_hole(piece_a: np.ndarray, piece_b: np.ndarray) -> np.ndarray | None:
    for _ in range(250):
        a = random.choice(_rotations(piece_a))
        b = random.choice(_rotations(piece_b))
        ah, aw = a.shape
        bh, bw = b.shape
        th_min = max(ah, bh, 3)
        tw_min = max(aw, bw, 4)
        if th_min > 4 or tw_min > 8:
            continue
        th = random.randint(th_min, 4)
        tw = random.randint(tw_min, 8)
        ar = random.randint(0, th - ah)
        ac = random.randint(0, tw - aw)
        br = random.randint(0, th - bh)
        bc = random.randint(0, tw - bw)

        ca = np.zeros((th, tw), dtype=bool)
        cb = np.zeros((th, tw), dtype=bool)
        ars, acs = np.nonzero(a)
        brs, bcs = np.nonzero(b)
        ca[ar + ars, ac + acs] = True
        cb[br + brs, bc + bcs] = True
        if np.any(ca & cb) or not _adjacent(ca, cb):
            continue
        union = _trim(ca | cb)
        if union.shape[0] > 4 or union.shape[1] > 8:
            continue
        if int(np.count_nonzero(union)) == union.size or not _is_connected(union):
            continue
        if len(_tile_solutions(union, piece_a, piece_b, limit=2)) != 1:
            continue
        return union
    return None


def _unique_shape_set(count: int) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    total_area = random.choice((10, 11, 12))
    pieces: list[np.ndarray] = []
    holes: list[np.ndarray] = []
    used_piece_sigs: set[tuple[tuple[int, int], bytes]] = set()
    used_hole_sigs: set[tuple[tuple[int, int], bytes]] = set()

    for _hole_index in range(count):
        made = False
        for _ in range(SHAPE_ATTEMPTS):
            a_area = random.randint(4, 7)
            b_area = total_area - a_area
            if b_area < 4 or b_area > 7:
                continue
            a = _random_connected_mask(a_area)
            b = _random_connected_mask(b_area)
            if a is None or b is None:
                continue
            siga = _canonical_rotation_sig(a)
            sigb = _canonical_rotation_sig(b)
            if siga == sigb or siga in used_piece_sigs or sigb in used_piece_sigs:
                continue
            hole = _compose_hole(a, b)
            if hole is None:
                continue
            sigh = _canonical_rotation_sig(hole)
            if sigh in used_hole_sigs:
                continue

            test_pieces = pieces + [a, b]
            test_holes = holes + [hole]
            if not _shapes_are_semantically_unique(test_holes, test_pieces):
                continue

            pieces.extend([a, b])
            holes.append(hole)
            used_piece_sigs.update((siga, sigb))
            used_hole_sigs.add(sigh)
            made = True
            break
        if not made:
            return None
    return holes, pieces


def _shapes_are_semantically_unique(holes: list[np.ndarray], pieces: list[np.ndarray]) -> bool:
    if len(pieces) != 2 * len(holes):
        return False
    used: set[int] = set()
    for hole in holes:
        sols: list[tuple[int, int]] = []
        h_area = int(np.count_nonzero(hole))
        for i, j in combinations(range(len(pieces)), 2):
            if int(np.count_nonzero(pieces[i])) + int(np.count_nonzero(pieces[j])) != h_area:
                continue
            if len(_tile_solutions(hole, pieces[i], pieces[j], limit=1)) == 1:
                sols.append((i, j))
                if len(sols) > 1:
                    return False
        if len(sols) != 1:
            return False
        used.update(sols[0])
    return used == set(range(len(pieces)))


def _random_free_location(canvas: np.ndarray, mask: np.ndarray, padding: int = 1) -> Coord | None:
    h, w = canvas.shape
    ph, pw = mask.shape
    candidates: list[Coord] = []
    for r0 in range(0, h - ph + 1):
        for c0 in range(0, w - pw + 1):
            r_lo = max(0, r0 - padding)
            c_lo = max(0, c0 - padding)
            r_hi = min(h, r0 + ph + padding)
            c_hi = min(w, c0 + pw + padding)
            if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != BG):
                continue
            candidates.append((int(r0), int(c0)))
    if not candidates:
        return None
    return random.choice(candidates)


def _paint_mask(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    rr, cc = np.nonzero(mask)
    grid[int(r0) + rr, int(c0) + cc] = int(color)


def _make_attempt() -> tuple[np.ndarray, dict] | None:
    bar_count = random.choice((2, 2, 3))
    shapes = _unique_shape_set(bar_count)
    if shapes is None:
        return None
    hole_masks, piece_masks = shapes

    max_hole_w = max(mask.shape[1] for mask in hole_masks)
    width = random.randint(max(22, max_hole_w + 9), 30)
    bar_colors = random.sample(COLORS, bar_count)
    bar_heights = [max(mask.shape[0] + 2, random.randint(5, 6)) for mask in hole_masks]
    top_h = random.randint(9, 12 if bar_count == 2 else 10)
    height = top_h + sum(bar_heights)
    if height > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    bars_meta: list[dict] = []
    holes_meta: list[dict] = []
    r = top_h
    body_c0, body_c1 = 1, width - 2
    for i, (bar_h, hole, color) in enumerate(zip(bar_heights, hole_masks, bar_colors)):
        r0, r1 = r, r + bar_h - 1
        grid[r0 : r1 + 1, body_c0 : body_c1 + 1] = int(color)
        hh, hw = hole.shape
        if body_c1 - body_c0 + 1 < hw + 4:
            return None
        hr = random.randint(r0 + 1, r1 - hh)
        hc = random.randint(body_c0 + 2, body_c1 - hw - 1)
        rr, cc = np.nonzero(hole)
        grid[hr + rr, hc + cc] = BG
        bars_meta.append({"r0": r0, "r1": r1, "c0": body_c0, "c1": body_c1, "color": int(color)})
        holes_meta.append({"bar_index": i, "mask": hole, "r0": hr, "c0": hc})
        r = r1 + 1

    sprite_count = len(piece_masks)
    sprite_colors = [None] * sprite_count
    decoy_index = random.randrange(sprite_count)
    decoy_target = decoy_index // 2
    wrong_colors = [bar_colors[i] for i in range(bar_count) if i != decoy_target]
    decoy_color = random.choice(wrong_colors)
    sprite_colors[decoy_index] = int(decoy_color)
    remaining = [c for c in COLORS if c != int(decoy_color)]
    for idx, color in zip([i for i in range(sprite_count) if sprite_colors[i] is None], random.sample(remaining, sprite_count - 1)):
        sprite_colors[idx] = int(color)

    order = list(range(sprite_count))
    random.shuffle(order)
    sprite_area = grid[: top_h - 1, :]
    sprites_meta: list[dict] = []
    for idx in order:
        display = random.choice(_rotations(piece_masks[idx]))
        loc = _random_free_location(sprite_area, display, padding=1)
        if loc is None:
            return None
        sr, sc = loc
        _paint_mask(sprite_area, sr, sc, display, int(sprite_colors[idx]))
        sprites_meta.append(
            {
                "piece_index": int(idx),
                "target_bar": int(idx // 2),
                "color": int(sprite_colors[idx]),
                "mask": display,
                "r0": int(sr),
                "c0": int(sc),
            }
        )

    meta = {"bars": bars_meta, "holes": holes_meta, "sprites": sprites_meta, "top_h": int(top_h)}
    return grid, meta


def _semantic_checks(inp: np.ndarray, meta: dict) -> bool:
    try:
        out = main(inp)
        bg, bars, holes, sprites, matches = _analyze(inp)
    except Exception:
        return False

    if inp.ndim != 2 or out.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if not np.array_equal(out[: min(bar["bbox"][0] for bar in bars), :], np.full_like(out[: min(bar["bbox"][0] for bar in bars), :], bg)):
        return False

    for bar in meta["bars"]:
        r0, r1, c0, c1 = bar["r0"], bar["r1"], bar["c0"], bar["c1"]
        if not np.all(out[r0 : r1 + 1, c0 : c1 + 1] == int(bar["color"])):
            return False

    if len(holes) < 2 or len(sprites) != 2 * len(holes) or len(matches) != len(holes):
        return False

    # Every generated example uses equal-area holes, so area-only pairing cannot
    # identify the matching sprite pair.
    if len({int(hole["area"]) for hole in holes}) != 1:
        return False

    decoys = [
        spr
        for spr in meta["sprites"]
        if any(int(spr["color"]) == int(bar["color"]) and int(spr["target_bar"]) != i for i, bar in enumerate(meta["bars"]))
    ]
    if not decoys:
        return False

    sig = (tuple(map(int, out.shape)), out.tobytes())
    if sig in _GENERATED_OUTPUT_SIGNATURES:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        attempt = _make_attempt()
        if attempt is None:
            continue
        grid, meta = attempt
        if not _semantic_checks(grid, meta):
            continue
        out = main(grid)
        _GENERATED_OUTPUT_SIGNATURES.add((tuple(map(int, out.shape)), out.tobytes()))
        return grid
    raise RuntimeError("failed to generate a valid two-piece keyhole input")
