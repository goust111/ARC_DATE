from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            bbox = _bbox_from_coords(coords)
            comps.append({"color": color, "coords": coords, "bbox": bbox})
    return comps


def _mask_from_component(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _classify(grid: np.ndarray):
    bg = _get_bg(grid)
    sprites = []
    lines = []
    for comp in _find_components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        item = {
            "color": comp["color"],
            "bbox": comp["bbox"],
            "mask": _mask_from_component(comp),
            "h": hh,
            "w": ww,
        }
        if hh > 1 and ww > 1:
            sprites.append(item)
        elif hh == 1 or ww == 1:
            lines.append(item)
    return bg, sprites, lines


def _edge_pattern(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask[:, 0].copy()
    if side == "right":
        return mask[:, -1].copy()
    if side == "top":
        return mask[0, :].copy()
    if side == "bottom":
        return mask[-1, :].copy()
    raise ValueError(side)


def _line_pattern_for_side(sprite: dict, line: dict, side: str) -> tuple[np.ndarray, int] | None:
    r0, c0, r1, c1 = sprite["bbox"]
    lr0, lc0, lr1, lc1 = line["bbox"]
    if side in {"left", "right"}:
        if lc0 != lc1:
            return None
        if lr0 < r0 or lr1 > r1:
            return None
        vec = np.zeros(sprite["h"], dtype=bool)
        vec[lr0 - r0 : lr1 - r0 + 1] = True
        d = c0 - lc0 if side == "left" else lc0 - c1
        return vec, int(d)
    if side in {"top", "bottom"}:
        if lr0 != lr1:
            return None
        if lc0 < c0 or lc1 > c1:
            return None
        vec = np.zeros(sprite["w"], dtype=bool)
        vec[lc0 - c0 : lc1 - c0 + 1] = True
        d = r0 - lr0 if side == "top" else lr0 - r1
        return vec, int(d)
    raise ValueError(side)


def _paste_mask(dst: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    block = dst[r0 : r0 + h, c0 : c0 + w]
    block[mask] = int(color)


def _valid_operations(grid: np.ndarray) -> list[dict]:
    _bg, sprites, lines = _classify(grid)
    ops: list[dict] = []
    for sprite_idx, sprite in enumerate(sprites):
        for line_idx, line in enumerate(lines):
            if sprite["color"] != line["color"]:
                continue
            for side in ("left", "right", "top", "bottom"):
                pair = _line_pattern_for_side(sprite, line, side)
                if pair is None:
                    continue
                pattern, d = pair
                if d <= 0 or not np.array_equal(pattern, _edge_pattern(sprite["mask"], side)):
                    continue
                step = sprite["w"] if side in {"left", "right"} else sprite["h"]
                if d % step != 0:
                    continue
                ops.append(
                    {
                        "sprite_idx": sprite_idx,
                        "line_idx": line_idx,
                        "side": side,
                        "copies": d // step,
                    }
                )
    return ops


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue unfinished sprite tiling chains by recognizing phase-aligned frontier relic lines.

    Input structure:
    - The grid has a black background, several monochromatic sprites, and several monochromatic 1-pixel-wide line relics.
    - Each sprite occupies a small rectangular bounding box with both height and width greater than 1, but its interior pattern may be partial.
    - Some relic lines share a sprite's row band or column band and have the same color; others are distractors.

    Rule:
    - For each sprite, test same-colored relics on its left, right, top, and bottom bands.
    - A relic is a candidate only when its 1D pattern exactly matches the sprite edge that faces it.
    - Let the sprite size along that direction be s, and let d be the distance from the sprite edge to the relic. Only when d is a positive multiple of s do we treat the relic as the frontier of an unfinished seamless tiling chain.
    - For every valid relic, stamp full copies of the sprite at offsets s, 2s, ..., d so the last stamped copy completes the relic itself. All invalid relics and unrelated objects stay unchanged.

    Key insight:
    - The short line is not an independent object: it is the visible leading edge of a periodic sprite chain, but it only counts when both the edge shape and the tiling phase agree.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    _bg, sprites, _lines = _classify(g)
    for op in _valid_operations(g):
        sprite = sprites[op["sprite_idx"]]
        r0, c0, _r1, _c1 = sprite["bbox"]
        h, w = sprite["h"], sprite["w"]
        dr = dc = 0
        if op["side"] == "left":
            dc = -w
        elif op["side"] == "right":
            dc = w
        elif op["side"] == "top":
            dr = -h
        else:
            dr = h
        for step_idx in range(1, op["copies"] + 1):
            rr = r0 + dr * step_idx
            cc = c0 + dc * step_idx
            _paste_mask(out, sprite["mask"], sprite["color"], rr, cc)
    return out


def _segment_length(total: int) -> int:
    if total <= 3:
        return random.randint(1, total - 1)
    options = list(range(1, total))
    weights = [3 if 1 < x < total - 1 else 1 for x in options]
    return random.choices(options, weights=weights, k=1)[0]


def _grow_interior(mask: np.ndarray, protected_side: str) -> None:
    h, w = mask.shape
    extra = random.randint(0, max(1, (h * w) // 6))
    for _ in range(extra):
        candidates = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if protected_side == "left" and c == 0:
                    continue
                if protected_side == "right" and c == w - 1:
                    continue
                if protected_side == "top" and r == 0:
                    continue
                if protected_side == "bottom" and r == h - 1:
                    continue
                if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                    candidates.append((r, c))
        if not candidates:
            return
        rr, cc = random.choice(candidates)
        mask[rr, cc] = True


def _make_sprite(side: str) -> np.ndarray:
    if side in {"left", "right"}:
        h = random.randint(3, 6)
        w = random.randint(3, 5)
        start = random.randint(0, h - 1)
        seg_len = _segment_length(h)
        if start + seg_len > h:
            start = h - seg_len
        trunk_c = random.randint(1, w - 2)
        mask = np.zeros((h, w), dtype=bool)
        mask[:, trunk_c] = True
        if side == "right":
            mask[start : start + seg_len, trunk_c:w] = True
            row_left = random.randint(0, h - 1)
            mask[row_left, : trunk_c + 1] = True
        else:
            mask[start : start + seg_len, : trunk_c + 1] = True
            row_right = random.randint(0, h - 1)
            mask[row_right, trunk_c:w] = True
    else:
        h = random.randint(3, 5)
        w = random.randint(3, 6)
        start = random.randint(0, w - 1)
        seg_len = _segment_length(w)
        if start + seg_len > w:
            start = w - seg_len
        trunk_r = random.randint(1, h - 2)
        mask = np.zeros((h, w), dtype=bool)
        mask[trunk_r, :] = True
        if side == "bottom":
            mask[trunk_r:h, start : start + seg_len] = True
            col_top = random.randint(0, w - 1)
            mask[: trunk_r + 1, col_top] = True
        else:
            mask[: trunk_r + 1, start : start + seg_len] = True
            col_bottom = random.randint(0, w - 1)
            mask[trunk_r:h, col_bottom] = True
    _grow_interior(mask, side)
    return mask


def _interval_from_vector(vec: np.ndarray) -> tuple[int, int]:
    idx = np.flatnonzero(vec)
    assert idx.size > 0
    assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1))
    return int(idx[0]), int(idx.size)


def _different_interval(total: int, start: int, seg_len: int) -> tuple[int, int]:
    options = []
    for new_len in range(1, total + 1):
        for new_start in range(0, total - new_len + 1):
            if new_start == start and new_len == seg_len:
                continue
            options.append((new_start, new_len))
    return random.choice(options)


def _vector_from_interval(total: int, start: int, seg_len: int) -> np.ndarray:
    vec = np.zeros(total, dtype=bool)
    vec[start : start + seg_len] = True
    return vec


def _render_group(role: str, axis: str | None, color: int) -> dict:
    if axis == "h":
        side = random.choice(["left", "right"])
    elif axis == "v":
        side = random.choice(["top", "bottom"])
    else:
        side = random.choice(["left", "right", "top", "bottom"])
    mask = _make_sprite(side)
    h, w = mask.shape
    edge_vec = _edge_pattern(mask, side)
    edge_start, edge_len = _interval_from_vector(edge_vec)
    step = w if side in {"left", "right"} else h
    if role == "legal":
        d = step * random.randint(1, 2)
        line_vec = edge_vec
    elif role == "phase_bad":
        bad_ds = [x for x in range(1, step * 2 + 1) if x % step != 0]
        d = random.choice(bad_ds)
        line_vec = edge_vec
    elif role == "shape_bad":
        d = step * random.randint(1, 2)
        other_start, other_len = _different_interval(h if side in {"left", "right"} else w, edge_start, edge_len)
        line_vec = _vector_from_interval(h if side in {"left", "right"} else w, other_start, other_len)
    else:
        raise ValueError(role)

    if side == "right":
        patch_h, patch_w = h + 2, w + d + 2
        sr, sc = 1, 1
        line_r0 = sr + int(np.flatnonzero(line_vec)[0])
        line_r1 = sr + int(np.flatnonzero(line_vec)[-1])
        line_c0 = sc + (w - 1) + d
        line_c1 = line_c0
    elif side == "left":
        patch_h, patch_w = h + 2, w + d + 2
        sr, sc = 1, 1 + d
        line_r0 = sr + int(np.flatnonzero(line_vec)[0])
        line_r1 = sr + int(np.flatnonzero(line_vec)[-1])
        line_c0 = 1
        line_c1 = 1
    elif side == "bottom":
        patch_h, patch_w = h + d + 2, w + 2
        sr, sc = 1, 1
        line_c0 = sc + int(np.flatnonzero(line_vec)[0])
        line_c1 = sc + int(np.flatnonzero(line_vec)[-1])
        line_r0 = sr + (h - 1) + d
        line_r1 = line_r0
    else:
        patch_h, patch_w = h + d + 2, w + 2
        sr, sc = 1 + d, 1
        line_c0 = sc + int(np.flatnonzero(line_vec)[0])
        line_c1 = sc + int(np.flatnonzero(line_vec)[-1])
        line_r0 = 1
        line_r1 = 1

    inp = np.zeros((patch_h, patch_w), dtype=int)
    out = np.zeros((patch_h, patch_w), dtype=int)
    _paste_mask(inp, mask, color, sr, sc)
    _paste_mask(out, mask, color, sr, sc)
    inp[line_r0 : line_r1 + 1, line_c0 : line_c1 + 1] = color
    out[line_r0 : line_r1 + 1, line_c0 : line_c1 + 1] = color
    if role == "legal":
        dr = dc = 0
        if side == "left":
            dc = -w
        elif side == "right":
            dc = w
        elif side == "top":
            dr = -h
        else:
            dr = h
        copies = d // step
        for step_idx in range(1, copies + 1):
            _paste_mask(out, mask, color, sr + dr * step_idx, sc + dc * step_idx)
    return {
        "role": role,
        "axis": axis,
        "side": side,
        "input": inp,
        "output": out,
    }


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_patches(shapes: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(shapes)), key=lambda idx: shapes[idx][0] * shapes[idx][1], reverse=True)
    placed: dict[int, tuple[int, int]] = {}
    rects: list[tuple[int, int, int, int]] = []
    for idx in order:
        ph, pw = shapes[idx]
        candidates = [
            (r, c)
            for r in range(0, h - ph + 1)
            for c in range(0, w - pw + 1)
        ]
        random.shuffle(candidates)
        found = None
        for r, c in candidates:
            rect = (r, c, r + ph - 1, c + pw - 1)
            if any(_rectangles_overlap(rect, old, pad=1) for old in rects):
                continue
            found = (r, c)
            rects.append(rect)
            placed[idx] = found
            break
        if found is None:
            return None
    return [placed[i] for i in range(len(shapes))]


def _shape_match_candidates(grid: np.ndarray) -> list[dict]:
    _bg, sprites, lines = _classify(grid)
    cand = []
    for sprite_idx, sprite in enumerate(sprites):
        for line_idx, line in enumerate(lines):
            if sprite["color"] != line["color"]:
                continue
            for side in ("left", "right", "top", "bottom"):
                pair = _line_pattern_for_side(sprite, line, side)
                if pair is None:
                    continue
                pattern, d = pair
                if d <= 0 or not np.array_equal(pattern, _edge_pattern(sprite["mask"], side)):
                    continue
                step = sprite["w"] if side in {"left", "right"} else sprite["h"]
                cand.append(
                    {
                        "sprite_idx": sprite_idx,
                        "line_idx": line_idx,
                        "side": side,
                        "divisible": d % step == 0,
                    }
                )
    return cand


def _build_scene() -> tuple[np.ndarray, np.ndarray]:
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)
    groups = [
        _render_group("legal", "h", colors[0]),
        _render_group("legal", "v", colors[1]),
        _render_group("phase_bad", None, colors[2]),
        _render_group("shape_bad", None, colors[3]),
    ]
    shapes = [group["output"].shape for group in groups]
    for _ in range(80):
        h = random.randint(24, 30)
        w = random.randint(24, 30)
        placements = _place_patches(shapes, h, w)
        if placements is None:
            continue
        inp = np.zeros((h, w), dtype=int)
        out = np.zeros((h, w), dtype=int)
        for (r, c), group in zip(placements, groups):
            ph, pw = group["input"].shape
            inp[r : r + ph, c : c + pw] = group["input"]
            out[r : r + ph, c : c + pw] = group["output"]
        return inp, out
    raise RuntimeError("failed to place scene patches")


def _validated_scene() -> np.ndarray | None:
    inp, expected = _build_scene()
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    valid_ops = _valid_operations(inp)
    if len(valid_ops) != 2:
        return None
    axes = {("h" if op["side"] in {"left", "right"} else "v") for op in valid_ops}
    if axes != {"h", "v"}:
        return None
    by_line = defaultdict(list)
    for op in valid_ops:
        by_line[op["line_idx"]].append(op)
    if any(len(v) != 1 for v in by_line.values()):
        return None
    shape_matches = _shape_match_candidates(inp)
    if sum(1 for cand in shape_matches if not cand["divisible"]) < 1:
        return None
    if np.array_equal(inp, actual):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _validated_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
