from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
FG_PALETTE = [1, 2, 3, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _contiguous(vals: list[int]) -> bool:
    return bool(vals) and vals == list(range(vals[0], vals[-1] + 1))


def _detect_axis(grid: np.ndarray, coords: np.ndarray) -> tuple[str, int, tuple[int, int]]:
    yellow = [(int(r), int(c)) for r, c in coords if int(grid[r, c]) == YELLOW]
    assert yellow, "expected yellow pixels"

    candidates: list[tuple[str, int, int]] = []
    row_to_cols: dict[int, list[int]] = {}
    col_to_rows: dict[int, list[int]] = {}
    for r, c in yellow:
        row_to_cols.setdefault(r, []).append(c)
        col_to_rows.setdefault(c, []).append(r)

    for row, cols in row_to_cols.items():
        cols = sorted(cols)
        if len(cols) >= 2 and _contiguous(cols):
            candidates.append(("h", int(row), len(cols)))
    for col, rows in col_to_rows.items():
        rows = sorted(rows)
        if len(rows) >= 2 and _contiguous(rows):
            candidates.append(("v", int(col), len(rows)))

    assert candidates, "expected one contiguous yellow axis"
    best_len = max(length for _orient, _axis, length in candidates)
    best = [(orient, axis) for orient, axis, length in candidates if length == best_len]
    assert len(best) == 1, "axis must be unique"
    orient, axis = best[0]

    if orient == "v":
        marker = [(r, c) for r, c in yellow if c != axis]
    else:
        marker = [(r, c) for r, c in yellow if r != axis]
    assert len(marker) == 1, "expected exactly one off-axis control marker"
    return orient, int(axis), (int(marker[0][0]), int(marker[0][1]))


def _extract_objects(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    objs = []
    for coords in _components_mask(grid != bg):
        if not np.any(grid[coords[:, 0], coords[:, 1]] == YELLOW):
            continue
        bbox = _bbox_coords(coords)
        orient, axis, marker = _detect_axis(grid, coords)
        colors = set(map(int, grid[coords[:, 0], coords[:, 1]])) - {bg, YELLOW}
        assert len(colors) == 1, "each object must have one non-yellow foreground color"
        obj = {
            "coords": coords,
            "bbox": bbox,
            "orient": orient,
            "axis": axis,
            "marker": marker,
            "color": next(iter(colors)),
        }
        objs.append(obj)

    assert objs, "expected at least one marked object"
    for obj in objs:
        r0, c0, r1, c1 = obj["bbox"]
        depth = 0
        for other in objs:
            if other is obj:
                continue
            rr0, cc0, rr1, cc1 = other["bbox"]
            if rr0 < r0 and cc0 < c0 and rr1 > r1 and cc1 > c1:
                depth += 1
        obj["depth"] = depth
    return sorted(objs, key=lambda d: (d["depth"], d["bbox"][0], d["bbox"][1]))


def _write_reflection(
    out: np.ndarray,
    src: np.ndarray,
    obj: dict,
    *,
    always_marker_is_source: bool = False,
) -> None:
    r0, c0, r1, c1 = obj["bbox"]
    orient = obj["orient"]
    axis = obj["axis"]
    mr, mc = obj["marker"]

    if orient == "v":
        half = axis - c0
        assert c1 - axis == half and half > 0
        marker_side = -1 if mc < axis else 1
        source_side = marker_side if (always_marker_is_source or obj["depth"] % 2 == 0) else -marker_side
        for r in range(r0, r1 + 1):
            for offset in range(half):
                src_c = axis + source_side * (offset + 1)
                dst_c = axis - source_side * (offset + 1)
                if src[r, src_c] == YELLOW or src[r, dst_c] == YELLOW:
                    continue
                out[r, dst_c] = src[r, src_c]
    else:
        half = axis - r0
        assert r1 - axis == half and half > 0
        marker_side = -1 if mr < axis else 1
        source_side = marker_side if (always_marker_is_source or obj["depth"] % 2 == 0) else -marker_side
        for c in range(c0, c1 + 1):
            for offset in range(half):
                src_r = axis + source_side * (offset + 1)
                dst_r = axis - source_side * (offset + 1)
                if src[src_r, c] == YELLOW or src[dst_r, c] == YELLOW:
                    continue
                out[dst_r, c] = src[src_r, c]


def _solve(
    grid: np.ndarray,
    *,
    always_marker_is_source: bool = False,
    reverse_order: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    objs = _extract_objects(g)
    out = g.copy()
    seq = list(reversed(objs)) if reverse_order else objs
    for obj in seq:
        _write_reflection(out, g, obj, always_marker_is_source=always_marker_is_source)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each nested marked object by mirroring the side implied by depth-parity-reversed control semantics.

    Input structure:
    - The background is black.
    - The grid contains 2 or 3 separated multicolor components; each valid object has one non-yellow foreground color.
    - Every object contains one full yellow horizontal or vertical symmetry axis and exactly one off-axis yellow control marker.
    - Some object bounding boxes are strictly nested inside others without touching them.

    Rule:
    - Detect each yellow-marked object and compute its nesting depth from strict bbox containment.
    - For even depths, the marker side is the source side; for odd depths, the marker side is the target side, so the opposite half is the source.
    - Mirror the entire rectangular source half across the yellow axis onto the other half, copying both foreground and background.
    - Keep all yellow axis and marker pixels unchanged, and process objects from outermost depth to innermost depth so deeper objects overwrite shallower reflected pixels.

    Key insight:
    - The yellow marker does not have a fixed meaning: nesting depth parity flips whether the marked side is trusted or must be overwritten.
    """
    return _solve(grid)


def _shortest_path(allowed: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = allowed.shape
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and (nr, nc) not in prev:
                prev[(nr, nc)] = (r, c)
                q.append((nr, nc))
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def _frontier(mask: np.ndarray, allowed: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = set()
    for r, c in np.argwhere(mask):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not mask[nr, nc]:
                out.add((int(nr), int(nc)))
    return list(out)


def _random_connected_mask(
    shape: tuple[int, int],
    axis_mode: str,
    *,
    blocked: np.ndarray | None = None,
    fill_lo: float = 0.35,
    fill_hi: float = 0.75,
) -> np.ndarray | None:
    h, w = shape
    if blocked is None:
        blocked = np.zeros((h, w), dtype=bool)
    allowed = ~blocked
    total = int(allowed.sum())
    if total < 5:
        return None

    for _ in range(80):
        if axis_mode == "col0":
            near = [(r, 0) for r in range(h) if allowed[r, 0]]
            far = [(r, w - 1) for r in range(h) if allowed[r, w - 1]]
            top = [(0, c) for c in range(w) if allowed[0, c]]
            bottom = [(h - 1, c) for c in range(w) if allowed[h - 1, c]]
        else:
            near = [(0, c) for c in range(w) if allowed[0, c]]
            far = [(h - 1, c) for c in range(w) if allowed[h - 1, c]]
            top = [(r, 0) for r in range(h) if allowed[r, 0]]
            bottom = [(r, w - 1) for r in range(h) if allowed[r, w - 1]]
        if not near or not far or not top or not bottom:
            continue

        anchors = [random.choice(near), random.choice(far), random.choice(top), random.choice(bottom)]
        uniq: list[tuple[int, int]] = []
        for pt in anchors:
            if pt not in uniq:
                uniq.append(pt)
        mask = np.zeros((h, w), dtype=bool)
        ok = True
        for a, b in zip(uniq, uniq[1:]):
            path = _shortest_path(allowed, a, b)
            if path is None:
                ok = False
                break
            for rr, cc in path:
                mask[rr, cc] = True
        if not ok or not mask.any():
            continue

        lo = max(int(mask.sum()), int(np.ceil(fill_lo * total)))
        hi = max(lo, int(np.floor(fill_hi * total)))
        target = random.randint(lo, hi)
        while int(mask.sum()) < target:
            cand = _frontier(mask, allowed)
            if not cand:
                break
            rr, cc = random.choice(cand)
            mask[rr, cc] = True

        rows, cols = np.where(mask)
        if axis_mode == "col0":
            if 0 not in cols or (w - 1) not in cols:
                continue
        else:
            if 0 not in rows or (h - 1) not in rows:
                continue
        if len(set(rows.tolist())) < min(3, h) or len(set(cols.tolist())) < min(3, w):
            continue
        return mask
    return None


def _object_dims(depth: int) -> dict:
    orient = random.choice(["v", "h"])
    if depth == 1:
        mirror = random.randint(2, 3)
        cross = random.randint(5, 7)
    else:
        mirror = random.randint(3, 5)
        cross = random.randint(7, 10)
    if orient == "v":
        h = cross
        w = 2 * mirror + 1
    else:
        h = 2 * mirror + 1
        w = cross
    return {"orient": orient, "h": h, "w": w, "half": mirror}


def _half_shape(spec: dict) -> tuple[int, int]:
    if spec["orient"] == "v":
        return spec["h"], spec["half"]
    return spec["half"], spec["w"]


def _half_to_global(spec: dict, side: int, lr: int, lc: int) -> tuple[int, int]:
    r0, c0 = spec["bbox"][0], spec["bbox"][1]
    axis = spec["axis"]
    if spec["orient"] == "v":
        r = r0 + lr
        c = axis - 1 - lc if side < 0 else axis + 1 + lc
        return int(r), int(c)
    r = axis - 1 - lr if side < 0 else axis + 1 + lr
    c = c0 + lc
    return int(r), int(c)


def _global_to_half(spec: dict, side: int, r: int, c: int) -> tuple[int, int] | None:
    r0, c0, r1, c1 = spec["bbox"]
    axis = spec["axis"]
    if not (r0 <= r <= r1 and c0 <= c <= c1):
        return None
    if spec["orient"] == "v":
        if side < 0:
            if not (c0 <= c < axis):
                return None
            return int(r - r0), int(axis - 1 - c)
        if not (axis < c <= c1):
            return None
        return int(r - r0), int(c - axis - 1)
    if side < 0:
        if not (r0 <= r < axis):
            return None
        return int(axis - 1 - r), int(c - c0)
    if not (axis < r <= r1):
        return None
    return int(r - axis - 1), int(c - c0)


def _marker_local(spec: dict) -> tuple[int, int]:
    if spec["orient"] == "v":
        return int(spec["marker_index"]), 0
    return 0, int(spec["marker_index"])


def _build_content(spec: dict, blocked_target: np.ndarray | None, target_heavier: bool) -> bool:
    shape = _half_shape(spec)
    axis_mode = "col0" if spec["orient"] == "v" else "row0"
    marker_cell = _marker_local(spec)

    blocked_source = np.zeros(shape, dtype=bool)
    if spec["marker_side"] == spec["source_side"]:
        blocked_source[marker_cell] = True

    blocked_target2 = np.zeros(shape, dtype=bool) if blocked_target is None else blocked_target.copy()
    if spec["marker_side"] != spec["source_side"]:
        blocked_target2[marker_cell] = True

    src = _random_connected_mask(shape, axis_mode, blocked=blocked_source, fill_lo=0.35, fill_hi=0.62)
    if src is None:
        return False
    src_area = int(src.sum())

    for _ in range(60):
        if target_heavier:
            tgt = _random_connected_mask(shape, axis_mode, blocked=blocked_target2, fill_lo=0.56, fill_hi=0.85)
        else:
            tgt = _random_connected_mask(shape, axis_mode, blocked=blocked_target2, fill_lo=0.30, fill_hi=0.80)
        if tgt is None:
            continue
        if np.array_equal(src, tgt):
            continue
        tgt_area = int(tgt.sum())
        if target_heavier and tgt_area < src_area:
            continue
        spec["source_mask"] = src
        spec["target_mask"] = tgt
        return True
    return False


def _render_input(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        for side, key in ((spec["source_side"], "source_mask"), (-spec["source_side"], "target_mask")):
            mask = spec[key]
            for lr, lc in np.argwhere(mask):
                r, c = _half_to_global(spec, side, int(lr), int(lc))
                grid[r, c] = spec["color"]

        r0, c0, r1, c1 = spec["bbox"]
        if spec["orient"] == "v":
            grid[r0 : r1 + 1, spec["axis"]] = YELLOW
            mr = r0 + spec["marker_index"]
            mc = spec["axis"] - 1 if spec["marker_side"] < 0 else spec["axis"] + 1
            grid[mr, mc] = YELLOW
        else:
            grid[spec["axis"], c0 : c1 + 1] = YELLOW
            mr = spec["axis"] - 1 if spec["marker_side"] < 0 else spec["axis"] + 1
            mc = c0 + spec["marker_index"]
            grid[mr, mc] = YELLOW
    return grid


def _metadata_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in sorted(specs, key=lambda d: (d["depth"], d["bbox"][0], d["bbox"][1])):
        for lr in range(_half_shape(spec)[0]):
            for lc in range(_half_shape(spec)[1]):
                sr, sc = _half_to_global(spec, spec["source_side"], lr, lc)
                tr, tc = _half_to_global(spec, -spec["source_side"], lr, lc)
                if inp[sr, sc] == YELLOW or inp[tr, tc] == YELLOW:
                    continue
                out[tr, tc] = inp[sr, sc]
    return out


def _padded_box(box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_outer(spec: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    oh, ow = spec["h"], spec["w"]
    if oh + 2 > h or ow + 2 > w:
        return False
    r0 = random.randint(1, h - oh - 1)
    c0 = random.randint(1, w - ow - 1)
    spec["bbox"] = (r0, c0, r0 + oh - 1, c0 + ow - 1)
    spec["axis"] = c0 + spec["half"] if spec["orient"] == "v" else r0 + spec["half"]
    limit = oh if spec["orient"] == "v" else ow
    spec["marker_index"] = random.randint(1, limit - 2)
    return True


def _inner_candidates(outer: dict, inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = outer["bbox"]
    axis = outer["axis"]
    starts = []
    if outer["orient"] == "v":
        rr = range(r0 + 2, r1 - 2 - inner_h + 2)
        if outer["target_side"] > 0:
            cc = range(axis + 2, c1 - 2 - inner_w + 2)
        else:
            cc = range(c0 + 2, axis - 2 - inner_w + 2)
    else:
        cc = range(c0 + 2, c1 - 2 - inner_w + 2)
        if outer["target_side"] > 0:
            rr = range(axis + 2, r1 - 2 - inner_h + 2)
        else:
            rr = range(r0 + 2, axis - 2 - inner_h + 2)
    for sr in rr:
        for sc in cc:
            starts.append((int(sr), int(sc)))
    return starts


def _blocked_target_from_inner(outer: dict, inner_box: tuple[int, int, int, int]) -> np.ndarray:
    block = np.zeros(_half_shape(outer), dtype=bool)
    padded = _padded_box(inner_box, 1, (outer["h"] + outer["bbox"][0] + 2, outer["w"] + outer["bbox"][1] + 2))
    for r in range(padded[0], padded[2] + 1):
        for c in range(padded[1], padded[3] + 1):
            loc = _global_to_half(outer, outer["target_side"], r, c)
            if loc is not None:
                block[loc] = True
    return block


def _place_extra(spec: dict, shape: tuple[int, int], forbidden: list[tuple[int, int, int, int]]) -> bool:
    h, w = shape
    eh, ew = spec["h"], spec["w"]
    starts = []
    for r0 in range(1, h - eh):
        for c0 in range(1, w - ew):
            box = (r0, c0, r0 + eh - 1, c0 + ew - 1)
            if any(_boxes_intersect(_padded_box(box, 2, shape), fb) for fb in forbidden):
                continue
            starts.append((r0, c0))
    if not starts:
        return False
    r0, c0 = random.choice(starts)
    spec["bbox"] = (r0, c0, r0 + eh - 1, c0 + ew - 1)
    spec["axis"] = c0 + spec["half"] if spec["orient"] == "v" else r0 + spec["half"]
    limit = eh if spec["orient"] == "v" else ew
    spec["marker_index"] = random.randint(1, limit - 2)
    return True


def _overlap_needed(outer: dict, inner: dict) -> bool:
    outer_color_cells = set()
    for lr, lc in np.argwhere(outer["source_mask"]):
        tr, tc = _half_to_global(outer, outer["target_side"], int(lr), int(lc))
        outer_color_cells.add((int(tr), int(tc)))
    inner_expected = set()
    for side in (inner["source_side"], -inner["source_side"]):
        for lr, lc in np.argwhere(inner["source_mask"]):
            rr, cc = _half_to_global(inner, side, int(lr), int(lc))
            inner_expected.add((int(rr), int(cc)))
    return bool(outer_color_cells & inner_expected)


def _object_changed(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    r0, c0, r1, c1 = spec["bbox"]
    before = inp[r0 : r1 + 1, c0 : c1 + 1]
    after = out[r0 : r1 + 1, c0 : c1 + 1]
    yellow_mask = before == YELLOW
    return np.any(before[~yellow_mask] != after[~yellow_mask])


def _validate(inp: np.ndarray, specs: list[dict]) -> bool:
    detected = _extract_objects(inp)
    if len(detected) != len(specs):
        return False
    if sorted(d["depth"] for d in detected).count(1) != 1:
        return False
    if sum(d["depth"] == 0 for d in detected) < 1:
        return False

    expected = _metadata_output(inp, specs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, actual):
        return False
    if np.array_equal(_solve(inp, always_marker_is_source=True), actual):
        return False
    if np.array_equal(_solve(inp, reverse_order=True), actual):
        return False
    if not any(_object_changed(inp, actual, spec) for spec in specs):
        return False
    if not all(_object_changed(inp, actual, spec) for spec in specs):
        return False
    if not any(int(spec["target_mask"].sum()) >= int(spec["source_mask"].sum()) for spec in specs):
        return False
    return True


def _attempt() -> np.ndarray | None:
    shape = (random.randint(18, 24), random.randint(18, 24))
    colors = random.sample(FG_PALETTE, random.choice([2, 3]))

    inner_dims = _object_dims(1)
    outer_dims = _object_dims(0)
    if outer_dims["orient"] == "v":
        outer_dims["half"] = random.randint(inner_dims["w"] + 2, inner_dims["w"] + 4)
        outer_dims["w"] = 2 * outer_dims["half"] + 1
        outer_dims["h"] = random.randint(inner_dims["h"] + 4, inner_dims["h"] + 6)
    else:
        outer_dims["half"] = random.randint(inner_dims["h"] + 2, inner_dims["h"] + 4)
        outer_dims["h"] = 2 * outer_dims["half"] + 1
        outer_dims["w"] = random.randint(inner_dims["w"] + 4, inner_dims["w"] + 6)
    if max(outer_dims["h"], outer_dims["w"]) > min(shape) - 2:
        return None

    outer = {
        **outer_dims,
        "depth": 0,
        "color": colors[0],
        "source_side": random.choice([-1, 1]),
    }
    outer["marker_side"] = outer["source_side"]
    outer["target_side"] = -outer["source_side"]
    if not _place_outer(outer, shape):
        return None

    inner = {
        **inner_dims,
        "depth": 1,
        "color": colors[1],
        "source_side": random.choice([-1, 1]),
    }
    inner["marker_side"] = -inner["source_side"]
    inner["target_side"] = -inner["source_side"]

    starts = _inner_candidates(outer, inner["h"], inner["w"])
    if not starts:
        return None
    sr, sc = random.choice(starts)
    inner["bbox"] = (sr, sc, sr + inner["h"] - 1, sc + inner["w"] - 1)
    inner["axis"] = sc + inner["half"] if inner["orient"] == "v" else sr + inner["half"]
    limit = inner["h"] if inner["orient"] == "v" else inner["w"]
    inner["marker_index"] = random.randint(1, limit - 2)

    specs = [outer, inner]
    if len(colors) == 3:
        extra_dims = _object_dims(0)
        extra = {
            **extra_dims,
            "depth": 0,
            "color": colors[2],
            "source_side": random.choice([-1, 1]),
        }
        extra["marker_side"] = extra["source_side"]
        extra["target_side"] = -extra["source_side"]
        forbidden = [_padded_box(outer["bbox"], 2, shape)]
        if not _place_extra(extra, shape, forbidden):
            return None
        specs.append(extra)

    heavy_index = random.randrange(len(specs))
    blocked_outer_target = _blocked_target_from_inner(outer, inner["bbox"])
    for idx, spec in enumerate(specs):
        blocked = blocked_outer_target if spec is outer else None
        if not _build_content(spec, blocked, idx == heavy_index):
            return None

    if not _overlap_needed(outer, inner):
        return None

    inp = _render_input(specs, shape)
    if not _validate(inp, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth mirror instance")
