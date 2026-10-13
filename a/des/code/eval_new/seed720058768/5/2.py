import random
from collections import Counter, deque

import numpy as np


BG = 0
TEMP_SIZE = 30
PALETTE = tuple(range(1, 10))
MAX_GENERATION_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _tight_crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _color_mask_to_array(mask: np.ndarray, color: int) -> np.ndarray:
    arr = np.zeros(mask.shape, dtype=int)
    arr[mask] = int(color)
    return arr


def _mask_components_by_color(grid: np.ndarray, bg: int) -> dict[int, list[np.ndarray]]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    comps: dict[int, list[np.ndarray]] = {}
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            rr = [p[0] for p in coords]
            cc = [p[1] for p in coords]
            r0, r1 = min(rr), max(rr)
            c0, c1 = min(cc), max(cc)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for pr, pc in coords:
                mask[pr - r0, pc - c0] = True
            comps.setdefault(color, []).append(mask)
    return comps


def _nonbg_runs(seq: np.ndarray, bg: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(map(int, seq)):
        if value != int(bg) and start is None:
            start = idx
        elif value == int(bg) and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(seq)))
    return runs


def _parse_stripe_pairs(values: list[int]) -> list[frozenset[int]]:
    pairs: list[frozenset[int]] = []
    i = 0
    n = len(values)
    while i < n:
        if i + 1 >= n or int(values[i]) == int(values[i + 1]):
            raise ValueError("invalid stripe segment")
        a, b = int(values[i]), int(values[i + 1])
        j = i + 2
        while j < n and int(values[j]) in (a, b) and int(values[j]) == int(values[j - 2]):
            j += 1
        if j - i < 4:
            raise ValueError("stripe segments must be long enough to show alternation")
        pairs.append(frozenset((a, b)))
        i = j
    if len(pairs) < 2:
        raise ValueError("expected multiple stripe pairs")
    return pairs


def _extract_stripe(grid: np.ndarray, bg: int):
    h, w = map(int, grid.shape)
    candidates = [
        ("top", np.asarray(grid[0, :], dtype=int)),
        ("bottom", np.asarray(grid[h - 1, :], dtype=int)),
        ("left", np.asarray(grid[:, 0], dtype=int)),
        ("right", np.asarray(grid[:, w - 1], dtype=int)),
    ]
    best = None
    for edge, seq in candidates:
        for start, end in _nonbg_runs(seq, bg):
            vals = list(map(int, seq[start:end]))
            try:
                pairs = _parse_stripe_pairs(vals)
            except ValueError:
                continue
            if best is None or (end - start) > best["length"]:
                best = {
                    "edge": edge,
                    "start": int(start),
                    "end": int(end),
                    "pairs": pairs,
                    "length": int(end - start),
                }
    assert best is not None, "expected a parseable border stripe"

    mask = np.zeros(grid.shape, dtype=bool)
    if best["edge"] == "top":
        mask[0, best["start"] : best["end"]] = True
        axis = "h"
    elif best["edge"] == "bottom":
        mask[h - 1, best["start"] : best["end"]] = True
        axis = "h"
    elif best["edge"] == "left":
        mask[best["start"] : best["end"], 0] = True
        axis = "v"
    else:
        mask[best["start"] : best["end"], w - 1] = True
        axis = "v"
    return best["pairs"], axis, mask


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs = set()
    for h in range(1, int(area ** 0.5) + 1):
        if area % h == 0:
            w = area // h
            pairs.add((h, w))
            pairs.add((w, h))
    return sorted(pairs)


def _assemble_pair(mask_a: np.ndarray, color_a: int, mask_b: np.ndarray, color_b: int) -> np.ndarray:
    coords_a = np.argwhere(mask_a)
    area = int(mask_a.sum() + mask_b.sum())
    best = None
    for rot in range(4):
        rot_b = np.rot90(mask_b, k=rot)
        coords_b = np.argwhere(rot_b)
        ha, wa = map(int, mask_a.shape)
        hb, wb = map(int, rot_b.shape)
        for out_h, out_w in _factor_pairs(area):
            if out_h < max(ha, hb) or out_w < max(wa, wb):
                continue
            for ra in range(out_h - ha + 1):
                for ca in range(out_w - wa + 1):
                    occ = np.zeros((out_h, out_w), dtype=bool)
                    shifted_a = coords_a + np.array([ra, ca])
                    occ[shifted_a[:, 0], shifted_a[:, 1]] = True
                    for rb in range(out_h - hb + 1):
                        for cb in range(out_w - wb + 1):
                            shifted_b = coords_b + np.array([rb, cb])
                            if occ[shifted_b[:, 0], shifted_b[:, 1]].any():
                                continue
                            arr = np.zeros((out_h, out_w), dtype=int)
                            arr[shifted_a[:, 0], shifted_a[:, 1]] = int(color_a)
                            arr[shifted_b[:, 0], shifted_b[:, 1]] = int(color_b)
                            if not np.all(arr != 0):
                                continue
                            signature = (rot, out_h, out_w, ra, ca, rb, cb)
                            if best is None or signature < best[0]:
                                best = (signature, arr)
    assert best is not None, "expected a complementary assembly for each stripe pair"
    return best[1]


def _seam_bias(arr: np.ndarray) -> str:
    vertical_steps = int(np.sum(arr[:-1, :] != arr[1:, :]))
    horizontal_steps = int(np.sum(arr[:, :-1] != arr[:, 1:]))
    return "h" if vertical_steps >= horizontal_steps else "v"


def _orient_completed_rectangle(arr: np.ndarray) -> np.ndarray:
    seam = _seam_bias(arr)
    h, w = map(int, arr.shape)
    if seam == "h" and h > w:
        return np.rot90(arr, k=1)
    if seam == "v" and w > h:
        return np.rot90(arr, k=1)
    return arr


def _render_strip(rects: list[np.ndarray], axis: str, bg: int) -> np.ndarray:
    assert rects, "expected at least one rectangle to render"
    if axis == "h":
        out_h = max(int(r.shape[0]) for r in rects)
        out_w = sum(int(r.shape[1]) for r in rects) + len(rects) - 1
        out = np.full((out_h, out_w), int(bg), dtype=int)
        cc = 0
        for idx, rect in enumerate(rects):
            h, w = map(int, rect.shape)
            out[:h, cc : cc + w] = rect
            cc += w
            if idx + 1 < len(rects):
                cc += 1
        return out

    out_h = sum(int(r.shape[0]) for r in rects) + len(rects) - 1
    out_w = max(int(r.shape[1]) for r in rects)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    rr = 0
    for idx, rect in enumerate(rects):
        h, w = map(int, rect.shape)
        out[rr : rr + h, :w] = rect
        rr += h
        if idx + 1 < len(rects):
            rr += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble complementary color fragments into filled rectangles and order them by a border color band.

    Input structure:
    - Background is black. Away from the border, each non-black connected component is a monochrome fragment.
    - Fragments come in color pairs; within a pair, the two shapes are complementary parts of one full rectangle, possibly after rotating one fragment.
    - One border row or border column contains consecutive alternating color bands, and each alternating two-color segment names one fragment pair.

    Rule:
    - Read the border band to recover the ordered list of fragment color pairs.
    - For each listed pair, find the two corresponding fragments, try the four rotations of one fragment, and place the two shapes so they exactly tile a rectangle without overlap or gaps.
    - Keep both fragment colors in the assembled rectangle. If the internal seam is mostly horizontal, place that rectangle in landscape orientation; if the seam is mostly vertical, place it in portrait orientation.
    - Concatenate the assembled rectangles in the border-band order: left-to-right for a horizontal band, or top-to-bottom for a vertical band, with one background cell between neighbors.

    Key insight:
    - The edge stripe gives the pair order, but each pair still has to be reconstructed by true shape complementation under rotation; the seam inside the completed rectangle then decides whether the output copy stays wide or becomes tall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    stripe_pairs, axis, stripe_mask = _extract_stripe(g, bg)
    work = g.copy()
    work[stripe_mask] = int(bg)
    comps = _mask_components_by_color(work, bg)

    rects: list[np.ndarray] = []
    for pair in stripe_pairs:
        colors = sorted(map(int, pair))
        assert len(colors) == 2, "each stripe segment must encode exactly two colors"
        color_a, color_b = colors
        assert color_a in comps and color_b in comps, "missing fragment color from stripe pair"
        assert len(comps[color_a]) == 1 and len(comps[color_b]) == 1, "expected one fragment per color"
        assembled = _assemble_pair(comps[color_a][0], color_a, comps[color_b][0], color_b)
        rects.append(_orient_completed_rectangle(assembled))

    out = _render_strip(rects, axis, bg)
    assert out.ndim == 2 and max(out.shape) <= 30
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen[start] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return count == int(mask.sum())


def _random_profile(length: int, limit: int, rng: random.Random) -> list[int]:
    vals = [rng.randint(1, limit)]
    for _ in range(1, length):
        delta = rng.choice([-1, 0, 1])
        vals.append(max(1, min(limit, vals[-1] + delta)))
    if len(set(vals)) == 1:
        idx = rng.randrange(length)
        if vals[idx] == 1:
            vals[idx] += 1
        elif vals[idx] == limit:
            vals[idx] -= 1
        else:
            vals[idx] += rng.choice([-1, 1])
    return vals


def _make_pair_masks(seam: str, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(80):
        if seam == "h":
            h = rng.randint(3, 6)
            w = rng.randint(4, 7)
            cuts = _random_profile(w, h - 1, rng)
            a = np.zeros((h, w), dtype=bool)
            for c, cut in enumerate(cuts):
                a[:cut, c] = True
        else:
            h = rng.randint(4, 7)
            w = rng.randint(3, 6)
            cuts = _random_profile(h, w - 1, rng)
            a = np.zeros((h, w), dtype=bool)
            for r, cut in enumerate(cuts):
                a[r, :cut] = True
        b = ~a
        if int(a.sum()) >= 4 and int(b.sum()) >= 4 and _is_connected(a) and _is_connected(b):
            return a, b
    raise RuntimeError("failed to build complementary masks")


def _stripe_values(ordered_pairs: list[tuple[int, int]], rng: random.Random) -> list[int]:
    values: list[int] = []
    for color_a, color_b in ordered_pairs:
        repeat = rng.choice([2, 3])
        if rng.random() < 0.5:
            segment = [color_a, color_b] * repeat
        else:
            segment = [color_b, color_a] * repeat
        values.extend(segment)
    return values


def _can_place(occupied: np.ndarray, arr: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    h, w = map(int, arr.shape)
    hh, ww = map(int, occupied.shape)
    if top < 0 or left < 0 or top + h > hh or left + w > ww:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _place_array(canvas: np.ndarray, occupied: np.ndarray, arr: np.ndarray, top: int, left: int) -> None:
    h, w = map(int, arr.shape)
    sub = canvas[top : top + h, left : left + w]
    mask = arr != 0
    sub[mask] = arr[mask]
    occupied[top : top + h, left : left + w] |= mask


def _crop_canvas(canvas: np.ndarray, edge: str) -> np.ndarray:
    coords = np.argwhere(canvas != 0)
    assert coords.size, "expected non-empty canvas"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = 0 if edge == "top" else max(0, int(r0) - 1)
    r1 = TEMP_SIZE - 1 if edge == "bottom" else min(TEMP_SIZE - 1, int(r1) + 1)
    c0 = 0 if edge == "left" else max(0, int(c0) - 1)
    c1 = TEMP_SIZE - 1 if edge == "right" else min(TEMP_SIZE - 1, int(c1) + 1)
    return canvas[r0 : r1 + 1, c0 : c1 + 1].copy()


def _build_expected(pair_specs: list[dict], axis: str) -> np.ndarray:
    rects = [spec["rect_out"] for spec in pair_specs]
    return _render_strip(rects, axis, BG)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_RETRIES):
        pair_count = rng.randint(2, 4)
        stripe_axis = rng.choice(["h", "v"])
        edge = rng.choice(["top", "bottom"] if stripe_axis == "h" else ["left", "right"])
        colors = rng.sample(PALETTE, 2 * pair_count)

        seam_choices = ["h", "v"]
        while len(seam_choices) < pair_count:
            seam_choices.append(rng.choice(["h", "v"]))
        rng.shuffle(seam_choices)

        pair_specs: list[dict] = []
        ordered_pairs: list[tuple[int, int]] = []
        color_idx = 0
        for seam in seam_choices:
            color_a = int(colors[color_idx])
            color_b = int(colors[color_idx + 1])
            color_idx += 2

            mask_a, mask_b = _make_pair_masks(seam, rng)
            full_rect = np.where(mask_a, color_a, color_b).astype(int)
            rect_out = _orient_completed_rectangle(full_rect)

            frag_a = np.rot90(_color_mask_to_array(mask_a, color_a), k=rng.randrange(4))
            frag_b = np.rot90(_color_mask_to_array(mask_b, color_b), k=rng.randrange(4))
            frag_a = _color_mask_to_array(_tight_crop_mask(frag_a != 0), color_a)
            frag_b = _color_mask_to_array(_tight_crop_mask(frag_b != 0), color_b)

            pair_specs.append(
                {
                    "colors": (color_a, color_b),
                    "rect_out": rect_out,
                    "frag_a": frag_a,
                    "frag_b": frag_b,
                }
            )
            ordered_pairs.append((color_a, color_b))

        rng.shuffle(pair_specs)
        ordered_pairs = [tuple(spec["colors"]) for spec in pair_specs]
        expected = _build_expected(pair_specs, stripe_axis)
        if max(expected.shape) > 30:
            continue

        stripe_vals = _stripe_values(ordered_pairs, rng)
        if len(stripe_vals) > 24:
            continue

        canvas = np.zeros((TEMP_SIZE, TEMP_SIZE), dtype=int)
        occupied = np.zeros((TEMP_SIZE, TEMP_SIZE), dtype=bool)
        if stripe_axis == "h":
            row = 0 if edge == "top" else TEMP_SIZE - 1
            start = rng.randint(1, TEMP_SIZE - len(stripe_vals) - 2)
            canvas[row, start : start + len(stripe_vals)] = np.array(stripe_vals, dtype=int)
            occupied[row, start : start + len(stripe_vals)] = True
        else:
            col = 0 if edge == "left" else TEMP_SIZE - 1
            start = rng.randint(1, TEMP_SIZE - len(stripe_vals) - 2)
            canvas[start : start + len(stripe_vals), col] = np.array(stripe_vals, dtype=int)
            occupied[start : start + len(stripe_vals), col] = True

        frags = []
        for spec in pair_specs:
            frags.append(spec["frag_a"])
            frags.append(spec["frag_b"])
        rng.shuffle(frags)

        ok = True
        for frag in frags:
            h, w = map(int, frag.shape)
            placed = False
            for _ in range(250):
                top = rng.randint(2, TEMP_SIZE - h - 2)
                left = rng.randint(2, TEMP_SIZE - w - 2)
                if _can_place(occupied, frag, top, left, pad=1):
                    _place_array(canvas, occupied, frag, top, left)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        grid = _crop_canvas(canvas, edge)
        if max(grid.shape) > 30:
            continue
        if len({frozenset(spec["colors"]) for spec in pair_specs}) != len(pair_specs):
            continue

        solved = main(grid)
        if not np.array_equal(solved, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input grid")
