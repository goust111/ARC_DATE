from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

SCALE_FACTORS = (2, 3)
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def component_coords(mask: np.ndarray) -> list[np.ndarray]:
    mask = mask.astype(bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for coords in component_coords(grid == color):
            bbox = bbox_coords(coords)
            mask = crop_bbox(grid == color, bbox)
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "mask": mask.astype(bool),
                    "area": int(coords.shape[0]),
                }
            )
    return objects


def centered_top_left(bbox: tuple[int, int, int, int], new_h: int, new_w: int) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    top = (r0 + r1 - (int(new_h) - 1)) // 2
    left = (c0 + c1 - (int(new_w) - 1)) // 2
    return int(top), int(left)


def paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = map(int, mask.shape)
    for r in range(h):
        gr = top + r
        if not (0 <= gr < grid.shape[0]):
            continue
        for c in range(w):
            if not mask[r, c]:
                continue
            gc = left + c
            if 0 <= gc < grid.shape[1]:
                grid[gr, gc] = int(color)


def placement_visible_mask(
    full_mask: np.ndarray, top: int, left: int, grid_shape: tuple[int, int]
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    h, w = map(int, full_mask.shape)
    gh, gw = map(int, grid_shape)
    rr0 = max(0, int(top))
    cc0 = max(0, int(left))
    rr1 = min(gh, int(top) + h)
    cc1 = min(gw, int(left) + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return None
    sub = full_mask[rr0 - top : rr1 - top, cc0 - left : cc1 - left]
    coords = np.argwhere(sub)
    if coords.size == 0:
        return None
    sr0, sc0 = coords.min(axis=0)
    sr1, sc1 = coords.max(axis=0)
    bbox = (rr0 + int(sr0), cc0 + int(sc0), rr0 + int(sr1), cc0 + int(sc1))
    cropped = sub[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    return bbox, cropped


def axis_candidates(start: int, end: int, full_len: int, limit: int) -> list[int]:
    if start > 0 and end < limit - 1:
        top = int(start)
        return [top] if top + full_len - 1 == end else []
    if start == 0 and end < limit - 1:
        top = int(end - full_len + 1)
        return [top] if top <= 0 else []
    if start > 0 and end == limit - 1:
        top = int(start)
        return [top] if top + full_len - 1 >= limit - 1 else []
    lo = limit - full_len
    hi = 0
    if lo > hi:
        lo, hi = hi, lo
    return list(range(int(lo), int(hi) + 1))


def match_candidates(grid_shape: tuple[int, int], small: dict, large: dict) -> list[dict]:
    if int(small["color"]) == int(large["color"]):
        return []
    large_bbox = tuple(map(int, large["bbox"]))
    large_mask = np.asarray(large["mask"], dtype=bool)
    out = []
    for factor in SCALE_FACTORS:
        full_mask = scale_mask(np.asarray(small["mask"], dtype=bool), factor)
        fh, fw = full_mask.shape
        if large_mask.shape[0] > fh or large_mask.shape[1] > fw:
            continue
        placements = []
        tops = axis_candidates(large_bbox[0], large_bbox[2], fh, grid_shape[0])
        lefts = axis_candidates(large_bbox[1], large_bbox[3], fw, grid_shape[1])
        for top in tops:
            for left in lefts:
                visible = placement_visible_mask(full_mask, top, left, grid_shape)
                if visible is None:
                    continue
                bbox, cropped = visible
                if bbox == large_bbox and np.array_equal(cropped, large_mask):
                    placements.append((int(top), int(left)))
        if len(placements) == 1:
            top, left = placements[0]
            out.append(
                {
                    "small": small,
                    "large": large,
                    "factor": int(factor),
                    "large_full_bbox": (top, left, top + fh - 1, left + fw - 1),
                    "large_full_mask": full_mask,
                }
            )
    return out


def all_matches(grid: np.ndarray, objects: list[dict]) -> list[dict]:
    out = []
    for i, small in enumerate(objects):
        for j, large in enumerate(objects):
            if i == j:
                continue
            out.extend(match_candidates(grid.shape, small, large))
    return out


def choose_unique_matching(matches: list[dict], objects: list[dict]) -> list[dict]:
    index_by_id = {id(obj): idx for idx, obj in enumerate(objects)}
    encoded = [
        (
            index_by_id[id(m["small"])],
            index_by_id[id(m["large"])],
            pos,
        )
        for pos, m in enumerate(matches)
    ]
    best: list[tuple[int, ...]] = []
    best_size = -1

    def rec(pos: int, used: set[int], chosen: list[int]) -> None:
        nonlocal best, best_size
        if len(chosen) + (len(encoded) - pos) < best_size:
            return
        if pos == len(encoded):
            key = tuple(sorted(chosen))
            size = len(chosen)
            if size > best_size:
                best_size = size
                best = [key]
            elif size == best_size:
                best.append(key)
            return
        s_idx, l_idx, match_idx = encoded[pos]
        rec(pos + 1, used, chosen)
        if s_idx not in used and l_idx not in used:
            used.add(s_idx)
            used.add(l_idx)
            chosen.append(match_idx)
            rec(pos + 1, used, chosen)
            chosen.pop()
            used.remove(s_idx)
            used.remove(l_idx)

    rec(0, set(), [])
    if not best:
        return []
    unique = sorted(set(best))
    assert len(unique) == 1, "pairing must be uniquely determined"
    return [matches[idx] for idx in unique[0]]


def solve_matches(grid: np.ndarray, bg: int | None = None) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    objects = extract_objects(g, int(bg))
    matches = choose_unique_matching(all_matches(g, objects), objects)
    return matches, objects


def apply_swap(grid: np.ndarray, matches: list[dict], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for match in matches:
        small = match["small"]
        large = match["large"]
        out[small["coords"][:, 0], small["coords"][:, 1]] = int(bg)
        out[large["coords"][:, 0], large["coords"][:, 1]] = int(bg)
    for match in matches:
        small = match["small"]
        large = match["large"]
        large_full_mask = np.asarray(match["large_full_mask"], dtype=bool)
        big_top, big_left = centered_top_left(tuple(map(int, small["bbox"])), *large_full_mask.shape)
        paint_mask(out, large_full_mask, big_top, big_left, int(small["color"]))

        small_mask = np.asarray(small["mask"], dtype=bool)
        small_top, small_left = centered_top_left(tuple(map(int, match["large_full_bbox"])), *small_mask.shape)
        paint_mask(out, small_mask, small_top, small_left, int(large["color"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the sizes of every cross-scale same-shape object pair in place.

    Input structure:
    - The grid has one dominant background color and several colored objects.
    - Some objects form pairs: ignoring color, one mask is a 2x or 3x integer scaling
      of the other, and the larger member may be clipped by the canvas edge.
    - Extra distractor objects may also be present and do not belong to any pair.

    Rule:
    - Split the foreground into monochromatic 8-connected objects.
    - Compare every object pair under 2x and 3x integer mask scaling, allowing the
      larger object's full bbox to extend outside the canvas when testing the match.
    - For each matched pair, swap their scales in place: enlarge the small object and
      shrink the large one, preserving each object's own color and anchoring the resize
      at the center of its completed bbox; enlarged parts that leave the canvas are clipped.

    Key insight:
    - The crucial relation is cross-scale shape kinship, not color or visible size: a
      clipped border object must first be mentally completed before the pair can exchange scale.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    matches, _objects = solve_matches(g, bg)
    return apply_swap(g, matches, bg)


def random_connected_mask(rng: random.Random, h: int, w: int) -> np.ndarray | None:
    target_area = rng.randint(max(h + w - 2, (h * w) // 2), h * w - 1)
    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        frontier = {start}
        while int(mask.sum()) < target_area:
            candidates = set()
            for r, c in frontier:
                for dr, dc in NEIGHBORS8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        candidates.add((nr, nc))
            if not candidates:
                break
            pick = rng.choice(tuple(candidates))
            mask[pick] = True
            frontier.add(pick)
        if int(mask.sum()) != target_area:
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        rows = tuple(tuple(row.tolist()) for row in mask)
        cols = tuple(tuple(col.tolist()) for col in mask.T)
        if len(set(rows)) == 1 or len(set(cols)) == 1:
            continue
        return mask
    return None


def paint_object(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    out = grid.copy()
    paint_mask(out, mask, top, left, color)
    return out


def visible_coords(mask: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    coords = set()
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            gr, gc = top + r, left + c
            if 0 <= gr < shape[0] and 0 <= gc < shape[1]:
                coords.add((int(gr), int(gc)))
    return coords


def padded(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    out.add((int(nr), int(nc)))
    return out


def choose_clip_top_left(
    rng: random.Random,
    shape: tuple[int, int],
    full_h: int,
    full_w: int,
    small_h: int,
    small_w: int,
    delta_r: int,
    delta_c: int,
    must_clip: bool,
) -> tuple[int, int] | None:
    h, w = shape
    r_lo = -delta_r
    r_hi = h - small_h - delta_r
    c_lo = -delta_c
    c_hi = w - small_w - delta_c
    if r_lo > r_hi or c_lo > c_hi:
        return None
    modes = ["none", "top", "bottom", "left", "right", "tl", "tr", "bl", "br"]
    rng.shuffle(modes)
    if must_clip:
        modes = [m for m in modes if m != "none"]
    for mode in modes:
        for _ in range(40):
            if mode == "none":
                top = rng.randint(0, h - full_h)
                left = rng.randint(0, w - full_w)
            else:
                top = rng.randint(r_lo, r_hi)
                left = rng.randint(c_lo, c_hi)
                if mode in {"top", "tl", "tr"}:
                    top = rng.randint(r_lo, -1)
                if mode in {"bottom", "bl", "br"}:
                    top = rng.randint(h - full_h + 1, r_hi)
                if mode in {"left", "tl", "bl"}:
                    left = rng.randint(c_lo, -1)
                if mode in {"right", "tr", "br"}:
                    left = rng.randint(w - full_w + 1, c_hi)
            clipped = top < 0 or left < 0 or top + full_h > h or left + full_w > w
            if clipped == must_clip:
                return int(top), int(left)
    return None


def object_signature(grid: np.ndarray, color: int) -> tuple[tuple[int, ...], ...]:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return tuple()
    r0, c0, r1, c1 = bbox_coords(coords)
    return tuple(tuple(int(v) for v in row) for row in grid[r0 : r1 + 1, c0 : c1 + 1])


def build_expected(
    shape: tuple[int, int],
    bg: int,
    pair_specs: list[dict],
    distractors: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, bg, dtype=int)
    out = np.full(shape, bg, dtype=int)
    for spec in pair_specs:
        inp = paint_object(inp, spec["small_mask"], spec["small_top"], spec["small_left"], spec["small_color"])
        inp = paint_object(inp, spec["large_mask"], spec["large_top"], spec["large_left"], spec["large_color"])
        out = paint_object(out, spec["large_full_mask"], spec["out_large_top"], spec["out_large_left"], spec["small_color"])
        out = paint_object(out, spec["small_mask"], spec["out_small_top"], spec["out_small_left"], spec["large_color"])
    for spec in distractors:
        inp = paint_object(inp, spec["mask"], spec["top"], spec["left"], spec["color"])
        out = paint_object(out, spec["mask"], spec["top"], spec["left"], spec["color"])
    return inp, out


def validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    pair_specs: list[dict],
    distractors: list[dict],
) -> bool:
    try:
        predicted = main(inp)
        bg = get_bg(inp)
        matches, objects = solve_matches(inp, bg)
    except Exception:
        return False
    if not np.array_equal(predicted, expected):
        return False
    if len(matches) != len(pair_specs):
        return False

    intended_pairs = {
        frozenset((int(spec["small_color"]), int(spec["large_color"])))
        for spec in pair_specs
    }
    seen_pairs = {
        frozenset((int(m["small"]["color"]), int(m["large"]["color"])))
        for m in matches
    }
    if seen_pairs != intended_pairs:
        return False

    factor_by_pair = {
        frozenset((int(spec["small_color"]), int(spec["large_color"]))): int(spec["factor"])
        for spec in pair_specs
    }
    for match in matches:
        key = frozenset((int(match["small"]["color"]), int(match["large"]["color"])))
        if int(match["factor"]) != factor_by_pair[key]:
            return False

    for spec in pair_specs:
        small_sig_in = object_signature(inp, int(spec["small_color"]))
        small_sig_out = object_signature(expected, int(spec["small_color"]))
        large_sig_in = object_signature(inp, int(spec["large_color"]))
        large_sig_out = object_signature(expected, int(spec["large_color"]))
        if small_sig_in == small_sig_out or large_sig_in == large_sig_out:
            return False
        if spec["must_clip"]:
            if not (
                spec["large_top"] < 0
                or spec["large_left"] < 0
                or spec["large_top"] + spec["large_full_mask"].shape[0] > inp.shape[0]
                or spec["large_left"] + spec["large_full_mask"].shape[1] > inp.shape[1]
            ):
                return False
            if int(spec["large_mask"].sum()) >= int(spec["large_full_mask"].sum()):
                return False

    for spec in distractors:
        color = int(spec["color"])
        if object_signature(inp, color) != object_signature(expected, color):
            return False

    if np.array_equal(inp, expected):
        return False
    if len(objects) != len(pair_specs) * 2 + len(distractors):
        return False
    return True


def gen_pair_spec(
    rng: random.Random,
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    small_color: int,
    large_color: int,
    factor: int,
    must_clip: bool,
) -> dict | None:
    if factor == 2:
        small_h = rng.choice((2, 4))
        small_w = rng.choice((2, 4))
    else:
        small_h = rng.randint(2, 4)
        small_w = rng.randint(2, 4)
    small_mask = random_connected_mask(rng, small_h, small_w)
    if small_mask is None:
        return None

    large_full_mask = scale_mask(small_mask, factor)
    full_h, full_w = large_full_mask.shape
    delta_r = (full_h - small_h) // 2
    delta_c = (full_w - small_w) // 2

    if shape[0] < full_h + 2 or shape[1] < full_w + 2:
        return None

    for _ in range(120):
        out_large_top = rng.randint(0, shape[0] - full_h)
        out_large_left = rng.randint(0, shape[1] - full_w)
        small_top = out_large_top + delta_r
        small_left = out_large_left + delta_c

        large_place = choose_clip_top_left(
            rng, shape, full_h, full_w, small_h, small_w, delta_r, delta_c, must_clip
        )
        if large_place is None:
            continue
        large_top, large_left = large_place
        out_small_top = large_top + delta_r
        out_small_left = large_left + delta_c

        input_small = visible_coords(small_mask, small_top, small_left, shape)
        input_large = visible_coords(large_full_mask, large_top, large_left, shape)
        output_large = visible_coords(large_full_mask, out_large_top, out_large_left, shape)
        output_small = visible_coords(small_mask, out_small_top, out_small_left, shape)

        if not input_small or not input_large or not output_large or not output_small:
            continue
        union = input_small | input_large | output_large | output_small
        if padded(union, shape) & reserved:
            continue
        if len(input_large) <= len(input_small):
            continue

        large_visible = placement_visible_mask(large_full_mask, large_top, large_left, shape)
        if large_visible is None:
            continue
        _bbox, large_mask = large_visible
        return {
            "factor": int(factor),
            "small_color": int(small_color),
            "large_color": int(large_color),
            "small_mask": small_mask,
            "large_full_mask": large_full_mask,
            "large_mask": large_mask,
            "small_top": int(small_top),
            "small_left": int(small_left),
            "large_top": int(large_top),
            "large_left": int(large_left),
            "out_large_top": int(out_large_top),
            "out_large_left": int(out_large_left),
            "out_small_top": int(out_small_top),
            "out_small_left": int(out_small_left),
            "must_clip": bool(must_clip),
            "reserved_union": union,
        }
    return None


def gen_distractor(
    rng: random.Random,
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    color: int,
) -> dict | None:
    for _ in range(80):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        mask = random_connected_mask(rng, h, w)
        if mask is None:
            continue
        top = rng.randint(0, shape[0] - h)
        left = rng.randint(0, shape[1] - w)
        coords = visible_coords(mask, top, left, shape)
        if padded(coords, shape) & reserved:
            continue
        return {"mask": mask, "top": int(top), "left": int(left), "color": int(color), "reserved_union": coords}
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 28)
    w = rng.randint(20, 28)
    bg = rng.randrange(10)
    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)

    pair_count = rng.choice((1, 2))
    distractor_count = rng.choice((1, 2))
    if pair_count == 2 and rng.random() < 0.7:
        factors = [2, 3]
        rng.shuffle(factors)
    else:
        factors = [rng.choice(SCALE_FACTORS) for _ in range(pair_count)]
    clip_flags = [rng.random() < 0.6 for _ in range(pair_count)]
    if not any(clip_flags):
        clip_flags[rng.randrange(pair_count)] = True

    if len(colors) < pair_count * 2 + distractor_count:
        return None

    reserved: set[tuple[int, int]] = set()
    pair_specs = []
    color_idx = 0
    for i in range(pair_count):
        spec = gen_pair_spec(
            rng,
            (h, w),
            reserved,
            colors[color_idx],
            colors[color_idx + 1],
            factors[i],
            clip_flags[i],
        )
        color_idx += 2
        if spec is None:
            return None
        pair_specs.append(spec)
        reserved |= padded(spec["reserved_union"], (h, w))

    distractors = []
    for _ in range(distractor_count):
        spec = gen_distractor(rng, (h, w), reserved, colors[color_idx])
        color_idx += 1
        if spec is None:
            return None
        distractors.append(spec)
        reserved |= padded(spec["reserved_union"], (h, w))

    inp, expected = build_expected((h, w), bg, pair_specs, distractors)
    if not validate_instance(inp, expected, pair_specs, distractors):
        return None
    return inp


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
