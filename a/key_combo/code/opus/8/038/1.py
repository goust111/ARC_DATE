from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
VECTORS = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (2, 2), (2, -2), (0, 3), (3, 0), (3, 3), (3, -3)]
MAX_GENERATION_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_of(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def crop_true_bbox(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def enclosed_black_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    empty = ~mask
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return int(np.count_nonzero(empty & ~outside))


def enclosed_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    empty = ~mask
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return components(empty & ~outside)


def dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = mask.shape[0] - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = mask.shape[1] - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c0 = max(0, dc)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def marker_step(markers: list[tuple[int, int]]) -> tuple[int, int] | None:
    if len(markers) != 3:
        return None
    pts = sorted(markers)
    dr1 = pts[1][0] - pts[0][0]
    dc1 = pts[1][1] - pts[0][1]
    dr2 = pts[2][0] - pts[1][0]
    dc2 = pts[2][1] - pts[1][1]
    if (dr1, dc1) != (dr2, dc2):
        return None
    if (dr1, dc1) == (0, 0):
        return None
    if not (dr1 == 0 or dc1 == 0 or abs(dr1) == abs(dc1)):
        return None
    return dr1, dc1


def parse_roles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(v) for v in np.unique(g) if int(v) != BG)
    assert RED not in colors, "input must reserve red for the output rays"

    parsed = []
    for color in colors:
        comps = components(g == color)
        assert comps, "each color must appear"
        object_candidates = []
        marker_cells: list[tuple[int, int]] = []
        for comp in comps:
            r0, c0, r1, c1 = bbox_of(comp)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                mask[r - r0, c - c0] = True
            holes = enclosed_black_count(mask)
            if holes > 0:
                object_candidates.append((comp, holes, (r0, c0, r1, c1), mask))
            else:
                marker_cells.extend(comp)

        assert len(object_candidates) == 1, "each color needs exactly one enclosed object"
        comp, holes, box, mask = object_candidates[0]
        assert 1 <= holes <= 5, "hole count must stay in the intended range"
        assert len(marker_cells) == 3, "each color needs exactly one marker triple"
        step = marker_step(marker_cells)
        assert step is not None, "marker cells must form an equally spaced straight triple"
        parsed.append(
            {
                "color": color,
                "object_coords": comp,
                "bbox": box,
                "object_mask": mask,
                "K": holes,
                "markers": sorted(marker_cells),
                "step": step,
            }
        )

    assert 2 <= len(parsed) <= 3, "expected 2-3 color-linked pairs"
    return parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a red ray whose length is set by the enclosed black area of the same-colored object.

    Input structure:
    - The background is black.
    - There are 2-3 distinct non-red colors; each color appears in one 4-connected foreground object and in one separate three-pixel marker triple.
    - Every foreground object encloses 1-5 black cells inside its own bounding box, while each triple is equally spaced on a horizontal, vertical, or diagonal line.

    Rule:
    - For each color, count the black cells inside that color's object bounding box that are not 4-connected to the bounding-box border through black cells; call this count K.
    - Order the three same-colored marker pixels in row-major order, take the repeated step vector between consecutive markers, and start from the row-major last marker.
    - Paint red along that step vector for exactly K steps, skipping any steps that fall outside the grid; keep all original objects and markers unchanged.

    Key insight:
    - The ray direction is encoded locally by the marker triple, but the ray length is encoded remotely by the hole area of the same-colored enclosed object, so color is the only visible link that combines the two subproblems.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0

    parsed = parse_roles(g)
    out = g.copy()
    for item in parsed:
        dr, dc = item["step"]
        r0, c0 = item["markers"][-1]
        for k in range(1, item["K"] + 1):
            rr = r0 + dr * k
            cc = c0 + dc * k
            if in_bounds(g.shape, rr, cc):
                assert out[rr, cc] == BG, "ray must only paint empty background"
                out[rr, cc] = RED
    return out


def hole_dims(area: int) -> list[tuple[int, int]]:
    dims = {(1, area), (area, 1)}
    if area == 2:
        dims.add((2, 1))
    if area == 3:
        dims.add((3, 1))
    if area == 4:
        dims.add((2, 2))
    return list(dims)


def make_ring_mask(hole_h: int, hole_w: int) -> np.ndarray:
    mask = np.zeros((hole_h + 2, hole_w + 2), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def add_external_protrusions(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    empty = ~padded
    outside = np.zeros_like(empty, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    h, w = empty.shape
    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    candidates = []
    for r in range(h):
        for c in range(w):
            if not outside[r, c]:
                continue
            if any(0 <= nr < h and 0 <= nc < w and padded[nr, nc] for nr, nc in neighbors4(r, c)):
                candidates.append((r, c))
    random.shuffle(candidates)
    for r, c in candidates[: random.randint(0, 2)]:
        padded[r, c] = True
    return crop_true_bbox(padded)


def make_single_hole_mask(area: int) -> np.ndarray:
    hh, hw = random.choice(hole_dims(area))
    mask = make_ring_mask(hh, hw)
    if random.random() < 0.8:
        mask = add_external_protrusions(mask)
    return mask


def combine_two_holes(a: int, b: int) -> np.ndarray:
    m1 = make_ring_mask(*random.choice(hole_dims(a)))
    m2 = make_ring_mask(*random.choice(hole_dims(b)))
    if random.random() < 0.5:
        align_top = random.random() < 0.5
        h = max(m1.shape[0], m2.shape[0])
        w = m1.shape[1] + 1 + m2.shape[1]
        out = np.zeros((h, w), dtype=bool)
        r1 = 0 if align_top else h - m1.shape[0]
        r2 = 0 if align_top else h - m2.shape[0]
        out[r1 : r1 + m1.shape[0], : m1.shape[1]] |= m1
        out[r2 : r2 + m2.shape[0], m1.shape[1] + 1 :] |= m2
        bridge_r = 0 if align_top else h - 1
        out[bridge_r, m1.shape[1]] = True
    else:
        align_left = random.random() < 0.5
        h = m1.shape[0] + 1 + m2.shape[0]
        w = max(m1.shape[1], m2.shape[1])
        out = np.zeros((h, w), dtype=bool)
        c1 = 0 if align_left else w - m1.shape[1]
        c2 = 0 if align_left else w - m2.shape[1]
        out[: m1.shape[0], c1 : c1 + m1.shape[1]] |= m1
        out[m1.shape[0] + 1 :, c2 : c2 + m2.shape[1]] |= m2
        bridge_c = 0 if align_left else w - 1
        out[m1.shape[0], bridge_c] = True
    if random.random() < 0.9:
        out = add_external_protrusions(out)
    return out


def build_object_mask(k: int, force_multi_hole: bool) -> np.ndarray:
    if k >= 2 and (force_multi_hole or random.random() < 0.35):
        splits = [(a, k - a) for a in range(1, k) if a <= k - a]
        if splits:
            mask = combine_two_holes(*random.choice(splits))
            if enclosed_black_count(mask) == k:
                if random.random() < 0.5:
                    mask = np.rot90(mask, random.randrange(4))
                if random.random() < 0.5:
                    mask = np.fliplr(mask)
                return crop_true_bbox(mask)
    mask = make_single_hole_mask(k)
    if random.random() < 0.5:
        mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return crop_true_bbox(mask)


def visible_ray_cells(shape: tuple[int, int], last: tuple[int, int], step: tuple[int, int], k: int) -> list[tuple[int, int]]:
    out = []
    for i in range(1, k + 1):
        rr = last[0] + step[0] * i
        cc = last[1] + step[1] * i
        if in_bounds(shape, rr, cc):
            out.append((rr, cc))
        else:
            break
    return out


def bundle_candidates(shape: tuple[int, int], k: int, truncated: bool) -> list[dict]:
    h, w = shape
    out = []
    vectors = VECTORS[:]
    random.shuffle(vectors)
    for dr, dc in vectors:
        local = []
        for last_r in range(h):
            for last_c in range(w):
                markers = [(last_r - 2 * dr, last_c - 2 * dc), (last_r - dr, last_c - dc), (last_r, last_c)]
                if not all(in_bounds(shape, r, c) for r, c in markers):
                    continue
                red = visible_ray_cells(shape, markers[-1], (dr, dc), k)
                if not red:
                    continue
                if truncated and len(red) >= k:
                    continue
                if not truncated and len(red) < k:
                    continue
                local.append({"markers": markers, "ray": red, "step": (dr, dc)})
        if local:
            random.shuffle(local)
            out.extend(local)
    return out


def place_bundles(shape: tuple[int, int], ks: list[int], must_have_truncated: bool) -> list[dict] | None:
    input_occ = np.zeros(shape, dtype=bool)
    any_occ = np.zeros(shape, dtype=bool)
    bundles: list[dict] = []
    trunc_targets = [k for k in ks if k > 1]
    forced_k = random.choice(trunc_targets) if must_have_truncated and trunc_targets else None

    for k in random.sample(ks, len(ks)):
        want_trunc = (k == forced_k) or (k > 1 and random.random() < 0.3)
        candidates = bundle_candidates(shape, k, want_trunc)
        if not candidates and want_trunc:
            candidates = bundle_candidates(shape, k, False)
            want_trunc = False
        if not candidates:
            return None

        placed = None
        forbidden_input = dilate8(input_occ) | any_occ
        for cand in candidates:
            marker_mask = np.zeros(shape, dtype=bool)
            ray_mask = np.zeros(shape, dtype=bool)
            for r, c in cand["markers"]:
                marker_mask[r, c] = True
            for r, c in cand["ray"]:
                ray_mask[r, c] = True
            if np.any(marker_mask & forbidden_input):
                continue
            if np.any(ray_mask & any_occ):
                continue
            placed = {
                "K": k,
                "markers": sorted(cand["markers"]),
                "ray": cand["ray"],
                "step": cand["step"],
                "truncated": want_trunc,
            }
            input_occ |= marker_mask
            any_occ |= marker_mask | ray_mask
            bundles.append(placed)
            break
        if placed is None:
            return None
    return bundles


def place_objects(shape: tuple[int, int], objects: list[dict], bundles: list[dict]) -> bool:
    input_occ = np.zeros(shape, dtype=bool)
    all_occ = np.zeros(shape, dtype=bool)
    for bundle in bundles:
        for r, c in bundle["markers"]:
            input_occ[r, c] = True
            all_occ[r, c] = True
        for r, c in bundle["ray"]:
            all_occ[r, c] = True

    for obj in sorted(objects, key=lambda item: item["mask"].sum(), reverse=True):
        forbidden = dilate8(input_occ) | all_occ
        mask = obj["mask"]
        mh, mw = mask.shape
        positions = []
        for r0 in range(shape[0] - mh + 1):
            for c0 in range(shape[1] - mw + 1):
                window = forbidden[r0 : r0 + mh, c0 : c0 + mw]
                if np.any(window & mask):
                    continue
                positions.append((r0, c0))
        if not positions:
            return False
        r0, c0 = random.choice(positions)
        obj["pos"] = (r0, c0)
        for rr, cc in np.argwhere(mask):
            input_occ[r0 + rr, c0 + cc] = True
            all_occ[r0 + rr, c0 + cc] = True
    return True


def bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def triple_center(markers: list[tuple[int, int]]) -> tuple[float, float]:
    return (sum(r for r, _ in markers) / 3.0, sum(c for _, c in markers) / 3.0)


def anti_shortcut_ok(objects: list[dict], bundles: list[dict]) -> bool:
    row_object = [item["K"] for item in sorted(objects, key=lambda item: item["bbox"][:2])]
    row_bundle = [item["K"] for item in sorted(bundles, key=lambda item: item["markers"][0])]
    if row_object == row_bundle:
        return False

    col_object = [item["K"] for item in sorted(objects, key=lambda item: (item["bbox"][1], item["bbox"][0]))]
    col_bundle = [item["K"] for item in sorted(bundles, key=lambda item: (item["markers"][0][1], item["markers"][0][0]))]
    if col_object == col_bundle:
        return False

    nearest_all_match = True
    for bundle in bundles:
        br, bc = triple_center(bundle["markers"])
        nearest = min(
            objects,
            key=lambda item: abs(br - bbox_center(item["bbox"])[0]) + abs(bc - bbox_center(item["bbox"])[1]),
        )
        if nearest["K"] != bundle["K"]:
            nearest_all_match = False
            break
    return not nearest_all_match


def render_case(shape: tuple[int, int], objects: list[dict], bundles: list[dict], color_of_k: dict[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        color = color_of_k[obj["K"]]
        r0, c0 = obj["pos"]
        for rr, cc in np.argwhere(obj["mask"]):
            grid[r0 + rr, c0 + cc] = color
    for bundle in bundles:
        color = color_of_k[bundle["K"]]
        for r, c in bundle["markers"]:
            assert grid[r, c] == BG
            grid[r, c] = color
    return grid


def validate_case(grid: np.ndarray, expected_ks: set[int], expect_multi: bool, expect_truncated: bool) -> bool:
    try:
        parsed = parse_roles(grid)
        out = main(grid)
    except AssertionError:
        return False

    if {item["K"] for item in parsed} != expected_ks:
        return False
    if len({item["K"] for item in parsed}) != len(parsed):
        return False

    red_added = np.argwhere((out == RED) & (grid == BG))
    if len(red_added) == 0:
        return False

    multi_seen = False
    truncated_seen = False
    for item in parsed:
        hole_regions = enclosed_regions(item["object_mask"])
        if len(hole_regions) > 1:
            multi_seen = True
        visible = visible_ray_cells(grid.shape, item["markers"][-1], item["step"], item["K"])
        if len(visible) < item["K"]:
            truncated_seen = True
        for r, c in visible:
            if out[r, c] != RED:
                return False
    if expect_multi and not multi_seen:
        return False
    if expect_truncated and not truncated_seen:
        return False
    return True


def attempt_generate() -> np.ndarray | None:
    shape = (random.randint(15, 19), random.randint(15, 19))
    count = random.choice((2, 3))
    ks = random.sample([1, 2, 3, 4, 5], count)

    multi_candidates = [k for k in ks if k >= 2]
    expect_multi = bool(multi_candidates and random.random() < 0.6)
    forced_multi = random.choice(multi_candidates) if expect_multi else None

    objects = []
    for k in ks:
        mask = build_object_mask(k, force_multi_hole=(k == forced_multi))
        if enclosed_black_count(mask) != k:
            return None
        objects.append({"K": k, "mask": mask, "bbox": None, "pos": None})

    expect_truncated = bool(any(k > 1 for k in ks) and random.random() < 0.55)
    bundles = place_bundles(shape, ks, must_have_truncated=expect_truncated)
    if bundles is None:
        return None

    if not place_objects(shape, objects, bundles):
        return None

    for obj in objects:
        r0, c0 = obj["pos"]
        mh, mw = obj["mask"].shape
        obj["bbox"] = (r0, c0, r0 + mh - 1, c0 + mw - 1)

    if not anti_shortcut_ok(objects, bundles):
        return None

    color_of_k = {k: color for k, color in zip(ks, random.sample(PALETTE, count))}
    grid = render_case(shape, objects, bundles, color_of_k)
    if not validate_case(grid, set(ks), expect_multi, expect_truncated):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
