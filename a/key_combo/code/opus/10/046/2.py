from __future__ import annotations

import random
from collections import defaultdict
from functools import lru_cache

import numpy as np


BACKGROUND = 0
UNMATCHED_COLOR = 2
MATCHED_COLOR = 3
INPUT_PALETTE = (1, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 600


def _gravity_settle(grid: np.ndarray) -> np.ndarray:
    """Drop each non-black column stack to the bottom, preserving top-to-bottom order."""
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    out = np.zeros_like(g)
    for c in range(W):
        values = [int(g[r, c]) for r in range(H) if int(g[r, c]) != BACKGROUND]
        if values:
            out[H - len(values) :, c] = values
    return out


def _find_components(grid: np.ndarray) -> list[dict]:
    """4-connected monochromatic components; returns bbox-cropped binary masks."""
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []

    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "bbox_area": mask.shape[0] * mask.shape[1],
                }
            )
    return comps


def _scale_mask2(mask: np.ndarray) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((2, 2), dtype=np.uint8)).astype(bool)


def _scale_related(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return np.array_equal(_scale_mask2(mask_a), mask_b) or np.array_equal(_scale_mask2(mask_b), mask_a)


def _candidate_edges(components: list[dict]) -> list[tuple[int, int, int]]:
    edges: list[tuple[int, int, int]] = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if _scale_related(components[i]["mask"], components[j]["mask"]):
                cost = int(components[i]["bbox_area"] + components[j]["bbox_area"])
                edges.append((i, j, cost))
    return edges


def _choose_pairs(components: list[dict]) -> set[tuple[int, int]]:
    edges = _candidate_edges(components)
    edge_map: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for i, j, cost in edges:
        edge_map[i].append((i, j, cost))
        edge_map[j].append((i, j, cost))

    @lru_cache(maxsize=None)
    def solve(used_mask: int) -> tuple[int, int, tuple[tuple[int, int], ...]]:
        first = -1
        for idx in range(len(components)):
            if not (used_mask & (1 << idx)):
                first = idx
                break
        if first == -1:
            return 0, 0, ()

        best_pairs, best_cost, best_edges = solve(used_mask | (1 << first))
        for i, j, cost in edge_map.get(first, []):
            other = j if i == first else i
            if used_mask & (1 << other):
                continue
            cand_pairs, cand_cost, cand_edges = solve(used_mask | (1 << first) | (1 << other))
            cand_pairs += 1
            cand_cost += cost
            cand_edges = ((min(first, other), max(first, other)),) + cand_edges
            if cand_pairs > best_pairs or (cand_pairs == best_pairs and cand_cost < best_cost):
                best_pairs, best_cost, best_edges = cand_pairs, cand_cost, cand_edges

        return best_pairs, best_cost, best_edges

    return set(solve(0)[2])


def _recolor_by_pairs(settled: np.ndarray, components: list[dict], pairs: set[tuple[int, int]]) -> np.ndarray:
    matched = {idx for pair in pairs for idx in pair}
    out = np.zeros_like(settled)
    for idx, comp in enumerate(components):
        color = MATCHED_COLOR if idx in matched else UNMATCHED_COLOR
        for r, c in comp["coords"]:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle all colored pixels by gravity, then recolor 1:2 same-shape pairs green and the rest red.

    Input structure:
    - Black background with several monochrome color groups scattered across disjoint column bands.
    - Within each band, pixels of one color are vertically lifted apart, so the meaningful object only appears after settling.
    - Different colors represent different candidate objects; output keeps the same canvas size.

    Rule:
    - For each column, drop all non-black cells to the lowest consecutive cells while preserving their top-to-bottom order.
    - In the settled grid, split monochrome 4-connected components and crop each one to its binary mask.
    - Pair components whose masks are exact 2x integer scalings of one another; each component can belong to at most one pair, and if multiple maximum pairings conflict, choose the one with the smallest total bbox-area sum.
    - Recolor every paired component green (3), every unpaired component red (2), and leave the background black.

    Key insight:
    - The input only reveals column counts; gravity must run first to assemble the real shapes, and only those assembled masks determine the final recoloring.
    """
    settled = _gravity_settle(np.asarray(grid, dtype=int))
    components = _find_components(settled)
    pairs = _choose_pairs(components)
    return _recolor_by_pairs(settled, components, pairs)


def _mask_from_heights(heights: tuple[int, ...]) -> np.ndarray:
    h = max(heights)
    w = len(heights)
    mask = np.zeros((h, w), dtype=bool)
    for c, height in enumerate(heights):
        mask[h - height :, c] = True
    return mask


def _scale_heights2(heights: tuple[int, ...]) -> tuple[int, ...]:
    out: list[int] = []
    for height in heights:
        out.extend((2 * height, 2 * height))
    return tuple(out)


def _shape_key(heights: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, int]]:
    mask = _mask_from_heights(heights)
    return tuple(int(v) for v in mask.ravel()), mask.shape


def _related_to_any(heights: tuple[int, ...], others: list[tuple[int, ...]]) -> bool:
    mask = _mask_from_heights(heights)
    return any(_scale_related(mask, _mask_from_heights(other)) for other in others)


def _random_small_heights() -> tuple[int, ...]:
    for _ in range(300):
        width = random.randint(2, 3)
        heights = tuple(random.randint(1, 3) for _ in range(width))
        area = sum(heights)
        if 3 <= area <= 6:
            return heights
    raise RuntimeError("failed to sample a small shape")


def _random_unmatched_heights(existing: list[tuple[int, ...]]) -> tuple[int, ...]:
    for _ in range(500):
        width = random.randint(2, 5)
        heights = tuple(random.randint(1, 5) for _ in range(width))
        area = sum(heights)
        if not (3 <= area <= 16):
            continue
        if _shape_key(heights) in {_shape_key(x) for x in existing}:
            continue
        if _related_to_any(heights, existing):
            continue
        return heights
    raise RuntimeError("failed to sample an unmatched shape")


def _near_miss_from_large(large: tuple[int, ...], existing: list[tuple[int, ...]]) -> tuple[int, ...]:
    base = list(large)
    for _ in range(300):
        idx = random.randrange(len(base))
        delta = random.choice((-1, 1))
        candidate = base.copy()
        candidate[idx] += delta
        if candidate[idx] < 1 or candidate[idx] > 7:
            continue
        heights = tuple(candidate)
        if _shape_key(heights) in {_shape_key(x) for x in existing}:
            continue
        if _related_to_any(heights, existing):
            continue
        return heights
    raise RuntimeError("failed to build a near-miss decoy")


def _build_objects() -> list[dict]:
    n_pairs = random.choice((1, 2))
    objects: list[dict] = []
    existing: list[tuple[int, ...]] = []

    for pair_id in range(n_pairs):
        for _ in range(300):
            small = _random_small_heights()
            large = _scale_heights2(small)
            if _shape_key(small) in {_shape_key(x) for x in existing}:
                continue
            if _shape_key(large) in {_shape_key(x) for x in existing}:
                continue
            if _related_to_any(small, existing) or _related_to_any(large, existing):
                continue
            objects.append({"heights": small, "pair_id": pair_id, "role": "small"})
            objects.append({"heights": large, "pair_id": pair_id, "role": "large"})
            existing.extend([small, large])
            break
        else:
            raise RuntimeError("failed to sample a paired shape set")

    decoy_source = random.choice([obj["heights"] for obj in objects if obj["role"] == "large"])
    near_miss = _near_miss_from_large(decoy_source, existing)
    objects.append({"heights": near_miss, "pair_id": None, "role": "near_miss"})
    existing.append(near_miss)

    extra_unmatched = random.choice((0, 1)) if n_pairs == 1 else 0
    for _ in range(extra_unmatched):
        unmatched = _random_unmatched_heights(existing)
        objects.append({"heights": unmatched, "pair_id": None, "role": "unmatched"})
        existing.append(unmatched)

    random.shuffle(objects)
    colors = random.sample(INPUT_PALETTE, len(objects))
    for obj, color in zip(objects, colors):
        obj["color"] = color
        obj["mask"] = _mask_from_heights(obj["heights"])
    return objects


def _settled_scene(objects: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    widths = [obj["mask"].shape[1] for obj in objects]
    heights = [obj["mask"].shape[0] for obj in objects]
    max_h = max(heights)

    gaps = [random.randint(1, 3) for _ in range(len(objects) + 1)]
    W = sum(widths) + sum(gaps)
    H = max(max_h + random.randint(4, 8), 10)
    if W > 30 or H > 30:
        raise RuntimeError("scene exceeds ARC size limits")

    settled = np.zeros((H, W), dtype=int)
    expected = np.zeros((H, W), dtype=int)
    cursor = gaps[0]
    for idx, obj in enumerate(objects):
        mask = obj["mask"]
        h, w = mask.shape
        r0 = H - h
        c0 = cursor
        obj["bbox"] = (r0, c0, H - 1, c0 + w - 1)
        obj["index"] = idx
        obj["columns"] = list(range(c0, c0 + w))
        settled_slice = settled[r0 : r0 + h, c0 : c0 + w]
        settled_slice[mask] = obj["color"]
        expected_slice = expected[r0 : r0 + h, c0 : c0 + w]
        expected_slice[mask] = MATCHED_COLOR if obj["pair_id"] is not None else UNMATCHED_COLOR
        cursor += w + gaps[idx + 1]

    return settled, expected


def _scatter_rows(H: int, count: int) -> list[int]:
    for _ in range(300):
        rows = sorted(random.sample(range(max(1, H - 1)), count))
        if rows == list(range(H - count, H)):
            continue
        if count == 1 or any(rows[i + 1] - rows[i] > 1 for i in range(count - 1)):
            return rows
    raise RuntimeError("failed to scatter a column")


def _input_from_objects(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    H, W = shape
    grid = np.zeros((H, W), dtype=int)
    for obj in objects:
        heights = obj["heights"]
        for dc, count in enumerate(heights):
            rows = _scatter_rows(H, count)
            col = obj["columns"][dc]
            for r in rows:
                grid[r, col] = obj["color"]
    return grid


def _pairs_by_color(components: list[dict], pairs: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for i, j in pairs:
        a = int(components[i]["color"])
        b = int(components[j]["color"])
        out.add((min(a, b), max(a, b)))
    return out


def _validate_semantics(inp: np.ndarray, settled: np.ndarray, expected: np.ndarray, objects: list[dict]) -> bool:
    if not np.array_equal(_gravity_settle(inp), settled):
        return False

    settled_components = _find_components(settled)
    if len(settled_components) != len(objects):
        return False

    per_color = defaultdict(int)
    for comp in settled_components:
        per_color[int(comp["color"])] += 1
    if any(count != 1 for count in per_color.values()):
        return False

    intended_pairs = set()
    color_to_pair = {}
    for obj in objects:
        if obj["pair_id"] is not None:
            if obj["pair_id"] in color_to_pair:
                intended_pairs.add(tuple(sorted((color_to_pair[obj["pair_id"]], obj["color"]))))
            else:
                color_to_pair[obj["pair_id"]] = obj["color"]

    actual_pairs = _pairs_by_color(settled_components, _choose_pairs(settled_components))
    if actual_pairs != intended_pairs:
        return False

    raw_components = _find_components(inp)
    if _choose_pairs(raw_components):
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    matched_mask = solved == MATCHED_COLOR
    unmatched_mask = solved == UNMATCHED_COLOR
    if not matched_mask.any() or not unmatched_mask.any():
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    try:
        objects = _build_objects()
        settled, expected = _settled_scene(objects)
        inp = _input_from_objects(settled.shape, objects)
        if not _validate_semantics(inp, settled, expected, objects):
            return None
        return inp
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
