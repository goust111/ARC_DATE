import random

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 500


def _background(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _prefix2(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(np.int16)
    return np.pad(arr, ((1, 0), (1, 0)), constant_values=0).cumsum(0).cumsum(1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _row_prefix(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(np.int16), ((0, 0), (1, 0)), constant_values=0).cumsum(1)


def _col_prefix(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(np.int16), ((1, 0), (0, 0)), constant_values=0).cumsum(0)


def _row_count(ps: np.ndarray, r: int, c0: int, c1: int) -> int:
    return int(ps[r, c1 + 1] - ps[r, c0])


def _col_count(ps: np.ndarray, c: int, r0: int, r1: int) -> int:
    return int(ps[r1 + 1, c] - ps[r0, c])


def _edge_count(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    width = c1 - c0 + 1
    height = r1 - r0 + 1
    total = 0
    if int(mask[r0, c0 : c1 + 1].sum()) == width:
        total += 1
    if int(mask[r1, c0 : c1 + 1].sum()) == width:
        total += 1
    if int(mask[r0 : r1 + 1, c0].sum()) == height:
        total += 1
    if int(mask[r0 : r1 + 1, c1].sum()) == height:
        total += 1
    return total


def _parse_order(row: np.ndarray, bg: int) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for value in map(int, row):
        if value == bg or value in seen:
            continue
        seen.add(value)
        order.append(value)
    return order


def _infer_bg_and_order(grid: np.ndarray) -> tuple[int, list[int]]:
    row = [int(v) for v in grid[-1]]
    counts: dict[int, int] = {}
    for value in row:
        counts[value] = counts.get(value, 0) + 1

    repeated = [value for value, count in counts.items() if count > 1]
    if repeated:
        bg = max(repeated, key=lambda value: counts[value])
        return int(bg), _parse_order(np.array(row, dtype=int), int(bg))

    deduped: list[int] = []
    seen: set[int] = set()
    for value in row:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)

    scene = np.asarray(grid[:-1], dtype=int)
    scene_colors = set(map(int, np.unique(scene)))
    bg_candidates = list(scene_colors - set(deduped))
    if bg_candidates:
        bg = max(bg_candidates, key=lambda value: int((scene == value).sum()))
        return int(bg), deduped

    bg = _background(scene)
    return int(bg), _parse_order(np.array(row, dtype=int), int(bg))


def _reconstruct_bbox(visible: np.ndarray, upper_mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(visible)
    assert pts.size > 0, "each rectangle must have visible cells"
    h, w = visible.shape
    min_r, min_c = pts.min(axis=0)
    max_r, max_c = pts.max(axis=0)
    bad = ~(visible | upper_mask)
    bad_ps = _prefix2(bad)
    row_ps = _row_prefix(visible)
    col_ps = _col_prefix(visible)

    candidates: list[tuple[int, int, int, int, int, int]] = []
    for r0 in range(int(min_r) + 1):
        for r1 in range(int(max_r), h):
            height = r1 - r0 + 1
            for c0 in range(int(min_c) + 1):
                for c1 in range(int(max_c), w):
                    if _rect_sum(bad_ps, r0, c0, r1, c1) != 0:
                        continue
                    width = c1 - c0 + 1
                    edges = 0
                    if _row_count(row_ps, r0, c0, c1) == width:
                        edges += 1
                    if _row_count(row_ps, r1, c0, c1) == width:
                        edges += 1
                    if _col_count(col_ps, c0, r0, r1) == height:
                        edges += 1
                    if _col_count(col_ps, c1, r0, r1) == height:
                        edges += 1
                    if edges < 2:
                        continue
                    area = width * height
                    hidden = area - int(visible.sum())
                    candidates.append((edges, area, hidden, r0, c0, r1, c1))
    assert candidates, "failed to reconstruct any rectangle"
    candidates.sort(reverse=True)
    _edges, _area, _hidden, r0, c0, r1, c1 = candidates[0]
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct overlapped solid rectangles and keep only the portions hidden by higher layers.

    Input structure:
    - A white-background grid contains several distinct-color solid axis-aligned rectangles in the top area.
    - Rectangles overlap, and each overlap shows only the topmost rectangle color.
    - The bottom row is a left-to-right color order strip giving the drawing order from lowest layer to highest layer.

    Rule:
    - Remove the bottom order strip and solve only on the scene above it.
    - Read the bottom strip to recover the stacking order, then process colors from top to bottom.
    - For each rectangle, infer its full bounding box from its visible cells plus the already reconstructed higher-layer occlusion mask; the correct box is the one whose non-occluded cells exactly match the visible color and that still has at least two fully visible edges.
    - Output a white background of the scene size and draw only the cells where each rectangle is covered by any higher-layer rectangle, using the hidden rectangle's own color.

    Key insight:
    - The visible pixels of a lower rectangle are exactly its full rectangle minus the union of higher rectangles, so once the order row is decoded the missing parts can be recovered by searching for the only box consistent with the visible edges and known occlusions.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] <= 1:
        return g.copy()

    bg, order = _infer_bg_and_order(g)
    scene = g[:-1]
    if not order:
        return scene.copy()

    rects: dict[int, tuple[int, int, int, int]] = {}
    upper_mask = np.zeros(scene.shape, dtype=bool)
    for color in reversed(order):
        visible = scene == int(color)
        assert np.any(visible), "every ordered color must appear in the scene"
        bbox = _reconstruct_bbox(visible, upper_mask)
        rects[int(color)] = bbox
        r0, c0, r1, c1 = bbox
        upper_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    out = np.full(scene.shape, bg, dtype=int)
    higher_union = np.zeros(scene.shape, dtype=bool)
    for color in reversed(order):
        bbox = rects[int(color)]
        mask = _bbox_mask(scene.shape, bbox)
        hidden = mask & higher_union
        out[hidden] = int(color)
        higher_union |= mask
    return out


def _intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)


def _render_scene(shape: tuple[int, int], rects: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for bbox, color in zip(rects, colors):
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _oracle_output(shape: tuple[int, int], rects: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    higher_union = np.zeros(shape, dtype=bool)
    for bbox, color in zip(reversed(rects), reversed(colors)):
        mask = _bbox_mask(shape, bbox)
        hidden = mask & higher_union
        out[hidden] = int(color)
        higher_union |= mask
    return out


def _visible_mask(rects: list[tuple[int, int, int, int]], index: int, shape: tuple[int, int]) -> np.ndarray:
    mask = _bbox_mask(shape, rects[index])
    upper = np.zeros(shape, dtype=bool)
    for bbox in rects[index + 1 :]:
        upper |= _bbox_mask(shape, bbox)
    return mask & ~upper


def _legend_row(width: int, colors: list[int], rng: random.Random) -> np.ndarray:
    row = np.full(width, BG, dtype=int)
    start = rng.randint(0, width - len(colors))
    row[start : start + len(colors)] = np.array(colors, dtype=int)
    return row


def _random_bbox(h: int, w: int, rng: random.Random, margin: int = 0) -> tuple[int, int, int, int]:
    max_h = min(9, h - 2 * margin)
    max_w = min(10, w - 2 * margin)
    rect_h = rng.randint(4, max_h)
    rect_w = rng.randint(4, max_w)
    r0 = rng.randint(margin, h - margin - rect_h)
    c0 = rng.randint(margin, w - margin - rect_w)
    return r0, c0, r0 + rect_h - 1, c0 + rect_w - 1


def _sample_corner_child(
    parent: tuple[int, int, int, int],
    scene_h: int,
    scene_w: int,
    mode: str,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    pr0, pc0, pr1, pc1 = parent
    ph = pr1 - pr0 + 1
    pw = pc1 - pc0 + 1
    for _ in range(50):
        child_h = rng.randint(3, min(8, scene_h))
        child_w = rng.randint(3, min(8, scene_w))
        max_oh = min(ph - 1, child_h - 1)
        max_ow = min(pw - 1, child_w - 1)
        if max_oh < 1 or max_ow < 1:
            continue
        oh = rng.randint(1, max_oh)
        ow = rng.randint(1, max_ow)
        if mode == "tl":
            r1 = pr0 + oh - 1
            c1 = pc0 + ow - 1
            r0 = r1 - child_h + 1
            c0 = c1 - child_w + 1
        elif mode == "tr":
            r1 = pr0 + oh - 1
            c0 = pc1 - ow + 1
            r0 = r1 - child_h + 1
            c1 = c0 + child_w - 1
        elif mode == "bl":
            r0 = pr1 - oh + 1
            c1 = pc0 + ow - 1
            r1 = r0 + child_h - 1
            c0 = c1 - child_w + 1
        else:
            r0 = pr1 - oh + 1
            c0 = pc1 - ow + 1
            r1 = r0 + child_h - 1
            c1 = c0 + child_w - 1
        if r0 < 0 or c0 < 0 or r1 >= scene_h or c1 >= scene_w:
            continue
        return r0, c0, r1, c1
    return None


def _sample_band_child(
    parent: tuple[int, int, int, int],
    axis: str,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    pr0, pc0, pr1, pc1 = parent
    ph = pr1 - pr0 + 1
    pw = pc1 - pc0 + 1
    if ph < 5 or pw < 5:
        return None
    for _ in range(50):
        if axis == "vertical":
            child_h = rng.randint(2, ph - 2)
            child_w = rng.randint(2, max(2, pw - 1))
            r0 = rng.randint(pr0 + 1, pr1 - child_h)
            c0 = rng.randint(pc0, pc1 - child_w + 1)
        else:
            child_h = rng.randint(2, max(2, ph - 1))
            child_w = rng.randint(2, pw - 2)
            r0 = rng.randint(pr0, pr1 - child_h + 1)
            c0 = rng.randint(pc0 + 1, pc1 - child_w)
        r1 = r0 + child_h - 1
        c1 = c0 + child_w - 1
        if r0 <= pr0 and r1 >= pr1:
            continue
        if c0 <= pc0 and c1 >= pc1:
            continue
        return r0, c0, r1, c1
    return None


def _candidate_child(
    parent: tuple[int, int, int, int],
    scene_h: int,
    scene_w: int,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    modes = ["tl", "tr", "bl", "br", "vertical", "horizontal"]
    rng.shuffle(modes)
    for mode in modes:
        if mode in {"tl", "tr", "bl", "br"}:
            bbox = _sample_corner_child(parent, scene_h, scene_w, mode, rng)
        elif mode == "vertical":
            bbox = _sample_band_child(parent, "vertical", rng)
        else:
            bbox = _sample_band_child(parent, "horizontal", rng)
        if bbox is not None:
            return bbox
    return None


def _place_rectangles(
    scene_h: int,
    scene_w: int,
    n_rects: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    cover_count = np.zeros((scene_h, scene_w), dtype=np.int8)
    first_margin = 1 if scene_h >= 10 and scene_w >= 10 else 0
    rects.append(_random_bbox(scene_h, scene_w, rng, margin=first_margin))
    r0, c0, r1, c1 = rects[0]
    cover_count[r0 : r1 + 1, c0 : c1 + 1] += 1
    overlap_count = 0

    for _ in range(1, n_rects):
        placed = False
        parent_order = list(range(len(rects)))
        rng.shuffle(parent_order)
        for parent_index in parent_order:
            parent = rects[parent_index]
            for _ in range(40):
                bbox = _candidate_child(parent, scene_h, scene_w, rng)
                if bbox is None:
                    continue
                rr0, cc0, rr1, cc1 = bbox
                if np.any(cover_count[rr0 : rr1 + 1, cc0 : cc1 + 1] >= 2):
                    continue
                overlaps = [i for i, other in enumerate(rects) if _intersection(bbox, other) is not None]
                if not overlaps or overlaps != [parent_index]:
                    continue
                rects.append(bbox)
                cover_count[rr0 : rr1 + 1, cc0 : cc1 + 1] += 1
                overlap_count += 1
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    if overlap_count == 0:
        return None
    return rects


def _valid_instance(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> bool:
    if len(set(rects)) != len(rects):
        return False
    cover = np.zeros(shape, dtype=np.int8)
    for bbox in rects:
        r0, c0, r1, c1 = bbox
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1
    if int(cover.max()) > 2:
        return False

    hidden_total = 0
    for idx, bbox in enumerate(rects):
        vis = _visible_mask(rects, idx, shape)
        if not np.any(vis):
            return False
        if _edge_count(vis, bbox) < 2:
            return False
        hidden_total += _bbox_area(bbox) - int(vis.sum())
    return hidden_total > 0


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rects = rng.choice([2, 3, 4])
    scene_h = rng.randint(9, 18)
    scene_w = rng.randint(max(12, n_rects + 3), 22)
    rects = _place_rectangles(scene_h, scene_w, n_rects, rng)
    if rects is None or not _valid_instance((scene_h, scene_w), rects):
        return None

    colors = rng.sample(COLOR_POOL, n_rects)
    scene = _render_scene((scene_h, scene_w), rects, colors)
    legend = _legend_row(scene_w, colors, rng)
    inp = np.vstack([scene, legend])
    expected = _oracle_output((scene_h, scene_w), rects, colors)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
