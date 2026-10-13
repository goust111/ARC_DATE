from __future__ import annotations

from collections import defaultdict, deque
from importlib.machinery import SourceFileLoader
from itertools import combinations
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_7_006_3",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
get_bg = _ARC_UTILS.get_bg
find_components = _ARC_UTILS.find_components
crop_bbox = _ARC_UTILS.crop_bbox
paste = _ARC_UTILS.paste

NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
MAX_GEN_ATTEMPTS = 500


def _mirror_cell(r: int, c: int, h: int, w: int) -> tuple[int, int]:
    return h - 1 - r, w - 1 - c


def _mask_point_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask.astype(bool), mask[::-1, ::-1].astype(bool)))


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        mask[0].any()
        and mask[-1].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _coords_to_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _sprite_patch(
    grid: np.ndarray,
    component: dict,
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    bbox = tuple(map(int, component["bbox"]))
    patch = np.full(crop_bbox(grid, bbox).shape, int(bg), dtype=int)
    mask = _coords_to_mask(component["coords"], bbox)
    patch[mask] = int(component["color"])
    return patch, mask


def _extract_sprites(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    sprites = []
    for component in components:
        patch, mask = _sprite_patch(g, component, bg)
        sprites.append(
            {
                "color": int(component["color"]),
                "bbox": tuple(map(int, component["bbox"])),
                "shape": tuple(map(int, patch.shape)),
                "patch": patch,
                "mask": mask,
            }
        )
    return bg, sprites


def _find_candidate_pairs(sprites: list[dict], bg: int) -> list[dict]:
    by_shape: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, sprite in enumerate(sprites):
        by_shape[sprite["shape"]].append(idx)

    candidates = []
    for shape, indices in by_shape.items():
        if len(indices) < 2:
            continue
        for i, j in combinations(indices, 2):
            mask_a = sprites[i]["mask"]
            mask_b = sprites[j]["mask"]
            if np.any(mask_a & mask_b):
                continue
            union_mask = mask_a | mask_b
            if not _mask_point_symmetric(union_mask):
                continue
            union = np.full(shape, int(bg), dtype=int)
            union[mask_a] = int(sprites[i]["color"])
            union[mask_b] = int(sprites[j]["color"])
            candidates.append(
                {
                    "pair": (int(i), int(j)),
                    "shape": shape,
                    "union": union,
                    "union_mask": union_mask,
                }
            )
    return candidates


def _detect_expected_output(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    bg, sprites = _extract_sprites(grid)
    candidates = _find_candidate_pairs(sprites, bg)
    assert len(candidates) == 1, "expected exactly one complementary point-symmetric pair"
    return candidates[0]["union"].copy(), {"bg": bg, "sprites": sprites, "candidate": candidates[0]}


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Find the unique complementary sprite pair whose aligned union becomes point-symmetric, and return the merged crop.

    Input structure:
    - The background is the dominant color and the foreground consists of several separated monochromatic sprites.
    - Multiple sprites share the same tight bounding-box size, so size alone does not identify the answer.
    - Exactly one same-size pair becomes a non-overlapping 180-degree point-symmetric pattern when their bbox crops are aligned.

    Rule:
    - Extract every sprite as its tight bbox crop while preserving internal background cells.
    - For every pair of sprites with the same bbox size, align their crops by bbox, reject overlaps, and test whether the union of occupied cells is 180-degree point-symmetric.
    - Return the unique valid union crop, keeping each sprite's original color.

    Key insight:
    - The answer is not a repeated sprite: it is the only pair of complementary fragments whose symmetry appears only after aligned overlay, while identical-shape duplicates are decoys because they overlap instead of complementing.
    """
    out, _meta = _detect_expected_output(np.asarray(input_grid, dtype=int))
    return out


def _random_side_anchor(h: int, w: int) -> list[tuple[int, int]]:
    return [
        (0, random.randint(1, w - 2)),
        (h - 1, random.randint(1, w - 2)),
        (random.randint(1, h - 2), 0),
        (random.randint(1, h - 2), w - 1),
    ]


def _has_mirror_conflict(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in cells:
        mr, mc = _mirror_cell(r, c, h, w)
        if (mr, mc) == (r, c):
            return True
        if (mr, mc) in cells:
            return True
    return False


def _bfs_path(
    h: int,
    w: int,
    sources: set[tuple[int, int]],
    target: tuple[int, int],
    used: set[tuple[int, int]],
    *,
    antisymmetric: bool,
) -> list[tuple[int, int]] | None:
    q = deque(sources)
    parent = {cell: None for cell in sources}
    while q:
        cur = q.popleft()
        if cur == target:
            break
        steps = NEIGHBORS8[:]
        random.shuffle(steps)
        for dr, dc in steps:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in parent:
                continue
            if nxt not in sources and nxt in used:
                continue
            if antisymmetric:
                mr, mc = _mirror_cell(nr, nc, h, w)
                if (mr, mc) == nxt:
                    continue
                if nxt not in sources and (mr, mc) in used:
                    continue
            parent[nxt] = cur
            q.append(nxt)
    if target not in parent:
        return None

    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()

    path_cells = set(path)
    if antisymmetric and _has_mirror_conflict(path_cells | used, h, w):
        return None
    return path


def _grow_connected_mask(
    h: int,
    w: int,
    *,
    antisymmetric: bool,
    force_not_symmetric: bool,
) -> np.ndarray | None:
    if h < 4 or w < 4:
        return None

    total_pairs = (h * w) // 2
    min_area = max(6, h + w - 3)
    max_area = max(min_area, min(h * w - 2, total_pairs - 1 if antisymmetric else int(0.7 * h * w)))
    if antisymmetric and max_area < min_area:
        return None

    for _ in range(200):
        anchors = _random_side_anchor(h, w)
        if len(set(anchors)) < 4:
            continue
        if antisymmetric and _has_mirror_conflict(set(anchors), h, w):
            continue

        order = anchors[:]
        random.shuffle(order)
        cells = {order[0]}
        ok = True
        for anchor in order[1:]:
            path = _bfs_path(h, w, cells, anchor, cells, antisymmetric=antisymmetric)
            if path is None:
                ok = False
                break
            cells.update(path)
        if not ok:
            continue

        target_area = random.randint(min_area, max_area)
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS8:
                    nr, nc = r + dr, c + dc
                    nxt = (nr, nc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if nxt in cells:
                        continue
                    if antisymmetric:
                        mr, mc = _mirror_cell(nr, nc, h, w)
                        if (mr, mc) == nxt or (mr, mc) in cells:
                            continue
                    frontier.append(nxt)
            if not frontier:
                break

            def _score(cell: tuple[int, int]) -> tuple[int, float]:
                r, c = cell
                hits = int(r == 0) + int(r == h - 1) + int(c == 0) + int(c == w - 1)
                return hits, random.random()

            frontier = sorted(set(frontier), key=_score, reverse=True)
            cells.add(frontier[0])
            if len(cells) >= min_area and _touches_all_sides(_cells_to_mask(cells, h, w)) and random.random() < 0.18:
                break

        mask = _cells_to_mask(cells, h, w)
        if not _touches_all_sides(mask):
            continue
        if antisymmetric and _has_mirror_conflict(cells, h, w):
            continue
        if force_not_symmetric and _mask_point_symmetric(mask):
            continue
        return mask
    return None


def _cells_to_mask(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _gen_answer_pair(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    mask_a = _grow_connected_mask(h, w, antisymmetric=True, force_not_symmetric=False)
    if mask_a is None:
        return None
    mask_b = mask_a[::-1, ::-1].copy()
    if np.any(mask_a & mask_b):
        return None
    if not (_touches_all_sides(mask_a) and _touches_all_sides(mask_b)):
        return None
    union = mask_a | mask_b
    if not _mask_point_symmetric(union):
        return None
    return mask_a, mask_b


def _gen_decoy_mask(shape: tuple[int, int], forbidden: list[np.ndarray]) -> np.ndarray | None:
    h, w = shape
    for _ in range(250):
        mask = _grow_connected_mask(h, w, antisymmetric=False, force_not_symmetric=True)
        if mask is None:
            continue
        if any(np.array_equal(mask, other) for other in forbidden):
            continue
        return mask
    return None


def _make_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _sample_other_sizes(answer_shape: tuple[int, int], count: int) -> list[tuple[int, int]]:
    choices = []
    for h in range(3, 8):
        for w in range(3, 8):
            shape = (h, w)
            if shape != answer_shape:
                choices.append(shape)
    random.shuffle(choices)
    out = []
    used = {answer_shape}
    for shape in choices:
        if shape in used:
            continue
        used.add(shape)
        out.append(shape)
        if len(out) == count:
            break
    return out


def _rect_is_clear(
    rect: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int]],
    gap: int = 1,
) -> bool:
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0):
            return False
    return True


def _place_patches(
    canvas_shape: tuple[int, int],
    patches: list[np.ndarray],
    bg: int,
) -> np.ndarray | None:
    H, W = canvas_shape
    grid = np.full((H, W), int(bg), dtype=int)
    placed: list[tuple[int, int, int, int]] = []
    order = list(range(len(patches)))
    order.sort(key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        h, w = map(int, patch.shape)
        candidates = []
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                if _rect_is_clear(rect, placed, gap=1):
                    candidates.append((r0, c0, rect))
        if not candidates:
            return None
        r0, c0, rect = random.choice(candidates)
        paste(grid, patch, r0, c0, transparent=bg)
        placed.append(rect)
    return grid


def _validate_instance(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    expected, meta = _detect_expected_output(grid)
    bg = meta["bg"]
    sprites = meta["sprites"]
    candidate = meta["candidate"]
    answer_i, answer_j = candidate["pair"]
    answer_shape = candidate["shape"]

    assert max(grid.shape) <= 30
    assert max(expected.shape) <= 30
    assert np.count_nonzero(expected != bg) >= 6
    assert len(np.unique(expected[expected != bg])) == 2

    same_shape_count = sum(1 for sprite in sprites if sprite["shape"] == answer_shape)
    assert same_shape_count >= 4, "need answer pair plus same-size decoy sprites"

    decoy_pairs = []
    for i, j in combinations(range(len(sprites)), 2):
        if sprites[i]["shape"] != answer_shape or sprites[j]["shape"] != answer_shape:
            continue
        if {i, j} == {answer_i, answer_j}:
            continue
        if np.array_equal(sprites[i]["mask"], sprites[j]["mask"]) and not _mask_point_symmetric(sprites[i]["mask"]):
            decoy_pairs.append((i, j))
    assert decoy_pairs, "missing identical non-symmetric decoy pair"

    actual = main(grid)
    assert isinstance(actual, np.ndarray) and actual.ndim == 2
    assert np.array_equal(actual, expected), "main() does not match the independent geometric check"
    return expected, meta


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    available_colors = [color for color in range(10) if color != bg]

    parity_shapes = [
        (4, 5),
        (5, 4),
        (5, 5),
        (4, 6),
        (6, 4),
        (5, 6),
        (6, 5),
        (6, 6),
        (7, 5),
        (5, 7),
    ]
    answer_shape = random.choice(parity_shapes)
    answer_pair = _gen_answer_pair(answer_shape)
    if answer_pair is None:
        return None
    answer_a, answer_b = answer_pair

    decoy = _gen_decoy_mask(answer_shape, [answer_a, answer_b])
    if decoy is None:
        return None

    masks = [answer_a, answer_b, decoy, decoy.copy()]

    if random.random() < 0.45:
        extra_same = _gen_decoy_mask(answer_shape, [answer_a, answer_b, decoy])
        if extra_same is None:
            return None
        masks.append(extra_same)

    other_count = random.randint(1, 3)
    for shape in _sample_other_sizes(answer_shape, other_count):
        mask = _gen_decoy_mask(shape, [])
        if mask is None:
            return None
        masks.append(mask)

    if len(masks) > len(available_colors):
        return None
    colors = random.sample(available_colors, len(masks))
    patches = [_make_patch(mask, color, bg) for mask, color in zip(masks, colors)]

    H = random.randint(max(18, answer_shape[0] * 2 + 5), 30)
    W = random.randint(max(18, answer_shape[1] * 2 + 5), 30)
    grid = _place_patches((H, W), patches, bg)
    if grid is None:
        return None

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-pair ARC input")
