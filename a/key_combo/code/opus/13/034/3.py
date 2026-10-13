from __future__ import annotations

from collections import Counter
import random

import numpy as np


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_ANOMALY_COLOR = 5
_PAIR_BRANCHES = ("same_mask", "same_color", "both")
_GEN_COUNTER = 0
_MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    pts = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in _DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(pts)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            r0, c0 = coords_arr.min(axis=0)
            r1, c1 = coords_arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _signature(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(map(int, row)) for row in np.asarray(arr, dtype=int))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.asarray(mask, dtype=bool))


def _center_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _build_output(mask: np.ndarray, pair_infos: list[dict], *, use_average: bool) -> np.ndarray:
    order_key = "avg_center" if use_average else "min_pos"
    ordered_pairs = sorted(pair_infos, key=lambda info: info[order_key])
    open_positions = sorted((int(r), int(c)) for r, c in np.argwhere(mask))
    out = np.zeros((9, 9), dtype=int)
    for (mr, mc), info in zip(open_positions, ordered_pairs):
        block = np.where(mask, int(info["color"]), 0)
        out[3 * mr : 3 * mr + 3, 3 * mc : 3 * mc + 3] = block
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique singleton sprite as a 3x3 Kronecker kernel and color each active copy by the repeated-pair order.

    Input structure:
    - The background is black.
    - Foreground consists of several separated 4-connected single-color sprites.
    - Every sprite has a 3x3 bounding box crop; exactly one exact colored crop appears once and every other exact colored crop appears exactly twice.

    Rule:
    - Crop each sprite to its 3x3 bounding box and count exact crop signatures, including black holes inside the box.
    - Select the only singleton crop as the anomaly sprite and convert it to a 3x3 binary mask M.
    - The number of on-cells in M equals the number of repeated crop-pairs.
    - Build a 9x9 output by replacing every on-cell of M with a copy of M and every off-cell with a black 3x3 block.
    - Traverse the on-cells of M in row-major reading order. Traverse the repeated pairs by the row-major order of the average of their two sprite centers. Color the i-th copied block with the i-th pair color.

    Key insight:
    - The singleton sprite is not the answer by itself; its shape provides the expansion kernel, while the duplicated sprites supply the ordered colors that fill the repeated kernel copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components

    by_signature: dict[tuple[tuple[int, ...], ...], list[dict]] = {}
    for comp in components:
        crop = _crop_bbox(g, comp["bbox"])
        assert crop.shape == (3, 3), "expected every sprite bbox to be 3x3"
        sig = _signature(crop)
        comp["crop"] = crop
        comp["signature"] = sig
        by_signature.setdefault(sig, []).append(comp)

    singleton_groups = [group for group in by_signature.values() if len(group) == 1]
    repeated_groups = [group for group in by_signature.values() if len(group) == 2]
    assert len(singleton_groups) == 1, "expected exactly one singleton crop"
    assert len(repeated_groups) >= 1
    assert len(repeated_groups) + 1 == len(by_signature), "non-singletons must all be exact pairs"

    anomaly = singleton_groups[0][0]
    mask = anomaly["crop"] != int(bg)
    assert mask.shape == (3, 3)
    k = int(mask.sum())
    assert len(repeated_groups) == k, "mask on-count must equal the number of repeated pairs"

    pair_infos: list[dict] = []
    for group in repeated_groups:
        colors = {int(comp["color"]) for comp in group}
        assert len(colors) == 1, "each repeated signature should be monochrome"
        centers = [_center_from_bbox(comp["bbox"]) for comp in group]
        avg_r = sum(rc[0] for rc in centers) / 2.0
        avg_c = sum(rc[1] for rc in centers) / 2.0
        min_pos = min((int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in group)
        pair_infos.append(
            {
                "color": int(group[0]["color"]),
                "avg_center": (avg_r, avg_c),
                "min_pos": min_pos,
            }
        )

    avg_keys = [info["avg_center"] for info in pair_infos]
    assert len(avg_keys) == len(set(avg_keys)), "pair average centers must be unique"
    return _build_output(mask, pair_infos, use_average=True)


def _enumerate_full_bbox_masks() -> dict[int, list[np.ndarray]]:
    masks_by_area: dict[int, list[np.ndarray]] = {}
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        if _mask_bbox(mask) != (0, 0, 2, 2):
            continue
        if not _is_connected(mask):
            continue
        masks_by_area.setdefault(int(mask.sum()), []).append(mask.copy())
    return masks_by_area


_MASKS_BY_AREA = _enumerate_full_bbox_masks()
_ALL_MASKS = [mask.copy() for masks in _MASKS_BY_AREA.values() for mask in masks]


def _sample_distinct_colors(k: int, branch: str) -> list[int]:
    palette = list(range(1, 10))
    if branch in {"same_color", "both"}:
        others = random.sample([c for c in palette if c != _ANOMALY_COLOR], k - 1)
        colors = [_ANOMALY_COLOR, *others]
        random.shuffle(colors)
        return colors
    return random.sample([c for c in palette if c != _ANOMALY_COLOR], k)


def _same_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return bool(np.array_equal(mask_a, mask_b))


def _choose_pair_masks(anomaly_mask: np.ndarray, k: int, branch: str, pair_colors: list[int]) -> list[np.ndarray] | None:
    pair_masks: list[np.ndarray | None] = [None] * k
    available_same_area = [m for m in _MASKS_BY_AREA[k] if not _same_mask(m, anomaly_mask)]

    same_mask_idx = None
    if branch in {"same_mask", "both"}:
        choices = [i for i, color in enumerate(pair_colors) if color != _ANOMALY_COLOR]
        if not choices:
            return None
        same_mask_idx = random.choice(choices)
        pair_masks[same_mask_idx] = anomaly_mask.copy()

    same_color_idx = None
    if branch in {"same_color", "both"}:
        choices = [i for i, color in enumerate(pair_colors) if color == _ANOMALY_COLOR]
        if not choices:
            return None
        same_color_idx = random.choice(choices)
        if not available_same_area:
            return None
        pair_masks[same_color_idx] = random.choice(available_same_area).copy()

    if all(mask is None for mask in pair_masks):
        if not available_same_area:
            return None
        forced_idx = random.randrange(k)
        pair_masks[forced_idx] = random.choice(available_same_area).copy()

    pool = _ALL_MASKS[:]
    random.shuffle(pool)
    usage: Counter[tuple[tuple[int, ...], ...]] = Counter(
        _mask_key(mask) for mask in pair_masks if mask is not None
    )

    for idx in range(k):
        if pair_masks[idx] is not None:
            continue
        for cand in pool:
            key = _mask_key(cand)
            if usage[key] >= 2:
                continue
            pair_masks[idx] = cand.copy()
            usage[key] += 1
            break
        if pair_masks[idx] is None:
            return None

    out = [mask for mask in pair_masks if mask is not None]
    if len(out) != k:
        return None
    return out


def _make_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), 0).astype(int)


def _can_place(existing: list[tuple[int, int]], r: int, c: int) -> bool:
    return all(abs(r - rr) > 3 or abs(c - cc) > 3 for rr, cc in existing)


def _place_objects(objects: list[dict], H: int, W: int) -> bool:
    candidates = [(r, c) for r in range(H - 2) for c in range(W - 2)]
    for _ in range(120):
        existing: list[tuple[int, int]] = []
        for obj in objects:
            obj["pos"] = None
        order = list(range(len(objects)))
        random.shuffle(order)
        random.shuffle(candidates)
        ok = True
        for idx in order:
            placed = False
            for r, c in candidates:
                if _can_place(existing, r, c):
                    existing.append((r, c))
                    objects[idx]["pos"] = (r, c)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return True
    return False


def _render_grid(objects: list[dict], H: int, W: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for obj in objects:
        r0, c0 = obj["pos"]
        patch = obj["patch"]
        grid[r0 : r0 + 3, c0 : c0 + 3] = patch
    return grid


def _sample_branch() -> str:
    global _GEN_COUNTER
    branch = _PAIR_BRANCHES[_GEN_COUNTER % len(_PAIR_BRANCHES)]
    _GEN_COUNTER += 1
    return branch


def _make_attempt(branch: str) -> np.ndarray | None:
    k = random.choice([5, 6, 7])
    anomaly_mask = random.choice(_MASKS_BY_AREA[k]).copy()
    pair_colors = _sample_distinct_colors(k, branch)
    pair_masks = _choose_pair_masks(anomaly_mask, k, branch, pair_colors)
    if pair_masks is None:
        return None

    signatures = Counter()
    anomaly_sig = _signature(_make_sprite(anomaly_mask, _ANOMALY_COLOR))
    signatures[anomaly_sig] += 1

    objects: list[dict] = [
        {
            "kind": "anomaly",
            "pair_idx": None,
            "patch": _make_sprite(anomaly_mask, _ANOMALY_COLOR),
            "mask": anomaly_mask.copy(),
            "color": _ANOMALY_COLOR,
        }
    ]

    for idx, (mask, color) in enumerate(zip(pair_masks, pair_colors)):
        patch = _make_sprite(mask, color)
        sig = _signature(patch)
        signatures[sig] += 2
        objects.append(
            {
                "kind": "pair",
                "pair_idx": idx,
                "patch": patch,
                "mask": mask.copy(),
                "color": int(color),
            }
        )
        objects.append(
            {
                "kind": "pair",
                "pair_idx": idx,
                "patch": patch.copy(),
                "mask": mask.copy(),
                "color": int(color),
            }
        )

    if list(signatures.values()).count(1) != 1:
        return None
    if any(count not in {1, 2} for count in signatures.values()):
        return None

    H = random.randint(22, 30)
    W = random.randint(22, 30)
    if not _place_objects(objects, H, W):
        return None

    grid = _render_grid(objects, H, W)
    components = _find_components(grid, 0)
    if len(components) != 1 + 2 * k:
        return None

    by_pair: dict[int, list[tuple[int, int]]] = {}
    for obj in objects:
        if obj["kind"] != "pair":
            continue
        by_pair.setdefault(int(obj["pair_idx"]), []).append(obj["pos"])
    if any(len(v) != 2 for v in by_pair.values()):
        return None

    pair_infos = []
    for idx in range(k):
        positions = by_pair[idx]
        centers = [(r + 1.0, c + 1.0) for r, c in positions]
        pair_infos.append(
            {
                "color": int(pair_colors[idx]),
                "avg_center": (
                    (centers[0][0] + centers[1][0]) / 2.0,
                    (centers[0][1] + centers[1][1]) / 2.0,
                ),
                "min_pos": min(positions),
            }
        )
    avg_keys = [info["avg_center"] for info in pair_infos]
    if len(avg_keys) != len(set(avg_keys)):
        return None

    correct = _build_output(anomaly_mask, pair_infos, use_average=True)
    wrong = _build_output(anomaly_mask, pair_infos, use_average=False)
    if np.array_equal(correct, wrong):
        return None

    has_same_mask_decoy = any(_same_mask(mask, anomaly_mask) for mask in pair_masks)
    has_same_color_decoy = any(color == _ANOMALY_COLOR for color in pair_colors)
    if branch == "same_mask" and not has_same_mask_decoy:
        return None
    if branch == "same_color" and not has_same_color_decoy:
        return None
    if branch == "both" and not (has_same_mask_decoy and has_same_color_decoy):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, correct):
        return None
    return grid


def generate_input() -> np.ndarray:
    branch = _sample_branch()
    for _ in range(_MAX_ATTEMPTS):
        grid = _make_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input for branch {branch}")
