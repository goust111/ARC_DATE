from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = BG) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _scale2(sprite: np.ndarray) -> np.ndarray:
    return np.kron(sprite, np.ones((2, 2), dtype=int))


def _apply_geom(sprite: np.ndarray, geom: str) -> np.ndarray:
    if geom == "rot90":
        return np.rot90(sprite, 1)
    if geom == "rot180":
        return np.rot90(sprite, 2)
    if geom == "rot270":
        return np.rot90(sprite, 3)
    if geom == "flip_lr":
        return np.fliplr(sprite)
    if geom == "flip_ud":
        return np.flipud(sprite)
    if geom == "transpose":
        return sprite.T.copy()
    if geom == "anti_transpose":
        return np.fliplr(np.rot90(sprite, 1))
    raise ValueError(f"unknown geometry: {geom}")


TRANSFORMS = (
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "anti_transpose",
)


def _recolor(sprite: np.ndarray, mapping: dict[int, int]) -> np.ndarray | None:
    out = sprite.copy()
    colors = set(map(int, np.unique(sprite))) - {BG}
    if not colors.issubset(mapping):
        return None
    for src, dst in mapping.items():
        out[sprite == src] = dst
    return out


def _component_sprites(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            r0, c0 = coords_arr.min(axis=0)
            r1, c1 = coords_arr.max(axis=0)
            sprite = _crop_bbox(grid, (int(r0), int(c0), int(r1), int(c1)))
            components.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "sprite": sprite,
                    "area": int(len(coords)),
                    "shape": tuple(map(int, sprite.shape)),
                    "colors": frozenset(map(int, np.unique(sprite[sprite != BG]))),
                }
            )
    return components


def _pair_mapping(src_big_ready: np.ndarray, dst_big: np.ndarray) -> dict[int, int] | None:
    if src_big_ready.shape != dst_big.shape:
        return None
    src_mask = src_big_ready != BG
    if not np.array_equal(src_mask, dst_big != BG):
        return None
    forward: dict[int, int] = {}
    reverse: dict[int, int] = {}
    rows, cols = np.where(src_mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        src = int(src_big_ready[r, c])
        dst = int(dst_big[r, c])
        if src in forward and forward[src] != dst:
            return None
        if dst in reverse and reverse[dst] != src:
            return None
        forward[src] = dst
        reverse[dst] = src
    return forward


def _merge_maps(
    forward: dict[int, int], reverse: dict[int, int], patch: dict[int, int]
) -> tuple[dict[int, int], dict[int, int]] | None:
    out_f = dict(forward)
    out_r = dict(reverse)
    for src, dst in patch.items():
        if src in out_f and out_f[src] != dst:
            return None
        if dst in out_r and out_r[dst] != src:
            return None
        out_f[src] = dst
        out_r[dst] = src
    return out_f, out_r


def _sig(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def _solve_outputs(grid: np.ndarray) -> list[np.ndarray]:
    comps = _component_sprites(np.asarray(grid, dtype=int))
    n = len(comps)
    if n < 3 or n % 2 == 0:
        return []
    pair_count = (n - 1) // 2
    comp_ids = tuple(range(n))
    transformed_cache: dict[tuple[str, int], np.ndarray] = {}
    edge_cache: dict[tuple[str, int, int], dict[int, int] | None] = {}
    for geom in TRANSFORMS:
        for i in comp_ids:
            transformed_cache[(geom, i)] = _apply_geom(_scale2(comps[i]["sprite"]), geom)
        for i in comp_ids:
            for j in comp_ids:
                if i == j:
                    continue
                if comps[j]["area"] != 4 * comps[i]["area"]:
                    edge_cache[(geom, i, j)] = None
                    continue
                edge_cache[(geom, i, j)] = _pair_mapping(transformed_cache[(geom, i)], comps[j]["sprite"])

    outputs: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for geom in TRANSFORMS:
        for small_ids in itertools.combinations(comp_ids, pair_count + 1):
            small_ids = tuple(sorted(small_ids))
            big_ids = tuple(i for i in comp_ids if i not in small_ids)
            big_areas = Counter(int(comps[i]["area"]) for i in big_ids)
            for orphan in small_ids:
                need_areas = Counter(4 * int(comps[i]["area"]) for i in small_ids if i != orphan)
                if need_areas != big_areas:
                    continue
                active_smalls = [i for i in small_ids if i != orphan]
                candidates: dict[int, list[tuple[int, dict[int, int]]]] = {}
                viable = True
                for big_id in big_ids:
                    matches = []
                    for small_id in active_smalls:
                        mapping = edge_cache[(geom, small_id, big_id)]
                        if mapping is not None:
                            matches.append((small_id, mapping))
                    if not matches:
                        viable = False
                        break
                    candidates[big_id] = matches
                if not viable:
                    continue
                ordered_bigs = sorted(big_ids, key=lambda idx: (len(candidates[idx]), idx))
                orphan_colors = set(map(int, comps[orphan]["colors"]))

                def dfs(
                    depth: int,
                    used_smalls: set[int],
                    forward: dict[int, int],
                    reverse: dict[int, int],
                ) -> None:
                    if depth == len(ordered_bigs):
                        if not orphan_colors.issubset(forward):
                            return
                        out = _recolor(transformed_cache[(geom, orphan)], forward)
                        if out is None:
                            return
                        outputs[_sig(out)] = out
                        return
                    big_id = ordered_bigs[depth]
                    for small_id, patch in candidates[big_id]:
                        if small_id in used_smalls:
                            continue
                        merged = _merge_maps(forward, reverse, patch)
                        if merged is None:
                            continue
                        new_f, new_r = merged
                        used_smalls.add(small_id)
                        dfs(depth + 1, used_smalls, new_f, new_r)
                        used_smalls.remove(small_id)

                dfs(0, set(), {}, {})
    return list(outputs.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the common recolor-scale-geometry transform from existing pairs and apply it to the lone unmatched sprite.

    Input structure:
    - A black background contains several separated multicolor 4-connected sprites.
    - The sprites split into smaller originals and larger counterparts, with the larger ones being exactly double-sized.
    - All larger sprites share one hidden transform from their matched smaller sprite: first recolor by one global color permutation, then upscale each pixel to a 2x2 block, then apply one common rotation or mirror.

    Rule:
    - Jointly infer which sprites are the smaller originals, which are the larger images, and which small-large pairings are compatible with one shared transform.
    - Exactly one smaller sprite is left unmatched after all larger sprites are paired.
    - Apply the inferred transform to that unmatched smaller sprite and return only the transformed sprite's tight bounding box.

    Key insight:
    - No single pair is enough by itself: the shared geometry and the shared color permutation must be recovered from the consistent set of all observed pairings, and only then can the missing transformed partner be produced.
    """
    g = np.asarray(grid, dtype=int)
    outputs = _solve_outputs(g)
    assert len(outputs) == 1, "input must determine exactly one missing transformed sprite"
    return outputs[0]


def _random_derangement(colors: list[int]) -> dict[int, int]:
    for _ in range(100):
        perm = colors[:]
        random.shuffle(perm)
        if all(a != b for a, b in zip(colors, perm)):
            return dict(zip(colors, perm))
    raise RuntimeError("failed to create a derangement")


def _grow_mask(h: int, w: int, target: int) -> np.ndarray | None:
    for _ in range(100):
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            base_r, base_c = random.choice(tuple(cells))
            neighbors = []
            for dr, dc in DIRS4:
                nr, nc = base_r + dr, base_c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    neighbors.append((nr, nc))
            if not neighbors:
                break
            cells.add(random.choice(neighbors))
        if len(cells) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        box = _bbox(mask)
        assert box is not None
        cropped = _crop_bbox(mask.astype(int), box).astype(bool)
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            continue
        if np.all(cropped):
            continue
        return cropped
    return None


def _random_sprite(palette: list[int]) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        if h * w <= 4:
            continue
        target = random.randint(4, min(h * w - 1, 8))
        mask = _grow_mask(h, w, target)
        if mask is None:
            continue
        coords = np.argwhere(mask)
        color_count = random.randint(2, min(3, len(palette), len(coords)))
        used_colors = random.sample(palette, color_count)
        assignments = used_colors[:]
        while len(assignments) < len(coords):
            assignments.append(random.choice(used_colors))
        random.shuffle(assignments)
        sprite = np.zeros(mask.shape, dtype=int)
        for (r, c), color in zip(coords.tolist(), assignments):
            sprite[int(r), int(c)] = int(color)
        return sprite
    return None


def _transformed_sprite(sprite: np.ndarray, geom: str, color_map: dict[int, int]) -> np.ndarray:
    recolored = _recolor(sprite, color_map)
    assert recolored is not None
    return _apply_geom(_scale2(recolored), geom)


def _pack_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    min_side = max(max_h, max_w) + 3
    for _ in range(40):
        rows = random.randint(max(18, min_side), 28)
        cols = random.randint(max(18, min_side), 28)
        grid = np.zeros((rows, cols), dtype=int)
        reserved = np.zeros((rows, cols), dtype=bool)
        order = list(range(len(patches)))
        random.shuffle(order)
        order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
        placements: dict[int, tuple[int, int]] = {}
        ok = True
        for idx in order:
            patch = patches[idx]
            h, w = map(int, patch.shape)
            positions = [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)]
            random.shuffle(positions)
            placed = False
            for r0, c0 in positions:
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(rows, r0 + h + 1)
                cc1 = min(cols, c0 + w + 1)
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                _paste(grid, patch, r0, c0, transparent=BG)
                reserved[rr0:rr1, cc0:cc1] = True
                placements[idx] = (r0, c0)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and len(_component_sprites(grid)) == len(patches):
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choices([2, 3], weights=[1, 3])[0]
    palette = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.choice([3, 4]))
    color_map = _random_derangement(palette[:])
    geom = random.choice(TRANSFORMS)

    sprites: list[np.ndarray] = []
    seen = set()
    for _ in range(200):
        sprite = _random_sprite(palette)
        if sprite is None:
            continue
        sig = _sig(sprite)
        if sig in seen:
            continue
        seen.add(sig)
        sprites.append(sprite)
        if len(sprites) == pair_count + 1:
            break
    if len(sprites) != pair_count + 1:
        return None

    all_colors = [set(map(int, np.unique(sprite[sprite != BG]))) for sprite in sprites]
    orphan_candidates = []
    for orphan in range(len(sprites)):
        other_colors = set().union(*(all_colors[i] for i in range(len(sprites)) if i != orphan))
        if not all_colors[orphan].issubset(other_colors):
            continue
        if set().union(*(all_colors[i] for i in range(len(sprites)) if i != orphan)) != set(palette):
            continue
        if _sig(_apply_geom(_scale2(sprites[orphan]), geom)) == _sig(_scale2(sprites[orphan])):
            continue
        non_orphan_areas = [int(np.count_nonzero(sprites[i])) for i in range(len(sprites)) if i != orphan]
        if len(set(non_orphan_areas)) == len(non_orphan_areas):
            continue
        orphan_candidates.append(orphan)
    if not orphan_candidates:
        return None
    orphan = random.choice(orphan_candidates)

    patches = [sprites[orphan]]
    expected = _transformed_sprite(sprites[orphan], geom, color_map)
    big_sigs = set()
    for idx, sprite in enumerate(sprites):
        if idx == orphan:
            continue
        big = _transformed_sprite(sprite, geom, color_map)
        sig = _sig(big)
        if sig == _sig(expected) or sig in big_sigs:
            return None
        big_sigs.add(sig)
        patches.append(sprite)
        patches.append(big)

    grid = _pack_patches(patches)
    if grid is None:
        return None
    outputs = _solve_outputs(grid)
    if len(outputs) != 1 or not np.array_equal(outputs[0], expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
