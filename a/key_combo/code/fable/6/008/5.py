from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SEP = 4
RULE = 5
SCENE_COLORS = (1, 2, 3, 6, 7, 8, 9)
GEN_RETRIES = 400


SHAPE_GROUPS: dict[tuple[int, int], list[tuple[tuple[int, int], ...]]] = {
    (2, 3): [
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
    ],
    (3, 3): [
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 0)),
        ((0, 2), (1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 2), (1, 2), (2, 0), (2, 1), (2, 2)),
    ],
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_coords(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in directions:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _shape_signature_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    shifted = np.asarray(coords, dtype=int) - np.asarray(coords, dtype=int).min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _shape_to_mask(shape: tuple[tuple[int, int], ...], dims: tuple[int, int]) -> np.ndarray:
    h, w = dims
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[int(r), int(c)] = True
    return mask


def _scene_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _component_coords(grid == int(color), connectivity=4):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "shape": _shape_signature_from_coords(coords),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _parse_task(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != BG]
    assert len(sep_rows) == 1, "expected exactly one full-width separator line"
    sep_r = sep_rows[0]
    scene = g[sep_r + 1 :, :].copy()
    rule_area = g[:sep_r, :]

    rule_comps = _component_coords(rule_area == RULE, connectivity=4)
    assert rule_comps and len(rule_comps) % 2 == 0, "expected an even number of legend shapes"

    comp_info = []
    for coords in rule_comps:
        r0, c0, r1, c1 = _bbox_coords(coords)
        comp_info.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": _shape_signature_from_coords(coords),
                "dims": (r1 - r0 + 1, c1 - c0 + 1),
            }
        )
    comp_info.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))

    rules = []
    for i in range(0, len(comp_info), 2):
        left = comp_info[i]
        right = comp_info[i + 1]
        assert left["dims"] == right["dims"], "rule pairs must share bbox size"
        rules.append((left["shape"], right["shape"], left["dims"]))
    return scene, rules


def _resolve_fixpoint(
    shape: tuple[tuple[int, int], ...],
    mapping: dict[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]],
) -> tuple[tuple[int, int], ...]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    cur = shape
    while cur in mapping:
        assert cur not in seen, "rewrite graph must be acyclic"
        seen.add(cur)
        cur = mapping[cur]
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """Rewrite each scene object by following the grey shape legend's cascade to a fixpoint.

    Input structure:
    - A full-width separator row splits the grid into a top legend area and a bottom scene area.
    - In the legend area, grey 4-connected shapes are laid out left-to-right in source/target pairs, with each pair sharing one bbox size.
    - In the scene area, separated colored 4-connected objects sit on a common background; some objects match legend source shapes up to translation.

    Rule:
    - Parse the grey legend shapes in order and pair them as source -> target rewrite rules.
    - For each scene object, compare its translation-normalized mask inside its own bbox against the source shapes.
    - If it matches, replace the mask by the paired target while preserving the object's color and bbox; if the new shape is itself another source, keep rewriting until no rule applies.
    - Leave unmatched scene objects unchanged, and output only the scene area below the separator.

    Key insight:
    - The legend is an acyclic rewrite system rather than a one-step lookup table, so some objects must travel through two or three successive shape changes before reaching their final form.
    """
    scene, rules = _parse_task(grid)
    bg = _get_bg(scene)

    mapping: dict[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = {}
    dims_by_shape: dict[tuple[tuple[int, int], ...], tuple[int, int]] = {}
    for src, dst, dims in rules:
        assert src not in mapping, "each source shape must be unique"
        mapping[src] = dst
        dims_by_shape[src] = dims
        dims_by_shape[dst] = dims

    out = scene.copy()
    for comp in _scene_components(scene, bg):
        r0, c0, r1, c1 = comp["bbox"]
        color = comp["color"]
        final_shape = _resolve_fixpoint(comp["shape"], mapping)
        if final_shape == comp["shape"]:
            continue
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        for dr, dc in final_shape:
            out[r0 + int(dr), c0 + int(dc)] = color
    return out


def _sample_chain_lengths(rng: random.Random) -> list[int]:
    return [3]


def _pick_chain_shapes(
    rng: random.Random,
    lengths: list[int],
) -> tuple[list[list[tuple[tuple[int, int], ...]]], set[tuple[tuple[int, int], ...]]]:
    used: set[tuple[tuple[int, int], ...]] = set()
    chains: list[list[tuple[tuple[int, int], ...]]] = []
    group_keys = list(SHAPE_GROUPS.keys())
    rng.shuffle(group_keys)

    for length in lengths:
        need = length + 1
        candidates = []
        for dims in group_keys:
            available = [shape for shape in SHAPE_GROUPS[dims] if shape not in used]
            if len(available) >= need:
                candidates.append((dims, available))
        if not candidates:
            return [], set()
        dims, available = rng.choice(candidates)
        picked = rng.sample(available, need)
        chains.append(picked)
        used.update(picked)
    return chains, used


def _rule_objects_from_chains(
    chains: list[list[tuple[tuple[int, int], ...]]]
) -> list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]]:
    objs = []
    for chain in chains:
        for shape in chain:
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)
            objs.append((shape, (max_r + 1, max_c + 1)))
    return objs


def _render_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> None:
    for dr, dc in shape:
        grid[top + int(dr), left + int(dc)] = int(color)


def _occupied_with_pad(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _build_rules_and_scene(rng: random.Random) -> tuple[np.ndarray, dict]:
    chain_lengths = _sample_chain_lengths(rng)
    chains, used_shapes = _pick_chain_shapes(rng, chain_lengths)
    if not chains:
        raise RuntimeError("failed to pick chain shapes")

    rule_pairs = []
    source_shapes = set()
    shape_dims: dict[tuple[tuple[int, int], ...], tuple[int, int]] = {}
    for chain in chains:
        for idx in range(len(chain) - 1):
            src = chain[idx]
            dst = chain[idx + 1]
            dims = (max(r for r, _ in src) + 1, max(c for _, c in src) + 1)
            assert dims == (max(r for r, _ in dst) + 1, max(c for _, c in dst) + 1)
            rule_pairs.append((src, dst, dims))
            source_shapes.add(src)
            shape_dims[src] = dims
            shape_dims[dst] = dims

    distractor_pool = [
        shape
        for shapes in SHAPE_GROUPS.values()
        for shape in shapes
        if shape not in used_shapes and shape not in source_shapes
    ]
    rng.shuffle(distractor_pool)
    assert len(distractor_pool) >= 2

    main_chain = chains[0]
    required_shapes = [main_chain[0], main_chain[1], main_chain[2]]
    extra_source_pool = [shape for shape in source_shapes if shape not in required_shapes]
    rng.shuffle(extra_source_pool)
    extra_count = rng.randint(1, 3)
    chosen_sources = required_shapes + extra_source_pool[:extra_count]

    unmatched_count = rng.randint(1, 2)
    chosen_unmatched = distractor_pool[:unmatched_count]

    scene_specs = []
    for shape in chosen_sources:
        scene_specs.append((shape, True))
    for shape in chosen_unmatched:
        scene_specs.append((shape, False))

    duplicate_source_pool = list(main_chain[:-1])
    rng.shuffle(duplicate_source_pool)
    for shape in duplicate_source_pool[: rng.randint(0, 2)]:
        scene_specs.append((shape, True))
    rng.shuffle(scene_specs)

    rule_height = 3 + max(dims[0] for _, _, dims in rule_pairs)
    scene_height = rng.randint(10, 15)
    pair_gap = 1
    between_gap = 2
    rule_width = sum(2 * dims[1] + pair_gap for _src, _dst, dims in rule_pairs)
    rule_width += between_gap * max(0, len(rule_pairs) - 1)
    grid_width = max(rule_width, rng.randint(18, 28))
    total_height = rule_height + 1 + scene_height
    if total_height > 30:
        scene_height -= total_height - 30
        total_height = rule_height + 1 + scene_height
    assert total_height <= 30 and grid_width <= 30

    grid = np.full((total_height, grid_width), BG, dtype=int)

    top = 1
    cur_c = 0
    for idx, (src, dst, dims) in enumerate(rule_pairs):
        h, w = dims
        _render_shape(grid, src, top, cur_c, RULE)
        cur_c += w + 1
        _render_shape(grid, dst, top, cur_c, RULE)
        cur_c += w
        if idx != len(rule_pairs) - 1:
            cur_c += between_gap

    sep_r = rule_height
    grid[sep_r, :] = SEP

    scene = grid[sep_r + 1 :, :]
    occupied = np.zeros(scene.shape, dtype=bool)
    scene_meta = []
    for shape, is_source in scene_specs:
        dims = (max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1)
        for _ in range(120):
            r0 = rng.randint(0, scene.shape[0] - dims[0])
            c0 = rng.randint(0, scene.shape[1] - dims[1])
            if _occupied_with_pad(occupied, r0, c0, dims[0], dims[1], pad=1):
                continue
            color = rng.choice(SCENE_COLORS)
            _render_shape(scene, shape, r0, c0, color)
            _mark_occupied(occupied, r0, c0, dims[0], dims[1], pad=1)
            scene_meta.append(
                {
                    "shape": shape,
                    "dims": dims,
                    "bbox": (r0, c0, r0 + dims[0] - 1, c0 + dims[1] - 1),
                    "color": color,
                    "is_source": is_source,
                }
            )
            break
        else:
            raise RuntimeError("failed to place scene object")

    meta = {
        "rule_pairs": rule_pairs,
        "chains": chains,
        "scene_meta": scene_meta,
        "source_shapes": source_shapes,
        "chain_lengths": chain_lengths,
    }
    return grid, meta


def _simulate_expected(meta: dict, scene_shape: tuple[int, int]) -> np.ndarray:
    expected = np.full(scene_shape, BG, dtype=int)
    mapping = {src: dst for src, dst, _dims in meta["rule_pairs"]}
    for spec in meta["scene_meta"]:
        final_shape = _resolve_fixpoint(spec["shape"], mapping)
        r0, c0, _r1, _c1 = spec["bbox"]
        for dr, dc in final_shape:
            expected[r0 + int(dr), c0 + int(dc)] = int(spec["color"])
    return expected


def _semantic_checks(grid: np.ndarray, meta: dict) -> bool:
    try:
        parsed_scene, parsed_rules = _parse_task(grid)
    except Exception:
        return False

    if len(parsed_rules) != len(meta["rule_pairs"]):
        return False

    mapping = {src: dst for src, dst, _dims in parsed_rules}
    if len(mapping) != len(parsed_rules):
        return False

    chain_depths = []
    for src in mapping:
        cur = src
        depth = 0
        seen = set()
        while cur in mapping:
            if cur in seen:
                return False
            seen.add(cur)
            cur = mapping[cur]
            depth += 1
        chain_depths.append(depth)
    if max(chain_depths) < 2:
        return False

    comps = _scene_components(parsed_scene, BG)
    rewrite_steps = []
    unmatched = 0
    for comp in comps:
        cur = comp["shape"]
        depth = 0
        seen = set()
        while cur in mapping:
            if cur in seen:
                return False
            seen.add(cur)
            cur = mapping[cur]
            depth += 1
        rewrite_steps.append(depth)
        if depth == 0:
            unmatched += 1
    if unmatched == 0 or max(rewrite_steps) < 2:
        return False

    expected = _simulate_expected(meta, parsed_scene.shape)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    final_shapes = []
    for comp in comps:
        final_shapes.append(_resolve_fixpoint(comp["shape"], mapping))
    if len({(shape, tuple(comp["bbox"])) for shape, comp in zip(final_shapes, comps)}) < 3:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, meta = _build_rules_and_scene(random.Random())
        except Exception:
            continue
        if _semantic_checks(grid, meta):
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
