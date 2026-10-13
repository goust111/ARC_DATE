from __future__ import annotations

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = DIRS4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))

ALL_TRANSFORMS = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "transpose",
    "flip_anti",
)
RULE_TRANSFORMS = ALL_TRANSFORMS[1:]

GEN_ATTEMPTS = 500
SPRITE_ATTEMPTS = 200


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
) -> np.ndarray:
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)
    assert 0 <= r0 and 0 <= c0 and r0 + h <= dst.shape[0] and c0 + w <= dst.shape[1]
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 8,
) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    dirs = DIRS4 if int(connectivity) == 4 else DIRS8
    comps: list[dict[str, object]] = []

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) == int(bg):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append({"coords": arr, "bbox": bbox_coords(arr), "area": int(arr.shape[0])})
    return comps


def transform_patch(patch: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(patch, dtype=int)
    if name == "identity":
        return a.copy()
    if name == "rot90":
        return np.rot90(a, 1)
    if name == "rot180":
        return np.rot90(a, 2)
    if name == "rot270":
        return np.rot90(a, 3)
    if name == "flip_lr":
        return np.fliplr(a)
    if name == "flip_ud":
        return np.flipud(a)
    if name == "transpose":
        return a.T.copy()
    if name == "flip_anti":
        return np.flipud(np.fliplr(a)).T.copy()
    raise ValueError(f"unknown transform: {name}")


def recolor_patch(patch: np.ndarray, mapping: dict[int, int], bg: int = BG) -> np.ndarray:
    out = np.asarray(patch, dtype=int).copy()
    for src, dst in mapping.items():
        out[out == int(src)] = int(dst)
    out[out == int(bg)] = int(bg)
    return out


def _component_patch(panel: np.ndarray, comp: dict[str, object]) -> np.ndarray:
    return crop_bbox(panel, comp["bbox"])


def _candidate_rules(
    demo_left: np.ndarray,
    demo_right: np.ndarray,
    *,
    bg: int,
    include_identity: bool,
) -> list[tuple[str, dict[int, int]]]:
    candidates: list[tuple[str, dict[int, int]]] = []
    transforms = ALL_TRANSFORMS if include_identity else RULE_TRANSFORMS
    right_mask = demo_right != int(bg)

    for name in transforms:
        moved = transform_patch(demo_left, name)
        if moved.shape != demo_right.shape:
            continue
        moved_mask = moved != int(bg)
        if not np.array_equal(moved_mask, right_mask):
            continue

        mapping: dict[int, int] = {}
        ok = True
        for src, dst in zip(moved[moved_mask], demo_right[right_mask]):
            s = int(src)
            d = int(dst)
            prev = mapping.get(s)
            if prev is None:
                mapping[s] = d
            elif prev != d:
                ok = False
                break
        if not ok:
            continue
        colors = {int(v) for v in np.unique(moved[moved_mask])}
        if set(mapping) != colors:
            continue
        candidates.append((name, mapping))
    return candidates


def infer_rule(
    demo_left: np.ndarray,
    demo_right: np.ndarray,
    *,
    bg: int,
) -> tuple[str, dict[int, int]]:
    candidates = _candidate_rules(demo_left, demo_right, bg=bg, include_identity=False)
    assert len(candidates) == 1, "demonstration must define one rigid transform and one recolor map"
    return candidates[0]


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    candidates = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != int(bg) and np.all(grid[r] == grid[r, 0])
    ]
    assert len(candidates) == 1, "expected one solid separator line"
    return int(candidates[0])


def _coords_to_patch(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return crop_bbox(mask.astype(int), bbox_coords(coords)).astype(bool)


def _random_connected_mask(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(h)), int(rng.integers(w)))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()
    for dr, dc in DIRS4:
        nr = start[0] + dr
        nc = start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while int(mask.sum()) < area:
        nr, nc = tuple(frontier)[int(rng.integers(len(frontier)))]
        frontier.remove((nr, nc))
        mask[nr, nc] = True
        for dr, dc in DIRS4:
            rr = nr + dr
            cc = nc + dc
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                frontier.add((rr, cc))
    return _coords_to_patch(mask)


def _mask_is_asymmetric(mask: np.ndarray) -> bool:
    encoded = mask.astype(int)
    for name in RULE_TRANSFORMS:
        moved = transform_patch(encoded, name) > 0
        if moved.shape == mask.shape and np.array_equal(moved, mask):
            return False
    return True


def _same_shape_up_to_transform(a: np.ndarray, b: np.ndarray) -> bool:
    aa = np.asarray(a, dtype=bool).astype(int)
    bb = np.asarray(b, dtype=bool)
    for name in ALL_TRANSFORMS:
        moved = transform_patch(aa, name) > 0
        if moved.shape == bb.shape and np.array_equal(moved, bb):
            return True
    return False


def _assign_colors(
    rng: np.random.Generator,
    mask: np.ndarray,
    colors: tuple[int, ...],
) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.shape[0] >= len(colors)
    order = rng.permutation(coords.shape[0])
    patch = np.full(mask.shape, BG, dtype=int)

    for idx, color in enumerate(colors):
        r, c = coords[order[idx]]
        patch[int(r), int(c)] = int(color)
    for idx in order[len(colors) :]:
        r, c = coords[idx]
        patch[int(r), int(c)] = int(rng.choice(colors))
    return patch


def _random_sprite(
    rng: np.random.Generator,
    colors: tuple[int, ...],
) -> np.ndarray:
    min_area = max(5, len(colors) + 2)
    for _ in range(SPRITE_ATTEMPTS):
        h = int(rng.integers(3, 7))
        w = int(rng.integers(3, 7))
        max_area = min(h * w - 1, 13)
        if max_area < min_area:
            continue
        area = int(rng.integers(min_area, max_area + 1))
        mask = _random_connected_mask(rng, h, w, area)
        if min(mask.shape) < 2 or not _mask_is_asymmetric(mask):
            continue
        return _assign_colors(rng, mask, colors)
    raise RuntimeError("failed to sample an asymmetric sprite")


def _make_demo_pair(
    rng: np.random.Generator,
    transform_name: str,
    source_colors: tuple[int, ...],
    target_colors: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    for _ in range(SPRITE_ATTEMPTS):
        demo_left = _random_sprite(rng, source_colors)
        permuted_targets = tuple(int(v) for v in rng.permutation(np.asarray(target_colors, dtype=int)))
        mapping = {int(src): int(dst) for src, dst in zip(source_colors, permuted_targets)}
        demo_right = recolor_patch(transform_patch(demo_left, transform_name), mapping)
        candidates = _candidate_rules(demo_left, demo_right, bg=BG, include_identity=True)
        if len(candidates) == 1 and candidates[0][0] == transform_name and candidates[0][1] == mapping:
            return demo_left, demo_right, mapping
    raise RuntimeError("failed to sample a unique demonstration pair")


def _make_query_sprite(
    rng: np.random.Generator,
    source_colors: tuple[int, ...],
    demo_left: np.ndarray,
    transform_name: str,
    mapping: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    demo_mask = demo_left != BG
    for _ in range(SPRITE_ATTEMPTS):
        query = _random_sprite(rng, source_colors)
        query_mask = query != BG
        if _same_shape_up_to_transform(query_mask, demo_mask):
            continue

        solved = recolor_patch(transform_patch(query, transform_name), mapping)
        if np.array_equal(query, solved):
            continue
        if np.array_equal(transform_patch(query, transform_name), query):
            continue
        if np.array_equal(recolor_patch(query, mapping), solved):
            continue

        same_outputs = 0
        for alt_name in ALL_TRANSFORMS:
            alt = recolor_patch(transform_patch(query, alt_name), mapping)
            if alt.shape == solved.shape and np.array_equal(alt, solved):
                same_outputs += 1
        if same_outputs != 1:
            continue
        return query, solved
    raise RuntimeError("failed to sample a non-degenerate query sprite")


def _render_instance(
    rng: np.random.Generator,
    demo_left: np.ndarray,
    demo_right: np.ndarray,
    query: np.ndarray,
    solved_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    top_pair_h = max(demo_left.shape[0], demo_right.shape[0])
    query_pair_h = max(query.shape[0], solved_query.shape[0])
    query_pair_w = max(query.shape[1], solved_query.shape[1])

    top_h = int(rng.integers(top_pair_h + 3, min(14, top_pair_h + 7)))
    bottom_h = int(rng.integers(query_pair_h + 3, min(15, query_pair_h + 8)))

    min_top_w = demo_left.shape[1] + demo_right.shape[1] + 5
    width_low = max(min_top_w + 2, query_pair_w + 6)
    width_high = min(31, width_low + 8)
    if width_low >= width_high:
        return None
    width = int(rng.integers(width_low, width_high))
    total_h = top_h + 1 + bottom_h
    if total_h > 30:
        return None

    pair_r = int(rng.integers(1, top_h - top_pair_h))
    gap = int(rng.integers(2, 5))
    left_c_max = width - demo_left.shape[1] - demo_right.shape[1] - gap - 1
    if left_c_max <= 1:
        return None
    left_c = int(rng.integers(1, left_c_max))
    right_c = left_c + demo_left.shape[1] + gap

    query_r_max = bottom_h - query_pair_h
    query_c_max = width - query_pair_w
    if query_r_max < 1 or query_c_max < 1:
        return None
    query_r = int(rng.integers(1, query_r_max + 1))
    query_c = int(rng.integers(1, query_c_max + 1))

    inp = np.full((total_h, width), BG, dtype=int)
    inp[top_h, :] = DIVIDER
    paste(inp, demo_left, pair_r, left_c, transparent=BG)
    paste(inp, demo_right, pair_r, right_c, transparent=BG)
    paste(inp, query, top_h + 1 + query_r, query_c, transparent=BG)

    out = np.full((bottom_h, width), BG, dtype=int)
    paste(out, solved_query, query_r, query_c, transparent=BG)
    return inp, out


def _valid_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    demo_left: np.ndarray,
    demo_right: np.ndarray,
    query: np.ndarray,
    transform_name: str,
    mapping: dict[int, int],
) -> bool:
    if get_bg(inp) != BG:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    top = inp[: _find_separator_row(inp, BG)]
    bottom = inp[_find_separator_row(inp, BG) + 1 :]
    top_comps = find_components(top, bg=BG, connectivity=8)
    bottom_comps = find_components(bottom, bg=BG, connectivity=8)
    if len(top_comps) != 2 or len(bottom_comps) != 1:
        return False

    full_candidates = _candidate_rules(demo_left, demo_right, bg=BG, include_identity=True)
    if len(full_candidates) != 1:
        return False
    if full_candidates[0][0] != transform_name or full_candidates[0][1] != mapping:
        return False

    query_colors = {int(v) for v in np.unique(query) if int(v) != BG}
    if query_colors != set(mapping):
        return False
    if any(mapping[color] == color for color in query_colors):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_colors = int(rng.integers(2, 5))
    palette = tuple(int(v) for v in rng.permutation(np.asarray(PALETTE, dtype=int)))
    source_colors = tuple(palette[:n_colors])
    target_colors = tuple(palette[n_colors : 2 * n_colors])
    transform_name = RULE_TRANSFORMS[int(rng.integers(len(RULE_TRANSFORMS)))]

    demo_left, demo_right, mapping = _make_demo_pair(
        rng, transform_name, source_colors, target_colors
    )
    query, solved_query = _make_query_sprite(
        rng, source_colors, demo_left, transform_name, mapping
    )
    rendered = _render_instance(rng, demo_left, demo_right, query, solved_query)
    if rendered is None:
        return None
    inp, expected = rendered
    if not _valid_instance(
        inp, expected, demo_left, demo_right, query, transform_name, mapping
    ):
        return None
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the demonstrated rigid transform and recolor rule, then apply them to the lower sprite.

    Input structure:
    - A solid horizontal separator splits the grid into an upper demonstration panel and a lower answer panel.
    - The upper panel contains exactly two multicolor sprites on the background; the right sprite is a transformed
      and recolored version of the left sprite.
    - The lower panel contains one multicolor sprite whose colors come from the upper-left demonstration sprite.

    Rule:
    - Crop the two upper sprites to their bounding boxes and compare them to recover the unique non-identity rigid
      transform and the per-color replacement table.
    - Remove the upper panel and separator; keep only the lower panel.
    - In that lower panel, crop the sprite, apply the same rigid transform around its own bounding-box top-left
      anchor, recolor it with the same table, and paste it back at the original bounding-box top-left.

    Key insight:
    - The upper pair is not content to copy; it is a pictorial instruction that encodes both geometry and recoloring,
      which must be induced before transferring the rule to a different sprite below.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    sep = _find_separator_row(g, bg)
    top = g[:sep, :]
    bottom = g[sep + 1 :, :]

    top_comps = find_components(top, bg=bg, connectivity=8)
    bottom_comps = find_components(bottom, bg=bg, connectivity=8)
    assert len(top_comps) == 2, "expected two demonstration sprites"
    assert len(bottom_comps) == 1, "expected one query sprite"

    top_comps.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    demo_left = _component_patch(top, top_comps[0])
    demo_right = _component_patch(top, top_comps[1])
    query_comp = bottom_comps[0]
    query = _component_patch(bottom, query_comp)

    transform_name, mapping = infer_rule(demo_left, demo_right, bg=bg)
    query_colors = {int(v) for v in np.unique(query) if int(v) != int(bg)}
    assert query_colors <= set(mapping), "query colors must be covered by the demonstration mapping"

    solved = recolor_patch(transform_patch(query, transform_name), mapping, bg=bg)
    out = np.full(bottom.shape, bg, dtype=int)
    r0, c0, _r1, _c1 = query_comp["bbox"]
    paste(out, solved, r0, c0, transparent=bg)
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid demonstration-transfer puzzle")
