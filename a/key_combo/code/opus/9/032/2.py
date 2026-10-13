from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
SPRITE_COLORS = [2, 3, 4, 6, 7]
DEPTH_RECOLOR = {2: 1, 3: 2, 4: 3}

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_N8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

_BASE_SHAPES = [
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (2, 1)],
    [(0, 0), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
    [(0, 1), (1, 1), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _translate(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _normalize_shape(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    out = sorted((r - min_r, c - min_c) for r, c in coords)
    return [(int(r), int(c)) for r, c in out]


def _transform_shape(coords: list[tuple[int, int]], rotate: int, flip: bool) -> list[tuple[int, int]]:
    cur = [(int(r), int(c)) for r, c in coords]
    for _ in range(rotate % 4):
        cur = [(-c, r) for r, c in cur]
    if flip:
        cur = [(r, -c) for r, c in cur]
    return _normalize_shape(cur)


def _shape_variants() -> list[list[tuple[int, int]]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    for base in _BASE_SHAPES:
        for rotate in range(4):
            for flip in (False, True):
                shape = tuple(_transform_shape(base, rotate, flip))
                h = 1 + max(r for r, _ in shape)
                w = 1 + max(c for _, c in shape)
                if h >= 2 and w >= 2:
                    variants.add(shape)
    return [list(shape) for shape in sorted(variants)]


_SHAPES = _shape_variants()


def _components_of_colors(
    grid: np.ndarray, colors: set[int]
) -> list[dict[str, object]]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _is_rect_frame(comp: dict[str, object]) -> bool:
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[index]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if int(coords.shape[0]) != expected:
        return False
    cells = {tuple(map(int, rc)) for rc in coords}
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if on_border != ((rr, cc) in cells):
                return False
    return True


def _frame_area(frame: dict[str, object]) -> int:
    r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _strictly_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = map(int, inner)
    or0, oc0, or1, oc1 = map(int, outer)
    return or0 + 1 <= ir0 and oc0 + 1 <= ic0 and ir1 <= or1 - 1 and ic1 <= oc1 - 1


def _coords_inside_frame(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return bool(
        np.all((r0 + 1 <= coords[:, 0]) & (coords[:, 0] <= r1 - 1) & (c0 + 1 <= coords[:, 1]) & (coords[:, 1] <= c1 - 1))
    )


def _inside_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    return r0 <= r <= r1 and c0 <= c <= c1


def _frame_tree(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(frames, key=_frame_area)
    for frame in ordered:
        frame["parent"] = None
        frame["children"] = []
        frame["depth"] = None
    for frame in ordered:
        containers = [outer for outer in ordered if outer is not frame and _strictly_inside(frame["bbox"], outer["bbox"])]  # type: ignore[index]
        if containers:
            parent = min(containers, key=_frame_area)
            frame["parent"] = parent
            parent["children"].append(frame)
    roots = [frame for frame in ordered if frame["parent"] is None]
    assert len(roots) == 1, "expected one nested frame chain"

    def assign(node: dict[str, object], depth: int) -> None:
        node["depth"] = int(depth)
        for child in node["children"]:  # type: ignore[index]
            assign(child, depth + 1)

    assign(roots[0], 1)
    return sorted(ordered, key=lambda frame: int(frame["depth"]))  # type: ignore[index]


def _layer_output_mask(
    shape: tuple[int, int], frame: dict[str, object], child_frames: list[dict[str, object]]
) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    for child in child_frames:
        cr0, cc0, cr1, cc1 = child["bbox"]  # type: ignore[index]
        mask[cr0 : cr1 + 1, cc0 : cc1 + 1] = False
    return mask


def _layer_input_mask(
    shape: tuple[int, int], frame: dict[str, object], child_frames: list[dict[str, object]]
) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
    if r1 - r0 < 4 or c1 - c0 < 4:
        return mask
    mask[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1] = True
    for child in child_frames:
        cr0, cc0, cr1, cc1 = child["bbox"]  # type: ignore[index]
        er0 = max(0, cr0 - 1)
        ec0 = max(0, cc0 - 1)
        er1 = min(H - 1, cr1 + 1)
        ec1 = min(W - 1, cc1 + 1)
        mask[er0 : er1 + 1, ec0 : ec1 + 1] = False
    return mask


def _effective_color(original: int, depth: int, recolor: bool) -> int:
    if not recolor:
        return int(original)
    return int(DEPTH_RECOLOR.get(int(depth), int(original)))


def _solve_variant(
    grid: np.ndarray,
    *,
    apply_shear: bool,
    apply_recolor: bool,
    use_true_depth: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = [comp for comp in _components_of_colors(g, {FRAME}) if _is_rect_frame(comp)]
    frames = _frame_tree(frames)
    sprites = _components_of_colors(g, set(SPRITE_COLORS))
    out = np.full_like(g, bg)

    for sprite in sprites:
        coords = np.asarray(sprite["coords"], dtype=int)
        containers = [frame for frame in frames if _coords_inside_frame(coords, frame["bbox"])]  # type: ignore[index]
        assert containers, "sprite must belong to a frame layer"
        container = min(containers, key=_frame_area)
        depth = int(container["depth"])  # type: ignore[index]
        shift_depth = depth if use_true_depth else 1
        valid_mask = _layer_output_mask(g.shape, container, list(container["children"]))  # type: ignore[arg-type,index]
        bottom = int(coords[:, 0].max())
        color = _effective_color(int(sprite["color"]), depth, apply_recolor)
        for r, c in coords:
            nr = int(r)
            nc = int(c) - (shift_depth * (bottom - int(r)) if apply_shear else 0)
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and valid_mask[nr, nc]:
                out[nr, nc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shear each colored sprite left by a layer-dependent amount, recolor it by nesting depth, and erase the grey frames.

    Input structure:
    - A black background contains 2 or 3 nested grey rectangular border frames.
    - Each ring-shaped layer bounded by one frame and its direct child frame contains 1 or 2 small single-color 4-connected sprites.
    - Sprite colors come from non-grey foreground colors, and sprites do not touch each other or the frames.

    Rule:
    - Detect the nested frame chain and assign each frame a depth, with the outermost frame at depth 1.
    - For each sprite, use the depth of its direct containing frame as the sprite depth, so sprites in the outer ring are depth 1.
    - Shear every sprite left around its bottom row: each pixel at (r, c) moves to (r, c - depth * (bottom - r)).
    - Keep only sheared pixels that stay inside that sprite's own layer region: the interior of its containing frame, excluding any direct child-frame rectangles.
    - Recolor depth-2 sprites blue, depth-3 sprites red, depth-4 sprites green, keep depth-1 sprites unchanged, and erase all grey frame pixels.

    Key insight:
    - The same hidden layer depth controls two visible outputs at once: the shear amount and the recolored result, while the frames matter only as temporary clipping boundaries.
    """
    return _solve_variant(np.asarray(grid, dtype=int), apply_shear=True, apply_recolor=True, use_true_depth=True)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _make_frames(H: int, W: int, n_frames: int) -> list[dict[str, object]] | None:
    outer_h_min = 16 if n_frames == 2 else 20
    outer_w_min = 16 if n_frames == 2 else 20
    if H < outer_h_min + 2 or W < outer_w_min + 2:
        return None
    outer_h = random.randint(outer_h_min, H - 2)
    outer_w = random.randint(outer_w_min, W - 2)
    r0 = random.randint(1, H - outer_h - 1)
    c0 = random.randint(1, W - outer_w - 1)
    frames: list[dict[str, object]] = [{"bbox": (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)}]

    for _ in range(1, n_frames):
        pr0, pc0, pr1, pc1 = frames[-1]["bbox"]  # type: ignore[index]
        ph = pr1 - pr0 + 1
        pw = pc1 - pc0 + 1
        gt = random.randint(3, 5)
        gb = max(3, min(5, gt + random.choice([-1, 0, 1])))
        gl = random.randint(3, 5)
        gr = max(3, min(5, gl + random.choice([-1, 0, 1])))
        child = (pr0 + gt, pc0 + gl, pr1 - gb, pc1 - gr)
        ch = child[2] - child[0] + 1
        cw = child[3] - child[1] + 1
        if ch < 6 or cw < 6 or ch >= ph or cw >= pw:
            return None
        frames.append({"bbox": child})

    return _frame_tree(frames)


def _pick_sprite_colors(counts_by_depth: list[int]) -> list[int]:
    total = sum(counts_by_depth)
    if total > len(SPRITE_COLORS):
        raise ValueError("too many sprites for the unique-color palette")
    remaining = list(SPRITE_COLORS)
    chosen: list[int] = []
    for depth, count in enumerate(counts_by_depth, start=1):
        preferred = [4, 6, 7, 2, 3] if depth == 1 else [7, 6, 4, 2, 3]
        depth_colors = [c for c in preferred if c in remaining][:count]
        if len(depth_colors) < count:
            extras = [c for c in remaining if c not in depth_colors][: count - len(depth_colors)]
            depth_colors.extend(extras)
        for color in depth_colors:
            remaining.remove(color)
        random.shuffle(depth_colors)
        chosen.extend(depth_colors)
    return chosen


def _coords_fit_mask(coords: list[tuple[int, int]], mask: np.ndarray) -> bool:
    return all(0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and mask[r, c] for r, c in coords)


def _touches_mask(coords: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    H, W = occupied.shape
    for r, c in coords:
        if occupied[r, c]:
            return True
        for dr, dc in _N8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc]:
                return True
    return False


def _mark_mask(coords: list[tuple[int, int]], occupied: np.ndarray) -> None:
    for r, c in coords:
        occupied[r, c] = True


def _transformed_coords(
    coords: list[tuple[int, int]], depth: int, valid_mask: np.ndarray
) -> list[tuple[int, int]]:
    bottom = max(r for r, _ in coords)
    out: list[tuple[int, int]] = []
    for r, c in coords:
        nc = c - depth * (bottom - r)
        if 0 <= r < valid_mask.shape[0] and 0 <= nc < valid_mask.shape[1] and valid_mask[r, nc]:
            out.append((int(r), int(nc)))
    return out


def _candidate_placements(
    shape_coords: list[tuple[int, int]],
    input_mask: np.ndarray,
    output_mask: np.ndarray,
    depth: int,
    occupied_in: np.ndarray,
    occupied_out: np.ndarray,
) -> list[dict[str, object]]:
    h = 1 + max(r for r, _ in shape_coords)
    w = 1 + max(c for _, c in shape_coords)
    H, W = input_mask.shape
    candidates: list[dict[str, object]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            coords = _translate(shape_coords, r0, c0)
            if not _coords_fit_mask(coords, input_mask):
                continue
            if _touches_mask(coords, occupied_in):
                continue
            shifted = _transformed_coords(coords, depth, output_mask)
            if len(shifted) < max(2, len(coords) - 2):
                continue
            if _touches_mask(shifted, occupied_out):
                continue
            candidates.append(
                {
                    "coords": coords,
                    "shifted": shifted,
                    "clipped": len(shifted) < len(coords),
                }
            )
    return candidates


def _plan_sprite_counts(n_frames: int) -> list[int]:
    counts = [1] * n_frames
    extra = random.randint(0, min(2, len(SPRITE_COLORS) - n_frames))
    for _ in range(extra):
        counts[random.randrange(n_frames)] += 1
    return counts


def _variant_outputs(grid: np.ndarray) -> list[np.ndarray]:
    erase_only = np.where(np.asarray(grid) == FRAME, BG, np.asarray(grid))
    return [
        erase_only,
        _solve_variant(grid, apply_shear=False, apply_recolor=True, use_true_depth=True),
        _solve_variant(grid, apply_shear=True, apply_recolor=False, use_true_depth=True),
        _solve_variant(grid, apply_shear=True, apply_recolor=True, use_true_depth=False),
    ]


def _validate_example(grid: np.ndarray, sprite_meta: list[dict[str, object]]) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape or not np.any(out != BG):
        return False

    depths = {int(meta["depth"]) for meta in sprite_meta}
    if len(depths) < 2 or not any(depth > 1 for depth in depths):
        return False
    if not any(bool(meta["clipped"]) for meta in sprite_meta):
        return False
    if not any(not bool(meta["clipped"]) for meta in sprite_meta):
        return False
    if not any(len({r for r, _ in meta["coords"]}) >= 2 for meta in sprite_meta):
        return False

    for meta in sprite_meta:
        color = int(meta["color"])
        depth = int(meta["depth"])
        if depth == 1 and DEPTH_RECOLOR.get(depth, color) != color:
            return False
        if depth > 1 and _effective_color(color, depth, True) == color and depth in DEPTH_RECOLOR:
            continue

    for wrong in _variant_outputs(grid):
        if np.array_equal(out, wrong):
            return False

    if int(np.count_nonzero(out)) < sum(len(meta["shifted"]) for meta in sprite_meta):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    n_frames = random.choice([2, 3])
    H = random.randint(20, 26 if n_frames == 2 else 28)
    W = random.randint(20, 26 if n_frames == 2 else 28)
    frames = _make_frames(H, W, n_frames)
    if frames is None:
        return None

    grid = np.full((H, W), BG, dtype=int)
    for frame in frames:
        _draw_frame(grid, frame["bbox"])  # type: ignore[arg-type]

    counts_by_depth = _plan_sprite_counts(n_frames)
    colors = _pick_sprite_colors(counts_by_depth)
    sprite_specs: list[dict[str, object]] = []
    color_idx = 0
    for depth, count in enumerate(counts_by_depth, start=1):
        for _ in range(count):
            sprite_specs.append({"depth": depth, "color": colors[color_idx]})
            color_idx += 1
    random.shuffle(sprite_specs)

    clip_target = random.randrange(len(sprite_specs))
    plain_target = random.randrange(len(sprite_specs))
    while plain_target == clip_target and len(sprite_specs) > 1:
        plain_target = random.randrange(len(sprite_specs))

    occupied_in = np.zeros_like(grid, dtype=bool)
    occupied_out = np.zeros_like(grid, dtype=bool)
    sprite_meta: list[dict[str, object]] = []

    for idx, spec in enumerate(sorted(sprite_specs, key=lambda item: -int(item["depth"]))):
        depth = int(spec["depth"])
        frame = frames[depth - 1]
        child_frames = list(frame["children"])  # type: ignore[index]
        input_mask = _layer_input_mask(grid.shape, frame, child_frames)
        output_mask = _layer_output_mask(grid.shape, frame, child_frames)
        if int(input_mask.sum()) < 6 or int(output_mask.sum()) < 6:
            return None

        shape_choices = random.sample(_SHAPES, k=min(len(_SHAPES), 12))
        clipped_pref = idx == clip_target
        plain_pref = idx == plain_target
        chosen: dict[str, object] | None = None
        for shape in shape_choices:
            candidates = _candidate_placements(shape, input_mask, output_mask, depth, occupied_in, occupied_out)
            if not candidates:
                continue
            clipped = [cand for cand in candidates if cand["clipped"]]
            plain = [cand for cand in candidates if not cand["clipped"]]
            pool = candidates
            if clipped_pref and clipped:
                pool = clipped
            elif plain_pref and plain:
                pool = plain
            elif clipped and plain:
                pool = clipped if random.random() < 0.5 else plain
            chosen = random.choice(pool)
            break
        if chosen is None:
            return None

        coords = chosen["coords"]  # type: ignore[index]
        shifted = chosen["shifted"]  # type: ignore[index]
        color = int(spec["color"])
        for r, c in coords:
            grid[r, c] = color
        _mark_mask(coords, occupied_in)
        _mark_mask(shifted, occupied_out)
        sprite_meta.append(
            {
                "depth": depth,
                "color": color,
                "coords": coords,
                "shifted": shifted,
                "clipped": bool(chosen["clipped"]),
            }
        )

    if not _validate_example(grid, sprite_meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-shear ARC input")
