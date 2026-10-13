from __future__ import annotations

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
RNG = np.random.default_rng()

MAX_SIDE = 30
MASTER_SIDE = 30
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
MIN_GROUP_GAP = 1
MIN_EDGE_MARGIN = 1
LAYER_OPTIONS = (1, 2, 3, 4)
GROUP_COUNT_OPTIONS = (2, 3)
PATTERN_DIM_OPTIONS = (
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 5),
    (5, 4),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(g: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return components


def _is_rect_frame(g: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    color = int(comp["color"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected_area = 2 * h + 2 * w - 4
    if int(comp["area"]) != expected_area:
        return False
    if not np.all(g[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(g[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(g[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(g[r0 : r1 + 1, c1] == color):
        return False
    return True


def _find_frames(g: np.ndarray) -> list[dict]:
    return [comp for comp in _find_components(g, bg=BG) if _is_rect_frame(g, comp)]


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _parent_indices(frames: list[dict]) -> list[int]:
    parents = [-1] * len(frames)
    frame_areas = [
        (int(frame["bbox"][2]) - int(frame["bbox"][0]) + 1) * (int(frame["bbox"][3]) - int(frame["bbox"][1]) + 1)
        for frame in frames
    ]
    for i, frame in enumerate(frames):
        candidates = [j for j, outer in enumerate(frames) if i != j and _contains(outer["bbox"], frame["bbox"])]
        if not candidates:
            continue
        parents[i] = min(candidates, key=lambda j: frame_areas[j])
    return parents


def _apply_flip_sequence(patch: np.ndarray, layers: int) -> np.ndarray:
    out = np.array(patch, copy=True)
    for step in range(layers):
        out = np.fliplr(out) if step % 2 == 0 else np.flipud(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flip each innermost multicolor pattern according to how many nested rectangular frames surround it.

    Input structure:
    - The background is black.
    - The grid contains several separated nested-rectangle groups made of one-cell-thick monochrome rectangular frames.
    - Each group's frames are concentric, and the innermost frame directly encloses a small asymmetric multicolor pattern.

    Rule:
    - Detect every monochrome rectangular frame and organize frames by strict geometric containment.
    - For each containment chain, keep all frames unchanged and find the innermost frame's interior patch.
    - Let N be the number of frames in that chain. Apply N alternating flips to the interior patch: odd-numbered flips are horizontal, even-numbered flips are vertical.

    Key insight:
    - The frame colors are only identifiers; the operative signal is the nesting depth of each rectangular-frame chain, which determines the alternating flip sequence for the innermost enclosed pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert frames
    parents = _parent_indices(frames)
    child_count = [0] * len(frames)
    for parent in parents:
        if parent != -1:
            child_count[parent] += 1

    out = g.copy()
    for idx, frame in enumerate(frames):
        if child_count[idx] != 0:
            continue
        depth = 1
        parent = parents[idx]
        while parent != -1:
            depth += 1
            parent = parents[parent]

        r0, c0, r1, c1 = map(int, frame["bbox"])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = _apply_flip_sequence(inner, depth)

    return out


def _d4_variant_count(arr: np.ndarray) -> int:
    cur = arr
    keys: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(4):
        keys.add((tuple(map(int, cur.shape)), cur.tobytes()))
        flip = np.fliplr(cur)
        keys.add((tuple(map(int, flip.shape)), flip.tobytes()))
        cur = np.rot90(cur)
    return len(keys)


def _make_pattern() -> np.ndarray:
    for _ in range(200):
        ph, pw = PATTERN_DIM_OPTIONS[int(RNG.integers(len(PATTERN_DIM_OPTIONS)))]
        color_count = int(RNG.integers(2, 5))
        colors = RNG.choice(PALETTE, size=color_count, replace=False)
        pat = RNG.choice(colors, size=(ph, pw), replace=True).astype(int)
        if len(set(map(int, pat.ravel()))) < 2:
            continue
        if _d4_variant_count(pat) != 8:
            continue
        if _find_frames(pat):
            continue
        return pat
    raise RuntimeError("failed to generate asymmetric pattern")


def _wrap_with_frame(content: np.ndarray, color: int, gap: int) -> tuple[np.ndarray, int]:
    h, w = map(int, content.shape)
    offset = gap + 1
    out = np.full((h + 2 * offset, w + 2 * offset), BG, dtype=int)
    out[offset : offset + h, offset : offset + w] = content
    out[0, :] = int(color)
    out[-1, :] = int(color)
    out[:, 0] = int(color)
    out[:, -1] = int(color)
    return out, offset


def _build_group(layers: int) -> dict:
    pattern = _make_pattern()
    pattern_colors = sorted(set(map(int, pattern.ravel())))
    available = [c for c in PALETTE if c not in pattern_colors]
    frame_colors = RNG.choice(available, size=layers, replace=False).astype(int)

    content = pattern.copy()
    pattern_bbox = (0, 0, content.shape[0] - 1, content.shape[1] - 1)
    leaf_bbox = None

    for level in range(layers):
        gap = 0 if level == 0 else int(RNG.choice([0, 0, 1]))
        content, offset = _wrap_with_frame(content, int(frame_colors[level]), gap)
        pr0, pc0, pr1, pc1 = pattern_bbox
        pattern_bbox = (pr0 + offset, pc0 + offset, pr1 + offset, pc1 + offset)
        if leaf_bbox is None:
            leaf_bbox = (0, 0, content.shape[0] - 1, content.shape[1] - 1)
        else:
            lr0, lc0, lr1, lc1 = leaf_bbox
            leaf_bbox = (lr0 + offset, lc0 + offset, lr1 + offset, lc1 + offset)

    assert leaf_bbox is not None
    transformed_pattern = _apply_flip_sequence(pattern, layers)
    return {
        "grid": content,
        "layers": layers,
        "leaf_bbox": leaf_bbox,
        "pattern_bbox": pattern_bbox,
        "pattern": pattern,
        "transformed_pattern": transformed_pattern,
    }


def _rect_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0
    return not separated


def _place_groups(groups: list[dict]) -> list[tuple[int, int]] | None:
    order = sorted(range(len(groups)), key=lambda i: int(groups[i]["grid"].shape[0] * groups[i]["grid"].shape[1]), reverse=True)
    placed_rects: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int] | None] = [None] * len(groups)

    for idx in order:
        h, w = map(int, groups[idx]["grid"].shape)
        max_r = MASTER_SIDE - h - MIN_EDGE_MARGIN
        max_c = MASTER_SIDE - w - MIN_EDGE_MARGIN
        if max_r < MIN_EDGE_MARGIN or max_c < MIN_EDGE_MARGIN:
            return None

        found = None
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(RNG.integers(MIN_EDGE_MARGIN, max_r + 1))
            c0 = int(RNG.integers(MIN_EDGE_MARGIN, max_c + 1))
            rect = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_rect_too_close(rect, other, MIN_GROUP_GAP) for other in placed_rects):
                continue
            found = (r0, c0)
            placed_rects.append(rect)
            positions[idx] = found
            break
        if found is None:
            return None

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None] if all(p is not None for p in positions) else None


def _render_case(groups: list[dict], positions: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((MASTER_SIDE, MASTER_SIDE), BG, dtype=int)
    out = np.full((MASTER_SIDE, MASTER_SIDE), BG, dtype=int)
    non_bg_rows: list[int] = []
    non_bg_cols: list[int] = []

    for group, (r0, c0) in zip(groups, positions):
        g = group["grid"]
        h, w = map(int, g.shape)
        inp[r0 : r0 + h, c0 : c0 + w] = g
        out[r0 : r0 + h, c0 : c0 + w] = g

        lr0, lc0, lr1, lc1 = map(int, group["leaf_bbox"])
        abs_leaf = (r0 + lr0, c0 + lc0, r0 + lr1, c0 + lc1)
        pr0, pc0, pr1, pc1 = map(int, group["pattern_bbox"])
        out[r0 + pr0 : r0 + pr1 + 1, c0 + pc0 : c0 + pc1 + 1] = group["transformed_pattern"]

        non_bg_rows.extend([r0, r0 + h - 1, abs_leaf[0], abs_leaf[2]])
        non_bg_cols.extend([c0, c0 + w - 1, abs_leaf[1], abs_leaf[3]])

    filled = np.argwhere(inp != BG)
    r_min, c_min = map(int, filled.min(axis=0))
    r_max, c_max = map(int, filled.max(axis=0))

    top_room = r_min
    left_room = c_min
    bottom_room = MASTER_SIDE - 1 - r_max
    right_room = MASTER_SIDE - 1 - c_max

    top = int(RNG.integers(1, min(4, top_room + 1)))
    left = int(RNG.integers(1, min(4, left_room + 1)))
    bottom = int(RNG.integers(1, min(4, bottom_room + 1)))
    right = int(RNG.integers(1, min(4, right_room + 1)))

    r0 = r_min - top
    c0 = c_min - left
    r1 = r_max + bottom
    c1 = c_max + right

    cropped_inp = inp[r0 : r1 + 1, c0 : c1 + 1]
    cropped_out = out[r0 : r1 + 1, c0 : c1 + 1]
    assert max(cropped_inp.shape) <= MAX_SIDE
    assert max(cropped_out.shape) <= MAX_SIDE
    return cropped_inp, cropped_out


def _gen_attempt() -> np.ndarray | None:
    group_count = int(RNG.choice(GROUP_COUNT_OPTIONS))
    layer_counts = RNG.choice(np.array(LAYER_OPTIONS, dtype=int), size=group_count, replace=False).tolist()
    groups = [_build_group(int(layers)) for layers in layer_counts]

    positions = _place_groups(groups)
    if positions is None:
        return None

    inp, expected = _render_case(groups, positions)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
