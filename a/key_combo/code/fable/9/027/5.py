import random
from collections import Counter, defaultdict

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(
                {
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": len(coords),
                }
            )
    return components


def axis_runs(pattern: np.ndarray, axis: int) -> list[tuple[int, int, tuple[int, ...]]]:
    lines = []
    if axis == 0:
        for i in range(pattern.shape[0]):
            lines.append(tuple(int(x) for x in pattern[i, :]))
    else:
        for i in range(pattern.shape[1]):
            lines.append(tuple(int(x) for x in pattern[:, i]))
    runs = []
    start = 0
    for i in range(1, len(lines) + 1):
        if i == len(lines) or lines[i] != lines[start]:
            runs.append((start, i, lines[start]))
            start = i
    return runs


def stretched_axis_indices(pattern: np.ndarray, axis: int, target_len: int) -> list[int]:
    runs = axis_runs(pattern, axis)
    assert len(runs) == 3, "expected exactly one interior run per axis"
    first, middle, last = runs
    fixed = (first[1] - first[0]) + (last[1] - last[0])
    middle_len = target_len - fixed
    assert middle_len >= 1, "target must fit the frozen boundary bands"

    indices = list(range(first[0], first[1]))
    indices.extend([middle[0]] * middle_len)
    indices.extend(range(last[0], last[1]))
    return indices


def stretch_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    row_idx = stretched_axis_indices(pattern, 0, int(shape[0]))
    col_idx = stretched_axis_indices(pattern, 1, int(shape[1]))
    return pattern[np.ix_(row_idx, col_idx)]


def is_exemplar_patch(patch: np.ndarray, bg: int) -> bool:
    if patch.size == 0 or np.any(patch == bg):
        return False
    if len(np.unique(patch)) < 2:
        return False
    border = np.concatenate((patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]))
    if len(border) == 0:
        return False
    outer = int(patch[0, 0])
    if not np.all(border == outer):
        return False
    return len(axis_runs(patch, 0)) == 3 and len(axis_runs(patch, 1)) == 3


def order_marker_pair(markers: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    assert len(markers) == 2
    (r1, c1), (r2, c2) = markers
    if r1 <= r2 and c1 <= c2:
        return (r1, c1), (r2, c2)
    if r2 <= r1 and c2 <= c1:
        return (r2, c2), (r1, c1)
    raise AssertionError("marker pair must define a top-left / bottom-right diagonal")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each multicolor frame exemplar into the virtual rectangle opened by its same-colored corner markers.

    Input structure:
    - The background separates several fully occupied multicolor rectangular exemplars and isolated single-pixel markers.
    - Every exemplar has a unique outer frame color, and each axis consists of a top boundary run, one interior run, and a bottom boundary run.
    - For each exemplar color there are exactly two isolated markers of that same color, standing for the unseen top-left and bottom-right corners of a target rectangle.

    Rule:
    - Match each marker pair to the exemplar whose outer frame uses the same color.
    - Use the two markers as the corners of the target bounding box.
    - Stretch the exemplar to that box by keeping both boundary runs fixed on each axis and lengthening only the single interior run, then stamp the result over the box and its markers.

    Key insight:
    - The target rectangle is never drawn directly; the only clues are a color-keyed exemplar and a same-colored diagonal marker pair, and the resizing is structural rather than geometric.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    components = find_components(g, bg)

    exemplars: dict[int, np.ndarray] = {}
    markers: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for comp in components:
        coords = comp["coords"]
        bbox = comp["bbox"]
        patch = crop_bbox(g, bbox)
        if comp["area"] == 1:
            r, c = coords[0]
            markers[int(g[r, c])].append((r, c))
            continue

        h = bbox[2] - bbox[0] + 1
        w = bbox[3] - bbox[1] + 1
        assert comp["area"] == h * w, "exemplars must be solid rectangles"
        assert is_exemplar_patch(patch, bg), "unexpected non-marker foreground object"
        outer = int(patch[0, 0])
        assert outer not in exemplars, "outer frame colors must be unique"
        exemplars[outer] = patch

    assert exemplars, "expected at least one exemplar"
    assert set(exemplars) == set(markers), "every exemplar color must have exactly one marker pair"

    out = g.copy()
    for color, exemplar in exemplars.items():
        tl, br = order_marker_pair(markers[color])
        target_shape = (br[0] - tl[0] + 1, br[1] - tl[1] + 1)
        stamp = stretch_pattern(exemplar, target_shape)
        out[tl[0] : br[0] + 1, tl[1] : br[1] + 1] = stamp

    return out


MAX_ATTEMPTS = 500
GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
EXEMPLAR_COUNT_CHOICES = (2, 3)
FRAME_THICKNESS_RANGE = (1, 3)
CENTER_SIZE_RANGE = (1, 4)
TARGET_DELTA_RANGE = (1, 6)
PADDING = 1


def build_exemplar(outer: int, inner: int) -> tuple[np.ndarray, dict]:
    top = random.randint(*FRAME_THICKNESS_RANGE)
    bottom = random.randint(*FRAME_THICKNESS_RANGE)
    left = random.randint(*FRAME_THICKNESS_RANGE)
    right = random.randint(*FRAME_THICKNESS_RANGE)
    center_h = random.randint(*CENTER_SIZE_RANGE)
    center_w = random.randint(*CENTER_SIZE_RANGE)

    h = top + center_h + bottom
    w = left + center_w + right
    patch = np.full((h, w), outer, dtype=int)
    patch[top : top + center_h, left : left + center_w] = inner
    return patch, {
        "outer": outer,
        "inner": inner,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "center_h": center_h,
        "center_w": center_w,
    }


def choose_target_shape(meta: dict, mode: str) -> tuple[int, int]:
    dh = 0
    dw = 0
    if mode in {"grow_h", "grow_both"}:
        dh = random.randint(*TARGET_DELTA_RANGE)
    if mode in {"grow_w", "grow_both"}:
        dw = random.randint(*TARGET_DELTA_RANGE)
    return (
        meta["top"] + meta["center_h"] + dh + meta["bottom"],
        meta["left"] + meta["center_w"] + dw + meta["right"],
    )


def free_positions(used: np.ndarray, h: int, w: int, margin: int = 1) -> list[tuple[int, int]]:
    H, W = used.shape
    out = []
    for r0 in range(margin, H - h - margin + 1):
        for c0 in range(margin, W - w - margin + 1):
            rr0 = max(0, r0 - PADDING)
            cc0 = max(0, c0 - PADDING)
            rr1 = min(H, r0 + h + PADDING)
            cc1 = min(W, c0 + w + PADDING)
            if not used[rr0:rr1, cc0:cc1].any():
                out.append((r0, c0))
    return out


def reserve_rect(used: np.ndarray, top_left: tuple[int, int], shape: tuple[int, int]) -> None:
    r0, c0 = top_left
    h, w = shape
    used[r0 : r0 + h, c0 : c0 + w] = True


def place_rectangles(rects: list[dict], H: int, W: int) -> dict[int, tuple[int, int]] | None:
    used = np.zeros((H, W), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    for rect in sorted(rects, key=lambda item: item["shape"][0] * item["shape"][1], reverse=True):
        h, w = rect["shape"]
        candidates = free_positions(used, h, w, margin=1)
        if not candidates:
            return None
        pos = random.choice(candidates)
        reserve_rect(used, pos, rect["shape"])
        placements[rect["id"]] = pos
    return placements


def validate_case(inp: np.ndarray, expected: np.ndarray, metas: list[dict]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    if not any(meta["top"] != meta["bottom"] or meta["left"] != meta["right"] for meta in metas):
        return False
    if not any(meta["mode"] != "grow_both" for meta in metas):
        return False

    stretched_patches = []
    for meta in metas:
        exemplar = meta["exemplar"]
        target_shape = meta["target_shape"]
        if target_shape == exemplar.shape:
            return False
        stretched = stretch_pattern(exemplar, target_shape)
        stretched_patches.append(stretched)
        if np.array_equal(stretched, exemplar):
            return False

    signatures = {(patch.shape, tuple(int(x) for x in patch.ravel())) for patch in stretched_patches}
    if len(signatures) != len(stretched_patches):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        case = generate_case()
        if case is None:
            continue
        inp, expected, metas = case
        if validate_case(inp, expected, metas):
            return inp
    raise RuntimeError("failed to generate a valid input")


def generate_case() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    n = random.choice(EXEMPLAR_COUNT_CHOICES)

    palette = random.sample(list(range(1, 10)), 2 * n)
    outers = palette[:n]
    inners = palette[n:]

    exemplars = []
    metas: list[dict] = []
    mode_cycle = ["grow_h", "grow_w", "grow_both"]
    random.shuffle(mode_cycle)

    for i in range(n):
        exemplar, meta = build_exemplar(outers[i], inners[i])
        mode = mode_cycle[i % len(mode_cycle)]
        target_shape = choose_target_shape(meta, mode)
        meta["mode"] = mode
        meta["target_shape"] = target_shape
        meta["exemplar"] = exemplar
        metas.append(meta)
        exemplars.append(exemplar)

    rects = []
    for idx, meta in enumerate(metas):
        rects.append({"id": idx, "shape": meta["exemplar"].shape, "kind": "exemplar"})
        rects.append({"id": idx + n, "shape": meta["target_shape"], "kind": "target"})

    placements = place_rectangles(rects, H, W)
    if placements is None:
        return None

    inp = np.zeros((H, W), dtype=int)
    expected = np.zeros((H, W), dtype=int)

    for idx, meta in enumerate(metas):
        exemplar = meta["exemplar"]
        ex_r0, ex_c0 = placements[idx]
        eh, ew = exemplar.shape
        inp[ex_r0 : ex_r0 + eh, ex_c0 : ex_c0 + ew] = exemplar
        expected[ex_r0 : ex_r0 + eh, ex_c0 : ex_c0 + ew] = exemplar

        tg_r0, tg_c0 = placements[idx + n]
        th, tw = meta["target_shape"]
        tg_r1 = tg_r0 + th - 1
        tg_c1 = tg_c0 + tw - 1
        color = meta["outer"]
        inp[tg_r0, tg_c0] = color
        inp[tg_r1, tg_c1] = color
        expected[tg_r0 : tg_r1 + 1, tg_c0 : tg_c1 + 1] = stretch_pattern(exemplar, meta["target_shape"])

    return inp, expected, metas
