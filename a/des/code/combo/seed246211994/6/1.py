import itertools
import numpy as np


GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()

ALL_HOLE_COUNTS = tuple(range(6))
LEGEND_HOLE_COUNTS = tuple(range(5))
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(points, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_points(points: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _holes_with_centers(mask: np.ndarray) -> list[tuple[float, float]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    centers: list[tuple[float, float]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                arr = np.asarray(pts, dtype=float)
                centers.append((float(arr[:, 0].mean()), float(arr[:, 1].mean())))
    centers.sort()
    return centers


def _flip_mode(centers: list[tuple[float, float]]) -> str:
    if len(centers) <= 1:
        return "none"
    rows = np.asarray([r for r, _ in centers], dtype=float)
    cols = np.asarray([c for _, c in centers], dtype=float)
    row_span = float(rows.max() - rows.min())
    col_span = float(cols.max() - cols.min())
    if col_span > row_span:
        return "lr"
    if row_span > col_span:
        return "ud"
    return "none"


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[int, int]:
    h, w = grid.shape
    mapping: dict[int, int] = {}
    for color in map(int, np.unique(grid)):
        if color in (bg, GREY):
            continue
        best: tuple[int, int] | None = None
        for pts in _components(grid == color):
            r0, c0, r1, c1 = _bbox(pts)
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if r1 - r0 < 1 or c1 - c0 < 1:
                continue
            mask = _mask_from_points(pts, (r0, c0, r1, c1))
            item = (len(pts), len(_holes_with_centers(mask)))
            if best is None or item[0] > best[0]:
                best = item
        if best is not None and best[1] not in mapping:
            mapping[int(best[1])] = int(color)
    return mapping


def _transformed_mask(mask: np.ndarray, centers: list[tuple[float, float]]) -> np.ndarray:
    mode = _flip_mode(centers)
    if mode == "lr":
        return np.fliplr(mask)
    if mode == "ud":
        return np.flipud(mask)
    return mask.copy()


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend = _legend_mapping(g, bg)

    out = g.copy()
    for pts in _components(g == GREY):
        box = _bbox(pts)
        mask = _mask_from_points(pts, box)
        centers = _holes_with_centers(mask)
        color = int(legend.get(len(centers), bg))
        transformed = _transformed_mask(mask, centers)

        r0, c0, r1, c1 = box
        patch = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[mask] = bg
        if color != bg:
            patch[transformed] = color
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _legend_mask(hole_count: int, axis: str) -> np.ndarray:
    span = 3 if hole_count == 0 else 2 * hole_count + 1
    if axis == "horizontal":
        mask = np.ones((3, span), dtype=bool)
        if hole_count:
            mask[1, 1::2] = False
        return mask
    mask = np.ones((span, 3), dtype=bool)
    if hole_count:
        mask[1::2, 1] = False
    return mask


def _line_span(hole_counts: list[int]) -> int:
    return sum(3 if h == 0 else 2 * h + 1 for h in hole_counts) + max(0, len(hole_counts) - 1)


def _choose_legend_holes(limit: int) -> list[int] | None:
    combos: list[tuple[int, ...]] = []
    for count in range(2, 5):
        for holes in itertools.combinations(LEGEND_HOLE_COUNTS, count):
            if max(holes) < 2:
                continue
            if _line_span(list(holes)) <= limit:
                combos.append(holes)
    if not combos:
        return None
    chosen = list(combos[int(RNG.integers(len(combos)))])
    RNG.shuffle(chosen)
    return [int(x) for x in chosen]


def _paste_mask(grid: np.ndarray, mask: np.ndarray, r: int, c: int, color: int) -> None:
    h, w = mask.shape
    sub = grid[r : r + h, c : c + w]
    sub[mask] = int(color)


def _mark_blocked(blocked: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, int(r) - pad)
    cc0 = max(0, int(c) - pad)
    rr1 = min(blocked.shape[0], int(r) + h + pad)
    cc1 = min(blocked.shape[1], int(c) + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _place_legends(
    grid: np.ndarray,
    expected: np.ndarray,
    blocked: np.ndarray,
    bg: int,
    hole_counts: list[int],
    colors: list[int],
    side: str,
) -> dict[int, int]:
    h, w = grid.shape
    axis = "horizontal" if side in ("top", "bottom") else "vertical"
    masks = [_legend_mask(hole_count, axis) for hole_count in hole_counts]
    mapping = {int(hole_count): int(color) for hole_count, color in zip(hole_counts, colors)}

    if axis == "horizontal":
        total = sum(mask.shape[1] for mask in masks) + len(masks) - 1
        c = int(RNG.integers(1, w - total))
        r = int(RNG.integers(1, 3)) if side == "top" else int(RNG.integers(h - 5, h - 3))
        cur_c = c
        top = r
        bottom = r + 3
        for mask, color in zip(masks, colors):
            _paste_mask(grid, mask, r, cur_c, color)
            _paste_mask(expected, mask, r, cur_c, color)
            cur_c += mask.shape[1] + 1
        if side == "top":
            blocked[: min(h, bottom + 2), :] = True
        else:
            blocked[max(0, top - 1) :, :] = True
    else:
        total = sum(mask.shape[0] for mask in masks) + len(masks) - 1
        r = int(RNG.integers(1, h - total))
        c = int(RNG.integers(1, 3)) if side == "left" else int(RNG.integers(w - 5, w - 3))
        cur_r = r
        left = c
        right = c + 3
        for mask, color in zip(masks, colors):
            _paste_mask(grid, mask, cur_r, c, color)
            _paste_mask(expected, mask, cur_r, c, color)
            cur_r += mask.shape[0] + 1
        if side == "left":
            blocked[:, : min(w, right + 2)] = True
        else:
            blocked[:, max(0, left - 1) :] = True

    return mapping


def _allowed_modes(hole_count: int) -> list[str]:
    if hole_count == 0:
        return ["solid"]
    if hole_count == 1:
        return ["single"]
    return ["horizontal", "vertical", "diagonal"]


def _desired_flip(mode: str) -> str:
    if mode == "horizontal":
        return "lr"
    if mode == "vertical":
        return "ud"
    return "none"


def _add_tab(canvas: np.ndarray, side: str, base_h: int, base_w: int) -> None:
    depth = int(RNG.integers(1, 3))
    if side in ("left", "right"):
        length = int(RNG.integers(1, base_h + 1))
        start = int(RNG.integers(0, base_h - length + 1))
        rr0 = 2 + start
        rr1 = rr0 + length
        if side == "left":
            canvas[rr0:rr1, 2 - depth : 2] = True
        else:
            canvas[rr0:rr1, 2 + base_w : 2 + base_w + depth] = True
    else:
        length = int(RNG.integers(1, base_w + 1))
        start = int(RNG.integers(0, base_w - length + 1))
        cc0 = 2 + start
        cc1 = cc0 + length
        if side == "top":
            canvas[2 - depth : 2, cc0:cc1] = True
        else:
            canvas[2 + base_h : 2 + base_h + depth, cc0:cc1] = True


def _build_object_mask(hole_count: int, mode: str) -> np.ndarray:
    desired = _desired_flip(mode)
    primary_pool = {
        "horizontal": ["left", "right"],
        "vertical": ["top", "bottom"],
        "diagonal": ["left", "right", "top", "bottom"],
        "single": ["left", "right", "top", "bottom"],
        "solid": ["left", "right", "top", "bottom"],
    }[mode]
    secondary_pool = {
        "horizontal": ["top", "bottom"],
        "vertical": ["left", "right"],
        "diagonal": ["left", "right", "top", "bottom"],
        "single": ["left", "right", "top", "bottom"],
        "solid": ["left", "right", "top", "bottom"],
    }[mode]

    for _ in range(40):
        if hole_count == 0:
            base_h = int(RNG.integers(4, 8))
            base_w = int(RNG.integers(4, 8))
            base = np.ones((base_h, base_w), dtype=bool)
        else:
            step = int(RNG.integers(2, 4))
            if mode == "horizontal":
                hole_positions = [(0, i * step) for i in range(hole_count)]
            elif mode == "vertical":
                hole_positions = [(i * step, 0) for i in range(hole_count)]
            elif mode == "diagonal":
                hole_positions = [(i * step, i * step) for i in range(hole_count)]
            else:
                hole_positions = [(0, 0)]
            top_pad = int(RNG.integers(1, 3))
            bottom_pad = int(RNG.integers(1, 4))
            left_pad = int(RNG.integers(1, 3))
            right_pad = int(RNG.integers(1, 4))
            max_r = max(r for r, _ in hole_positions)
            max_c = max(c for _, c in hole_positions)
            base_h = top_pad + max_r + 1 + bottom_pad
            base_w = left_pad + max_c + 1 + right_pad
            base = np.ones((base_h, base_w), dtype=bool)
            for rr, cc in hole_positions:
                base[top_pad + rr, left_pad + cc] = False

        canvas = np.zeros((base_h + 4, base_w + 4), dtype=bool)
        canvas[2 : 2 + base_h, 2 : 2 + base_w] = base
        primary_side = primary_pool[int(RNG.integers(len(primary_pool)))]
        _add_tab(canvas, primary_side, base_h, base_w)
        if RNG.random() < 0.6:
            secondary_side = secondary_pool[int(RNG.integers(len(secondary_pool)))]
            _add_tab(canvas, secondary_side, base_h, base_w)

        coords = np.argwhere(canvas)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        mask = canvas[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        centers = _holes_with_centers(mask)
        if len(centers) != hole_count:
            continue
        if _flip_mode(centers) != desired:
            continue
        if desired == "lr" and np.array_equal(mask, np.fliplr(mask)):
            continue
        if desired == "ud" and np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    raise RuntimeError(f"failed to build object with {hole_count=} {mode=}")


def _object_plan(legend_holes: list[int]) -> list[tuple[int, str, bool]]:
    rich = [h for h in legend_holes if h >= 2]
    distractors = [h for h in ALL_HOLE_COUNTS if h not in legend_holes]
    if not rich or not distractors:
        raise RuntimeError("invalid legend set for object planning")

    neutral_h = int(legend_holes[int(RNG.integers(len(legend_holes)))])
    if neutral_h >= 2:
        neutral_mode = "diagonal"
    elif neutral_h == 1:
        neutral_mode = "single"
    else:
        neutral_mode = "solid"

    plan = [
        (int(rich[int(RNG.integers(len(rich)))]), "horizontal", True),
        (int(rich[int(RNG.integers(len(rich)))]), "vertical", True),
        (neutral_h, neutral_mode, True),
    ]

    bad_h = int(distractors[int(RNG.integers(len(distractors)))])
    bad_modes = _allowed_modes(bad_h)
    plan.append((bad_h, bad_modes[int(RNG.integers(len(bad_modes)))], False))

    extra_count = int(RNG.integers(1, 4))
    for _ in range(extra_count):
        matched = bool(RNG.integers(0, 2))
        pool = legend_holes if matched else distractors
        hole_count = int(pool[int(RNG.integers(len(pool)))])
        modes = _allowed_modes(hole_count)
        plan.append((hole_count, modes[int(RNG.integers(len(modes)))], matched))

    RNG.shuffle(plan)
    return plan


def _find_location(blocked: np.ndarray, shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = shape
    grid_h, grid_w = blocked.shape
    if h >= grid_h - 1 or w >= grid_w - 1:
        return None
    candidates = [(r, c) for r in range(1, grid_h - h) for c in range(1, grid_w - w)]
    RNG.shuffle(candidates)
    for r, c in candidates:
        if not blocked[r : r + h, c : c + w].any():
            return int(r), int(c)
    return None


def _place_objects(
    grid: np.ndarray,
    expected: np.ndarray,
    blocked: np.ndarray,
    bg: int,
    legend_map: dict[int, int],
) -> bool:
    specs = []
    for hole_count, mode, matched in _object_plan(list(legend_map.keys())):
        mask = _build_object_mask(hole_count, mode)
        centers = _holes_with_centers(mask)
        transformed = _transformed_mask(mask, centers)
        specs.append(
            {
                "mask": mask,
                "out_mask": transformed,
                "color": int(legend_map[hole_count]) if matched else bg,
            }
        )

    specs.sort(key=lambda item: item["mask"].shape[0] * item["mask"].shape[1], reverse=True)
    for spec in specs:
        mask = spec["mask"]
        out_mask = spec["out_mask"]
        h, w = mask.shape
        loc = _find_location(blocked, (h, w))
        if loc is None:
            return False
        r, c = loc
        _paste_mask(grid, mask, r, c, GREY)
        if spec["color"] != bg:
            _paste_mask(expected, out_mask, r, c, int(spec["color"]))
        _mark_blocked(blocked, r, c, h, w, pad=1)
    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(24, 35))
    w = int(RNG.integers(24, 35))
    side = ("top", "bottom", "left", "right")[int(RNG.integers(4))]
    span_limit = (w - 2) if side in ("top", "bottom") else (h - 2)
    legend_holes = _choose_legend_holes(span_limit)
    if legend_holes is None:
        return None

    palette = [c for c in range(10) if c != GREY]
    bg = int(palette[int(RNG.integers(len(palette)))])
    legend_colors_pool = [c for c in palette if c != bg]
    if len(legend_colors_pool) < len(legend_holes):
        return None
    legend_colors = list(map(int, RNG.choice(legend_colors_pool, size=len(legend_holes), replace=False)))

    grid = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    legend_map = _place_legends(grid, expected, blocked, bg, legend_holes, legend_colors, side)
    if not _place_objects(grid, expected, blocked, bg, legend_map):
        return None
    if int(np.sum(grid == bg)) <= grid.size // 2:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
