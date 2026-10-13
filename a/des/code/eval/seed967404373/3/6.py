import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
CORNERS = ("tl", "tr", "bl", "br")
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _draw_frame(canvas: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    canvas[top, left : left + width] = color
    canvas[top + height - 1, left : left + width] = color
    canvas[top : top + height, left] = color
    canvas[top : top + height, left + width - 1] = color


def _corner_positions(height: int, width: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (0, 0),
        "tr": (0, width - 1),
        "bl": (height - 1, 0),
        "br": (height - 1, width - 1),
    }


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _fit_dimensions(src_h: int, src_w: int, max_h: int, max_w: int) -> tuple[int, int]:
    assert max_h >= 3 and max_w >= 3
    best: tuple[float, float, int, int, int] | None = None
    best_dims: tuple[int, int] | None = None
    for out_h in range(3, max_h + 1):
        for out_w in range(3, max_w + 1):
            scale_h = out_h / src_h
            scale_w = out_w / src_w
            score = (
                min(scale_h, scale_w),
                -abs(scale_h - scale_w),
                out_h * out_w,
                out_h + out_w,
                -abs(out_h * src_w - out_w * src_h),
            )
            if best is None or score > best:
                best = score
                best_dims = (out_h, out_w)
    assert best_dims is not None
    return best_dims


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid != BG):
        r0 = int(comp[:, 0].min())
        c0 = int(comp[:, 1].min())
        r1 = int(comp[:, 0].max())
        c1 = int(comp[:, 1].max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3

        mask = np.zeros((h, w), dtype=bool)
        mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
        perim = np.zeros((h, w), dtype=bool)
        perim[0, :] = True
        perim[-1, :] = True
        perim[:, 0] = True
        perim[:, -1] = True
        assert np.array_equal(mask, perim), "each component must be a 1-pixel hollow rectangle"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_vals = patch[perim]
        color_counts = Counter(map(int, border_vals))
        base_color = int(max(color_counts.items(), key=lambda kv: (kv[1], kv[0]))[0])
        assert color_counts[base_color] >= border_vals.size - 1

        corners = _corner_positions(h, w)
        pointer_corner = None
        pointer_color = None
        for name, (rr, cc) in corners.items():
            value = int(patch[rr, cc])
            if value != base_color:
                assert pointer_corner is None, "at most one corner may be recolored"
                pointer_corner = name
                pointer_color = value
        for rr in range(h):
            for cc in range(w):
                if not perim[rr, cc]:
                    assert patch[rr, cc] == BG
                    continue
                if (rr, cc) in corners.values():
                    continue
                assert int(patch[rr, cc]) == base_color

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "height": h,
                "width": w,
                "base_color": base_color,
                "pointer_color": pointer_color,
                "pointer_corner": pointer_corner,
            }
        )

    base_colors = [frame["base_color"] for frame in frames]
    assert len(base_colors) == len(set(base_colors)), "frame colors must be unique"
    return frames


def _ordered_chain(frames: list[dict]) -> list[dict]:
    by_color = {frame["base_color"]: frame for frame in frames}
    indegree = {frame["base_color"]: 0 for frame in frames}

    for frame in frames:
        target = frame["pointer_color"]
        if target in by_color and target != frame["base_color"]:
            frame["next_color"] = int(target)
            indegree[int(target)] += 1
        else:
            frame["next_color"] = None

    assert all(v <= 1 for v in indegree.values()), "nesting must be a chain, not a branch"
    roots = [color for color, deg in indegree.items() if deg == 0]
    assert len(roots) == 1, "expected exactly one outermost frame"

    chain: list[dict] = []
    seen: set[int] = set()
    cur = by_color[int(roots[0])]
    while True:
        color = int(cur["base_color"])
        assert color not in seen, "pointer chain must not cycle"
        seen.add(color)
        chain.append(cur)
        nxt = cur["next_color"]
        if nxt is None:
            break
        cur = by_color[int(nxt)]

    assert len(chain) == len(frames), "all frames must belong to one nesting chain"
    return chain


def _compose_output(chain: list[dict]) -> np.ndarray:
    outer = chain[0]
    out = np.full((int(outer["height"]), int(outer["width"])), BG, dtype=int)
    top = 0
    left = 0
    height = int(outer["height"])
    width = int(outer["width"])
    _draw_frame(out, top, left, height, width, int(outer["base_color"]))

    for frame in chain[1:]:
        max_h = height - 4
        max_w = width - 4
        assert max_h >= 3 and max_w >= 3, "container too small for the next nested frame"
        new_h, new_w = _fit_dimensions(int(frame["height"]), int(frame["width"]), max_h, max_w)
        top = top + 2 + (max_h - new_h) // 2
        left = left + 2 + (max_w - new_w) // 2
        height = new_h
        width = new_w
        _draw_frame(out, top, left, height, width, int(frame["base_color"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the pointer-defined frame chain and restack it as centered nested frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames = _extract_frames(g)
    assert len(frames) >= 2
    chain = _ordered_chain(frames)
    return _compose_output(chain)


def _sample_size(min_dim: int, max_dim: int, mode: str) -> tuple[int, int]:
    if mode == "square":
        h = random.randint(min_dim, max_dim)
        w = random.randint(max(min_dim, h - 2), min(max_dim, h + 2))
        return h, w
    if mode == "wide":
        h = random.randint(min_dim, max_dim - 3)
        w = random.randint(max(h + 3, min_dim), max_dim)
        return h, w
    h = random.randint(min_dim, max_dim)
    w = random.randint(min_dim, max(max_dim - 3, min_dim))
    if h <= w + 2:
        h = min(max_dim, w + random.randint(3, max(3, max_dim - w)))
    return h, w


def _random_frame_sizes(count: int) -> list[tuple[int, int]]:
    min_outer = 4 * (count - 1) + 3
    outer_mode = random.choice(("square", "wide", "tall"))
    outer_h, outer_w = _sample_size(min_outer, min_outer + 5, outer_mode)
    sizes = [(outer_h, outer_w)]

    modes = ["square", "wide", "tall"]
    while len(sizes) < count:
        mode = random.choice(modes)
        h, w = _sample_size(5, min(15, min_outer + 2), mode)
        if (h, w) not in sizes:
            sizes.append((h, w))
    return sizes


def _random_pointer_spec(frame_colors: list[int]) -> list[dict]:
    count = len(frame_colors)
    terminal_mode = random.choice(("none", "unmatched"))
    spare_colors = [c for c in PALETTE if c not in frame_colors]
    unmatched_color = random.choice(spare_colors) if terminal_mode == "unmatched" else None

    specs: list[dict] = []
    for i, color in enumerate(frame_colors):
        if i < count - 1:
            pointer_color = frame_colors[i + 1]
            pointer_corner = random.choice(CORNERS)
        elif terminal_mode == "none":
            pointer_color = None
            pointer_corner = None
        else:
            pointer_color = int(unmatched_color)
            pointer_corner = random.choice(CORNERS)
        specs.append(
            {
                "base_color": int(color),
                "pointer_color": pointer_color,
                "pointer_corner": pointer_corner,
            }
        )
    return specs


def _place_boxes(shape: tuple[int, int], sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    grid_h, grid_w = shape
    placements: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    placed_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    for idx in order:
        h, w = sizes[idx]
        row_lo = 1
        row_hi = grid_h - h - 1
        col_lo = 1
        col_hi = grid_w - w - 1
        if row_lo > row_hi or col_lo > col_hi:
            return None
        success = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_boxes_separate(box, other, gap=1) for other in placed_boxes):
                placements[idx] = box
                placed_boxes.append(box)
                success = True
                break
        if not success:
            return None
    return [box for box in placements if box is not None]


def _render_input(frame_defs: list[dict], boxes: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for frame, (r0, c0, r1, c1) in zip(frame_defs, boxes):
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        _draw_frame(grid, r0, c0, h, w, int(frame["base_color"]))
        if frame["pointer_color"] is not None:
            rr, cc = _corner_positions(h, w)[str(frame["pointer_corner"])]
            grid[r0 + rr, c0 + cc] = int(frame["pointer_color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice((3, 4, 5))
    colors = random.sample(PALETTE, frame_count)
    specs = _random_pointer_spec(colors)
    sizes = _random_frame_sizes(frame_count)

    for spec, (h, w) in zip(specs, sizes):
        spec["height"] = h
        spec["width"] = w

    min_h = max(max(h for h, _ in sizes) + 4, 26)
    min_w = max(max(w for _, w in sizes) + 4, 26)
    total_area = sum(h * w for h, w in sizes)
    grid_h = random.randint(min_h, min(38, max(min_h, int(np.ceil(np.sqrt(total_area))) + 18)))
    grid_w = random.randint(min_w, min(38, max(min_w, int(np.ceil(np.sqrt(total_area))) + 18)))

    boxes = _place_boxes((grid_h, grid_w), sizes)
    if boxes is None:
        return None

    grid = _render_input(specs, boxes, (grid_h, grid_w))
    try:
        solved = main(grid)
    except Exception:
        return None

    chain = []
    for spec in specs:
        chain.append(
            {
                "base_color": int(spec["base_color"]),
                "pointer_color": spec["pointer_color"],
                "height": int(spec["height"]),
                "width": int(spec["width"]),
            }
        )
    expected = _compose_output(_ordered_chain(chain))
    return grid if np.array_equal(solved, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-chain input")
