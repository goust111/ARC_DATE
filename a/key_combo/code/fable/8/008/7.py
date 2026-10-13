from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components_mask(grid == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _analyze_frame(grid: np.ndarray, component: dict, bg: int = BG) -> dict | None:
    color = int(component["color"])
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    expected_border_area = 2 * h + 2 * w - 4
    if int(component["area"]) != expected_border_area:
        return None

    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _border_mask(h, w)
    if not np.all(region[border] == color):
        return None
    if np.any(region[~border] == color):
        return None

    interior = region[1:-1, 1:-1]
    pointer_cells = np.argwhere(interior != bg)
    if pointer_cells.shape[0] > 1:
        return None

    pointer_color = None
    pointer_pos = None
    if pointer_cells.shape[0] == 1:
        pr, pc = map(int, pointer_cells[0])
        pointer_color = int(interior[pr, pc])
        if pointer_color == color or pointer_color == bg:
            return None
        pointer_pos = (pr + 1, pc + 1)

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "area": int(component["area"]),
        "height": h,
        "width": w,
        "pointer_color": pointer_color,
        "pointer_pos": pointer_pos,
    }


def _extract_frames(grid: np.ndarray, bg: int = BG) -> list[dict]:
    frames = []
    for component in _find_components(grid, bg=bg):
        frame = _analyze_frame(grid, component, bg=bg)
        if frame is not None:
            frames.append(frame)
    return frames


def _chain_order(frames: list[dict]) -> list[int]:
    assert frames, "expected at least one frame"
    by_color = {}
    for frame in frames:
        color = int(frame["color"])
        assert color not in by_color, "frame colors must be unique"
        by_color[color] = frame

    referenced = []
    tail_count = 0
    for frame in frames:
        pointer_color = frame["pointer_color"]
        if pointer_color is None:
            tail_count += 1
            continue
        assert pointer_color in by_color, "pointer must reference another frame color"
        assert pointer_color != frame["color"], "self-loops are not allowed"
        referenced.append(int(pointer_color))

    assert tail_count == 1, "expected exactly one empty frame"
    heads = [color for color in by_color if color not in set(referenced)]
    assert len(heads) == 1, "expected exactly one unreferenced head"

    order: list[int] = []
    seen: set[int] = set()
    cur = int(heads[0])
    for _ in range(len(frames)):
        assert cur not in seen, "pointer chain must not cycle"
        seen.add(cur)
        order.append(cur)
        nxt = by_color[cur]["pointer_color"]
        if nxt is None:
            break
        cur = int(nxt)

    assert len(order) == len(frames), "pointer chain must visit every frame exactly once"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the color-pointer chain inside hollow rectangles and output the frame colors in chain order.

    Input structure:
    - The grid has a black background with several different-colored hollow rectangular frames.
    - Each non-tail frame contains exactly one interior pixel whose color matches another frame color, while one tail frame has an empty interior.
    - One or two solid square distractors may also appear and must be ignored.

    Rule:
    - Detect monochrome 4-connected components and keep only those whose bbox is a complete 1-pixel-thick rectangle border.
    - For each kept frame, treat its single interior pixel as a pointer to the frame with that border color; the empty frame is the tail.
    - Find the unique frame color never referenced by any interior pixel, follow the chain to the empty frame, and output a 1xn row of visited frame colors.

    Key insight:
    - The answer order is encoded by the hidden color links inside the frames, not by position, size, or the distractor squares.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _extract_frames(g, bg=BG)
    assert len(frames) >= 2, "expected at least two frames"

    order = _chain_order(frames)
    return np.array([order], dtype=int)


def _make_frame_patch(color: int, h: int, w: int, pointer_color: int | None) -> tuple[np.ndarray, tuple[int, int] | None]:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    pointer_pos = None
    if pointer_color is not None:
        pr = random.randint(1, h - 2)
        pc = random.randint(1, w - 2)
        patch[pr, pc] = int(pointer_color)
        pointer_pos = (pr, pc)
    return patch, pointer_pos


def _make_square_patch(color: int, size: int) -> np.ndarray:
    return np.full((size, size), int(color), dtype=int)


def _rect_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    view = dst[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def _place_items(rows: int, cols: int, items: list[dict], padding: int = 2, border: int = 1) -> np.ndarray | None:
    grid = np.zeros((rows, cols), dtype=int)
    placed_rects: list[tuple[int, int, int, int]] = []

    for item in sorted(items, key=lambda x: (x["patch"].shape[0] * x["patch"].shape[1], random.random()), reverse=True):
        patch = item["patch"]
        h, w = patch.shape
        candidates = []
        for r0 in range(border, rows - h - border + 1):
            for c0 in range(border, cols - w - border + 1):
                rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                if all(_rect_separated(rect, other, padding) for other in placed_rects):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        _paste(grid, patch, r0, c0)
        item["placed_bbox"] = (r0, c0, r0 + h - 1, c0 + w - 1)
        placed_rects.append(item["placed_bbox"])

    return grid


def _expected_output(chain_colors: list[int]) -> np.ndarray:
    return np.array([chain_colors], dtype=int)


def _frame_orders(frames: list[dict]) -> list[list[int]]:
    return [
        [f["color"] for f in sorted(frames, key=lambda f: (f["bbox"][0], f["bbox"][1]))],
        [f["color"] for f in sorted(frames, key=lambda f: (f["bbox"][1], f["bbox"][0]))],
        [f["color"] for f in sorted(frames, key=lambda f: (f["height"] * f["width"], f["bbox"][0], f["bbox"][1]))],
        [f["color"] for f in sorted(frames, key=lambda f: (f["height"] + f["width"], f["bbox"][0], f["bbox"][1]))],
        sorted([f["color"] for f in frames]),
    ]


def _passes_filters(grid: np.ndarray, chain_colors: list[int]) -> bool:
    expected = _expected_output(chain_colors)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    frames = _extract_frames(grid, bg=BG)
    if len(frames) != len(chain_colors):
        return False
    if sum(frame["pointer_color"] is None for frame in frames) != 1:
        return False
    if any(frame["pointer_color"] == frame["color"] for frame in frames):
        return False

    for simple_order in _frame_orders(frames):
        if simple_order == chain_colors or simple_order[::-1] == chain_colors:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(3, 5)
    n_distractors = random.randint(1, 2)
    chosen_colors = random.sample(PALETTE, n_frames + n_distractors)
    chain_colors = chosen_colors[:n_frames]
    distractor_colors = chosen_colors[n_frames:]

    items = []
    for i, color in enumerate(chain_colors):
        h = random.randint(3, 7)
        w = random.randint(3, 7)
        if random.random() < 0.35:
            w = h
        pointer_color = None if i == n_frames - 1 else chain_colors[i + 1]
        patch, pointer_pos = _make_frame_patch(color, h, w, pointer_color)
        items.append(
            {
                "kind": "frame",
                "color": color,
                "patch": patch,
                "pointer_color": pointer_color,
                "pointer_pos": pointer_pos,
            }
        )

    for color in distractor_colors:
        size = random.randint(2, 5)
        items.append(
            {
                "kind": "distractor",
                "color": color,
                "patch": _make_square_patch(color, size),
            }
        )

    rows = random.randint(18, 30)
    cols = random.randint(18, 30)
    grid = _place_items(rows, cols, items, padding=random.randint(1, 3), border=1)
    if grid is None:
        return None

    if not _passes_filters(grid, chain_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain puzzle")
