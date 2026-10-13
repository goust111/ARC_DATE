import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a stack of overlapping solid rectangles and list the full rectangles
    from bottom to top, arranged left-to-right with one blank column between them.

    Assumptions enforced by the generator and expected by the solver:
    - Background is black.
    - Each foreground color belongs to exactly one solid axis-aligned rectangle.
    - Every rectangle keeps at least one visible pixel on each of its four sides,
      so its full bbox is recoverable from visible pixels.
    - Every covering relation leaves visible evidence inside the lower rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    colors = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert colors, "expected at least one foreground rectangle"

    rects = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        rects.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )

    n = len(rects)
    edges = [set() for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _has_visible_overlap_evidence(g, rects[i]["bbox"], rects[j]["color"]):
                edges[i].add(j)

    for i in range(n):
        for j in edges[i]:
            indeg[j] += 1

    order = _toposort(rects, edges, indeg)
    return _rectangles_to_row([rects[idx] for idx in order], bg)


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _has_visible_overlap_evidence(grid: np.ndarray, lower_bbox: tuple[int, int, int, int], upper_color: int) -> bool:
    r0, c0, r1, c1 = lower_bbox
    return bool(np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == upper_color))


def _toposort(rects: list[dict], edges: list[set[int]], indeg: list[int]) -> list[int]:
    queue = [
        idx
        for idx, deg in enumerate(indeg)
        if deg == 0
    ]
    queue.sort(key=lambda idx: (-rects[idx]["area"], rects[idx]["bbox"][0], rects[idx]["bbox"][1]))

    order = []
    while queue:
        idx = queue.pop(0)
        order.append(idx)
        for nxt in sorted(edges[idx]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
        queue.sort(key=lambda k: (-rects[k]["area"], rects[k]["bbox"][0], rects[k]["bbox"][1]))

    assert len(order) == len(rects), "covering graph must be acyclic and fully sortable"
    return order


def _rectangles_to_row(rects: list[dict], bg: int) -> np.ndarray:
    heights = [bbox[2] - bbox[0] + 1 for bbox in [r["bbox"] for r in rects]]
    widths = [bbox[3] - bbox[1] + 1 for bbox in [r["bbox"] for r in rects]]
    out_h = max(heights)
    out_w = sum(widths) + max(0, len(widths) - 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    c = 0
    for rect, h, w in zip(rects, heights, widths):
        out[:h, c : c + w] = rect["color"]
        c += w + 1
    return out


# --------------- generator ---------------

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
RECT_COUNT_CHOICES = (2, 3, 4, 5)
TOP_H_RANGE = (2, 6)
TOP_W_RANGE = (2, 6)
MARGIN_RANGE = (1, 4)
GEN_RETRIES = 400


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    gh = rng.randint(*GRID_H_RANGE)
    gw = rng.randint(*GRID_W_RANGE)
    count = rng.choice(RECT_COUNT_CHOICES)

    top_h = rng.randint(*TOP_H_RANGE)
    top_w = rng.randint(*TOP_W_RANGE)
    rects = [{"bbox": (0, 0, top_h - 1, top_w - 1)}]

    for _ in range(count - 1):
        r0, c0, r1, c1 = rects[0]["bbox"]
        mt = rng.randint(*MARGIN_RANGE)
        ml = rng.randint(*MARGIN_RANGE)
        mb = rng.randint(*MARGIN_RANGE)
        mr = rng.randint(*MARGIN_RANGE)
        rects.insert(0, {"bbox": (r0 - mt, c0 - ml, r1 + mb, c1 + mr)})

    min_r = min(b["bbox"][0] for b in rects)
    min_c = min(b["bbox"][1] for b in rects)
    max_r = max(b["bbox"][2] for b in rects)
    max_c = max(b["bbox"][3] for b in rects)
    stack_h = max_r - min_r + 1
    stack_w = max_c - min_c + 1
    if stack_h > gh or stack_w > gw:
        return None

    pad_top = rng.randint(0, gh - stack_h)
    pad_left = rng.randint(0, gw - stack_w)
    shifted = []
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        shifted.append((r0 - min_r + pad_top, c0 - min_c + pad_left, r1 - min_r + pad_top, c1 - min_c + pad_left))

    colors = rng.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], count)
    latent = []
    for color, bbox in zip(colors, shifted):
        latent.append({"color": color, "bbox": bbox})

    inp = np.zeros((gh, gw), dtype=int)
    for rect in latent:
        r0, c0, r1, c1 = rect["bbox"]
        inp[r0 : r1 + 1, c0 : c1 + 1] = rect["color"]

    if len(set(inp.ravel())) < 3:
        return None

    if not _visible_sides_ok(inp, latent):
        return None

    expected = _rectangles_to_row(latent, 0)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    return inp


def _visible_sides_ok(grid: np.ndarray, rects: list[dict]) -> bool:
    for rect in rects:
        color = rect["color"]
        r0, c0, r1, c1 = rect["bbox"]
        top = np.any(grid[r0, c0 : c1 + 1] == color)
        bottom = np.any(grid[r1, c0 : c1 + 1] == color)
        left = np.any(grid[r0 : r1 + 1, c0] == color)
        right = np.any(grid[r0 : r1 + 1, c1] == color)
        if not (top and bottom and left and right):
            return False
    return True
