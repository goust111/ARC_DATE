import random
from collections import Counter

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """沿指针像素构成的颜色链决定层序，把等尺寸面板透明叠合成单块输出。

    Input structure:
    - 黑色背景上分散着若干块同尺寸矩形面板。
    - 每块面板以自己的主色为主，内部带有黑色孔洞；除一块外，其余每块都恰有
      一个单像素指针，颜色等于另一块面板的主色。
    - 所有指针关系覆盖全部面板并构成一条单链，面板的左右位置与真实层序无关。

    Rule:
    - 先把每块散落面板裁成自己的外接矩形。
    - 每块面板里占比最多的非黑色即该面板主色；若还存在唯一的另一种非黑色，
      那个单像素就是指向紧邻上层面板的指针。
    - 没有被任何指针指向的面板是最底层，没有指针像素的面板是最顶层。按这条
      链自底向上叠印，只复制各面板的主色像素，黑色孔洞和指针像素都视为透明。

    Key insight:
    - 指针像素不是图案而是元数据；真正决定覆盖优先级的是颜色链表，而不是面板
      在画布上的摆放位置。
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)

    panels = _extract_panels(g, bg)
    assert len(panels) >= 2, "expected multiple scattered panels"
    shapes = {panel["arr"].shape for panel in panels}
    assert len(shapes) == 1, "all panels must share the same size"

    for panel in panels:
        arr = panel["arr"]
        colors, counts = np.unique(arr[arr != bg], return_counts=True)
        assert len(colors) in {1, 2}, "each panel must have one main color plus optional pointer"
        main_color = int(colors[np.argmax(counts)])
        pointer_colors = [int(c) for c in colors if int(c) != main_color]
        pointer_color = None
        if pointer_colors:
            pointer_color = pointer_colors[0]
            assert int(np.sum(arr == pointer_color)) == 1, "pointer color must be a single pixel"
        panel["main_color"] = main_color
        panel["pointer_color"] = pointer_color

    main_to_panel = {panel["main_color"]: panel for panel in panels}
    assert len(main_to_panel) == len(panels), "panel main colors must be unique"

    inbound = {color: 0 for color in main_to_panel}
    tops = []
    for panel in panels:
        pointer = panel["pointer_color"]
        if pointer is None:
            tops.append(panel["main_color"])
            continue
        assert pointer in main_to_panel, "pointer must name another panel's main color"
        assert pointer != panel["main_color"], "panel cannot point to itself"
        inbound[pointer] += 1
        assert inbound[pointer] == 1, "pointer targets must form a single chain"

    bottoms = [color for color, indeg in inbound.items() if indeg == 0]
    assert len(bottoms) == 1, "there must be exactly one bottom panel"
    assert len(tops) == 1, "there must be exactly one top panel"

    chain = []
    seen = set()
    cur = bottoms[0]
    while True:
        assert cur not in seen, "pointer relation must not contain a cycle"
        seen.add(cur)
        panel = main_to_panel[cur]
        chain.append(panel)
        nxt = panel["pointer_color"]
        if nxt is None:
            break
        cur = nxt

    assert len(chain) == len(panels), "pointer chain must cover every panel"
    assert chain[-1]["main_color"] == tops[0]

    out = np.full(chain[0]["arr"].shape, bg, dtype=int)
    for panel in chain:
        mask = panel["arr"] == panel["main_color"]
        out[mask] = panel["main_color"]
    return out


# --------------- generator (gpt-5.4) ---------------

BG = 0
PANEL_COUNT_RANGE = (3, 4)
PANEL_H_RANGE = (4, 6)
PANEL_W_RANGE = (4, 5)
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
INTERNAL_GAP = 2
GENERATION_RETRIES = 300


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain overlay input")


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    panel_count = rng.randint(*PANEL_COUNT_RANGE)
    panel_h = rng.randint(*PANEL_H_RANGE)
    panel_w = rng.randint(*PANEL_W_RANGE)
    main_colors = rng.sample(COLOR_POOL, panel_count)

    masks = []
    pointer_slots = []
    for _ in range(panel_count):
        mask = _random_panel_mask(panel_h, panel_w, rng)
        if mask is None:
            return None
        slots = _pointer_slots(mask)
        if not slots:
            return None
        masks.append(mask)
        pointer_slots.append(slots)

    chain = list(range(panel_count))
    rng.shuffle(chain)
    x_order = list(range(panel_count))
    rng.shuffle(x_order)
    if x_order == chain:
        x_order[0], x_order[1] = x_order[1], x_order[0]

    min_canvas_w = panel_count * panel_w + (panel_count - 1) * INTERNAL_GAP + 2
    if min_canvas_w > 30:
        return None
    canvas_w = rng.randint(min_canvas_w, min(30, min_canvas_w + 4))
    canvas_h = rng.randint(panel_h + 2, min(30, panel_h + 8))

    xs = _row_positions(canvas_w, panel_count, panel_w, INTERNAL_GAP, rng)
    ys = [rng.randint(1, canvas_h - panel_h - 1) for _ in range(panel_count)]

    next_color = {}
    for idx, panel_id in enumerate(chain[:-1]):
        next_color[panel_id] = main_colors[chain[idx + 1]]

    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    for rank, panel_id in enumerate(x_order):
        arr = np.full((panel_h, panel_w), BG, dtype=int)
        arr[masks[panel_id]] = main_colors[panel_id]
        if panel_id in next_color:
            pr, pc = rng.choice(pointer_slots[panel_id])
            arr[pr, pc] = next_color[panel_id]
        y = ys[panel_id]
        x = xs[rank]
        if np.any(grid[y : y + panel_h, x : x + panel_w] != BG):
            return None
        grid[y : y + panel_h, x : x + panel_w] = arr

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (panel_h, panel_w):
        return None
    visible_counts = Counter(map(int, out[out != BG].ravel()))
    if set(visible_counts) != set(main_colors):
        return None
    if min(visible_counts.values()) < 2:
        return None
    return grid


def _random_panel_mask(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    area = h * w
    min_keep = max(h + w - 1, int(np.ceil(area * 0.45)))
    max_keep = max(min_keep, int(np.floor(area * 0.72)))
    cells = [(r, c) for r in range(h) for c in range(w)]

    for _ in range(120):
        mask = np.ones((h, w), dtype=bool)
        target = rng.randint(min_keep, max_keep)
        shuffled = cells[:]
        rng.shuffle(shuffled)
        for r, c in shuffled:
            if int(mask.sum()) <= target:
                break
            if int(mask[r, :].sum()) <= 1 or int(mask[:, c].sum()) <= 1:
                continue
            mask[r, c] = False
            if not _is_connected(mask):
                mask[r, c] = True

        kept = int(mask.sum())
        if kept < min_keep or kept > max_keep:
            continue
        if not _pointer_slots(mask):
            continue
        if len(np.unique(mask.astype(int), axis=0)) == 1 and len(np.unique(mask.astype(int), axis=1)) == 1:
            continue
        return mask
    return None


def _row_positions(
    canvas_w: int,
    panel_count: int,
    panel_w: int,
    min_gap: int,
    rng: random.Random,
) -> list[int]:
    min_total = panel_count * panel_w + (panel_count - 1) * min_gap + 2
    slack = canvas_w - min_total
    gaps = [1] + [min_gap] * (panel_count - 1) + [1]
    for _ in range(slack):
        gaps[rng.randrange(len(gaps))] += 1

    xs = []
    x = gaps[0]
    for i in range(panel_count):
        xs.append(x)
        x += panel_w
        if i < panel_count - 1:
            x += gaps[i + 1]
    return xs


def _pointer_slots(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    slots = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    slots.append((r, c))
                    break
    return slots


def _extract_panels(grid: np.ndarray, bg: int) -> list[dict]:
    mask = grid != bg
    panels = []
    for coords in _components(mask):
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        panels.append({"arr": grid[r0 : r1 + 1, c0 : c1 + 1].copy(), "bbox": (r0, c0, r1, c1)})
    return panels


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])
