from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
YELLOW = 4
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARROW_TO_DIR = {
    ((0, 0), (0, 1), (1, 1)): "right",
    ((0, 1), (1, 0), (1, 1)): "down",
    ((0, 0), (1, 0), (1, 1)): "left",
    ((0, 0), (0, 1), (1, 0)): "up",
}
DIR_TO_ARROW = {v: k for k, v in ARROW_TO_DIR.items()}
PALETTE = [0, 1, 3, 5, 6, 7, 8, 9]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _paint_rect(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[top : top + height, left : left + width] = int(color)


def _extract_arrows(grid: np.ndarray) -> list[dict[str, int]]:
    arrows: list[dict[str, int]] = []
    for comp in _components(grid == RED):
        if int(comp.shape[0]) != 3:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        local = tuple(sorted((int(r - r0), int(c - c0)) for r, c in comp))
        direction = ARROW_TO_DIR.get(local)
        if direction is None:
            continue
        arrows.append({"top": r0, "left": c0, "dir": direction})
    arrows.sort(key=lambda a: (int(a["top"]), int(a["left"]), str(a["dir"])))
    return arrows


def _extract_blocks(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    mask = (grid != int(bg)) & (grid != RED)
    blocks: list[dict[str, int]] = []
    for comp in _components(mask):
        if int(comp.shape[0]) != 4:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        color = int(sub[0, 0])
        if color == int(bg) or color == RED:
            continue
        if bool((sub == color).all()):
            blocks.append({"top": r0, "left": c0, "color": color})
    blocks.sort(key=lambda b: (int(b["top"]), int(b["left"]), int(b["color"])))
    return blocks


def _targets(arrow: dict[str, int], block: dict[str, int]) -> bool:
    at = int(arrow["top"])
    al = int(arrow["left"])
    bt = int(block["top"])
    bl = int(block["left"])
    direction = str(arrow["dir"])
    if direction == "right":
        return bt == at and bl > al + 1
    if direction == "left":
        return bt == at and bl + 1 < al
    if direction == "down":
        return bl == al and bt > at + 1
    return bl == al and bt + 1 < at


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat each red 3-pixel arrow as a pusher. It pushes every 2x2 block on its
    forward row-band or column-band toward the boundary. Adjacent moved blocks
    merge into a larger yellow rectangle, while arrows disappear.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    arrows = _extract_arrows(g)
    blocks = _extract_blocks(g, bg)

    out = np.full_like(g, int(bg))

    assignments: dict[int, list[dict[str, int]]] = {i: [] for i in range(len(arrows))}
    assigned_blocks = set()
    for bi, block in enumerate(blocks):
        hits = [ai for ai, arrow in enumerate(arrows) if _targets(arrow, block)]
        assert len(hits) <= 1, "a block may not be targeted by multiple arrows"
        if hits:
            assignments[hits[0]].append(block)
            assigned_blocks.add(bi)
        else:
            _paint_rect(out, int(block["top"]), int(block["left"]), 2, 2, int(block["color"]))

    for ai, arrow in enumerate(arrows):
        movers = assignments[ai]
        if not movers:
            continue
        direction = str(arrow["dir"])
        if direction == "right":
            movers.sort(key=lambda b: int(b["left"]))
            start = w - 2 * len(movers)
            for i, block in enumerate(movers):
                color = YELLOW if len(movers) > 1 else int(block["color"])
                _paint_rect(out, int(block["top"]), start + 2 * i, 2, 2, color)
        elif direction == "left":
            movers.sort(key=lambda b: int(b["left"]))
            for i, block in enumerate(movers):
                color = YELLOW if len(movers) > 1 else int(block["color"])
                _paint_rect(out, int(block["top"]), 2 * i, 2, 2, color)
        elif direction == "down":
            movers.sort(key=lambda b: int(b["top"]))
            start = h - 2 * len(movers)
            for i, block in enumerate(movers):
                color = YELLOW if len(movers) > 1 else int(block["color"])
                _paint_rect(out, start + 2 * i, int(block["left"]), 2, 2, color)
        else:
            movers.sort(key=lambda b: int(b["top"]))
            for i, block in enumerate(movers):
                color = YELLOW if len(movers) > 1 else int(block["color"])
                _paint_rect(out, 2 * i, int(block["left"]), 2, 2, color)

    return out


def _distribute(total: int, slots: int) -> list[int]:
    values = [0] * slots
    for _ in range(total):
        values[random.randrange(slots)] += 1
    return values


def _band_starts(total_size: int, count: int) -> list[int]:
    min_used = 2 * count + max(0, count - 1)
    extra = total_size - min_used
    if extra < 0:
        raise ValueError("not enough space for bands")
    gaps = [0] + [1] * max(0, count - 1) + [0]
    add = _distribute(extra, len(gaps))
    gaps = [g + a for g, a in zip(gaps, add)]
    starts: list[int] = []
    cur = gaps[0]
    for i in range(count):
        starts.append(cur)
        cur += 2
        if i < count - 1:
            cur += gaps[i + 1]
    return starts


def _line_layout(length: int, block_count: int, toward_positive: bool) -> tuple[int, list[int]]:
    base_gaps = [0, 1] + [1] * max(0, block_count - 1) + [1]
    if not toward_positive:
        base_gaps = [1] + [1] * max(0, block_count - 1) + [1, 0]
    extra = length - (2 + 2 * block_count + sum(base_gaps))
    if extra < 0:
        raise ValueError("not enough space for lane")
    gaps = [g + a for g, a in zip(base_gaps, _distribute(extra, len(base_gaps)))]

    if toward_positive:
        arrow = gaps[0]
        blocks: list[int] = []
        cur = arrow + 2 + gaps[1]
        for i in range(block_count):
            blocks.append(cur)
            cur += 2
            if i < block_count - 1:
                cur += gaps[2 + i]
        return arrow, blocks

    first_block = gaps[0]
    blocks = []
    cur = first_block
    for i in range(block_count):
        blocks.append(cur)
        cur += 2
        if i < block_count - 1:
            cur += gaps[1 + i]
    arrow = cur + gaps[-2]
    return arrow, blocks


def _draw_arrow(grid: np.ndarray, top: int, left: int, direction: str) -> None:
    for dr, dc in DIR_TO_ARROW[direction]:
        grid[top + dr, left + dc] = RED


def _build_axis_scene(axis: str) -> np.ndarray | None:
    arrow_count = random.randint(1, 3)
    counts = []
    while len(counts) < arrow_count:
        count = 1 if random.random() < 0.5 else random.randint(2, 3)
        counts.append(count)
    total_blocks = sum(counts)
    if total_blocks > 7:
        return None

    if axis == "h":
        min_h = max(8, 2 * arrow_count + max(0, arrow_count - 1) + 2)
        min_w = max(10, max(2 + 2 * k + (k + 1) for k in counts))
        if min_h > 18 or min_w > 20:
            return None
        h = random.randint(min_h, 18)
        w = random.randint(min_w, 20)
    else:
        min_w = max(8, 2 * arrow_count + max(0, arrow_count - 1) + 2)
        min_h = max(10, max(2 + 2 * k + (k + 1) for k in counts))
        if min_h > 20 or min_w > 18:
            return None
        h = random.randint(min_h, 20)
        w = random.randint(min_w, 18)

    bg = random.choice(PALETTE)
    colors = [c for c in PALETTE if c != bg]
    block_colors = random.sample(colors, total_blocks)
    grid = np.full((h, w), int(bg), dtype=int)

    if axis == "h":
        band_rows = _band_starts(h, arrow_count)
        random.shuffle(band_rows)
        color_idx = 0
        for top, count in zip(band_rows, counts):
            direction = random.choice(["left", "right"])
            arrow_left, block_lefts = _line_layout(w, count, toward_positive=(direction == "right"))
            _draw_arrow(grid, int(top), int(arrow_left), direction)
            for left in block_lefts:
                _paint_rect(grid, int(top), int(left), 2, 2, int(block_colors[color_idx]))
                color_idx += 1
    else:
        band_cols = _band_starts(w, arrow_count)
        random.shuffle(band_cols)
        color_idx = 0
        for left, count in zip(band_cols, counts):
            direction = random.choice(["up", "down"])
            arrow_top, block_tops = _line_layout(h, count, toward_positive=(direction == "down"))
            _draw_arrow(grid, int(arrow_top), int(left), direction)
            for top in block_tops:
                _paint_rect(grid, int(top), int(left), 2, 2, int(block_colors[color_idx]))
                color_idx += 1

    return grid


def _valid_generated(grid: np.ndarray) -> bool:
    bg = _most_common_color(grid)
    arrows = _extract_arrows(grid)
    blocks = _extract_blocks(grid, bg)
    if not arrows or not blocks:
        return False
    assigned = [0] * len(blocks)
    saw_collision = False
    saw_single = False
    for arrow in arrows:
        hits = [i for i, block in enumerate(blocks) if _targets(arrow, block)]
        if not hits:
            return False
        if len(hits) == 1:
            saw_single = True
        if len(hits) > 1:
            saw_collision = True
        for idx in hits:
            assigned[idx] += 1
    if any(v != 1 for v in assigned):
        return False
    out = main(grid)
    if out.shape != grid.shape or int((out == RED).sum()) != 0:
        return False
    if np.array_equal(out, grid):
        return False
    return saw_single or saw_collision


def generate_input() -> np.ndarray:
    for _ in range(300):
        axis = random.choice(["h", "v"])
        grid = _build_axis_scene(axis)
        if grid is not None and _valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate valid input")
