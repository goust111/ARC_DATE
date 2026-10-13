from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
CHAIN_COLORS = [BLUE, RED, GREEN, YELLOW, PINK]
MAX_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords = []
        while stack:
            cr, cc2 = stack.pop()
            coords.append((cr, cc2))
            for nr, nc in _neighbors4(cr, cc2):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_frames(grid: np.ndarray) -> list[dict]:
    mask = np.isin(grid, (GREY, ORANGE))
    frames = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            raise AssertionError("frame too small")
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        sub_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(h, w)
        if not np.array_equal(sub_mask, border):
            raise AssertionError("component is not a 1-pixel hollow rectangle border")
        if not np.all(np.isin(sub[border], (GREY, ORANGE))):
            raise AssertionError("border contains invalid colors")
        if np.any(sub[~border] != BG):
            raise AssertionError("frame interior must be black")
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "top": frozenset(int(x) for x in np.where(sub[0, :] == ORANGE)[0]),
                "bottom": frozenset(int(x) for x in np.where(sub[-1, :] == ORANGE)[0]),
                "left": frozenset(int(x) for x in np.where(sub[:, 0] == ORANGE)[0]),
                "right": frozenset(int(x) for x in np.where(sub[:, -1] == ORANGE)[0]),
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _successors(frames: list[dict]) -> dict[int, int]:
    n = len(frames)
    succ: dict[int, int] = {}
    for i in range(n):
        r0a, c0a, r1a, c1a = frames[i]["bbox"]
        hits = []
        for j in range(n):
            if i == j:
                continue
            r0b, c0b, r1b, c1b = frames[j]["bbox"]
            if c1a < c0b and frames[i]["right"] & frames[j]["left"]:
                hits.append(j)
            if r1a < r0b and frames[i]["bottom"] & frames[j]["top"]:
                hits.append(j)
        if len(hits) > 1:
            raise AssertionError(f"frame {i} has multiple outgoing matches: {hits}")
        if hits:
            succ[i] = hits[0]
    return succ


def _chain_order(frames: list[dict]) -> list[int]:
    n = len(frames)
    if n < 2 or n > len(CHAIN_COLORS):
        raise AssertionError("expected 4-5 frames")
    succ = _successors(frames)
    indeg = [0] * n
    outdeg = [0] * n
    for src, dst in succ.items():
        outdeg[src] += 1
        indeg[dst] += 1
        if indeg[dst] > 1:
            raise AssertionError(f"frame {dst} has multiple incoming matches")
    starts = [i for i in range(n) if indeg[i] == 0 and outdeg[i] == 1]
    ends = [i for i in range(n) if indeg[i] == 1 and outdeg[i] == 0]
    mids = [i for i in range(n) if indeg[i] == 1 and outdeg[i] == 1]
    if len(starts) != 1 or len(ends) != 1 or len(mids) != n - 2:
        raise AssertionError("matches do not form a unique directed chain")
    order = []
    seen = set()
    cur = starts[0]
    while True:
        if cur in seen:
            raise AssertionError("cycle detected")
        order.append(cur)
        seen.add(cur)
        if cur not in succ:
            break
        cur = succ[cur]
    if len(order) != n or len(succ) != n - 1:
        raise AssertionError("chain does not cover every frame exactly once")
    return order


def _port_coord(side: str, offset: int, height: int, width: int) -> tuple[int, int]:
    if side == "top":
        return 0, int(offset)
    if side == "bottom":
        return height - 1, int(offset)
    if side == "left":
        return int(offset), 0
    if side == "right":
        return int(offset), width - 1
    raise ValueError(side)


def _solve(grid: np.ndarray) -> tuple[list[dict], list[int]]:
    frames = _extract_frames(grid)
    order = _chain_order(frames)
    return frames, order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame interior using the order induced by edge-contact matches.

    Input structure:
    - A black background contains 4-5 non-overlapping hollow rectangular frames.
    - Each frame is a 1-pixel border made of grey cells with some orange edge cells.
    - Frame interiors are black, and only edge-contact offset matches relate frames.

    Rule:
    - Detect every hollow frame and compare orange offsets on facing sides: a frame points
      to a frame on its right if right-edge and left-edge orange offsets intersect, and
      similarly points downward if bottom-edge and top-edge offsets intersect.
    - These directed matches form one unique chain; follow it from the unique head.
    - Flood-fill each frame interior by chain position with colors blue, red, green,
      yellow, then magenta, while leaving grey borders and orange edge cells unchanged.

    Key insight:
    - Spatial position alone does not determine the output; only the directed topology
      induced by matching orange edge offsets reveals which frame is first, second, etc.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert _get_bg(g) == BG

    frames, order = _solve(g)
    out = g.copy()
    for pos, idx in enumerate(order):
        color = CHAIN_COLORS[pos]
        r0, c0, r1, c1 = frames[idx]["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BG] = color
    return out


def _choose_dirs(rng: random.Random, n: int) -> list[str]:
    mode = rng.random()
    if mode < 0.22:
        return ["R"] * (n - 1)
    if mode < 0.44:
        return ["D"] * (n - 1)
    dirs = [rng.choice(["R", "D"]) for _ in range(n - 1)]
    if len(set(dirs)) == 1:
        dirs[rng.randrange(n - 1)] = "D" if dirs[0] == "R" else "R"
    return dirs


def _assign_ports(
    rng: random.Random, heights: list[int], widths: list[int], dirs: list[str]
) -> list[dict[str, int]] | None:
    n = len(heights)
    for _ in range(120):
        ports = [dict() for _ in range(n)]
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        ok = True
        h_conns = [i for i, d in enumerate(dirs) if d == "R"]
        rng.shuffle(h_conns)
        for i in h_conns:
            limit = min(heights[i], heights[i + 1])
            choices = [k for k in range(limit) if k not in used_rows]
            rng.shuffle(choices)
            if not choices:
                ok = False
                break
            offset = choices[0]
            ports[i]["right"] = offset
            ports[i + 1]["left"] = offset
            used_rows.add(offset)
        if not ok:
            continue
        v_conns = [i for i, d in enumerate(dirs) if d == "D"]
        rng.shuffle(v_conns)
        for i in v_conns:
            limit = min(widths[i], widths[i + 1])
            choices = [k for k in range(limit) if k not in used_cols]
            rng.shuffle(choices)
            if not choices:
                ok = False
                break
            offset = choices[0]
            ports[i]["bottom"] = offset
            ports[i + 1]["top"] = offset
            used_cols.add(offset)
        if not ok:
            continue
        clash = False
        for i in range(n):
            pts = [
                _port_coord(side, offset, heights[i], widths[i])
                for side, offset in ports[i].items()
            ]
            if len(pts) != len(set(pts)):
                clash = True
                break
        if not clash:
            return ports
    return None


def _disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ra, ca, ha, wa = a
    rb, cb, hb, wb = b
    return (
        ra + ha - 1 + gap < rb
        or rb + hb - 1 + gap < ra
        or ca + wa - 1 + gap < cb
        or cb + wb - 1 + gap < ca
    )


def _layout_boxes(
    rng: random.Random, heights: list[int], widths: list[int], dirs: list[str]
) -> list[tuple[int, int, int, int]] | None:
    n = len(heights)
    for _ in range(160):
        boxes = [(0, 0, heights[0], widths[0])]
        ok = True
        for i, d in enumerate(dirs):
            cur_r, cur_c, cur_h, cur_w = boxes[-1]
            nxt_h, nxt_w = heights[i + 1], widths[i + 1]
            placed = None
            for _ in range(80):
                if d == "R":
                    gap = rng.randint(1, 3)
                    nr = cur_r + rng.randint(-(nxt_h - 1), cur_h - 1)
                    nc = cur_c + cur_w + gap
                else:
                    gap = rng.randint(1, 3)
                    nr = cur_r + cur_h + gap
                    nc = cur_c + rng.randint(-(nxt_w - 1), cur_w - 1)
                candidate = (nr, nc, nxt_h, nxt_w)
                if all(_disjoint(candidate, prev, gap=1) for prev in boxes):
                    placed = candidate
                    break
            if placed is None:
                ok = False
                break
            boxes.append(placed)
        if not ok:
            continue
        min_r = min(r for r, _, _, _ in boxes)
        min_c = min(c for _, c, _, _ in boxes)
        norm = [(r - min_r, c - min_c, h, w) for r, c, h, w in boxes]
        ext_h = max(r + h for r, _, h, _ in norm)
        ext_w = max(c + w for _, c, _, w in norm)
        if ext_h <= 28 and ext_w <= 28:
            return norm
    return None


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], ports: dict[str, int]) -> None:
    r, c, h, w = box
    grid[r, c : c + w] = GREY
    grid[r + h - 1, c : c + w] = GREY
    grid[r : r + h, c] = GREY
    grid[r : r + h, c + w - 1] = GREY
    for side, offset in ports.items():
        dr, dc = _port_coord(side, offset, h, w)
        grid[r + dr, c + dc] = ORANGE


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice([4, 5])
    heights = [rng.randint(4, 6) for _ in range(n)]
    widths = [rng.randint(4, 6) for _ in range(n)]
    dirs = _choose_dirs(rng, n)
    ports = _assign_ports(rng, heights, widths, dirs)
    if ports is None:
        return None
    boxes = _layout_boxes(rng, heights, widths, dirs)
    if boxes is None:
        return None

    ext_h = max(r + h for r, _, h, _ in boxes)
    ext_w = max(c + w for _, c, _, w in boxes)
    min_h = max(18, ext_h + 2)
    max_h = min(30, ext_h + 6)
    min_w = max(18, ext_w + 2)
    max_w = min(30, ext_w + 6)
    if min_h > max_h or min_w > max_w:
        return None

    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    row_shift = rng.randint(1, h - ext_h - 1)
    col_shift = rng.randint(1, w - ext_w - 1)
    shifted = [(r + row_shift, c + col_shift, hh, ww) for r, c, hh, ww in boxes]

    grid = np.zeros((h, w), dtype=int)
    for box, port_map in zip(shifted, ports):
        _draw_frame(grid, box, port_map)

    try:
        frames, order = _solve(grid)
        if len(frames) != n or len(order) != n:
            return None
        out = main(grid)
    except AssertionError:
        return None

    used_fill = {color for color in CHAIN_COLORS[:n] if np.any(out == color)}
    if len(used_fill) != n:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
