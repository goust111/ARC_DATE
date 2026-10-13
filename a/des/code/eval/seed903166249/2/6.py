import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
BOX_SIZE = 5
BOX_HALF = BOX_SIZE // 2
OPS = ("reverse", "swap_pairs", "rotate_left")
GEN_RETRIES = 400


def _apply_op(seq: list[int], op: str) -> list[int]:
    if op == "reverse":
        return list(reversed(seq))
    if op == "swap_pairs":
        out = list(seq)
        for i in range(0, len(out) - 1, 2):
            out[i], out[i + 1] = out[i + 1], out[i]
        return out
    if op == "rotate_left":
        return list(seq[1:] + seq[:1]) if seq else []
    raise ValueError(op)


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
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _find_runs(values: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    start = None
    for i, value in enumerate(values.tolist()):
        if value != BG and start is None:
            start = i
        if value == BG and start is not None:
            runs.append(values[start:i].astype(int).tolist())
            start = None
    if start is not None:
        runs.append(values[start:].astype(int).tolist())
    return runs


def _legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == GREY)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    assert np.all(grid[r0, c0 : c1 + 1] == GREY)
    assert np.all(grid[r1, c0 : c1 + 1] == GREY)
    assert np.all(grid[r0 : r1 + 1, c0] == GREY)
    assert np.all(grid[r0 : r1 + 1, c1] == GREY)
    return int(r0), int(c0), int(r1), int(c1)


def _is_swatch_frame(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    ring = np.ones((3, 3), dtype=bool)
    ring[1, 1] = False
    return bool(sub[1, 1] == BG and np.all(sub[ring] == color))


def _infer_op(inp: list[int], out: list[int]) -> str:
    matches = [op for op in OPS if _apply_op(inp, op) == out]
    assert len(matches) == 1
    return matches[0]


def _parse_legend(grid: np.ndarray) -> tuple[dict[int, str], tuple[int, int, int, int]]:
    lr0, lc0, lr1, lc1 = _legend_bbox(grid)
    inside = np.zeros_like(grid, dtype=bool)
    inside[lr0 + 1 : lr1, lc0 + 1 : lc1] = True
    mapping: dict[int, str] = {}
    colors = [int(c) for c in np.unique(grid[inside]) if c not in (BG, GREY)]
    for color in colors:
        mask = (grid == color) & inside
        for comp in _components(mask):
            if not _is_swatch_frame(grid, comp, color):
                continue
            r0, c0, _r1, c1 = _bbox(comp)
            row = r0 + 1
            runs = _find_runs(grid[row, c1 + 1 : lc1])
            assert len(runs) == 2
            mapping[color] = _infer_op(runs[0], runs[1])
    assert mapping
    return mapping, (lr0, lc0, lr1, lc1)


def _is_box_frame(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5 or h % 2 == 0 or w % 2 == 0:
        return False
    if len(coords) != 2 * h + 2 * w - 4:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(sub[0, :] == color):
        return False
    if not np.all(sub[-1, :] == color):
        return False
    if not np.all(sub[:, 0] == color):
        return False
    if not np.all(sub[:, -1] == color):
        return False
    return bool(np.all(sub[1:-1, 1:-1] == BG))


def _find_boxes(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> list[dict[str, int]]:
    lr0, lc0, lr1, lc1 = legend_bbox
    outside = np.ones_like(grid, dtype=bool)
    outside[lr0 : lr1 + 1, lc0 : lc1 + 1] = False
    boxes: list[dict[str, int]] = []
    colors = [int(c) for c in np.unique(grid[outside]) if c not in (BG, GREY)]
    for color in colors:
        mask = (grid == color) & outside
        for comp in _components(mask):
            if not _is_box_frame(grid, comp, color):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            boxes.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "cr": (r0 + r1) // 2,
                    "cc": (c0 + c1) // 2,
                }
            )
    assert boxes
    return boxes


def _h_neighbor(grid: np.ndarray, boxes: list[dict[str, int]], idx: int, side: str) -> int | None:
    box = boxes[idx]
    row = box["cr"]
    candidates: list[tuple[int, int]] = []
    for j, other in enumerate(boxes):
        if j == idx or other["cr"] != row:
            continue
        if side == "left" and other["c1"] < box["c0"]:
            if np.all(grid[row, other["c1"] + 1 : box["c0"]] == BG):
                candidates.append((other["c1"], j))
        if side == "right" and other["c0"] > box["c1"]:
            if np.all(grid[row, box["c1"] + 1 : other["c0"]] == BG):
                candidates.append((other["c0"], j))
    if not candidates:
        return None
    if side == "left":
        return max(candidates)[1]
    return min(candidates)[1]


def _v_neighbor(grid: np.ndarray, boxes: list[dict[str, int]], idx: int, side: str) -> int | None:
    box = boxes[idx]
    col = box["cc"]
    candidates: list[tuple[int, int]] = []
    for j, other in enumerate(boxes):
        if j == idx or other["cc"] != col:
            continue
        if side == "up" and other["r1"] < box["r0"]:
            if np.all(grid[other["r1"] + 1 : box["r0"], col] == BG):
                candidates.append((other["r1"], j))
        if side == "down" and other["r0"] > box["r1"]:
            if np.all(grid[box["r1"] + 1 : other["r0"], col] == BG):
                candidates.append((other["r0"], j))
    if not candidates:
        return None
    if side == "up":
        return max(candidates)[1]
    return min(candidates)[1]


def _annotate_boxes(grid: np.ndarray, boxes: list[dict[str, int]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for i, box in enumerate(boxes):
        row = box["cr"]
        col = box["cc"]
        left_source = box["c0"] > 0 and bool(np.all(grid[row, : box["c0"]] != BG))
        top_source = box["r0"] > 0 and bool(np.all(grid[: box["r0"], col] != BG))
        left_neighbor = _h_neighbor(grid, boxes, i, "left")
        right_neighbor = _h_neighbor(grid, boxes, i, "right")
        up_neighbor = _v_neighbor(grid, boxes, i, "up")
        down_neighbor = _v_neighbor(grid, boxes, i, "down")
        horizontal = left_source or left_neighbor is not None or right_neighbor is not None
        vertical = top_source or up_neighbor is not None or down_neighbor is not None
        assert horizontal ^ vertical
        info: dict[str, object] = dict(box)
        if horizontal:
            info["orient"] = "h"
            info["pred"] = left_neighbor
            info["succ"] = right_neighbor
            info["source"] = grid[row, : box["c0"]].astype(int).tolist() if left_source else None
        else:
            info["orient"] = "v"
            info["pred"] = up_neighbor
            info["succ"] = down_neighbor
            info["source"] = grid[: box["r0"], col].astype(int).tolist() if top_source else None
        enriched.append(info)
    return enriched


def _paint_box_output(out: np.ndarray, box: dict[str, object], seq: list[int]) -> None:
    if box["orient"] == "h":
        row = int(box["cr"])
        start = int(box["c1"]) + 1
        assert start + len(seq) <= out.shape[1]
        out[row, start : start + len(seq)] = np.array(seq, dtype=int)
    else:
        col = int(box["cc"])
        start = int(box["r1"]) + 1
        assert start + len(seq) <= out.shape[0]
        out[start : start + len(seq), col] = np.array(seq, dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mapping, legend_bbox = _parse_legend(g)
    boxes = _annotate_boxes(g, _find_boxes(g, legend_bbox))
    out = g.copy()
    sources = [i for i, box in enumerate(boxes) if box["pred"] is None]
    assert sources
    visited: set[int] = set()
    for source in sources:
        seq = boxes[source]["source"]
        assert isinstance(seq, list) and seq
        cur = source
        while cur is not None:
            assert cur not in visited
            visited.add(cur)
            box = boxes[cur]
            seq = _apply_op(seq, mapping[int(box["color"])])
            _paint_box_output(out, box, seq)
            succ = box["succ"]
            cur = int(succ) if succ is not None else None
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    pts = np.argwhere(mask)
    for r, c in pts:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _legend_grid(
    color_to_op: dict[int, str], sample_seq: list[int], corner: str
) -> tuple[np.ndarray, tuple[int, int]]:
    entries = list(color_to_op.items())
    random.shuffle(entries)
    sample_len = len(sample_seq)
    height = 2 + len(entries) * 3 + (len(entries) - 1)
    width = 2 + 3 + 1 + sample_len + 1 + sample_len
    legend = np.full((height, width), BG, dtype=int)
    legend[0, :] = GREY
    legend[-1, :] = GREY
    legend[:, 0] = GREY
    legend[:, -1] = GREY
    for i, (color, op) in enumerate(entries):
        top = 1 + i * 4
        legend[top, 1:4] = color
        legend[top + 2, 1:4] = color
        legend[top : top + 3, 1] = color
        legend[top : top + 3, 3] = color
        in_seq = np.array(sample_seq, dtype=int)
        out_seq = np.array(_apply_op(sample_seq, op), dtype=int)
        legend[top + 1, 5 : 5 + sample_len] = in_seq
        legend[top + 1, 6 + sample_len : 6 + 2 * sample_len] = out_seq
    return legend, (height, width)


def _empty_grid(h: int, w: int) -> np.ndarray:
    return np.full((h, w), BG, dtype=int)


def _place_legend(grid: np.ndarray, legend: np.ndarray, corner: str) -> np.ndarray:
    h, w = grid.shape
    lh, lw = legend.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - lw
    elif corner == "bl":
        r0, c0 = h - lh, 0
    else:
        r0, c0 = h - lh, w - lw
    grid[r0 : r0 + lh, c0 : c0 + lw] = legend
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r0 + lh, c0 : c0 + lw] = True
    return mask


def _chain_total(length: int, stripe_len: int) -> int:
    return length * BOX_SIZE + (length + 1) * stripe_len


def _horizontal_mask(shape: tuple[int, int], center_r: int, length: int, stripe_len: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w = shape
    total_w = _chain_total(length, stripe_len)
    mask = np.zeros((h, w), dtype=bool)
    mask[center_r, :total_w] = True
    boxes: list[tuple[int, int]] = []
    col = stripe_len
    for _ in range(length):
        r0 = center_r - BOX_HALF
        c0 = col
        mask[r0 : r0 + BOX_SIZE, c0 : c0 + BOX_SIZE] = True
        boxes.append((r0, c0))
        col += BOX_SIZE + stripe_len
    return mask, boxes


def _vertical_mask(shape: tuple[int, int], center_c: int, length: int, stripe_len: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w = shape
    total_h = _chain_total(length, stripe_len)
    mask = np.zeros((h, w), dtype=bool)
    mask[:total_h, center_c] = True
    boxes: list[tuple[int, int]] = []
    row = stripe_len
    for _ in range(length):
        r0 = row
        c0 = center_c - BOX_HALF
        mask[r0 : r0 + BOX_SIZE, c0 : c0 + BOX_SIZE] = True
        boxes.append((r0, c0))
        row += BOX_SIZE + stripe_len
    return mask, boxes


def _draw_box(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    grid[r0, c0 : c0 + BOX_SIZE] = color
    grid[r0 + BOX_SIZE - 1, c0 : c0 + BOX_SIZE] = color
    grid[r0 : r0 + BOX_SIZE, c0] = color
    grid[r0 : r0 + BOX_SIZE, c0 + BOX_SIZE - 1] = color


def _assign_box_colors(border_colors: list[int], total_boxes: int) -> list[int]:
    colors: list[int] = []
    base = border_colors[:]
    while len(colors) < total_boxes:
        random.shuffle(base)
        colors.extend(base)
    return colors[:total_boxes]


def _render_expected(
    inp: np.ndarray, chains: list[dict[str, object]], color_to_op: dict[int, str]
) -> np.ndarray:
    out = inp.copy()
    for chain in chains:
        seq = list(chain["source"])
        for box in chain["boxes"]:
            seq = _apply_op(seq, color_to_op[int(box["color"])])
            if chain["orient"] == "h":
                row = int(box["r0"]) + BOX_HALF
                start = int(box["c0"]) + BOX_SIZE
                out[row, start : start + len(seq)] = np.array(seq, dtype=int)
            else:
                col = int(box["c0"]) + BOX_HALF
                start = int(box["r0"]) + BOX_SIZE
                out[start : start + len(seq), col] = np.array(seq, dtype=int)
    return out


def _try_place_chain(
    occ: np.ndarray, orient: str, length: int, stripe_len: int
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    h, w = occ.shape
    candidates: list[tuple[np.ndarray, list[tuple[int, int]]]] = []
    if orient == "h":
        total_w = _chain_total(length, stripe_len)
        if total_w > w:
            return None
        for center_r in range(BOX_HALF, h - BOX_HALF):
            mask, boxes = _horizontal_mask((h, w), center_r, length, stripe_len)
            if np.any(_dilate(mask) & occ):
                continue
            candidates.append((mask, boxes))
    else:
        total_h = _chain_total(length, stripe_len)
        if total_h > h:
            return None
        for center_c in range(BOX_HALF, w - BOX_HALF):
            mask, boxes = _vertical_mask((h, w), center_c, length, stripe_len)
            if np.any(_dilate(mask) & occ):
                continue
            candidates.append((mask, boxes))
    return random.choice(candidates) if candidates else None


def _build_chain(
    grid: np.ndarray,
    occ: np.ndarray,
    orient: str,
    length: int,
    stripe_len: int,
    box_colors: list[int],
    stripe_palette: list[int],
) -> dict[str, object] | None:
    placed = _try_place_chain(occ, orient, length, stripe_len)
    if placed is None:
        return None
    mask, coords = placed
    occ |= _dilate(mask)
    source = random.sample(stripe_palette, stripe_len)
    boxes: list[dict[str, int]] = []
    for (r0, c0), color in zip(coords, box_colors):
        _draw_box(grid, r0, c0, color)
        boxes.append({"r0": r0, "c0": c0, "color": color})
    if orient == "h":
        row = coords[0][0] + BOX_HALF
        grid[row, :stripe_len] = np.array(source, dtype=int)
    else:
        col = coords[0][1] + BOX_HALF
        grid[:stripe_len, col] = np.array(source, dtype=int)
    return {"orient": orient, "boxes": boxes, "source": source}


def _gen_attempt() -> np.ndarray | None:
    stripe_len = random.choice([4, 5])
    border_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    stripe_palette = [c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c not in border_colors]
    color_to_op = dict(zip(border_colors, OPS))
    sample_seq = random.sample(stripe_palette, 5)

    chain_count = random.choice([1, 2])
    if chain_count == 1:
        chain_specs = [(random.choice(["h", "v"]), 3)]
    else:
        chain_specs = [("h", random.choice([2, 3])), ("v", random.choice([2, 3]))]
        random.shuffle(chain_specs)

    legend_corner = random.choice(["tl", "tr", "bl", "br"])
    legend, (legend_h, legend_w) = _legend_grid(color_to_op, sample_seq, legend_corner)

    horiz_totals = [_chain_total(length, stripe_len) for orient, length in chain_specs if orient == "h"]
    vert_totals = [_chain_total(length, stripe_len) for orient, length in chain_specs if orient == "v"]
    min_h = max([legend_h + 6, 18] + vert_totals)
    min_w = max([legend_w + 6, 18] + horiz_totals)
    h = random.randint(min_h, min_h + 4)
    w = random.randint(min_w, min_w + 4)
    grid = _empty_grid(h, w)
    occ = _place_legend(grid, legend, legend_corner)
    occ = _dilate(occ)

    total_boxes = sum(length for _orient, length in chain_specs)
    assigned = _assign_box_colors(border_colors, total_boxes)

    chains: list[dict[str, object]] = []
    offset = 0
    for orient, length in chain_specs:
        chain = _build_chain(
            grid,
            occ,
            orient,
            length,
            stripe_len,
            assigned[offset : offset + length],
            stripe_palette,
        )
        if chain is None:
            return None
        chains.append(chain)
        offset += length

    expected = _render_expected(grid, chains, color_to_op)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
