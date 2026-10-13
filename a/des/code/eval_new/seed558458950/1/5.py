from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue

            q: deque[tuple[int, int]] = deque([(sr, sc)])
            visited[sr, sc] = True
            cells: list[tuple[int, int]] = []

            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            comps.append(np.asarray(cells, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max()) + 1
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max()) + 1
    return r0, r1, c0, c1


def _is_full_frame(grid: np.ndarray, frame_color: int, bg: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    if r1 - r0 < 3 or c1 - c0 < 3:
        return False

    patch = grid[r0:r1, c0:c1]
    if not np.all(patch[0, :] == frame_color):
        return False
    if not np.all(patch[-1, :] == frame_color):
        return False
    if not np.all(patch[:, 0] == frame_color):
        return False
    if not np.all(patch[:, -1] == frame_color):
        return False

    inner = patch[1:-1, 1:-1]
    if inner.size == 0:
        return False
    return bool(np.all(inner == bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    将左侧多色对象按隐含深度拆分并在右侧对应的目标框内按层重建。

    Input structure:
    - bg = 全图中最常见颜色（黑底）。
    - 左半区有若干连通多色对象，每个对象由2到3种颜色组成，像素是多层矩形叠加后的可见结果。
    - 右半区有一列从上到下的单色图块作为深度图例；右半区还含有与每个对象大小一致的灰色框，内部为空。

    Rule:
    - 对每个左侧对象提取每种颜色的连通像素簇，取每簇最小外接矩形并按“图例上到下=底到顶”的颜色顺序绘制。
    - 以对象在右侧对应目标框的内部区域为画布，按顺序填充这些矩形；后画层覆盖先画层。
    - 目标框外内容（原图中的其他区域与图例）保持不变。

    Key insight:
    - 仅有可见表面不足以直接知道每层完整形状，必须先按颜色分层并用最小外接矩形补全，再结合统一的深度顺序进行重叠合成。
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    if h == 0 or w == 0:
        return g.copy()

    out = g.copy()
    flat, counts = np.unique(g.ravel(), return_counts=True)
    bg = int(flat[int(np.argmax(counts))])
    split = w // 2

    # 1) Detect source objects in the left half.
    left_mask = g[:, :split] != bg
    left_components = _connected_components(left_mask)
    objects: list[dict[str, Any]] = []

    for comp in left_components:
        if comp.size == 0:
            continue
        r0, r1, c0, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        sub = g[r0:r1, c0:c1]
        colors = [int(v) for v in np.unique(sub) if int(v) != bg]
        if not colors:
            continue

        objects.append(
            {
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "h": r1 - r0,
                "w": c1 - c0,
                "colors": colors,
            }
        )

    if not objects:
        return out

    # 2) Find frame color and target boxes in the right half.
    right = g[:, split:]
    right_vals = right[right != bg]
    if right_vals.size == 0:
        return out

    vals, cnts = np.unique(right_vals, return_counts=True)
    frame_color = int(vals[int(np.argmax(cnts))])

    frame_mask = (g == frame_color) & (np.arange(w)[None, :] >= split)
    right_components = _connected_components(frame_mask)

    boxes: list[dict[str, Any]] = []
    for comp in right_components:
        if comp.size == 0:
            continue
        r0, r1, c0, c1 = _bbox(comp)
        if c0 < split:
            continue
        if not _is_full_frame(g, frame_color, bg, r0, c0, r1, c1):
            continue

        ih = r1 - r0 - 2
        iw = c1 - c0 - 2
        if ih <= 0 or iw <= 0:
            continue

        boxes.append(
            {
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "h": ih,
                "w": iw,
            }
        )

    if not boxes:
        return out

    # 3) Detect legend colors from a one-column vertical column in the right half.
    legend_candidates: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for rr in range(h):
        for cc in range(split, w):
            v = int(g[rr, cc])
            if v == bg or v == frame_color:
                continue
            legend_candidates[cc].append((rr, v))

    legend_colors: list[int] = []
    if legend_candidates:
        legend_col = max(legend_candidates.items(), key=lambda kv: len(kv[1]))[0]
        legend_colors = [int(v) for _, v in sorted(legend_candidates[legend_col], key=lambda p: p[0])]
    legend_index = {v: i for i, v in enumerate(legend_colors)}
    unknown_depth = len(legend_colors)

    # 4) Match each source object to one target box.
    obj_by_size: dict[tuple[int, int], list[int]] = defaultdict(list)
    box_by_size: dict[tuple[int, int], list[int]] = defaultdict(list)

    for oi, obj in enumerate(objects):
        obj_by_size[(obj["h"], obj["w"])].append(oi)
    for bi, box in enumerate(boxes):
        box_by_size[(box["h"], box["w"])].append(bi)

    matched: list[tuple[int, int]] = []
    used_o: set[int] = set()
    used_b: set[int] = set()

    for size in list(obj_by_size.keys()):
        oi_list = sorted(obj_by_size[size], key=lambda i: (objects[i]["r0"], objects[i]["c0"]))
        bi_list = sorted(box_by_size.get(size, []), key=lambda i: (boxes[i]["r0"], boxes[i]["c0"]))
        for oi, bi in zip(oi_list, bi_list):
            matched.append((oi, bi))
            used_o.add(oi)
            used_b.add(bi)

    rem_o = sorted((i for i in range(len(objects)) if i not in used_o), key=lambda i: (objects[i]["r0"], objects[i]["c0"]))
    rem_b = sorted((i for i in range(len(boxes)) if i not in used_b), key=lambda i: (boxes[i]["r0"], boxes[i]["c0"]))
    for oi, bi in zip(rem_o, rem_b):
        matched.append((oi, bi))

    # 5) Rebuild each pair by layering rectangle-completed color layers.
    for oi, bi in matched:
        obj = objects[oi]
        box = boxes[bi]
        if obj["h"] != box["h"] or obj["w"] != box["w"]:
            continue

        or0, or1 = obj["r0"], obj["r1"]
        oc0, oc1 = obj["c0"], obj["c1"]
        br0, br1 = box["r0"] + 1, box["r1"] - 1
        bc0, bc1 = box["c0"] + 1, box["c1"] - 1

        out[br0:br1, bc0:bc1] = bg

        obj_patch = g[or0:or1, oc0:oc1]
        layers: list[tuple[int, int, int, int, int]] = []  # color, r0, r1, c0, c1
        for color in [c for c in np.unique(obj_patch) if c != bg]:
            color_mask = obj_patch == color
            for cc in _connected_components(color_mask):
                if cc.size == 0:
                    continue
                lr0, lr1, lc0, lc1 = _bbox(cc)
                layers.append((int(color), lr0, lr1, lc0, lc1))

        layers.sort(key=lambda x: (legend_index.get(x[0], unknown_depth), x[1], x[3]))

        for color, lr0, lr1, lc0, lc1 in layers:
            rr0 = br0 + lr0
            rr1 = br0 + lr1
            cc0 = bc0 + lc0
            cc1 = bc0 + lc1
            out[rr0:rr1, cc0:cc1] = color

    return out


def generate_input() -> np.ndarray:
    bg = 0
    frame_color = 1
    palette = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=int)

    def _rects_from_colors(base_top: int, base_left: int, base_h: int, base_w: int, colors: list[int], rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
        rects = [(base_top, base_left, base_h, base_w)]
        top, left = base_top, base_left
        h, w = base_h, base_w
        for _ in range(1, len(colors)):
            if h <= 1 and w <= 1:
                rects.append((top, left, 1, 1))
                continue

            chosen = None
            for _ in range(20):
                dt = int(rng.integers(0, min(3, h + 1)))
                db = int(rng.integers(0, min(3, h + 1 - dt)))
                dl = int(rng.integers(0, min(3, w + 1)))
                dr = int(rng.integers(0, min(3, w + 1 - dl)))

                if dt + db + dl + dr == 0:
                    if h >= 2:
                        dt = 1
                    elif w >= 2:
                        dl = 1
                        dr = 0
                    else:
                        dt = 0

                nh = h - dt - db
                nw = w - dl - dr
                if nh >= 1 and nw >= 1:
                    chosen = (dt, db, dl, dr, nh, nw)
                    break

            if chosen is None:
                chosen = (0, 0, 0, 0, h, w)

            dt, db, dl, dr, h, w = chosen
            top += dt
            left += dl
            rects.append((top, left, h, w))
        return rects

    for _ in range(400):
        H = int(np.random.default_rng().integers(25, 31))
        W = int(np.random.default_rng().integers(24, 31))
        rng = np.random.default_rng()
        split = W // 2
        if split < 10:
            continue
        if W - split < 8:
            continue

        grid = np.full((H, W), bg, dtype=int)
        occupied = np.zeros((H, W), dtype=bool)

        rng.shuffle(palette)
        palette_shuffled = palette.copy()
        num_legend = int(rng.integers(4, 7))
        legend_colors = [int(v) for v in palette_shuffled[:num_legend]]

        num_objects = int(rng.integers(1, 4))
        objects: list[dict[str, int]] = []
        used_sizes: set[tuple[int, int]] = set()

        for _obj in range(num_objects):
            placed = False
            for _ in range(250):
                obj_h = int(rng.integers(3, 7))
                obj_w = int(rng.integers(3, 7))
                if (obj_h, obj_w) in used_sizes and num_objects > 1:
                    continue
                if split - obj_w - 1 < 2 or H - obj_h - 1 < 2:
                    continue

                top = int(rng.integers(1, H - obj_h - 1))
                left = int(rng.integers(1, split - obj_w - 1))

                t0 = max(0, top - 1)
                t1 = min(H, top + obj_h + 1)
                l0 = max(0, left - 1)
                l1 = min(split, left + obj_w + 1)
                if occupied[t0:t1, l0:l1].any():
                    continue

                layer_count = int(rng.choice([2, 3], p=np.array([0.55, 0.45])))
                color_idx = np.sort(rng.choice(len(legend_colors), size=layer_count, replace=False))
                colors = [legend_colors[int(i)] for i in color_idx]

                rects = _rects_from_colors(top, left, obj_h, obj_w, colors, rng)

                for color, rect in zip(colors, rects):
                    r, c, hh, ww = rect
                    grid[r : r + hh, c : c + ww] = color
                    occupied[r : r + hh, c : c + ww] = True

                objects.append(
                    {
                        "h": obj_h,
                        "w": obj_w,
                        "top": top,
                        "left": left,
                    }
                )
                used_sizes.add((obj_h, obj_w))
                placed = True
                break

            if not placed:
                break

        if len(objects) != num_objects:
            continue

        # Place target boxes in the right half with a fixed column for each object.
        max_box_w = W - split - 2
        if any(obj["w"] + 2 > max_box_w for obj in objects):
            continue

        box_order = np.arange(num_objects)
        rng.shuffle(box_order)
        top_cursor = 1
        placed_boxes = 0
        for idx in box_order:
            obj = objects[idx]
            box_h = obj["h"] + 2
            box_w = obj["w"] + 2
            box_left = split + 1
            if top_cursor + box_h > H:
                break
            if box_left + box_w >= W - 1:
                break

            br0, bc0 = top_cursor, box_left
            br1, bc1 = top_cursor + box_h, box_left + box_w

            grid[br0, bc0:bc1] = frame_color
            grid[br1 - 1, bc0:bc1] = frame_color
            grid[br0:br1, bc0] = frame_color
            grid[br0:br1, bc1 - 1] = frame_color
            top_cursor = br1 + 1
            placed_boxes += 1

        if placed_boxes != num_objects:
            continue

        # Draw legend in the rightmost column.
        for i, color in enumerate(legend_colors):
            r = i + 1
            if r >= H:
                break
            grid[r, W - 1] = color

        return grid

    return np.full((25, 30), bg, dtype=int)
