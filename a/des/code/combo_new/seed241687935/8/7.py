from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GRAY = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BOARD_LIMIT = 30
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _line_like(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return (h == 1 or w == 1) and len(coords) == h * w


def _infer_bridge_color(grid: np.ndarray, bg: int) -> int | None:
    candidates: list[int] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components(grid == color)
        if comps and all(_line_like(comp) for comp in comps):
            candidates.append(int(color))
    if len(candidates) == 1:
        return int(candidates[0])
    if len(candidates) == 0:
        return None
    if GRAY in candidates:
        return GRAY
    raise AssertionError("expected a unique bridge color")


def _bridge_side(rect: dict, bridge_cell: tuple[int, int]) -> str:
    r0, c0, r1, c1 = rect["bbox"]
    br, bc = bridge_cell
    if br == r0 - 1 and c0 <= bc <= c1:
        return "top"
    if br == r1 + 1 and c0 <= bc <= c1:
        return "bottom"
    if bc == c0 - 1 and r0 <= br <= r1:
        return "left"
    if bc == c1 + 1 and r0 <= br <= r1:
        return "right"
    raise AssertionError("bridge must touch a rectangle side orthogonally")


def _collect_rectangles(
    grid: np.ndarray,
    bg: int,
    bridge_color: int | None,
) -> tuple[list[dict], np.ndarray]:
    h, w = map(int, grid.shape)
    rect_id_grid = np.full((h, w), -1, dtype=int)
    rectangles: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {bg, bridge_color}):
        comps = _components(grid == color)
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            assert hh >= 2 and ww >= 2, "every color block must be a rectangle of at least 2x2"
            assert len(comp) == hh * ww, "color blocks must be solid rectangles"
            assert bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color))
            rect_id = len(rectangles)
            rect_id_grid[r0 : r1 + 1, c0 : c1 + 1] = rect_id
            rectangles.append(
                {
                    "id": rect_id,
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "height": int(hh),
                    "width": int(ww),
                    "area": int(hh * ww),
                }
            )
    assert rectangles, "expected at least one rectangle"
    return rectangles, rect_id_grid


def _collect_bridges(
    grid: np.ndarray,
    bridge_color: int | None,
    rect_id_grid: np.ndarray,
    rectangles: list[dict],
) -> list[dict]:
    if bridge_color is None:
        return []

    h, w = map(int, grid.shape)
    bridges: list[dict] = []
    for comp in _components(grid == bridge_color):
        assert _line_like(comp), "each bridge must be a straight one-cell-wide line"
        touched: dict[int, str] = {}
        for br, bc in comp:
            for dr, dc in NEIGHBORS4:
                nr, nc = int(br + dr), int(bc + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                rect_id = int(rect_id_grid[nr, nc])
                if rect_id < 0:
                    continue
                side = _bridge_side(rectangles[rect_id], (int(br), int(bc)))
                prev = touched.get(rect_id)
                if prev is None:
                    touched[rect_id] = side
                else:
                    assert prev == side, "bridge must contact one side of each rectangle"
        assert len(touched) == 2, "each bridge must connect exactly two rectangles"
        items = list(touched.items())
        a, side_a = items[0]
        b, side_b = items[1]
        bridges.append(
            {
                "coords": comp,
                "a": int(a),
                "b": int(b),
                "side_a": side_a,
                "side_b": side_b,
            }
        )
    return bridges


def _graph(rectangles: list[dict], bridges: list[dict]) -> dict[int, list[tuple[int, str]]]:
    graph = {rect["id"]: [] for rect in rectangles}
    for bridge in bridges:
        a = int(bridge["a"])
        b = int(bridge["b"])
        graph[a].append((b, str(bridge["side_a"])))
        graph[b].append((a, str(bridge["side_b"])))
    return graph


def _groups(graph: dict[int, list[tuple[int, str]]]) -> list[list[int]]:
    seen: set[int] = set()
    groups: list[list[int]] = []
    for start in sorted(graph):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp: list[int] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt, _side in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        groups.append(comp)
    return groups


def _entry_sides(
    group_ids: list[int],
    body_id: int,
    graph: dict[int, list[tuple[int, str]]],
) -> dict[int, str]:
    side_by_id: dict[int, str] = {}
    q = deque([body_id])
    visited = {body_id}
    while q:
        cur = q.popleft()
        for nxt, side_at_cur in graph[cur]:
            candidate = side_at_cur if cur == body_id else side_by_id[cur]
            if nxt not in visited:
                visited.add(nxt)
                side_by_id[nxt] = candidate
                q.append(nxt)
            else:
                if nxt != body_id:
                    assert side_by_id[nxt] == candidate, "entry side must be unambiguous"
    assert visited == set(group_ids), "each rectangle in a group must reach the body"
    return side_by_id


def _embed_bbox(body: dict, child: dict, side: str) -> tuple[int, int, int, int]:
    br0, bc0, br1, bc1 = body["bbox"]
    bh = int(body["height"])
    bw = int(body["width"])
    ch = int(child["height"])
    cw = int(child["width"])
    assert ch <= bh and cw <= bw, "embedded block must fit inside the body"

    if side == "top":
        r0 = br0
        c0 = bc0 + (bw - cw) // 2
    elif side == "bottom":
        r0 = br1 - ch + 1
        c0 = bc0 + (bw - cw) // 2
    elif side == "left":
        r0 = br0 + (bh - ch) // 2
        c0 = bc0
    else:
        assert side == "right"
        r0 = br0 + (bh - ch) // 2
        c0 = bc1 - cw + 1
    return (int(r0), int(c0), int(r0 + ch - 1), int(c0 + cw - 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group bridge-connected rectangles and stack the smaller members inside each unique largest body rectangle.

    Input structure:
    - The background is black.
    - Foreground objects are separated solid monochrome rectangles of different colors.
    - Some rectangle pairs are linked by one-cell-wide straight grey bridges, and the bridge graph
      splits the rectangles into connected groups.
    - Every group has exactly one rectangle with strictly largest area.

    Rule:
    - Identify the bridge-connected groups and keep the unique largest rectangle of each group as the body.
    - For every other rectangle in the group, trace its path to the body; the side where that path enters
      the body determines the embedding side inside the body: top, right, bottom, or left.
    - Draw the non-body rectangles inside the body in increasing area order, aligned to the chosen side
      and centered along the other axis; later, larger rectangles overwrite earlier ones.
    - Erase every original non-body rectangle and every bridge pixel to black, leaving only the bodies with
      their embedded interior stacks.

    Key insight:
    - The grey bridges define hidden graph groups, and a non-body rectangle inherits its placement side from
      the final bridge segment that reaches the group's largest rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BG
    bridge_color = _infer_bridge_color(g, bg)
    rectangles, rect_id_grid = _collect_rectangles(g, bg, bridge_color)
    bridges = _collect_bridges(g, bridge_color, rect_id_grid, rectangles)
    graph = _graph(rectangles, bridges)

    out = np.full_like(g, bg)
    rect_by_id = {rect["id"]: rect for rect in rectangles}
    for group_ids in _groups(graph):
        group_rects = [rect_by_id[idx] for idx in group_ids]
        ranked = sorted(group_rects, key=lambda rect: int(rect["area"]), reverse=True)
        assert len(ranked) == 1 or ranked[0]["area"] > ranked[1]["area"], "body must be uniquely largest"
        body = ranked[0]
        _paint_rect(out, body["bbox"], body["color"])

        if len(group_ids) == 1:
            continue

        accessories = ranked[1:]
        areas = [int(rect["area"]) for rect in accessories]
        assert len(set(areas)) == len(areas), "accessory areas must be distinct to define a paint order"
        side_by_id = _entry_sides(group_ids, int(body["id"]), graph)
        for rect in sorted(accessories, key=lambda item: int(item["area"])):
            side = side_by_id[int(rect["id"])]
            _paint_rect(out, _embed_bbox(body, rect, side), rect["color"])

    return out


def _sample_group_sizes() -> list[int]:
    group_count = random.choices([2, 3, 1], weights=[5, 3, 1])[0]
    sizes = [2] * group_count
    max_extra = min(8 - sum(sizes), sum(4 - size for size in sizes))
    extra = random.randint(1 if max_extra > 0 else 0, max_extra)
    for _ in range(extra):
        choices = [idx for idx, size in enumerate(sizes) if size < 4]
        sizes[random.choice(choices)] += 1
    if len(sizes) == 1:
        sizes[0] = max(3, sizes[0])
    return sizes


def _branch_length_options(accessory_count: int, require_depth: bool) -> list[tuple[int, ...]]:
    options: list[tuple[int, ...]] = []
    for n_branches in range(1, min(4, accessory_count) + 1):
        starts = [1] * n_branches
        extra = accessory_count - n_branches

        def rec(pos: int, current: list[int], remaining: int) -> None:
            if pos == n_branches:
                if remaining == 0:
                    candidate = tuple(current)
                    if max(candidate) <= 2:
                        if not require_depth or max(candidate) >= 2:
                            options.append(candidate)
                return
            cap = 1
            add_max = min(cap, remaining)
            for add in range(add_max + 1):
                current[pos] += add
                rec(pos + 1, current, remaining - add)
                current[pos] -= add

        rec(0, starts[:], extra)
    assert options, "expected at least one branch decomposition"
    deduped = sorted(set(tuple(sorted(option, reverse=True)) for option in options))
    return deduped


def _choose_sides(n_sides: int, side_counts: dict[str, int]) -> list[str]:
    available = list(SIDES)
    chosen: list[str] = []
    for _ in range(n_sides):
        min_count = min(side_counts[side] for side in available)
        candidates = [side for side in available if side_counts[side] == min_count]
        side = random.choice(candidates)
        chosen.append(side)
        available.remove(side)
        side_counts[side] += 1
    return chosen


def _accessory_choices(body_h: int, body_w: int) -> dict[int, list[tuple[int, int]]]:
    body_area = body_h * body_w
    by_area: dict[int, list[tuple[int, int]]] = {}
    for h in range(2, min(4, body_h) + 1):
        for w in range(2, min(4, body_w) + 1):
            area = h * w
            if area < body_area:
                by_area.setdefault(area, []).append((h, w))
    return by_area


def _sample_body(accessory_count: int) -> tuple[int, int, dict[int, list[tuple[int, int]]]]:
    options: list[tuple[int, int, dict[int, list[tuple[int, int]]]]] = []
    for h in range(4, 7):
        for w in range(4, 7):
            choices = _accessory_choices(h, w)
            if len(choices) >= accessory_count:
                options.append((h, w, choices))
    assert options, "expected body sizes that can host all accessories"
    return random.choice(options)


def _sample_axis(length: int) -> int:
    low = max(0, length // 4)
    high = min(length - 1, length - 1 - length // 4)
    if low > high:
        low, high = 0, length - 1
    return random.randint(low, high)


def _place_child(
    parent_bbox: tuple[int, int, int, int],
    side: str,
    child_h: int,
    child_w: int,
    axis: int,
    bridge_len: int,
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    pr0, pc0, pr1, pc1 = parent_bbox
    if side == "top":
        c0 = axis - child_w // 2
        r0 = pr0 - bridge_len - child_h
        bridge = [(rr, axis) for rr in range(r0 + child_h, pr0)]
    elif side == "bottom":
        c0 = axis - child_w // 2
        r0 = pr1 + bridge_len + 1
        bridge = [(rr, axis) for rr in range(pr1 + 1, r0)]
    elif side == "left":
        r0 = axis - child_h // 2
        c0 = pc0 - bridge_len - child_w
        bridge = [(axis, cc) for cc in range(c0 + child_w, pc0)]
    else:
        assert side == "right"
        r0 = axis - child_h // 2
        c0 = pc1 + bridge_len + 1
        bridge = [(axis, cc) for cc in range(pc1 + 1, c0)]
    bbox = (int(r0), int(c0), int(r0 + child_h - 1), int(c0 + child_w - 1))
    assert bridge and len(bridge) == bridge_len
    return bbox, bridge


def _render_group(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    rects = [spec["body"], *spec["accessories"]]
    all_rows = []
    all_cols = []
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        all_rows.extend([r0, r1])
        all_cols.extend([c0, c1])
    for bridge in spec["bridges"]:
        for r, c in bridge["coords"]:
            all_rows.append(r)
            all_cols.append(c)
    min_r = min(all_rows)
    max_r = max(all_rows)
    min_c = min(all_cols)
    max_c = max(all_cols)

    h = max_r - min_r + 1
    w = max_c - min_c + 1
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    shifted_body = dict(spec["body"])
    br0, bc0, br1, bc1 = spec["body"]["bbox"]
    shifted_body["bbox"] = (br0 - min_r, bc0 - min_c, br1 - min_r, bc1 - min_c)

    shifted_accessories: list[dict] = []
    for bridge in spec["bridges"]:
        for r, c in bridge["coords"]:
            rr = r - min_r
            cc = c - min_c
            assert inp[rr, cc] == BG
            inp[rr, cc] = GRAY

    for rect in rects:
        rr0, cc0, rr1, cc1 = rect["bbox"]
        rr0 -= min_r
        rr1 -= min_r
        cc0 -= min_c
        cc1 -= min_c
        assert bool(np.all(inp[rr0 : rr1 + 1, cc0 : cc1 + 1] == BG))
        inp[rr0 : rr1 + 1, cc0 : cc1 + 1] = rect["color"]
        if rect is not spec["body"]:
            shifted = dict(rect)
            shifted["bbox"] = (rr0, cc0, rr1, cc1)
            shifted_accessories.append(shifted)

    _paint_rect(out, shifted_body["bbox"], shifted_body["color"])
    for rect in sorted(shifted_accessories, key=lambda item: int(item["area"])):
        _paint_rect(out, _embed_bbox(shifted_body, rect, rect["root_side"]), rect["color"])

    return inp, out


def _make_group(
    size: int,
    colors: list[int],
    require_depth: bool,
    side_counts: dict[str, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    accessory_count = size - 1
    body_h, body_w, choices_by_area = _sample_body(accessory_count)
    branch_lengths = list(random.choice(_branch_length_options(accessory_count, require_depth)))
    random.shuffle(branch_lengths)
    sides = _choose_sides(len(branch_lengths), side_counts)

    area_values = random.sample(sorted(choices_by_area), accessory_count)
    accessory_specs: list[dict] = []
    for color, area in zip(colors[1:], area_values):
        h, w = random.choice(choices_by_area[area])
        accessory_specs.append(
            {
                "color": int(color),
                "height": int(h),
                "width": int(w),
                "area": int(area),
            }
        )
    random.shuffle(accessory_specs)

    spec = {
        "body": {
            "color": int(colors[0]),
            "bbox": (0, 0, body_h - 1, body_w - 1),
            "height": int(body_h),
            "width": int(body_w),
            "area": int(body_h * body_w),
        },
        "accessories": [],
        "bridges": [],
    }

    cursor = 0
    for side, branch_len in zip(sides, branch_lengths):
        if side in {"top", "bottom"}:
            axis = _sample_axis(body_w - 1)
        else:
            axis = _sample_axis(body_h - 1)
        parent_bbox = spec["body"]["bbox"]
        if side in {"top", "bottom"}:
            axis += spec["body"]["bbox"][1]
        else:
            axis += spec["body"]["bbox"][0]

        for _depth in range(branch_len):
            child = dict(accessory_specs[cursor])
            cursor += 1
            bridge_len = random.randint(1, 2)
            child_bbox, bridge_coords = _place_child(
                parent_bbox,
                side,
                int(child["height"]),
                int(child["width"]),
                int(axis),
                bridge_len,
            )
            child["bbox"] = child_bbox
            child["root_side"] = side
            spec["accessories"].append(child)
            spec["bridges"].append({"coords": bridge_coords})
            parent_bbox = child_bbox

    inp, out = _render_group(spec)
    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return inp, out


def _boxes_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_groups(
    groups: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    board_in = np.full((BOARD_LIMIT, BOARD_LIMIT), BG, dtype=int)
    board_out = np.full((BOARD_LIMIT, BOARD_LIMIT), BG, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []

    order = sorted(range(len(groups)), key=lambda idx: int(np.prod(groups[idx][0].shape)), reverse=True)
    for idx in order:
        inp, out = groups[idx]
        h, w = map(int, inp.shape)
        placed = False
        for _ in range(200):
            r0 = random.randint(0, BOARD_LIMIT - h)
            c0 = random.randint(0, BOARD_LIMIT - w)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if not all(_boxes_separated(box, other, pad=1) for other in boxes):
                continue
            board_in[r0 : r0 + h, c0 : c0 + w] = np.maximum(board_in[r0 : r0 + h, c0 : c0 + w], inp)
            board_out[r0 : r0 + h, c0 : c0 + w] = np.maximum(board_out[r0 : r0 + h, c0 : c0 + w], out)
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    coords = np.argwhere(board_in != BG)
    if coords.size == 0:
        return None
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    top_margin = random.randint(0, min(3, r0))
    left_margin = random.randint(0, min(3, c0))
    bottom_margin = random.randint(0, min(3, BOARD_LIMIT - 1 - r1))
    right_margin = random.randint(0, min(3, BOARD_LIMIT - 1 - c1))
    rr0 = r0 - top_margin
    cc0 = c0 - left_margin
    rr1 = r1 + bottom_margin
    cc1 = c1 + right_margin
    return (
        board_in[rr0 : rr1 + 1, cc0 : cc1 + 1],
        board_out[rr0 : rr1 + 1, cc0 : cc1 + 1],
    )


def _gen_attempt() -> np.ndarray | None:
    sizes = _sample_group_sizes()
    total_rects = sum(sizes)
    colors = random.sample(RECT_COLORS, total_rects)
    require_depth_candidates = [idx for idx, size in enumerate(sizes) if size >= 3]
    require_depth_idx = random.choice(require_depth_candidates) if require_depth_candidates else None

    groups: list[tuple[np.ndarray, np.ndarray]] = []
    side_counts = {side: 0 for side in SIDES}
    cursor = 0
    for idx, size in enumerate(sizes):
        chunk = colors[cursor : cursor + size]
        cursor += size
        group = _make_group(
            size,
            chunk,
            require_depth=(idx == require_depth_idx),
            side_counts=side_counts,
        )
        if group is None:
            return None
        groups.append(group)

    placed = _place_groups(groups)
    if placed is None:
        return None
    inp, expected = placed

    if inp.ndim != 2 or expected.ndim != 2:
        return None
    if max(inp.shape) > BOARD_LIMIT or max(expected.shape) > BOARD_LIMIT:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-stacking puzzle")
