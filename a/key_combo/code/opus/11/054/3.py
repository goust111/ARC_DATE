from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_marker_top_left(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    hits = []
    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(grid[r : r + 2, c : c + 2] == BLACK):
                hits.append((r, c))
    assert len(hits) == 1, "expected a unique 2x2 black marker"
    return hits[0]


def _erase_cross(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    g = np.array(grid, dtype=int)
    mr, mc = _find_marker_top_left(g)
    out = g.copy()
    row_mask = np.zeros(g.shape, dtype=bool)
    row_mask[mr : mr + 2, :] = True
    row_mask[:, mc : mc + 2] = True
    out[row_mask & (g != GREY)] = BLACK
    return out, (mr, mc)


def _extract_surviving_components(grid: np.ndarray) -> list[dict]:
    items = []
    for color in sorted(set(int(x) for x in np.unique(grid)) - {BLACK, GREY}):
        for coords in _components_of_color(grid, color):
            bbox = _bbox_from_coords(coords)
            patch = _crop(grid, bbox)
            anchors = np.argwhere(patch == GREY)
            assert anchors.shape[0] == 1, "each surviving component bbox must contain exactly one gray anchor"
            items.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "patch": patch,
                    "anchor": (int(anchors[0, 0]), int(anchors[0, 1])),
                }
            )
    return items


def _overlay_group(items: list[dict]) -> np.ndarray:
    min_r = min(-item["anchor"][0] for item in items)
    min_c = min(-item["anchor"][1] for item in items)
    max_r = max(item["patch"].shape[0] - 1 - item["anchor"][0] for item in items)
    max_c = max(item["patch"].shape[1] - 1 - item["anchor"][1] for item in items)
    canvas = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for item in items:
        patch = item["patch"]
        ar, ac = item["anchor"]
        r0 = -min_r - ar
        c0 = -min_c - ac
        view = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        mask = patch != BLACK
        view[mask] = patch[mask]
    nz = np.argwhere(canvas != BLACK)
    return _crop(canvas, _bbox_from_coords(nz))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase a marker-defined cross, then overlay same-colored surviving components by their gray anchors.

    Input structure:
    - The grid contains exactly one solid 2x2 black marker block.
    - Away from that marker-defined cross, non-black colored regions each contain a single gray anchor pixel.
    - Gray pixels are protected from erasure and act as anchor points for the surviving regions.

    Rule:
    - Use the 2x2 black marker to select two rows and two columns, then repaint every selected non-gray
      cell black.
    - In the erased grid, find every surviving 4-connected monochrome component of colors other than black
      and gray; each component's bounding box contains exactly one gray pixel.
    - Crop each component to its bounding box, group components by color, and overlay each color group by
      aligning those gray anchor pixels.
    - If only one color group survives, return its overlaid crop. Otherwise stack the per-color overlays
      vertically in descending order of group size, breaking ties by the group's topmost source position,
      with one black row between groups.

    Key insight:
    - The same gray pixel serves two jobs: it survives the cross erasure and later becomes the registration
      anchor that reveals how same-colored fragments should be merged.
    """
    erased, _marker = _erase_cross(np.asarray(grid, dtype=int))
    parts = _extract_surviving_components(erased)
    groups: dict[int, list[dict]] = defaultdict(list)
    for item in parts:
        groups[item["color"]].append(item)

    rendered = []
    for color, items in groups.items():
        sprite = _overlay_group(items)
        top_r = min(int(item["bbox"][0]) for item in items)
        left_c = min(int(item["bbox"][1]) for item in items)
        rendered.append((len(items), top_r, left_c, color, sprite))

    rendered.sort(key=lambda x: (-x[0], x[1], x[2]))
    if len(rendered) == 1:
        return rendered[0][4]

    gap = 1
    out_h = sum(sprite.shape[0] for _n, _r, _c, _color, sprite in rendered) + gap * (len(rendered) - 1)
    out_w = max(sprite.shape[1] for _n, _r, _c, _color, sprite in rendered)
    out = np.zeros((out_h, out_w), dtype=int)
    cur = 0
    for idx, (_n, _r, _c, _color, sprite) in enumerate(rendered):
        out[cur : cur + sprite.shape[0], : sprite.shape[1]] = sprite
        cur += sprite.shape[0]
        if idx + 1 < len(rendered):
            cur += gap
    return out


def _component_relative_coords(item: dict) -> tuple[tuple[int, int], ...]:
    patch = item["patch"]
    ar, ac = item["anchor"]
    coords = []
    for r, c in np.argwhere(patch == item["color"]):
        coords.append((int(r) - ar, int(c) - ac))
    return tuple(sorted(coords))


def _make_group_shapes(count: int) -> list[dict] | None:
    for _ in range(200):
        union_h = random.randint(3, 5)
        union_w = random.randint(3, 5)
        anchor_r = random.randint(1, union_h - 2)
        anchor_c = random.randint(1, union_w - 2)
        top = -anchor_r
        bottom = union_h - 1 - anchor_r
        left = -anchor_c
        right = union_w - 1 - anchor_c
        sides = random.sample(["top", "bottom", "left", "right"], count)
        shapes = []
        rects = []
        for side in sides:
            t, b, l, r = top, bottom, left, right
            if side == "top":
                t += 1
            elif side == "bottom":
                b -= 1
            elif side == "left":
                l += 1
            else:
                r -= 1
            if b - t + 1 < 2 or r - l + 1 < 2:
                break
            rects.append((t, b, l, r))
        if len(rects) != count or len(set(rects)) != count:
            continue
        union_coords = set()
        for t, b, l, r in rects:
            h = b - t + 1
            w = r - l + 1
            ar = -t
            ac = -l
            shapes.append({"h": h, "w": w, "anchor": (ar, ac)})
            for rr in range(t, b + 1):
                for cc in range(l, r + 1):
                    if (rr, cc) != (0, 0):
                        union_coords.add((rr, cc))
        if any(shape["h"] * shape["w"] - 1 >= len(union_coords) for shape in shapes):
            continue
        return shapes
    return None


def _region_specs(h: int, w: int, mr: int, mc: int) -> list[dict]:
    return [
        {"name": "tl", "r0": 0, "r1": mr - 1, "c0": 0, "c1": mc - 1, "flushes": ("south", "east")},
        {"name": "tr", "r0": 0, "r1": mr - 1, "c0": mc + 2, "c1": w - 1, "flushes": ("south", "west")},
        {"name": "bl", "r0": mr + 2, "r1": h - 1, "c0": 0, "c1": mc - 1, "flushes": ("north", "east")},
        {"name": "br", "r0": mr + 2, "r1": h - 1, "c0": mc + 2, "c1": w - 1, "flushes": ("north", "west")},
    ]


def _candidate_positions(shape: dict, region: dict, mr: int, mc: int) -> list[dict]:
    h = shape["h"]
    w = shape["w"]
    r0_lo, r0_hi = region["r0"], region["r1"] - h + 1
    c0_lo, c0_hi = region["c0"], region["c1"] - w + 1
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return []
    out = []
    for r0 in range(r0_lo, r0_hi + 1):
        for c0 in range(c0_lo, c0_hi + 1):
            flushes = []
            if "south" in region["flushes"] and r0 + h - 1 == mr - 1:
                flushes.append("south")
            if "north" in region["flushes"] and r0 == mr + 2:
                flushes.append("north")
            if "east" in region["flushes"] and c0 + w - 1 == mc - 1:
                flushes.append("east")
            if "west" in region["flushes"] and c0 == mc + 2:
                flushes.append("west")
            if flushes:
                out.append({"r0": r0, "c0": c0, "flushes": tuple(flushes), "region": region["name"]})
    random.shuffle(out)
    return out


def _expanded_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0)


def _build_patch(shape: dict, color: int) -> np.ndarray:
    patch = np.full((shape["h"], shape["w"]), int(color), dtype=int)
    ar, ac = shape["anchor"]
    patch[ar, ac] = GREY
    return patch


def _place_components(h: int, w: int, mr: int, mc: int, groups: list[tuple[int, list[dict]]]) -> list[dict] | None:
    regions = _region_specs(h, w, mr, mc)
    entries = []
    for color, shapes in groups:
        for shape in shapes:
            entries.append({"color": color, "shape": shape, "patch": _build_patch(shape, color)})
    random.shuffle(entries)
    placed = []
    used_boxes: list[tuple[int, int, int, int]] = []
    for entry in entries:
        candidates = []
        for region in regions:
            candidates.extend(_candidate_positions(entry["shape"], region, mr, mc))
        random.shuffle(candidates)
        choice = None
        for cand in candidates:
            bbox = (
                cand["r0"],
                cand["c0"],
                cand["r0"] + entry["patch"].shape[0] - 1,
                cand["c0"] + entry["patch"].shape[1] - 1,
            )
            if any(_expanded_intersects(bbox, prev) for prev in used_boxes):
                continue
            choice = cand
            used_boxes.append(bbox)
            break
        if choice is None:
            return None
        placed.append(
            {
                "color": entry["color"],
                "patch": entry["patch"],
                "r0": choice["r0"],
                "c0": choice["c0"],
                "flushes": choice["flushes"],
                "bbox": used_boxes[-1],
            }
        )
    return placed


def _paint_components(grid: np.ndarray, placed: list[dict]) -> None:
    for item in placed:
        r0, c0 = item["r0"], item["c0"]
        patch = item["patch"]
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch


def _edge_attachment_cells(item: dict, side: str) -> list[tuple[int, int]]:
    patch = item["patch"]
    r0, c0 = item["r0"], item["c0"]
    cells = []
    if side == "south":
        rr = r0 + patch.shape[0] - 1
        for cc in range(patch.shape[1]):
            if int(patch[patch.shape[0] - 1, cc]) == int(item["color"]):
                cells.append((rr, c0 + cc))
    elif side == "north":
        rr = r0
        for cc in range(patch.shape[1]):
            if int(patch[0, cc]) == int(item["color"]):
                cells.append((rr, c0 + cc))
    elif side == "east":
        cc = c0 + patch.shape[1] - 1
        for rr in range(patch.shape[0]):
            if int(patch[rr, patch.shape[1] - 1]) == int(item["color"]):
                cells.append((r0 + rr, cc))
    else:
        cc = c0
        for rr in range(patch.shape[0]):
            if int(patch[rr, 0]) == int(item["color"]):
                cells.append((r0 + rr, cc))
    return cells


def _add_component_tails(inp: np.ndarray, placed: list[dict], mr: int, mc: int) -> int:
    changed = 0
    for item in placed:
        side = random.choice(item["flushes"])
        attachments = _edge_attachment_cells(item, side)
        if not attachments:
            continue
        ar, ac = random.choice(attachments)
        if side == "south":
            for rr in (mr, mr + 1):
                if int(inp[rr, ac]) != GREY:
                    continue
                inp[rr, ac] = int(item["color"])
                changed += 1
        elif side == "north":
            for rr in (mr + 1, mr):
                if int(inp[rr, ac]) != GREY:
                    continue
                inp[rr, ac] = int(item["color"])
                changed += 1
        elif side == "east":
            for cc in (mc, mc + 1):
                if int(inp[ar, cc]) != GREY:
                    continue
                inp[ar, cc] = int(item["color"])
                changed += 1
        else:
            for cc in (mc + 1, mc):
                if int(inp[ar, cc]) != GREY:
                    continue
                inp[ar, cc] = int(item["color"])
                changed += 1
    return changed


def _add_cross_clutter(inp: np.ndarray, mr: int, mc: int, colors: list[int]) -> int:
    h, w = inp.shape
    changed = 0
    for r in (mr, mr + 1):
        cols = list(range(w))
        random.shuffle(cols)
        for c in cols[: random.randint(max(4, w // 4), max(5, w // 2))]:
            if mc <= c <= mc + 1:
                continue
            if int(inp[r, c]) == GREY:
                inp[r, c] = random.choice(colors)
                changed += 1
    for c in (mc, mc + 1):
        rows = list(range(h))
        random.shuffle(rows)
        for r in rows[: random.randint(max(4, h // 4), max(5, h // 2))]:
            if mr <= r <= mr + 1:
                continue
            if int(inp[r, c]) == GREY:
                inp[r, c] = random.choice(colors)
                changed += 1
    return changed


def _group_output_stats(erased: np.ndarray) -> list[tuple[int, int]]:
    parts = _extract_surviving_components(erased)
    groups: dict[int, list[dict]] = defaultdict(list)
    for part in parts:
        groups[part["color"]].append(part)
    return sorted((int(color), len(items)) for color, items in groups.items())


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    mr = random.randint(4, h - 7)
    mc = random.randint(4, w - 7)

    if random.random() < 0.45:
        group_counts = [random.choice([2, 3])]
    else:
        group_counts = [3, 2]

    colors = random.sample(PALETTE, len(group_counts))
    groups = []
    for color, count in zip(colors, group_counts):
        shapes = _make_group_shapes(count)
        if shapes is None:
            return None
        groups.append((color, shapes))

    placed = _place_components(h, w, mr, mc, groups)
    if placed is None:
        return None

    base_grid = np.full((h, w), GREY, dtype=int)
    _paint_components(base_grid, placed)

    inp = base_grid.copy()
    tail_changes = _add_component_tails(inp, placed, mr, mc)
    clutter_changes = _add_cross_clutter(inp, mr, mc, colors)
    inp[mr : mr + 2, mc : mc + 2] = BLACK

    try:
        erased, marker = _erase_cross(inp)
    except AssertionError:
        return None
    if marker != (mr, mc):
        return None

    if _group_output_stats(erased) != sorted(zip(colors, group_counts)):
        return None

    parts = _extract_surviving_components(erased)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for part in parts:
        grouped[part["color"]].append(part)

    if set(grouped) != set(colors):
        return None
    for color, count in zip(colors, group_counts):
        items = grouped[color]
        if len(items) != count:
            return None
        rels = [_component_relative_coords(item) for item in items]
        if len(set(rels)) != len(rels):
            return None
        union_size = len(set().union(*[set(rel) for rel in rels]))
        if union_size <= max(len(rel) for rel in rels):
            return None

    if tail_changes == 0 or clutter_changes == 0:
        return None

    out = main(inp)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if len(group_counts) == 1:
        if np.count_nonzero(out == colors[0]) <= max(len(_component_relative_coords(item)) for item in grouped[colors[0]]):
            return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
