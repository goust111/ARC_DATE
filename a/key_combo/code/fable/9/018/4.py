from __future__ import annotations

from collections import Counter
import numpy as np


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_coords(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _edge_singleton_tail(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int]) -> tuple[str, tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    edges = [
        ("U", [p for p in border_set if p[0] == r0]),
        ("D", [p for p in border_set if p[0] == r1]),
        ("L", [p for p in border_set if p[1] == c0]),
        ("R", [p for p in border_set if p[1] == c1]),
    ]
    candidates = [(d, cells[0]) for d, cells in edges if len(cells) == 1 and cells[0] not in corners]
    if len(candidates) != 1:
        return None
    return candidates[0]


def _mouth_edge_cells(mouth: set[tuple[int, int]], bbox: tuple[int, int, int, int], tail_dir: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    mouth_side = OPPOSITE[tail_dir]
    if mouth_side == "U":
        cells = [(r, c) for (r, c) in mouth if r == r0]
    elif mouth_side == "D":
        cells = [(r, c) for (r, c) in mouth if r == r1]
    elif mouth_side == "L":
        cells = [(r, c) for (r, c) in mouth if c == c0]
    else:
        cells = [(r, c) for (r, c) in mouth if c == c1]
    return sorted(cells)


def _parse_cups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    cups: list[dict] = []
    next_id = 0
    for col in sorted(set(map(int, np.unique(g))) - {bg}):
        for coords in _components_coords(g == col, connectivity=8):
            if len(coords) < 5:
                continue
            bbox = _bbox_coords(coords)
            border_set = {(int(r), int(c)) for r, c in coords}
            tail = _edge_singleton_tail(border_set, bbox)
            if tail is None:
                continue
            tail_dir, spike = tail
            mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
            mouth_edge = _mouth_edge_cells(mouth, bbox, tail_dir)
            if len(mouth) < 2 or not mouth_edge:
                continue
            cups.append(
                {
                    "id": next_id,
                    "border_color": int(col),
                    "bbox": bbox,
                    "border": border_set,
                    "tail_dir": tail_dir,
                    "spike": spike,
                    "mouth": mouth,
                    "mouth_edge": mouth_edge,
                }
            )
            next_id += 1
    return cups


def _content_color(grid: np.ndarray, cups: list[dict]) -> int | None:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    border_mask = np.zeros_like(g, dtype=bool)
    for cup in cups:
        for r, c in cup["border"]:
            border_mask[r, c] = True
    vals = [
        int(g[r, c])
        for r, c in np.argwhere((g != bg) & ~border_mask)
    ]
    if not vals:
        return None
    return int(Counter(vals).most_common(1)[0][0])


def _trace_next(cup: dict, mouth_owner: dict[tuple[int, int], int], border_owner: dict[tuple[int, int], int], shape: tuple[int, int]) -> int | None:
    h, w = shape
    dr, dc = DIR4[cup["tail_dir"]]
    r, c = cup["spike"]
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        key = (r, c)
        if key in mouth_owner and mouth_owner[key] != cup["id"]:
            return int(mouth_owner[key])
        if key in border_owner and border_owner[key] != cup["id"]:
            return None
        r += dr
        c += dc
    return None


def _resolve_sink(start: int, next_map: dict[int, int | None]) -> int | None:
    seen: set[int] = set()
    cur = start
    while cur is not None and cur not in seen:
        seen.add(cur)
        nxt = next_map.get(cur)
        if nxt is None:
            return cur
        cur = nxt
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route all content pixels along the tail-to-mouth cup chain and emit them once from the unique reachable exit cup.

    Input structure:
    - The background contains several hollow cup frames, each with a unique border color, a one-pixel tail spike, and an open mouth on the opposite side.
    - Some cups contain content pixels inside the mouth interior; all content pixels share one color that is not used by any frame.
    - Extending a ray from each tail spike in its tail direction either enters another cup through its mouth or exits the grid.
    - Several cups may point out of the grid, but every content-bearing cup belongs to the same flow component and ultimately reaches one exit cup.

    Rule:
    - Identify every cup from its frame geometry, then recover the directed cup graph from spike rays entering downstream mouths.
    - Remove all original content pixels from every cup interior.
    - Sum the original content pixels over the whole grid, follow only the cups reachable from content-bearing cups, and find their unique sink cup.
    - From that sink cup's spike, draw a straight content-colored ray outward in the tail direction with length equal to the total original content count; cells beyond the boundary are discarded.

    Key insight:
    - The visible content only marks where flow starts; the true destination is determined by the hidden tail-to-mouth graph, and the emitted line preserves the total content count.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _get_bg(g)
    cups = _parse_cups(g)
    if not cups:
        return out

    content_col = _content_color(g, cups)
    if content_col is None:
        return out

    mouth_owner: dict[tuple[int, int], int] = {}
    border_owner: dict[tuple[int, int], int] = {}
    for cup in cups:
        for cell in cup["mouth"]:
            mouth_owner[cell] = int(cup["id"])
        for cell in cup["border"]:
            border_owner[cell] = int(cup["id"])

    next_map = {int(cup["id"]): _trace_next(cup, mouth_owner, border_owner, g.shape) for cup in cups}
    by_id = {int(cup["id"]): cup for cup in cups}

    total_count = 0
    content_cups: list[int] = []
    for cup in cups:
        content_cells = sorted(cell for cell in cup["mouth"] if int(g[cell]) == content_col)
        cup["content_cells"] = content_cells
        total_count += len(content_cells)
        if content_cells:
            content_cups.append(int(cup["id"]))

    if not content_cups or total_count == 0:
        return out

    sinks = {_resolve_sink(cup_id, next_map) for cup_id in content_cups}
    sinks.discard(None)
    if len(sinks) != 1:
        return out
    sink_id = next(iter(sinks))
    sink = by_id[sink_id]

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg

    dr, dc = DIR4[sink["tail_dir"]]
    sr, sc = sink["spike"]
    for step in range(1, total_count + 1):
        rr = sr + dr * step
        cc = sc + dc * step
        if not _in_bounds(out.shape, rr, cc):
            break
        out[rr, cc] = content_col
    return out


def _build_canonical_cup(l_arm: int, r_arm: int, h_l: int, h_r: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    w = l_arm + r_arm + 1
    h = max(l_arm + h_l, r_arm + h_r) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[0, l_arm] = color
    patch[np.arange(1, l_arm + 1), l_arm - np.arange(1, l_arm + 1)] = color
    patch[np.arange(1, r_arm + 1), l_arm + np.arange(1, r_arm + 1)] = color
    if h_l:
        patch[l_arm + 1 : l_arm + h_l + 1, 0] = color
    if h_r:
        patch[r_arm + 1 : r_arm + h_r + 1, w - 1] = color
    return patch, (0, l_arm)


def _mouth_interior_local(patch: np.ndarray, border_color: int, side: str) -> list[tuple[int, int]]:
    h, w = patch.shape
    border = patch == border_color
    if side == "U":
        seeds = [(0, c) for c in range(w) if not border[0, c]]
    elif side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border[h - 1, c]]
    elif side == "L":
        seeds = [(r, 0) for r in range(h) if not border[r, 0]]
    else:
        seeds = [(r, w - 1) for r in range(h) if not border[r, w - 1]]
    seen = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and not border[nr, nc]:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return sorted(seen)


def _rotate_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for (r, c) in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for (r, c) in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for (r, c) in mouth]
    raise ValueError(direction)


def _sample_right_cup(rng: np.random.Generator, color: int, min_slots: int) -> dict | None:
    for _ in range(120):
        l_arm = int(rng.integers(1, 4))
        r_arm = int(rng.integers(1, 4))
        h_l = int(rng.integers(1, 4))
        h_r = int(rng.integers(1, 4))
        patch0, spike0 = _build_canonical_cup(l_arm, r_arm, h_l, h_r, color)
        mouth0 = _mouth_interior_local(patch0, color, "D")
        patch, spike, mouth = _rotate_to_direction(patch0, spike0, mouth0, "R")
        if patch.shape[0] > 7 or patch.shape[1] > 7:
            continue
        coords = np.argwhere(patch == color)
        bbox = _bbox_coords(coords)
        border_set = {(int(r), int(c)) for r, c in coords}
        tail = _edge_singleton_tail(border_set, bbox)
        if tail is None or tail[0] != "R":
            continue
        mouth_set = set(mouth)
        mouth_edge = _mouth_edge_cells(mouth_set, bbox, "R")
        if len(mouth_edge) < min_slots or len(mouth) < 3:
            continue
        return {
            "patch": patch,
            "spike": spike,
            "mouth": sorted(mouth),
            "mouth_edge": mouth_edge,
            "shape": patch.shape,
            "color": int(color),
        }
    return None


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0


def _placeable(box: tuple[int, int, int, int], placed: dict[str, tuple[int, int, int, int]]) -> bool:
    return all(_boxes_separate(box, other, margin=1) for other in placed.values())


def _sorted_edge_rows(cup: dict) -> list[int]:
    return [int(r) for r, _c in sorted(cup["mouth_edge"])]


def _path_clear(source: dict, src_pos: tuple[int, int], target_cell: tuple[int, int], boxes: dict[str, tuple[int, int, int, int]], allow: set[str]) -> bool:
    ph, pw = source["shape"]
    sr = src_pos[0] + int(source["spike"][0])
    sc = src_pos[1] + int(source["spike"][1])
    tr, tc = target_cell
    if sr != tr or tc <= sc:
        return False
    for c in range(sc + 1, tc):
        for name, (r0, c0, r1, c1) in boxes.items():
            if name in allow:
                continue
            if r0 <= sr <= r1 and c0 <= c <= c1:
                return False
    for name, (r0, c0, r1, c1) in boxes.items():
        if name in allow:
            continue
        if r0 <= tr <= r1 and c0 <= tc <= c1:
            return False
    return True


def _sink_clear(source: dict, src_pos: tuple[int, int], run_len: int, boxes: dict[str, tuple[int, int, int, int]], name_self: str) -> bool:
    sr = src_pos[0] + int(source["spike"][0])
    sc = src_pos[1] + int(source["spike"][1])
    for c in range(sc + 1, sc + 1 + run_len):
        for name, (r0, c0, r1, c1) in boxes.items():
            if name == name_self:
                continue
            if r0 <= sr <= r1 and c0 <= c <= c1:
                return False
    return True


def _content_count_for(cup: dict, salt: int) -> int:
    cap = min(4, len(cup["mouth"]))
    base = len(cup["mouth"]) + cup["shape"][0] + 2 * cup["shape"][1] + salt
    return 1 + (base % cap)


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != 0
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _rotate_grid(grid: np.ndarray, k: int) -> np.ndarray:
    if k % 4 == 0:
        return grid.copy()
    return np.rot90(grid, k=k).copy()


def _template_spec(rng: np.random.Generator) -> dict:
    deep = None
    if rng.random() < 0.6:
        deep = str(rng.choice(["A", "B"]))
    edges = {"A": "M", "B": "M", "M": "E"}
    content_nodes = ["A", "B"]
    if deep is not None:
        edges["C"] = deep
        other = "B" if deep == "A" else "A"
        content_nodes = ["C", other]
    nodes = ["E", "M", "A", "B", "D"] + (["C"] if deep is not None else [])
    indegree = {name: 0 for name in nodes}
    for src, dst in edges.items():
        indegree[dst] += 1
    return {"nodes": nodes, "edges": edges, "content_nodes": content_nodes, "indegree": indegree, "deep": deep}


def _place_template_right(rng: np.random.Generator, cups: dict[str, dict], spec: dict, h: int, w: int, exit_run: int, decoy_run: int) -> dict[str, tuple[int, int]] | None:
    e = cups["E"]
    m = cups["M"]
    e_h, e_w = e["shape"]
    m_h, m_w = m["shape"]
    e_c0 = w - exit_run - e_w
    if e_c0 < 4:
        return None
    e_slots = _sorted_edge_rows(e)
    m_slots = _sorted_edge_rows(m)

    slot_pairs = sorted(
        [(m_slots[i], m_slots[j]) for i in range(len(m_slots)) for j in range(len(m_slots)) if i != j],
        key=lambda pair: -abs(pair[0] - pair[1]),
    )
    e_slots = sorted(e_slots, key=lambda v: abs(v - np.median(e_slots)))

    for gap_em in (1, 2, 3):
        m_c0 = e_c0 - m_w - gap_em
        if m_c0 < 0:
            continue
        for gap_a, gap_b in ((1, 1), (1, 2), (2, 1), (2, 2)):
            c0_a = m_c0 - cups["A"]["shape"][1] - gap_a
            c0_b = m_c0 - cups["B"]["shape"][1] - gap_b
            if c0_a < 0 or c0_b < 0:
                continue
            c0_c = None
            for slot_a, slot_b in slot_pairs:
                for e_slot in e_slots:
                    low = 0
                    high = h - m_h
                    low = max(low, int(e_slot - m["spike"][0]))
                    high = min(high, int(h - e_h + e_slot - m["spike"][0]))
                    low = max(low, int(cups["A"]["spike"][0] - slot_a))
                    high = min(high, int(h - cups["A"]["shape"][0] - slot_a + cups["A"]["spike"][0]))
                    low = max(low, int(cups["B"]["spike"][0] - slot_b))
                    high = min(high, int(h - cups["B"]["shape"][0] - slot_b + cups["B"]["spike"][0]))

                    parent_slot = None
                    if spec["deep"] is not None:
                        parent = spec["deep"]
                        child = "C"
                        parent_slot_choices = _sorted_edge_rows(cups[parent])
                        parent_slot_choices = sorted(parent_slot_choices, key=lambda v: abs(v - np.median(parent_slot_choices)))
                    else:
                        parent_slot_choices = [None]

                    for p_slot in parent_slot_choices:
                        gap_c = 1
                        if spec["deep"] == "A":
                            c0_c = c0_a - cups["C"]["shape"][1] - gap_c
                            if c0_c < 0:
                                continue
                            low2 = int(cups["A"]["spike"][0] - slot_a + cups["C"]["spike"][0] - p_slot)
                            high2 = int(h - cups["C"]["shape"][0] + cups["A"]["spike"][0] - slot_a + cups["C"]["spike"][0] - p_slot)
                            low_use = max(low, low2)
                            high_use = min(high, high2)
                        elif spec["deep"] == "B":
                            c0_c = c0_b - cups["C"]["shape"][1] - gap_c
                            if c0_c < 0:
                                continue
                            low2 = int(cups["B"]["spike"][0] - slot_b + cups["C"]["spike"][0] - p_slot)
                            high2 = int(h - cups["C"]["shape"][0] + cups["B"]["spike"][0] - slot_b + cups["C"]["spike"][0] - p_slot)
                            low_use = max(low, low2)
                            high_use = min(high, high2)
                        else:
                            low_use, high_use = low, high

                        if low_use > high_use:
                            continue

                        m_r0 = int((low_use + high_use) // 2)
                        pos = {
                            "M": (m_r0, m_c0),
                            "E": (m_r0 + int(m["spike"][0]) - int(e_slot), e_c0),
                            "A": (m_r0 + int(slot_a) - int(cups["A"]["spike"][0]), c0_a),
                            "B": (m_r0 + int(slot_b) - int(cups["B"]["spike"][0]), c0_b),
                        }
                        if spec["deep"] == "A":
                            pos["C"] = (
                                pos["A"][0] + int(p_slot) - int(cups["C"]["spike"][0]),
                                c0_c,
                            )
                        elif spec["deep"] == "B":
                            pos["C"] = (
                                pos["B"][0] + int(p_slot) - int(cups["C"]["spike"][0]),
                                c0_c,
                            )

                        boxes = {}
                        valid = True
                        for name in pos:
                            ch, cw = cups[name]["shape"]
                            r0, c0 = pos[name]
                            if not (0 <= r0 <= h - ch and 0 <= c0 <= w - cw):
                                valid = False
                                break
                            box = (r0, c0, r0 + ch - 1, c0 + cw - 1)
                            if not _placeable(box, boxes):
                                valid = False
                                break
                            boxes[name] = box
                        if not valid:
                            continue

                        if not _path_clear(m, pos["M"], (pos["E"][0] + int(e_slot), e_c0), boxes, {"M", "E"}):
                            continue
                        if not _path_clear(cups["A"], pos["A"], (m_r0 + int(slot_a), m_c0), boxes, {"A", "M"}):
                            continue
                        if not _path_clear(cups["B"], pos["B"], (m_r0 + int(slot_b), m_c0), boxes, {"B", "M"}):
                            continue
                        if spec["deep"] == "A":
                            if not _path_clear(cups["C"], pos["C"], (pos["A"][0] + int(p_slot), pos["A"][1]), boxes, {"C", "A"}):
                                continue
                        elif spec["deep"] == "B":
                            if not _path_clear(cups["C"], pos["C"], (pos["B"][0] + int(p_slot), pos["B"][1]), boxes, {"C", "B"}):
                                continue

                        d = cups["D"]
                        dh, dw = d["shape"]
                        d_c0 = w - decoy_run - dw
                        if d_c0 < 0:
                            continue
                        rows = list(range(0, h - dh + 1))
                        rng.shuffle(rows)
                        for d_r0 in rows:
                            d_box = (d_r0, d_c0, d_r0 + dh - 1, d_c0 + dw - 1)
                            if not _placeable(d_box, boxes):
                                continue
                            test_boxes = dict(boxes)
                            test_boxes["D"] = d_box
                            if not _sink_clear(d, (d_r0, d_c0), decoy_run, test_boxes, "D"):
                                continue
                            if not _sink_clear(e, pos["E"], exit_run, test_boxes, "E"):
                                continue
                            pos["D"] = (d_r0, d_c0)
                            return pos
    return None


def _semantic_ok(grid: np.ndarray) -> bool:
    cups = _parse_cups(grid)
    if len(cups) < 5:
        return False
    content_col = _content_color(grid, cups)
    if content_col is None:
        return False
    border_cols = {cup["border_color"] for cup in cups}
    if content_col in border_cols:
        return False

    mouth_owner: dict[tuple[int, int], int] = {}
    border_owner: dict[tuple[int, int], int] = {}
    for cup in cups:
        for cell in cup["mouth"]:
            mouth_owner[cell] = int(cup["id"])
        for cell in cup["border"]:
            border_owner[cell] = int(cup["id"])
    next_map = {int(cup["id"]): _trace_next(cup, mouth_owner, border_owner, grid.shape) for cup in cups}

    indeg = Counter(v for v in next_map.values() if v is not None)
    if max(indeg.values(), default=0) < 2:
        return False
    sinks = [cid for cid, nxt in next_map.items() if nxt is None]
    if len(sinks) < 2:
        return False

    total_content = 0
    content_cups: list[int] = []
    active_noncontent_on_path = False
    by_id = {cup["id"]: cup for cup in cups}
    for cup in cups:
        cells = [cell for cell in cup["mouth"] if int(grid[cell]) == content_col]
        if any(int(grid[cell]) != content_col and int(grid[cell]) != _get_bg(grid) for cell in cup["mouth"]):
            return False
        total_content += len(cells)
        if cells:
            content_cups.append(int(cup["id"]))
    if len(content_cups) < 2 or total_content < 2:
        return False

    reached = []
    for cid in content_cups:
        cur = cid
        seen: set[int] = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            nxt = next_map[cur]
            if nxt is None:
                break
            if nxt not in content_cups:
                active_noncontent_on_path = True
            cur = nxt
        reached.append(_resolve_sink(cid, next_map))
    if len(set(reached)) != 1 or reached[0] is None:
        return False
    if not active_noncontent_on_path:
        return False

    sink = by_id[reached[0]]
    dr, dc = DIR4[sink["tail_dir"]]
    sr, sc = sink["spike"]
    run = 0
    rr, cc = sr + dr, sc + dc
    while _in_bounds(grid.shape, rr, cc):
        run += 1
        rr += dr
        cc += dc
    if run < 1:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    spec = _template_spec(rng)
    n_cups = len(spec["nodes"])
    palette = list(rng.permutation(10))
    bg = int(palette[0])
    content_col = int(palette[1])
    border_cols = [int(v) for v in palette[2 : 2 + n_cups]]
    cups: dict[str, dict] = {}
    for name, color in zip(spec["nodes"], border_cols):
        cup = _sample_right_cup(rng, color, max(1, spec["indegree"][name]))
        if cup is None:
            return None
        cups[name] = cup

    overflow = bool(rng.integers(0, 2))
    exit_run = int(rng.integers(1, 4)) if overflow else int(rng.integers(4, 9))
    decoy_run = int(rng.integers(1, 7))
    a_w = cups["A"]["shape"][1]
    b_w = cups["B"]["shape"][1]
    c_w = cups["C"]["shape"][1] if "C" in cups else 0
    m_w = cups["M"]["shape"][1]
    e_w = cups["E"]["shape"][1]
    d_w = cups["D"]["shape"][1]
    left_branch = max(
        a_w + (c_w + 1 if spec["deep"] == "A" else 0),
        b_w + (c_w + 1 if spec["deep"] == "B" else 0),
    )
    min_w = max(e_w + m_w + left_branch + exit_run + 5, e_w + d_w + max(exit_run, decoy_run) + 2, 18)
    if min_w > 30:
        return None
    w = int(rng.integers(max(min_w, 24), 31))

    heights = [cups[name]["shape"][0] for name in spec["nodes"]]
    min_h = max(max(heights) + 4, 14)
    if min_h > 30:
        return None
    h = int(rng.integers(max(min_h, 18), 31))
    pos = _place_template_right(rng, cups, spec, h, w, exit_run, decoy_run)
    if pos is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for name in spec["nodes"]:
        _paint_patch(grid, cups[name]["patch"], pos[name][0], pos[name][1])

    total_count = 0
    for idx, name in enumerate(spec["content_nodes"]):
        cup = cups[name]
        count = _content_count_for(cup, idx + 1)
        local_cells = sorted(cup["mouth"], key=lambda rc: (rc[1], rc[0]))[:count]
        for r, c in local_cells:
            grid[pos[name][0] + r, pos[name][1] + c] = content_col
        total_count += count

    if overflow != (total_count > exit_run):
        return None

    rot_k = int(rng.integers(0, 4))
    grid = _rotate_grid(grid, rot_k)

    if not _semantic_ok(grid):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None

    cups = _parse_cups(grid)
    sink_line_cells = [
        (r, c)
        for r, c in np.argwhere((out == content_col) & (grid != content_col))
    ]
    if len(sink_line_cells) != min(total_count, exit_run):
        return None
    if len(set(map(tuple, sink_line_cells))) != len(sink_line_cells):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 500 attempts")
