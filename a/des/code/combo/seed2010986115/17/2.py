from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
EDGES = ("top", "bottom", "left", "right")
MAX_GEN_RETRIES = 400
PAIR_GEN_RETRIES = 120
LAYOUT_RETRIES = 120
FRAGMENT_GEN_RETRIES = 120


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))


def _color_counts(arr: np.ndarray) -> Counter:
    return Counter(map(int, arr.reshape(-1)))


def _get_bg(arr: np.ndarray) -> int:
    counts = _color_counts(arr)
    top = counts.most_common()
    assert len(top) > 0
    best = top[0][1]
    winners = [color for color, cnt in top if cnt == best]
    assert len(winners) == 1, "expected a unique dominant background color"
    return int(winners[0])


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            comp = []
            queue = deque([(int(r), int(c))])
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nxt = (int(r + dr), int(c + dc))
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(coords)


def _is_full_rectangle(coords: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(list(coords))
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _mask_to_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    window = grid[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    if transparent is None:
        window[:] = patch
    else:
        mask = patch != int(transparent)
        window[mask] = patch[mask]


def _classify_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    h, w = grid.shape
    fragments: list[dict] = []
    marker_singletons: list[dict] = []

    for color in sorted(set(map(int, grid.reshape(-1))) - {int(bg)}):
        for coords in _component_coords(grid == int(color)):
            touch_edge = any(
                r == 0 or c == 0 or r == h - 1 or c == w - 1
                for r, c in coords
            )
            if len(coords) == 1 and touch_edge:
                marker_singletons.append({"color": int(color), "cell": coords[0]})
                continue

            assert not touch_edge, "fragments must stay off the border"
            assert 2 <= len(coords) <= 6, "fragments must be small interior monochrome components"
            norm = _normalize_coords(coords)
            rr = [r for r, _ in norm]
            cc = [c for _, c in norm]
            fragments.append(
                {
                    "color": int(color),
                    "coords": norm,
                    "shape": (max(rr) + 1, max(cc) + 1),
                    "area": int(len(norm)),
                }
            )

    assert fragments, "expected interior fragments"
    assert marker_singletons, "expected boundary markers"
    return fragments, marker_singletons


def _assembled_patterns(a: dict, b: dict) -> list[np.ndarray]:
    h1, w1 = map(int, a["shape"])
    h2, w2 = map(int, b["shape"])
    cells1 = {(int(r), int(c)) for r, c in a["coords"]}
    cells2 = {(int(r), int(c)) for r, c in b["coords"]}
    area = int(a["area"] + b["area"])

    patterns: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for rect_h in range(2, area + 1):
        if area % rect_h != 0:
            continue
        rect_w = area // rect_h
        if rect_w < 2:
            continue
        if rect_h < h1 or rect_h < h2 or rect_w < w1 or rect_w < w2:
            continue

        for ar in range(rect_h - h1 + 1):
            for ac in range(rect_w - w1 + 1):
                placed_a = {(int(r + ar), int(c + ac)) for r, c in cells1}
                for br in range(rect_h - h2 + 1):
                    for bc in range(rect_w - w2 + 1):
                        placed_b = {(int(r + br), int(c + bc)) for r, c in cells2}
                        if not placed_a.isdisjoint(placed_b):
                            continue
                        arr = np.full((rect_h, rect_w), -1, dtype=int)
                        for r, c in placed_a:
                            arr[r, c] = int(a["color"])
                        for r, c in placed_b:
                            arr[r, c] = int(b["color"])
                        if bool(np.any(arr < 0)):
                            continue
                        key = (tuple(map(int, arr.shape)), arr.tobytes())
                        patterns[key] = arr

    return list(patterns.values())


def _perfect_matchings(n: int, edges: dict[tuple[int, int], np.ndarray]) -> list[list[tuple[int, int]]]:
    assert n % 2 == 0
    solutions: list[list[tuple[int, int]]] = []

    def rec(remaining: tuple[int, ...], current: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(current.copy())
            return

        i = remaining[0]
        rem_set = set(remaining)
        for j in remaining[1:]:
            key = (i, j) if i < j else (j, i)
            if key not in edges:
                continue
            nxt = tuple(sorted(rem_set - {i, j}))
            current.append(key)
            rec(nxt, current)
            current.pop()

    rec(tuple(range(n)), [])
    return solutions


def _marker_groups(shape: tuple[int, int], marker_singletons: list[dict]) -> list[dict]:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    color_at: dict[tuple[int, int], int] = {}
    for marker in marker_singletons:
        cell = tuple(map(int, marker["cell"]))
        assert not bool(mask[cell]), "marker cells must be unique"
        mask[cell] = True
        color_at[cell] = int(marker["color"])

    groups = []
    for coords in _component_coords(mask):
        assert len(coords) == 2, "markers must form adjacent pairs"
        rows = {int(r) for r, _ in coords}
        cols = {int(c) for _, c in coords}

        if len(rows) == 1 and next(iter(rows)) == 0:
            ordered = sorted(coords, key=lambda rc: rc[1])
            assert ordered[0][1] + 1 == ordered[1][1]
            edge = "top"
            anchor = int(ordered[0][1])
        elif len(rows) == 1 and next(iter(rows)) == h - 1:
            ordered = sorted(coords, key=lambda rc: rc[1])
            assert ordered[0][1] + 1 == ordered[1][1]
            edge = "bottom"
            anchor = int(ordered[0][1])
        elif len(cols) == 1 and next(iter(cols)) == 0:
            ordered = sorted(coords, key=lambda rc: rc[0])
            assert ordered[0][0] + 1 == ordered[1][0]
            edge = "left"
            anchor = int(ordered[0][0])
        elif len(cols) == 1 and next(iter(cols)) == w - 1:
            ordered = sorted(coords, key=lambda rc: rc[0])
            assert ordered[0][0] + 1 == ordered[1][0]
            edge = "right"
            anchor = int(ordered[0][0])
        else:
            raise AssertionError("marker pairs must be adjacent on exactly one grid edge")

        groups.append(
            {
                "edge": edge,
                "anchor": anchor,
                "colors": tuple(int(color_at[cell]) for cell in ordered),
                "color_bag": tuple(sorted(int(color_at[cell]) for cell in ordered)),
            }
        )

    return groups


def _transform_for_edge(pattern: np.ndarray, edge: str) -> np.ndarray:
    if edge in {"top", "bottom"}:
        return np.array(pattern, copy=True)
    if edge == "left":
        return np.rot90(pattern, 1)
    if edge == "right":
        return np.rot90(pattern, 3)
    raise ValueError(edge)


def _edge_signature(pattern: np.ndarray, edge: str) -> tuple[int, int]:
    if edge == "top":
        return int(pattern[0, 0]), int(pattern[0, 1])
    if edge == "bottom":
        return int(pattern[-1, 0]), int(pattern[-1, 1])
    if edge == "left":
        return int(pattern[0, 0]), int(pattern[1, 0])
    if edge == "right":
        return int(pattern[0, -1]), int(pattern[1, -1])
    raise ValueError(edge)


def _placement_from_marker(grid_shape: tuple[int, int], pattern_shape: tuple[int, int], edge: str, anchor: int) -> tuple[int, int]:
    h, w = map(int, grid_shape)
    ph, pw = map(int, pattern_shape)
    if edge == "top":
        return 0, int(anchor)
    if edge == "bottom":
        return int(h - ph), int(anchor)
    if edge == "left":
        return int(anchor), 0
    if edge == "right":
        return int(anchor), int(w - pw)
    raise ValueError(edge)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary monochrome fragments, rebuild each full rectangle, and move it to its
    marker-indicated border position.

    The solver first ignores the border markers and finds the unique perfect matching of fragment
    pairs where each pair can be translated, without rotation, into one solid rectangle. It then
    uses the two adjacent border markers of the pair to choose the destination edge and anchor.
    Top/bottom markers keep the rebuilt rectangle's orientation; left/right markers rotate it
    ninety degrees so the rectangle follows the marker direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    fragments, marker_singletons = _classify_objects(g, bg)
    assert len(marker_singletons) == len(fragments), "each fragment color must appear once as a border marker"
    groups = _marker_groups(g.shape, marker_singletons)
    assert len(groups) * 2 == len(marker_singletons), "markers must be partitioned into border-adjacent pairs"

    pair_patterns: dict[tuple[int, int], np.ndarray] = {}
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            patterns = _assembled_patterns(fragments[i], fragments[j])
            if len(patterns) == 1:
                pair_patterns[(i, j)] = patterns[0]

    solutions = _perfect_matchings(len(fragments), pair_patterns)
    assert len(solutions) == 1, "fragment pairing must be unique"

    pair_entries = []
    for i, j in solutions[0]:
        colors = (int(fragments[i]["color"]), int(fragments[j]["color"]))
        pair_entries.append(
            {
                "pattern": pair_patterns[(i, j)],
                "colors": colors,
                "color_bag": tuple(sorted(colors)),
            }
        )

    out = np.full(g.shape, int(bg), dtype=int)
    used_pairs: set[int] = set()
    for group in groups:
        candidates = []
        for idx, entry in enumerate(pair_entries):
            if idx in used_pairs or entry["color_bag"] != group["color_bag"]:
                continue
            placed = _transform_for_edge(entry["pattern"], group["edge"])
            if _edge_signature(placed, group["edge"]) != group["colors"]:
                continue
            candidates.append((idx, placed))

        assert len(candidates) == 1, "marker pair must identify exactly one rebuilt rectangle"
        idx, placed = candidates[0]
        used_pairs.add(idx)
        r0, c0 = _placement_from_marker(g.shape, placed.shape, group["edge"], group["anchor"])
        ph, pw = placed.shape
        assert 0 <= r0 <= g.shape[0] - ph and 0 <= c0 <= g.shape[1] - pw
        window = out[r0 : r0 + ph, c0 : c0 + pw]
        assert not bool(np.any(window != bg)), "rebuilt rectangles must not overlap"
        window[:] = placed

    assert len(used_pairs) == len(pair_entries)
    return out


RECT_DIMS = [
    (h, w)
    for h in range(2, 7)
    for w in range(2, 7)
    if 4 <= h * w <= 12 and max(2, h * w - 6) <= min(6, h * w - 2)
]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    for dr, dc in DIR8:
        rs_src = slice(max(0, -dr), mask.shape[0] - max(0, dr))
        cs_src = slice(max(0, -dc), mask.shape[1] - max(0, dc))
        rs_dst = slice(max(0, dr), mask.shape[0] - max(0, -dr))
        cs_dst = slice(max(0, dc), mask.shape[1] - max(0, -dc))
        out[rs_dst, cs_dst] |= mask[rs_src, cs_src]
    return out


def _random_connected_subset(h: int, w: int, size: int) -> set[tuple[int, int]] | None:
    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}
    frontier = set()
    while len(cells) < size:
        frontier.clear()
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _random_dihedral(pattern: np.ndarray) -> np.ndarray:
    arr = np.array(pattern, copy=True)
    if random.random() < 0.5:
        arr = np.fliplr(arr)
    if random.random() < 0.5:
        arr = np.flipud(arr)
    return np.rot90(arr, random.randrange(4))


def _generate_pair_pattern(color_a: int, color_b: int) -> np.ndarray | None:
    for _ in range(PAIR_GEN_RETRIES):
        rect_h, rect_w = random.choice(RECT_DIMS)
        area = rect_h * rect_w
        size_lo = max(2, area - 6)
        size_hi = min(6, area - 2)
        if size_lo > size_hi:
            continue
        size_a = random.randint(size_lo, size_hi)
        cells_a = _random_connected_subset(rect_h, rect_w, size_a)
        if cells_a is None:
            continue
        all_cells = {(r, c) for r in range(rect_h) for c in range(rect_w)}
        cells_b = all_cells - cells_a
        if not _connected(cells_b):
            continue
        if _is_full_rectangle(cells_a) and _is_full_rectangle(cells_b):
            continue

        pat = np.full((rect_h, rect_w), int(color_b), dtype=int)
        for r, c in cells_a:
            pat[r, c] = int(color_a)
        pat = _random_dihedral(pat)

        ok = False
        for edge in EDGES:
            placed = _transform_for_edge(pat, edge)
            sig = _edge_signature(placed, edge)
            if sig[0] != sig[1]:
                ok = True
                break
        if ok:
            return pat
    return None


def _fragment_specs_from_pattern(pattern: np.ndarray, colors: tuple[int, int], bg: int) -> list[dict]:
    specs = []
    for color in colors:
        coords = list(map(tuple, np.argwhere(pattern == int(color))))
        r0, c0, r1, c1 = _bbox([(int(r), int(c)) for r, c in coords])
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            mask[int(r) - r0, int(c) - c0] = True
        specs.append(
            {
                "color": int(color),
                "mask": mask,
                "patch": _mask_to_patch(mask, int(color), int(bg)),
            }
        )
    return specs


def _marker_cells(edge: str, anchor: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    if edge == "top":
        return [(0, int(anchor)), (0, int(anchor + 1))]
    if edge == "bottom":
        return [(h - 1, int(anchor)), (h - 1, int(anchor + 1))]
    if edge == "left":
        return [(int(anchor), 0), (int(anchor + 1), 0)]
    if edge == "right":
        return [(int(anchor), w - 1), (int(anchor + 1), w - 1)]
    raise ValueError(edge)


def _edge_options(pattern: np.ndarray) -> list[dict]:
    opts = []
    for edge in EDGES:
        placed = _transform_for_edge(pattern, edge)
        sig = _edge_signature(placed, edge)
        if sig[0] == sig[1]:
            continue
        opts.append({"edge": edge, "placed": placed, "signature": sig})
    return opts


def _reserve_marker_buffer(buffer: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = buffer.shape
    for r, c in cells:
        for dr, dc in [(0, 0)] + DIR4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                buffer[nr, nc] = True


def _can_place_fragment(dilated: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    return not bool((dilated[int(r0) : int(r0) + h, int(c0) : int(c0) + w] & mask).any())


def _draw_full_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = patch


def _attempt() -> np.ndarray | None:
    h = random.randint(15, 22)
    w = random.randint(15, 22)
    pair_count = random.randint(2, 4)

    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    colors = random.sample(palette, 2 * pair_count)

    pair_specs = []
    for i in range(pair_count):
        color_a = int(colors[2 * i])
        color_b = int(colors[2 * i + 1])
        pattern = _generate_pair_pattern(color_a, color_b)
        if pattern is None:
            return None
        pair_specs.append(
            {
                "colors": (color_a, color_b),
                "pattern": pattern,
                "fragments": _fragment_specs_from_pattern(pattern, (color_a, color_b), bg),
                "options": _edge_options(pattern),
            }
        )
        if not pair_specs[-1]["options"]:
            return None

    for _ in range(LAYOUT_RETRIES):
        placed_specs = [dict(spec) for spec in pair_specs]
        random.shuffle(placed_specs)
        placed_specs.sort(key=lambda spec: spec["pattern"].size, reverse=True)

        occ = np.zeros((h, w), dtype=bool)
        marker_buffer = np.zeros((h, w), dtype=bool)
        ok = True
        for spec in placed_specs:
            options = spec["options"].copy()
            random.shuffle(options)
            choice = None
            for opt in options:
                placed = opt["placed"]
                ph, pw = placed.shape
                if opt["edge"] in {"top", "bottom"}:
                    anchors = list(range(1, w - pw))
                else:
                    anchors = list(range(1, h - ph))
                random.shuffle(anchors)
                for anchor in anchors:
                    r0, c0 = _placement_from_marker((h, w), placed.shape, opt["edge"], anchor)
                    cells = _marker_cells(opt["edge"], anchor, (h, w))
                    if bool(occ[r0 : r0 + ph, c0 : c0 + pw].any()):
                        continue
                    if any(bool(marker_buffer[r, c]) for r, c in cells):
                        continue
                    choice = {
                        "edge": opt["edge"],
                        "anchor": int(anchor),
                        "placed": placed,
                        "signature": tuple(map(int, opt["signature"])),
                        "r0": int(r0),
                        "c0": int(c0),
                        "marker_cells": cells,
                    }
                    break
                if choice is not None:
                    break
            if choice is None:
                ok = False
                break
            spec.update(choice)
            occ[spec["r0"] : spec["r0"] + spec["placed"].shape[0], spec["c0"] : spec["c0"] + spec["placed"].shape[1]] = True
            _reserve_marker_buffer(marker_buffer, spec["marker_cells"])
        if not ok:
            continue

        grid = np.full((h, w), int(bg), dtype=int)
        frag_occ = np.zeros((h, w), dtype=bool)
        frag_list = []
        for spec in placed_specs:
            frag_list.extend(spec["fragments"])
        random.shuffle(frag_list)
        frag_list.sort(key=lambda frag: int(frag["mask"].sum()), reverse=True)

        for frag in frag_list:
            mask = frag["mask"]
            fh, fw = mask.shape
            rows = list(range(1, h - fh))
            cols = list(range(1, w - fw))
            if not rows or not cols:
                ok = False
                break
            candidates = [(r0, c0) for r0 in rows for c0 in cols]
            random.shuffle(candidates)
            dilated = _dilate8(frag_occ)
            placed = False
            for r0, c0 in candidates:
                if not _can_place_fragment(dilated, mask, r0, c0):
                    continue
                _paste(grid, frag["patch"], r0, c0, transparent=bg)
                frag_occ[int(r0) : int(r0) + fh, int(c0) : int(c0) + fw] |= mask
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        expected = np.full((h, w), int(bg), dtype=int)
        for spec in placed_specs:
            for (mr, mc), color in zip(spec["marker_cells"], spec["signature"]):
                grid[int(mr), int(mc)] = int(color)
            _draw_full_patch(expected, spec["placed"], spec["r0"], spec["c0"])

        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
