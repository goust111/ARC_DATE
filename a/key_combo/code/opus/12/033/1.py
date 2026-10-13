import random
from collections import Counter, deque

import numpy as np


BG = 0
FILL = 2
RAIL = 5
PALETTE = (1, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    if connectivity == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_anchors(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    h, w = grid.shape
    anchors = {}
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(grid[r, c])
            if color in (BG, RAIL):
                continue
            if all(
                not (0 <= nr < h and 0 <= nc < w) or int(grid[nr, nc]) == BG
                for nr, nc in neighbors4(r, c)
            ):
                anchors[color] = (r, c)
    return anchors


def _find_segments(grid: np.ndarray) -> list[dict]:
    segments = []
    for coords in _components(grid == RAIL, connectivity=8):
        ks = {int(r + c) for r, c in coords}
        if len(ks) != 1:
            continue
        k = next(iter(ks))
        bs = [int(r - c) for r, c in coords]
        segments.append(
            {
                "coords": coords,
                "k": int(k),
                "b0": int(min(bs)),
                "b1": int(max(bs)),
            }
        )
    return segments


def _find_sprites(grid: np.ndarray, anchors: dict[int, tuple[int, int]]) -> list[dict]:
    mask = (grid != BG) & (grid != RAIL)
    for r, c in anchors.values():
        mask[r, c] = False
    sprites = []
    for coords in _components(mask, connectivity=4):
        values = [int(grid[r, c]) for r, c in coords]
        counts = Counter(values)
        marker_colors = [color for color, cnt in counts.items() if cnt == 1]
        assert len(marker_colors) == 1, "expected exactly one marker color"
        marker_color = int(marker_colors[0])
        marker = next((int(r), int(c)) for r, c in coords if int(grid[r, c]) == marker_color)
        sprites.append(
            {
                "coords": coords,
                "colors": np.array(values, dtype=int),
                "marker_color": marker_color,
                "marker": marker,
            }
        )
    return sprites


def _pair_candidates_for_sprite(sprite: dict, segments: list[dict]) -> list[dict]:
    cells = sprite["coords"]
    candidates = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            a = segments[i]
            b = segments[j]
            low_k = min(a["k"], b["k"])
            high_k = max(a["k"], b["k"])
            overlap0 = max(a["b0"], b["b0"])
            overlap1 = min(a["b1"], b["b1"])
            if overlap0 > overlap1:
                continue
            ok = True
            for r, c in cells:
                b_val = int(r - c)
                k_val = int(r + c)
                if not (overlap0 <= b_val <= overlap1 and low_k < k_val < high_k):
                    ok = False
                    break
            if ok:
                candidates.append(
                    {
                        "pair": (i, j),
                        "k0": low_k,
                        "k1": high_k,
                        "b0": overlap0,
                        "b1": overlap1,
                        "gap": high_k - low_k,
                    }
                )
    candidates.sort(key=lambda item: (item["gap"], item["pair"]))
    return candidates


def _scene(grid: np.ndarray):
    anchors = _find_anchors(grid)
    segments = _find_segments(grid)
    sprites = _find_sprites(grid, anchors)
    assignments = []
    for sprite in sprites:
        candidates = _pair_candidates_for_sprite(sprite, segments)
        assert len(candidates) == 1, "each sprite must belong to a unique rail pair"
        match = candidates[0]
        marker_color = sprite["marker_color"]
        assert marker_color in anchors, "missing matching anchor"
        assignments.append((sprite, match, anchors[marker_color]))
    return anchors, segments, assignments


def _move_sprite(sprite: dict, anchor: tuple[int, int], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    marker_r, marker_c = sprite["marker"]
    anchor_r, anchor_c = anchor
    if anchor_r in (0, h - 1) and anchor_c not in (0, w - 1):
        t = anchor_c - marker_c
    else:
        t = anchor_r - marker_r
    moved = sprite["coords"] + np.array([t, t], dtype=int)
    assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < h) & (0 <= moved[:, 1]) & (moved[:, 1] < w))
    return moved, sprite["colors"]


def _fill_pairs(out: np.ndarray, assignments) -> None:
    for _sprite, match, _anchor in assignments:
        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                b_val = r - c
                k_val = r + c
                if match["b0"] <= b_val <= match["b1"] and match["k0"] < k_val < match["k1"] and out[r, c] == BG:
                    out[r, c] = FILL


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each multicolor sprite along its diagonal rail and fill the rail-bounded strip in red.

    Input structure:
    - The background is black, with grey anti-diagonal line segments appearing in parallel pairs.
    - Each rail pair encloses exactly one small 4-connected multicolor sprite whose unique marker pixel has its own color.
    - The border contains isolated anchor pixels; each anchor color matches exactly one sprite marker color.

    Rule:
    - Pair the grey anti-diagonal segments and use each pair both as a motion rail and as the boundary of a fill strip.
    - Match every sprite to the border anchor with the same marker color, then translate the whole sprite only along the main diagonal direction `(t, t)`.
    - If the anchor is on the top or bottom edge, choose `t` so the marker column matches the anchor column; otherwise choose `t` so the marker row matches the anchor row.
    - After all sprites move, fill every background cell strictly between the paired rails with red while leaving the sprites and rails unchanged, creating hollow holes.

    Key insight:
    - The same diagonal rail pair simultaneously identifies the correct sprite, constrains its only legal slide direction, and defines the exact region that is red-filled after motion.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _anchors, _segments, assignments = _scene(g)

    for sprite, _match, _anchor in assignments:
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = BG

    for sprite, _match, anchor in assignments:
        moved, colors = _move_sprite(sprite, anchor, g.shape)
        for idx, (r, c) in enumerate(moved):
            assert out[r, c] == BG, "sprites should not collide after sliding"
            out[r, c] = int(colors[idx])

    _fill_pairs(out, assignments)
    return out


SHAPES = [
    ([(0, 0), (0, 1), (1, 0)], (0, 0)),
    ([(0, 0), (0, 1), (1, 0), (1, 1)], (0, 0)),
    ([(0, 0), (1, 0), (1, 1), (2, 0)], (0, 0)),
    ([(0, 0), (0, 1), (0, 2), (1, 1)], (0, 1)),
    ([(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)], (1, 1)),
    ([(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)], (0, 0)),
]


def _transform(rel_cells, marker, op: int):
    def apply(cell):
        r, c = cell
        if op == 0:
            return r, c
        if op == 1:
            return c, -r
        if op == 2:
            return -r, -c
        if op == 3:
            return -c, r
        if op == 4:
            return r, -c
        if op == 5:
            return -r, c
        if op == 6:
            return c, r
        return -c, -r

    transformed = [apply(cell) for cell in rel_cells]
    mr, mc = apply(marker)
    normalized = sorted((r - mr, c - mc) for r, c in transformed)
    return normalized


def _all_shape_variants():
    variants = []
    seen = set()
    for rel_cells, marker in SHAPES:
        for op in range(8):
            cells = tuple(_transform(rel_cells, marker, op))
            if cells not in seen:
                seen.add(cells)
                variants.append(cells)
    return variants


SHAPE_VARIANTS = _all_shape_variants()


def _abs_cells(marker: tuple[int, int], rel_cells) -> list[tuple[int, int]]:
    return [(marker[0] + dr, marker[1] + dc) for dr, dc in rel_cells]


def _inside_strict_strip(cells, k0: int, k1: int, b0: int, b1: int) -> bool:
    for r, c in cells:
        if r <= 0 or c <= 0:
            return False
        b_val = r - c
        k_val = r + c
        if not (b0 <= b_val <= b1 and k0 + 1 < k_val < k1 - 1):
            return False
    return True


def _touching_any(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr, cc in neighbors8(r, c):
            if (rr, cc) in occupied:
                return True
    return False


def _choose_anchor(
    marker: tuple[int, int],
    branch: str,
    used: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
    rng: random.Random,
):
    candidates = []
    if branch == "tb" and 0 < marker[1] < w - 1:
        candidates = [(0, marker[1]), (h - 1, marker[1])]
    if branch == "lr" and 0 < marker[0] < h - 1:
        candidates = [(marker[0], 0), (marker[0], w - 1)]
    rng.shuffle(candidates)
    for anchor in candidates:
        if anchor in used or anchor in blocked:
            continue
        ok = True
        for nr, nc in neighbors4(*anchor):
            if (nr, nc) in used or (nr, nc) in blocked:
                ok = False
                break
        if ok:
            return anchor
    return None


def _render_grid(h: int, w: int, plans: list[dict], use_input: bool) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for plan in plans:
        for r, c in plan["seg_a"]:
            grid[r, c] = RAIL
        for r, c in plan["seg_b"]:
            grid[r, c] = RAIL
        ar, ac = plan["anchor"]
        grid[ar, ac] = plan["marker_color"]
        cells = plan["input_cells"] if use_input else plan["final_cells"]
        for idx, (r, c) in enumerate(cells):
            grid[r, c] = plan["marker_color"] if idx == plan["marker_index"] else plan["body_color"]
    return grid


def _expected_output(h: int, w: int, plans: list[dict]) -> np.ndarray:
    out = _render_grid(h, w, plans, use_input=False)
    for plan in plans:
        for r in range(h):
            for c in range(w):
                b_val = r - c
                k_val = r + c
                if plan["b0"] <= b_val <= plan["b1"] and plan["k0"] < k_val < plan["k1"] and out[r, c] == BG:
                    out[r, c] = FILL
    return out


def _plan_pair(
    h: int,
    w: int,
    branch: str,
    marker_color: int,
    body_color: int,
    used_bands: list[tuple[int, int]],
    used_anchors: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
    occupied_static: set[tuple[int, int]],
    rng: random.Random,
):
    for _ in range(150):
        length = rng.randint(3, 5)
        gap = rng.choice((6, 8))
        offset = gap // 2
        r0_max = h - 2 - offset - (length - 1)
        c0_min = length
        c0_max = w - 2 - offset
        if r0_max < 1 or c0_min > c0_max:
            continue
        r0 = rng.randint(1, r0_max)
        c0 = rng.randint(c0_min, c0_max)
        b0 = r0 - c0
        b1 = b0 + 2 * (length - 1)
        if any(not (b1 < lo - 2 or b0 > hi + 2) for lo, hi in used_bands):
            continue

        seg_a = [(r0 + i, c0 - i) for i in range(length)]
        seg_b = [(r0 + offset + i, c0 + offset - i) for i in range(length)]
        if any(cell in occupied_static for cell in seg_a + seg_b):
            continue
        k0 = seg_a[0][0] + seg_a[0][1]
        k1 = seg_b[0][0] + seg_b[0][1]

        variants = SHAPE_VARIANTS[:]
        rng.shuffle(variants)
        for rel_cells in variants:
            marker_index = rel_cells.index((0, 0))
            markers = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]
            rng.shuffle(markers)
            for marker in markers:
                final_cells = _abs_cells(marker, rel_cells)
                if not _inside_strict_strip(final_cells, k0, k1, b0, b1):
                    continue
                if any(cell in occupied_static for cell in final_cells):
                    continue
                if _touching_any(final_cells, occupied_output):
                    continue
                shifts = [-2, -1, 1, 2]
                rng.shuffle(shifts)
                for t in shifts:
                    input_cells = [(r - t, c - t) for r, c in final_cells]
                    if not _inside_strict_strip(input_cells, k0, k1, b0, b1):
                        continue
                    if any(cell in occupied_static for cell in input_cells):
                        continue
                    if _touching_any(input_cells, occupied_input):
                        continue
                    blocked = occupied_static | set(seg_a) | set(seg_b) | set(input_cells)
                    anchor = _choose_anchor(marker, branch, used_anchors, blocked, h, w, rng)
                    if anchor is None:
                        continue
                    return {
                        "seg_a": seg_a,
                        "seg_b": seg_b,
                        "k0": k0,
                        "k1": k1,
                        "b0": b0,
                        "b1": b1,
                        "anchor": anchor,
                        "marker_color": marker_color,
                        "body_color": body_color,
                        "final_cells": final_cells,
                        "input_cells": input_cells,
                        "marker_index": marker_index,
                        "branch": branch,
                    }
    return None


def _semantic_checks(inp: np.ndarray, out: np.ndarray, plans: list[dict]) -> None:
    anchors, segments, assignments = _scene(inp)
    assert len(segments) == 2 * len(plans)
    assert len(assignments) == len(plans)
    assert len(anchors) == len(plans)
    assert np.any(out == FILL), "output must contain red fill"
    for plan in plans:
        hole = sum(1 for r, c in plan["final_cells"] if plan["k0"] < r + c < plan["k1"] and out[r, c] != FILL)
        assert hole == len(plan["final_cells"]), "final sprite must remain as a hole inside the filled strip"
    assert not np.array_equal(inp, out), "input and output should differ"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        h = rng.randint(16, 22)
        w = rng.randint(16, 22)
        pair_count = rng.choice((2, 2, 3))
        branch_types = ["tb", "lr"]
        while len(branch_types) < pair_count:
            branch_types.append(rng.choice(("tb", "lr")))
        rng.shuffle(branch_types)

        marker_colors = rng.sample(PALETTE, pair_count)
        remaining = [c for c in PALETTE if c not in marker_colors]
        body_colors = rng.sample(remaining, pair_count)

        plans = []
        used_bands = []
        used_anchors = set()
        occupied_input = set()
        occupied_output = set()
        occupied_static = set()

        ok = True
        for idx in range(pair_count):
            plan = _plan_pair(
                h=h,
                w=w,
                branch=branch_types[idx],
                marker_color=marker_colors[idx],
                body_color=body_colors[idx],
                used_bands=used_bands,
                used_anchors=used_anchors,
                occupied_input=occupied_input,
                occupied_output=occupied_output,
                occupied_static=occupied_static,
                rng=rng,
            )
            if plan is None:
                ok = False
                break
            plans.append(plan)
            used_bands.append((plan["b0"], plan["b1"]))
            used_anchors.add(plan["anchor"])
            occupied_input.update(plan["input_cells"])
            occupied_output.update(plan["final_cells"])
            occupied_static.update(plan["seg_a"])
            occupied_static.update(plan["seg_b"])
            occupied_static.add(plan["anchor"])
            for nr, nc in neighbors4(*plan["anchor"]):
                if 0 <= nr < h and 0 <= nc < w:
                    occupied_static.add((nr, nc))

        if not ok:
            continue

        inp = _render_grid(h, w, plans, use_input=True)
        expected = _expected_output(h, w, plans)
        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue
        _semantic_checks(inp, actual, plans)
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
