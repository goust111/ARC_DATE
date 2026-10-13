from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
ARROW_COLOR = 2
OBJECT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GENERATE_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _pad_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _scaled_bbox(box: tuple[int, int, int, int], k: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    new_h = h * k
    new_w = w * k
    # When parity prevents an exact discrete centering, bias toward top/left consistently.
    nr0 = (r0 + r1 - new_h + 1) // 2
    nc0 = (c0 + c1 - new_w + 1) // 2
    return nr0, nc0, nr0 + new_h - 1, nc0 + new_w - 1


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if v not in (BACKGROUND, ARROW_COLOR)):
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h not in (1, 2) or w not in (1, 2):
                continue
            if len(comp) != h * w:
                continue
            if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
                continue
            objects.append({"bbox": (r0, c0, r1, c1), "color": color})
    return objects


def _classify_arrow(comp: list[tuple[int, int]]) -> dict | None:
    cells = set(comp)
    deg: dict[tuple[int, int], int] = {}
    for cell in cells:
        deg[cell] = sum((nbr in cells) for nbr in _neighbors4(*cell))

    junctions = [cell for cell, d in deg.items() if d == 3]
    if len(junctions) != 1:
        return None
    jr, jc = junctions[0]
    rel = {(nr - jr, nc - jc) for nr, nc in _neighbors4(jr, jc) if (nr, nc) in cells}

    if rel == {(-1, 0), (1, 0), (0, -1)}:
        direction = "right"
        shaft = sorted(c for r, c in cells if r == jr)
        if {(r, c) for r, c in cells if r != jr} != {(jr - 1, jc), (jr + 1, jc)}:
            return None
        if shaft[-1] != jc or shaft != list(range(shaft[0], jc + 1)):
            return None
    elif rel == {(-1, 0), (1, 0), (0, 1)}:
        direction = "left"
        shaft = sorted(c for r, c in cells if r == jr)
        if {(r, c) for r, c in cells if r != jr} != {(jr - 1, jc), (jr + 1, jc)}:
            return None
        if shaft[0] != jc or shaft != list(range(jc, shaft[-1] + 1)):
            return None
    elif rel == {(0, -1), (0, 1), (-1, 0)}:
        direction = "down"
        shaft = sorted(r for r, c in cells if c == jc)
        if {(r, c) for r, c in cells if c != jc} != {(jr, jc - 1), (jr, jc + 1)}:
            return None
        if shaft[-1] != jr or shaft != list(range(shaft[0], jr + 1)):
            return None
    elif rel == {(0, -1), (0, 1), (1, 0)}:
        direction = "up"
        shaft = sorted(r for r, c in cells if c == jc)
        if {(r, c) for r, c in cells if c != jc} != {(jr, jc - 1), (jr, jc + 1)}:
            return None
        if shaft[0] != jr or shaft != list(range(jr, shaft[-1] + 1)):
            return None
    else:
        return None

    shaft_len = len(shaft)
    if not 3 <= shaft_len <= 5:
        return None
    if len(cells) != shaft_len + 2:
        return None
    return {"junction": (jr, jc), "direction": direction, "cells": cells}


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for comp in _components(np.asarray(grid, dtype=int) == ARROW_COLOR):
        arrow = _classify_arrow(comp)
        if arrow is not None:
            arrows.append(arrow)
    return arrows


def _ray_target(arrow: dict, objects: list[dict]) -> int | None:
    jr, jc = arrow["junction"]
    direction = arrow["direction"]
    best: tuple[int, int] | None = None
    for idx, obj in enumerate(objects):
        r0, c0, r1, c1 = obj["bbox"]
        dist: int | None = None
        if direction == "right" and r0 <= jr <= r1 and c0 > jc:
            dist = c0 - jc
        elif direction == "left" and r0 <= jr <= r1 and c1 < jc:
            dist = jc - c1
        elif direction == "down" and c0 <= jc <= c1 and r0 > jr:
            dist = r0 - jr
        elif direction == "up" and c0 <= jc <= c1 and r1 < jr:
            dist = jr - r1
        if dist is None:
            continue
        if best is None or dist < best[0]:
            best = (dist, idx)
    return None if best is None else best[1]


def _render_scaled_objects(shape: tuple[int, int], objects: list[dict], counts: dict[int, int]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BACKGROUND, dtype=int)
    for idx, obj in enumerate(objects):
        k = int(counts.get(idx, 0))
        if k <= 0:
            continue
        nr0, nc0, nr1, nc1 = _scaled_bbox(obj["bbox"], k)
        rr0, cc0 = max(0, nr0), max(0, nc0)
        rr1, cc1 = min(h - 1, nr1), min(w - 1, nc1)
        if rr0 <= rr1 and cc0 <= cc1:
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count red arrows per target object, then enlarge each target in place by that count.
    """
    g = np.asarray(grid, dtype=int)
    objects = _find_objects(g)
    arrows = _find_arrows(g)

    counts = {idx: 0 for idx in range(len(objects))}
    for arrow in arrows:
        target = _ray_target(arrow, objects)
        if target is not None:
            counts[target] += 1

    return _render_scaled_objects(g.shape, objects, counts)


def _box_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _cheby_pad(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    padded.add((nr, nc))
    return padded


def _clip_box(box: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
    return max(0, box[0]), max(0, box[1]), min(h - 1, box[2]), min(w - 1, box[3])


def _can_force_clip(size: int, k: int) -> bool:
    return (size == 1 and k >= 4) or (size == 2 and k >= 3)


def _place_objects(
    h: int,
    w: int,
    sizes: list[int],
    counts: list[int],
    colors: list[int],
    clip_idx: int | None,
) -> list[dict] | None:
    order = sorted(range(len(sizes)), key=lambda i: (sizes[i] * counts[i], sizes[i]), reverse=True)
    placed: list[dict | None] = [None] * len(sizes)
    used_row_spans: list[tuple[int, int]] = []
    used_col_spans: list[tuple[int, int]] = []
    used_out_boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        size = sizes[idx]
        k = counts[idx]
        candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
        for r0 in range(1, h - size):
            for c0 in range(1, w - size):
                r1 = r0 + size - 1
                c1 = c0 + size - 1
                if any(not (r1 < ur0 or r0 > ur1) for ur0, ur1 in used_row_spans):
                    continue
                if any(not (c1 < uc0 or c0 > uc1) for uc0, uc1 in used_col_spans):
                    continue
                box = (r0, c0, r1, c1)
                out_box = _clip_box(_scaled_bbox(box, k), h, w)
                if clip_idx == idx and _scaled_bbox(box, k) == out_box:
                    continue
                if any(_boxes_overlap(_pad_box(out_box, 1, h, w), _pad_box(prev, 1, h, w)) for prev in used_out_boxes):
                    continue
                candidates.append((r0, c0, out_box))
        if not candidates:
            return None

        if clip_idx == idx:
            candidates.sort(key=lambda item: min(item[0], item[1], h - 1 - (item[0] + size - 1), w - 1 - (item[1] + size - 1)))
            shortlist = candidates[: max(1, len(candidates) // 3)]
            r0, c0, out_box = random.choice(shortlist)
        else:
            candidates.sort(
                key=lambda item: -min(item[0], item[1], h - 1 - (item[0] + size - 1), w - 1 - (item[1] + size - 1))
            )
            shortlist = candidates[: max(1, len(candidates) // 2)]
            r0, c0, out_box = random.choice(shortlist)

        r1 = r0 + size - 1
        c1 = c0 + size - 1
        obj = {"bbox": (r0, c0, r1, c1), "color": colors[idx], "size": size, "k": k}
        placed[idx] = obj
        used_row_spans.append((r0, r1))
        used_col_spans.append((c0, c1))
        used_out_boxes.append(out_box)

    return [obj for obj in placed if obj is not None]


def _arrow_candidate(
    h: int,
    w: int,
    box: tuple[int, int, int, int],
    direction: str,
    lane: int,
    length: int,
    gap: int,
    occupied_objects: set[tuple[int, int]],
) -> dict | None:
    r0, c0, r1, c1 = box
    cells: set[tuple[int, int]] = set()
    ray: set[tuple[int, int]] = set()

    if direction == "right":
        junction = c0 - gap - 1
        if lane - 1 < 0 or lane + 1 >= h or junction - length + 1 < 0:
            return None
        cells |= {(lane, c) for c in range(junction - length + 1, junction + 1)}
        cells |= {(lane - 1, junction), (lane + 1, junction)}
        ray |= {(lane, c) for c in range(junction + 1, c0)}
    elif direction == "left":
        junction = c1 + gap + 1
        if lane - 1 < 0 or lane + 1 >= h or junction + length - 1 >= w:
            return None
        cells |= {(lane, c) for c in range(junction, junction + length)}
        cells |= {(lane - 1, junction), (lane + 1, junction)}
        ray |= {(lane, c) for c in range(c1 + 1, junction)}
    elif direction == "down":
        junction = r0 - gap - 1
        if lane - 1 < 0 or lane + 1 >= w or junction - length + 1 < 0:
            return None
        cells |= {(r, lane) for r in range(junction - length + 1, junction + 1)}
        cells |= {(junction, lane - 1), (junction, lane + 1)}
        ray |= {(r, lane) for r in range(junction + 1, r0)}
    else:
        assert direction == "up"
        junction = r1 + gap + 1
        if lane - 1 < 0 or lane + 1 >= w or junction + length - 1 >= h:
            return None
        cells |= {(r, lane) for r in range(junction, junction + length)}
        cells |= {(junction, lane - 1), (junction, lane + 1)}
        ray |= {(r, lane) for r in range(r1 + 1, junction)}

    if len(cells) != length + 2:
        return None
    footprint = _cheby_pad(cells, h, w) | ray
    if footprint & occupied_objects:
        return None
    return {
        "direction": direction,
        "cells": cells,
        "ray": ray,
        "footprint": footprint,
    }


def _enumerate_arrow_candidates(h: int, w: int, objects: list[dict]) -> list[list[dict]]:
    occupied_objects = set().union(*(_box_cells(obj["bbox"]) for obj in objects))
    all_candidates: list[list[dict]] = []
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        lanes = {
            "right": list(range(r0, r1 + 1)),
            "left": list(range(r0, r1 + 1)),
            "down": list(range(c0, c1 + 1)),
            "up": list(range(c0, c1 + 1)),
        }
        obj_candidates: list[dict] = []
        for direction in ("up", "down", "left", "right"):
            for lane in lanes[direction]:
                for length in (3, 4, 5):
                    for gap in (1, 2, 3):
                        cand = _arrow_candidate(h, w, obj["bbox"], direction, lane, length, gap, occupied_objects)
                        if cand is not None:
                            obj_candidates.append(cand)
        random.shuffle(obj_candidates)
        all_candidates.append(obj_candidates)
    return all_candidates


def _assign_arrows(candidates_by_obj: list[list[dict]], counts: list[int]) -> list[dict] | None:
    for _ in range(80):
        remaining = counts[:]
        forbidden: set[tuple[int, int]] = set()
        used_cells: set[tuple[int, int]] = set()
        used_dirs = [set() for _ in counts]
        chosen: list[dict] = []
        failed = False

        while sum(remaining) > 0:
            best_obj = None
            best_valid: list[dict] | None = None
            for obj_idx, need in enumerate(remaining):
                if need == 0:
                    continue
                valid = [
                    cand
                    for cand in candidates_by_obj[obj_idx]
                    if cand["cells"].isdisjoint(forbidden) and cand["footprint"].isdisjoint(used_cells)
                ]
                if len(valid) < need:
                    failed = True
                    break
                if best_valid is None or len(valid) < len(best_valid):
                    best_obj = obj_idx
                    best_valid = valid
            if failed:
                break

            assert best_obj is not None and best_valid is not None
            random.shuffle(best_valid)
            best_valid.sort(key=lambda cand: (cand["direction"] in used_dirs[best_obj], len(cand["ray"])))
            pool = best_valid[: min(8, len(best_valid))]
            cand = random.choice(pool)
            chosen.append({"object_idx": best_obj, **cand})
            remaining[best_obj] -= 1
            used_dirs[best_obj].add(cand["direction"])
            forbidden |= cand["footprint"]
            used_cells |= cand["cells"]

        if not failed and sum(remaining) == 0:
            return chosen
    return None


def _render_input(shape: tuple[int, int], objects: list[dict], arrows: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(obj["color"])
    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[r, c] = ARROW_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    n_obj = random.randint(2, 4)
    sizes = [random.choice((1, 2)) for _ in range(n_obj)]
    counts = [random.choice((1, 2, 3)) for _ in range(n_obj)]

    colors = random.sample(list(OBJECT_COLORS), n_obj)
    clip_idx = None
    if random.random() < 0.3:
        clip_idx = random.randrange(n_obj)
        sizes[clip_idx] = 2
        counts[clip_idx] = 3
    if not (3 <= sum(counts) <= 7):
        return None

    objects = _place_objects(h, w, sizes, counts, colors, clip_idx)
    if objects is None:
        return None

    candidates_by_obj = _enumerate_arrow_candidates(h, w, objects)
    if any(len(cands) < need for cands, need in zip(candidates_by_obj, counts)):
        return None

    arrows = _assign_arrows(candidates_by_obj, counts)
    if arrows is None:
        return None

    grid = _render_input((h, w), objects, arrows)
    expected = _render_scaled_objects((h, w), objects, {i: obj["k"] for i, obj in enumerate(objects)})

    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_RETRIES} attempts")
