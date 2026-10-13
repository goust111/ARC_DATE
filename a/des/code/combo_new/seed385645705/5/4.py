import numpy as np

from collections import deque


BLACK = 0
GREY = 5
AZURE = 8
RECT_COLORS = (1, 2, 3, 4, 6, 7, 9)

GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
PAIR_COUNT_CHOICES = (2, 3)
RECT_SIZE_CHOICES = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (2, 5),
    (5, 2),
    (3, 4),
    (4, 3),
)
PRE_GAPS = {
    0: (0, 2, 4),
    1: (1, 3),
}
POST_GAPS = (0, 1, 2, 3)
GEN_ATTEMPTS = 500
PAIR_PLACEMENT_ATTEMPTS = 240

ARROW_SPECS = {
    "R": {
        "delta": (0, 1),
        "pattern": frozenset({(1, 0), (1, 1), (1, 2), (0, 3), (2, 3)}),
        "back": (1, 0),
        "front": (1, 2),
    },
    "L": {
        "delta": (0, -1),
        "pattern": frozenset({(0, 0), (2, 0), (1, 1), (1, 2), (1, 3)}),
        "back": (1, 3),
        "front": (1, 1),
    },
    "U": {
        "delta": (-1, 0),
        "pattern": frozenset({(0, 0), (0, 2), (1, 1), (2, 1), (3, 1)}),
        "back": (3, 1),
        "front": (1, 1),
    },
    "D": {
        "delta": (1, 0),
        "pattern": frozenset({(0, 1), (1, 1), (2, 1), (3, 0), (3, 2)}),
        "back": (0, 1),
        "front": (2, 1),
    },
}


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        deltas = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if _inside(h, w, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_rectangles(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    mask = (g != BLACK) & (g != GREY) & (g != AZURE)
    comps = _components(mask, connectivity=4)
    rects: list[dict] = []
    rect_index = -np.ones(g.shape, dtype=int)

    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if len(comp) != area or area < 6:
            continue
        rid = len(rects)
        for r, c in comp:
            rect_index[r, c] = rid
        rects.append(
            {
                "id": rid,
                "color": int(g[comp[0][0], comp[0][1]]),
                "bbox": (r0, c0, r1, c1),
                "cells": tuple(comp),
            }
        )
    return rects, rect_index


def _find_arrows(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    arrows: list[dict] = []
    for comp in _components(g == GREY, connectivity=8):
        if len(comp) != 5:
            continue
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, c0 = min(rs), min(cs)
        rel = frozenset((r - r0, c - c0) for r, c in comp)
        for name, spec in ARROW_SPECS.items():
            if rel != spec["pattern"]:
                continue
            br, bc = spec["back"]
            fr, fc = spec["front"]
            arrows.append(
                {
                    "dir": name,
                    "delta": spec["delta"],
                    "back": (r0 + br, c0 + bc),
                    "front": (r0 + fr, c0 + fc),
                    "cells": tuple(comp),
                }
            )
            break
    arrows.sort(key=lambda item: (item["back"][0], item["back"][1], item["dir"]))
    return arrows


def _extract_relations(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rects, rect_index = _find_rectangles(g)
    arrows = _find_arrows(g)
    relations: list[dict] = []

    for arrow in arrows:
        dr, dc = arrow["delta"]
        back_r, back_c = arrow["back"]
        src_r = back_r - dr
        src_c = back_c - dc
        assert _inside(h, w, src_r, src_c), "arrow must touch a source rectangle"
        source_id = int(rect_index[src_r, src_c])
        assert source_id >= 0, "arrow back must be adjacent to a solid rectangle"

        r = arrow["front"][0] + dr
        c = arrow["front"][1] + dc
        gap_black = 0
        while _inside(h, w, r, c) and int(g[r, c]) == BLACK:
            gap_black += 1
            r += dr
            c += dc
        assert _inside(h, w, r, c), "ray must encounter a gate before leaving the board"
        assert int(g[r, c]) == AZURE, "ray must encounter an Azure gate"
        gate = (r, c)

        r += dr
        c += dc
        while _inside(h, w, r, c) and int(g[r, c]) == BLACK:
            r += dr
            c += dc
        assert _inside(h, w, r, c), "ray must continue from the gate into a target rectangle"
        target_id = int(rect_index[r, c])
        assert target_id >= 0, "ray must hit a solid target rectangle"
        assert target_id != source_id, "source and target rectangles must be distinct"

        relations.append(
            {
                "arrow": arrow,
                "source_id": source_id,
                "target_id": target_id,
                "gate": gate,
                "gap_black": gap_black,
                "open": (gap_black % 2 == 0),
            }
        )

    return rects, relations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Selectively propagate a source rectangle's color through grey arrows when the Azure gate parity is open.

    Input structure:
    - Black background with several large solid colored rectangles.
    - Some rectangles have an adjacent Grey 5-pixel arrow: a 3-pixel shaft plus a two-pixel wedge that indicates up, down, left, or right.
    - Along each arrow ray there is exactly one Azure gate pixel and, farther on the same line, one solid target rectangle.

    Rule:
    - Read the arrow direction from its geometry and treat the rectangle touching the back of the arrow as the source.
    - Count the Black cells strictly between the arrow's front shaft pixel and the Azure gate on that ray.
    - If the count is even, recolor the entire target rectangle to the source rectangle's color; if the count is odd, leave the target unchanged.
    - Keep the arrows, gate pixels, and source rectangles unchanged.

    Key insight:
    - The arrow shape determines which distant rectangle is linked to which source, while the gate offset contributes a separate even/odd switch that decides whether color transfer happens.
    """
    g = np.asarray(grid, dtype=int)
    rects, relations = _extract_relations(g)

    out = g.copy()
    for rel in relations:
        if not rel["open"]:
            continue
        src = rects[rel["source_id"]]
        tgt = rects[rel["target_id"]]
        for r, c in tgt["cells"]:
            out[r, c] = src["color"]
    return out


def _offset_choices(length: int) -> tuple[int, ...]:
    if length >= 3:
        return tuple(range(1, length - 1))
    return tuple(range(length))


def _pair_parities(rng: np.random.Generator, n_pairs: int) -> list[int]:
    if n_pairs == 2:
        result = [0, 1]
    else:
        result = [0, 0, 1] if int(rng.integers(0, 2)) == 0 else [0, 1, 1]
    rng.shuffle(result)
    return result


def _sample_size(rng: np.random.Generator) -> tuple[int, int]:
    idx = int(rng.integers(0, len(RECT_SIZE_CHOICES)))
    return RECT_SIZE_CHOICES[idx]


def _cells_ok(used: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    h, w = used.shape
    for r, c in cells:
        if not (0 < r < h - 1 and 0 < c < w - 1):
            return False
        if used[r, c]:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if _inside(h, w, nr, nc) and used[nr, nc]:
                    return False
    return True


def _commit_cells(grid: np.ndarray, used: np.ndarray, lane: dict) -> None:
    for r, c in lane["source_cells"]:
        grid[r, c] = lane["source_color"]
        used[r, c] = True
    for r, c in lane["target_cells"]:
        grid[r, c] = lane["target_color"]
        used[r, c] = True
    for r, c in lane["arrow_cells"]:
        grid[r, c] = GREY
        used[r, c] = True
    gr, gc = lane["gate"]
    grid[gr, gc] = AZURE
    used[gr, gc] = True


def _build_horizontal_lane(
    direction: str,
    anchor_row: int,
    source_left: int,
    src_h: int,
    src_w: int,
    src_off: int,
    tgt_h: int,
    tgt_w: int,
    tgt_off: int,
    pre_gap: int,
    post_gap: int,
    source_color: int,
    target_color: int,
) -> dict:
    src_top = anchor_row - src_off
    tgt_top = anchor_row - tgt_off
    source_cells = {
        (src_top + dr, source_left + dc)
        for dr in range(src_h)
        for dc in range(src_w)
    }

    if direction == "R":
        target_left = source_left + src_w + 4 + pre_gap + post_gap
        target_cells = {
            (tgt_top + dr, target_left + dc)
            for dr in range(tgt_h)
            for dc in range(tgt_w)
        }
        arrow_cells = {
            (anchor_row, source_left + src_w + dc)
            for dc in range(3)
        }
        arrow_cells |= {
            (anchor_row - 1, source_left + src_w + 3),
            (anchor_row + 1, source_left + src_w + 3),
        }
        gate = (anchor_row, source_left + src_w + 3 + pre_gap)
    else:
        target_left = source_left - tgt_w - 4 - pre_gap - post_gap
        target_cells = {
            (tgt_top + dr, target_left + dc)
            for dr in range(tgt_h)
            for dc in range(tgt_w)
        }
        arrow_cells = {
            (anchor_row, source_left - 1 - dc)
            for dc in range(3)
        }
        arrow_cells |= {
            (anchor_row - 1, source_left - 4),
            (anchor_row + 1, source_left - 4),
        }
        gate = (anchor_row, source_left - 4 - pre_gap)

    all_cells = set(source_cells) | set(target_cells) | set(arrow_cells) | {gate}
    return {
        "source_cells": source_cells,
        "target_cells": target_cells,
        "arrow_cells": arrow_cells,
        "gate": gate,
        "source_color": source_color,
        "target_color": target_color,
        "all_cells": all_cells,
    }


def _build_vertical_lane(
    direction: str,
    anchor_col: int,
    source_top: int,
    src_h: int,
    src_w: int,
    src_off: int,
    tgt_h: int,
    tgt_w: int,
    tgt_off: int,
    pre_gap: int,
    post_gap: int,
    source_color: int,
    target_color: int,
) -> dict:
    src_left = anchor_col - src_off
    tgt_left = anchor_col - tgt_off
    source_cells = {
        (source_top + dr, src_left + dc)
        for dr in range(src_h)
        for dc in range(src_w)
    }

    if direction == "D":
        target_top = source_top + src_h + 4 + pre_gap + post_gap
        target_cells = {
            (target_top + dr, tgt_left + dc)
            for dr in range(tgt_h)
            for dc in range(tgt_w)
        }
        arrow_cells = {
            (source_top + src_h + dr, anchor_col)
            for dr in range(3)
        }
        arrow_cells |= {
            (source_top + src_h + 3, anchor_col - 1),
            (source_top + src_h + 3, anchor_col + 1),
        }
        gate = (source_top + src_h + 3 + pre_gap, anchor_col)
    else:
        target_top = source_top - tgt_h - 4 - pre_gap - post_gap
        target_cells = {
            (target_top + dr, tgt_left + dc)
            for dr in range(tgt_h)
            for dc in range(tgt_w)
        }
        arrow_cells = {
            (source_top - 1 - dr, anchor_col)
            for dr in range(3)
        }
        arrow_cells |= {
            (source_top - 4, anchor_col - 1),
            (source_top - 4, anchor_col + 1),
        }
        gate = (source_top - 4 - pre_gap, anchor_col)

    all_cells = set(source_cells) | set(target_cells) | set(arrow_cells) | {gate}
    return {
        "source_cells": source_cells,
        "target_cells": target_cells,
        "arrow_cells": arrow_cells,
        "gate": gate,
        "source_color": source_color,
        "target_color": target_color,
        "all_cells": all_cells,
    }


def _place_pair(
    rng: np.random.Generator,
    grid: np.ndarray,
    used: np.ndarray,
    direction: str,
    parity: int,
    source_color: int,
    target_color: int,
) -> bool:
    h, w = grid.shape
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        src_h, src_w = _sample_size(rng)
        tgt_h, tgt_w = _sample_size(rng)
        src_offsets = _offset_choices(src_h if direction in ("R", "L") else src_w)
        tgt_offsets = _offset_choices(tgt_h if direction in ("R", "L") else tgt_w)
        src_off = int(src_offsets[int(rng.integers(0, len(src_offsets)))])
        tgt_off = int(tgt_offsets[int(rng.integers(0, len(tgt_offsets)))])
        pre_gap_choices = PRE_GAPS[parity]
        pre_gap = int(pre_gap_choices[int(rng.integers(0, len(pre_gap_choices)))])
        post_gap = int(POST_GAPS[int(rng.integers(0, len(POST_GAPS)))])

        if direction in ("R", "L"):
            low_row = max(2, 1 + src_off, 1 + tgt_off)
            high_row = min(
                h - 3,
                h - src_h - 1 + src_off,
                h - tgt_h - 1 + tgt_off,
            )
            if low_row > high_row:
                continue
            anchor_row = int(rng.integers(low_row, high_row + 1))
            if direction == "R":
                low_col = 1
                high_col = w - src_w - tgt_w - pre_gap - post_gap - 5
            else:
                low_col = tgt_w + pre_gap + post_gap + 5
                high_col = w - src_w - 1
            if low_col > high_col:
                continue
            source_left = int(rng.integers(low_col, high_col + 1))
            lane = _build_horizontal_lane(
                direction=direction,
                anchor_row=anchor_row,
                source_left=source_left,
                src_h=src_h,
                src_w=src_w,
                src_off=src_off,
                tgt_h=tgt_h,
                tgt_w=tgt_w,
                tgt_off=tgt_off,
                pre_gap=pre_gap,
                post_gap=post_gap,
                source_color=source_color,
                target_color=target_color,
            )
        else:
            low_col = max(2, 1 + src_off, 1 + tgt_off)
            high_col = min(
                w - 3,
                w - src_w - 1 + src_off,
                w - tgt_w - 1 + tgt_off,
            )
            if low_col > high_col:
                continue
            anchor_col = int(rng.integers(low_col, high_col + 1))
            if direction == "D":
                low_row = 1
                high_row = h - src_h - tgt_h - pre_gap - post_gap - 5
            else:
                low_row = tgt_h + pre_gap + post_gap + 5
                high_row = h - src_h - 1
            if low_row > high_row:
                continue
            source_top = int(rng.integers(low_row, high_row + 1))
            lane = _build_vertical_lane(
                direction=direction,
                anchor_col=anchor_col,
                source_top=source_top,
                src_h=src_h,
                src_w=src_w,
                src_off=src_off,
                tgt_h=tgt_h,
                tgt_w=tgt_w,
                tgt_off=tgt_off,
                pre_gap=pre_gap,
                post_gap=post_gap,
                source_color=source_color,
                target_color=target_color,
            )

        if not _cells_ok(used, lane["all_cells"]):
            continue
        _commit_cells(grid, used, lane)
        return True

    return False


def _valid_generated_grid(grid: np.ndarray, n_pairs: int) -> bool:
    try:
        rects, relations = _extract_relations(grid)
    except AssertionError:
        return False

    if len(rects) != 2 * n_pairs:
        return False
    if len(relations) != n_pairs:
        return False

    source_ids = {rel["source_id"] for rel in relations}
    target_ids = {rel["target_id"] for rel in relations}
    if len(source_ids) != n_pairs or len(target_ids) != n_pairs:
        return False
    if source_ids & target_ids:
        return False

    opens = [rel["open"] for rel in relations]
    if not any(opens) or all(opens):
        return False

    out = main(grid)
    for rel in relations:
        src_color = rects[rel["source_id"]]["color"]
        tgt_cells = rects[rel["target_id"]]["cells"]
        expected = src_color if rel["open"] else rects[rel["target_id"]]["color"]
        if any(int(out[r, c]) != expected for r, c in tgt_cells):
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    grid = np.full((h, w), BLACK, dtype=int)
    used = np.zeros((h, w), dtype=bool)

    n_pairs = int(PAIR_COUNT_CHOICES[int(rng.integers(0, len(PAIR_COUNT_CHOICES)))])
    parities = _pair_parities(rng, n_pairs)
    dir_pool = np.array(["R", "L", "U", "D"], dtype=object)
    directions = list(dir_pool[rng.choice(len(dir_pool), size=n_pairs, replace=False)])
    colors = list(np.array(RECT_COLORS)[rng.choice(len(RECT_COLORS), size=2 * n_pairs, replace=False)])

    for idx in range(n_pairs):
        if not _place_pair(
            rng=rng,
            grid=grid,
            used=used,
            direction=str(directions[idx]),
            parity=int(parities[idx]),
            source_color=int(colors[2 * idx]),
            target_color=int(colors[2 * idx + 1]),
        ):
            return None

    if not _valid_generated_grid(grid, n_pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
