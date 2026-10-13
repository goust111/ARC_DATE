from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 7
RED = 2
GREY = 5
TAIL = 9
BODY_COLORS = (RED, GREY)
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    )


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _build_body(h: int, w: int, start_color: int) -> np.ndarray:
    body = np.empty((h, w), dtype=int)
    other = GREY if start_color == RED else RED
    for r in range(h):
        body[r, :] = start_color if r % 2 == 0 else other
    return body


def _swap_body_colors(sprite: np.ndarray) -> np.ndarray:
    swapped = sprite.copy()
    swapped[sprite == RED] = GREY
    swapped[sprite == GREY] = RED
    return swapped


def _clip_paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(dst.shape[0], r0 + h)
    cc1 = min(dst.shape[1], c0 + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    sr0 = rr0 - r0
    sc0 = cc0 - c0
    patch = src[sr0 : sr0 + (rr1 - rr0), sc0 : sc0 + (cc1 - cc0)]
    region = dst[rr0:rr1, cc0:cc1]
    assert np.all(region == BG), "stamps from different objects must not overlap"
    region[:, :] = patch


def _tail_cells(r0: int, c0: int, h: int, w: int, direction: str, tail_len: int) -> list[tuple[int, int]]:
    if direction == "U":
        col = c0 + w // 2
        return [(r0 - d, col) for d in range(tail_len, 0, -1)]
    if direction == "D":
        col = c0 + w // 2
        return [(r0 + h - 1 + d, col) for d in range(1, tail_len + 1)]
    if direction == "L":
        row = r0 + h // 2
        return [(row, c0 - d) for d in range(tail_len, 0, -1)]
    row = r0 + h // 2
    return [(row, c0 + w - 1 + d) for d in range(1, tail_len + 1)]


def _body_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _simulate_object(
    r0: int,
    c0: int,
    h: int,
    w: int,
    direction: str,
    grid_shape: tuple[int, int],
) -> dict:
    dr, dc = DIRS[direction]
    H, W = grid_shape
    copies: list[dict] = []
    all_output = set(_body_cells(r0, c0, h, w))
    k = 1
    while True:
        if direction in ("U", "D"):
            copy_w = w - 2 * k
            if copy_w <= 0:
                return {"copies": copies, "cause": "shrink", "output_cells": all_output}
            rr0 = r0 + dr * h * k
            cc0 = c0 + k
            rr1 = rr0 + h - 1
            cc1 = cc0 + copy_w - 1
        else:
            copy_h = h - 2 * k
            if copy_h <= 0:
                return {"copies": copies, "cause": "shrink", "output_cells": all_output}
            rr0 = r0 + k
            cc0 = c0 + dc * w * k
            rr1 = rr0 + copy_h - 1
            cc1 = cc0 + w - 1

        if rr1 < 0 or rr0 >= H or cc1 < 0 or cc0 >= W:
            return {"copies": copies, "cause": "offgrid", "output_cells": all_output}

        partial = rr0 < 0 or rr1 >= H or cc0 < 0 or cc1 >= W
        cells = {
            (r, c)
            for r in range(max(0, rr0), min(H - 1, rr1) + 1)
            for c in range(max(0, cc0), min(W - 1, cc1) + 1)
        }
        copies.append(
            {
                "k": k,
                "r0": rr0,
                "c0": cc0,
                "r1": rr1,
                "c1": cc1,
                "partial": partial,
                "cells": cells,
            }
        )
        all_output |= cells
        k += 1


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({BG, RED, GREY, TAIL}), "unexpected colors"
    components = _find_components(g != BG)
    assert 2 <= len(components) <= 3, "expected 2-3 tailed rectangles"

    objects: list[dict] = []
    for comp in components:
        vals = np.array([g[r, c] for r, c in comp], dtype=int)
        tail_coords = comp[vals == TAIL]
        body_coords = comp[vals != TAIL]
        assert len(tail_coords) >= 2, "tail must have length at least 2"
        assert len(body_coords) >= 1, "body must exist"

        r0, c0, r1, c1 = _bbox(body_coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 4 and w >= 5, "body rectangle too small"
        assert len(body_coords) == h * w, "body must be a solid rectangle"

        body = g[r0 : r1 + 1, c0 : c1 + 1]
        assert set(np.unique(body)).issubset(BODY_COLORS), "body must use red/grey only"
        for rr in range(h):
            assert np.all(body[rr] == body[rr, 0]), "body rows must be uniform stripes"
            if rr > 0:
                assert body[rr, 0] != body[rr - 1, 0], "body stripes must alternate"

        tail_set = {(int(r), int(c)) for r, c in tail_coords}
        body_set = {(int(r), int(c)) for r, c in body_coords}
        touching = [
            (r, c)
            for r, c in tail_set
            if any((nr, nc) in body_set for nr, nc in _neighbors4(r, c))
        ]
        assert len(touching) == 1, "tail must attach at exactly one border cell"

        tail_rows = sorted({int(r) for r, _ in tail_set})
        tail_cols = sorted({int(c) for _, c in tail_set})
        if len(tail_rows) == 1:
            row = tail_rows[0]
            assert row == r0 + h // 2 and h % 2 == 1, "horizontal tail must leave from the center row"
            assert tail_cols == list(range(tail_cols[0], tail_cols[-1] + 1)), "tail must be contiguous"
            if tail_cols[-1] == c0 - 1:
                direction = "L"
            else:
                assert tail_cols[0] == c1 + 1, "horizontal tail must sit outside the body"
                direction = "R"
        else:
            assert len(tail_cols) == 1, "tail must be straight"
            col = tail_cols[0]
            assert col == c0 + w // 2 and w % 2 == 1, "vertical tail must leave from the center column"
            assert tail_rows == list(range(tail_rows[0], tail_rows[-1] + 1)), "tail must be contiguous"
            if tail_rows[-1] == r0 - 1:
                direction = "U"
            else:
                assert tail_rows[0] == r1 + 1, "vertical tail must sit outside the body"
                direction = "D"

        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "h": h,
                "w": w,
                "body": body.copy(),
                "direction": direction,
                "tail_len": len(tail_coords),
            }
        )

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each maroon tail, then stamp the striped rectangle repeatedly in the tail direction while each copy shrinks across the perpendicular axis and flips red/grey.

    Input structure:
    - The background is orange.
    - There are 2-3 disconnected foreground components.
    - Each component is a solid red/grey horizontally striped rectangle with a one-pixel-thick maroon tail extending from the center of one side in a cardinal direction.

    Rule:
    - For each component, delete the tail and keep the striped rectangle in place.
    - The tail direction is the stamping direction; the step size is the rectangle height for vertical stamping and the rectangle width for horizontal stamping.
    - The k-th copy is offset by k steps, shrinks by k cells from each side of the perpendicular axis, swaps red and grey if k is odd, and is clipped to the grid if only partly in bounds.
    - Stop a rectangle's sequence once the shrinking dimension becomes non-positive or the next copy lies fully outside the grid.

    Key insight:
    - The tail supplies direction only, the rectangle dimensions supply stride, and the visible copies are centered crops of the original stripe pattern with alternating color polarity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    objects = _extract_objects(g)

    out = np.full_like(g, BG)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = obj["body"]

    for obj in objects:
        body = obj["body"]
        r0, c0, _r1, _c1 = obj["bbox"]
        h = obj["h"]
        w = obj["w"]
        direction = obj["direction"]
        dr, dc = DIRS[direction]
        k = 1
        while True:
            if direction in ("U", "D"):
                if w - 2 * k <= 0:
                    break
                sprite = body[:, k : w - k]
                rr0 = r0 + dr * h * k
                cc0 = c0 + k
            else:
                if h - 2 * k <= 0:
                    break
                sprite = body[k : h - k, :]
                rr0 = r0 + k
                cc0 = c0 + dc * w * k

            if rr0 + sprite.shape[0] - 1 < 0 or rr0 >= g.shape[0] or cc0 + sprite.shape[1] - 1 < 0 or cc0 >= g.shape[1]:
                break
            stamped = _swap_body_colors(sprite) if k % 2 == 1 else sprite.copy()
            _clip_paste(out, stamped, rr0, cc0)
            k += 1

    return out


def _sample_dimensions(direction: str, termination: str) -> tuple[int, int]:
    if direction in ("U", "D"):
        h = random.choice([4, 5, 6])
        w = random.choice([5, 7] if termination == "shrink" else [7, 9])
    else:
        h = random.choice([5, 7] if termination == "shrink" else [7, 9])
        w = random.choice([5, 6, 7] if termination == "shrink" else [6, 7, 8, 9])
    return h, w


def _dilate(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    H, W = shape
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < H and 0 <= nc < W:
                out.add((nr, nc))
    return out


def _choose_tail_len(direction: str, h: int, w: int) -> int:
    step = h if direction in ("U", "D") else w
    return random.randint(2, min(4, step - 2))


def _candidate_positions(
    spec: dict,
    shape: tuple[int, int],
    forbidden_input: set[tuple[int, int]],
    forbidden_output: set[tuple[int, int]],
) -> list[dict]:
    H, W = shape
    h = spec["h"]
    w = spec["w"]
    direction = spec["direction"]
    termination = spec["termination"]
    tail_len = spec["tail_len"]
    positions: list[dict] = []

    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            tails = _tail_cells(r0, c0, h, w, direction, tail_len)
            if not all(0 <= r < H and 0 <= c < W for r, c in tails):
                continue
            input_cells = _body_cells(r0, c0, h, w) | set(tails)
            if input_cells & forbidden_input:
                continue

            sim = _simulate_object(r0, c0, h, w, direction, shape)
            copies = sim["copies"]
            cause = sim["cause"]
            output_cells = sim["output_cells"]
            if output_cells & forbidden_output:
                continue
            if len(copies) < 2:
                continue
            if termination == "shrink":
                if cause != "shrink" or any(copy["partial"] for copy in copies):
                    continue
            else:
                if cause != "offgrid":
                    continue
                if not any(copy["partial"] for copy in copies):
                    continue
                if not any(not copy["partial"] for copy in copies):
                    continue

            positions.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "input_cells": input_cells,
                    "output_cells": output_cells,
                    "copies": copies,
                    "cause": cause,
                }
            )
    random.shuffle(positions)
    return positions


def _render_grid(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in objects:
        body = _build_body(obj["h"], obj["w"], obj["start_color"])
        grid[obj["r0"] : obj["r0"] + obj["h"], obj["c0"] : obj["c0"] + obj["w"]] = body
        for r, c in _tail_cells(obj["r0"], obj["c0"], obj["h"], obj["w"], obj["direction"], obj["tail_len"]):
            grid[r, c] = TAIL
    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        objects = _extract_objects(grid)
        out = main(grid)
    except AssertionError:
        return False

    directions = {obj["direction"] for obj in objects}
    if not directions & {"U", "D"} or not directions & {"L", "R"}:
        return False
    if len({obj["tail_len"] for obj in objects}) < 2:
        return False
    if len({obj["h"] if obj["direction"] in ("U", "D") else obj["w"] for obj in objects}) < 2:
        return False

    causes = set()
    saw_partial = False
    for obj in objects:
        r0, c0, _r1, _c1 = obj["bbox"]
        sim = _simulate_object(r0, c0, obj["h"], obj["w"], obj["direction"], grid.shape)
        if len(sim["copies"]) < 2:
            return False
        causes.add(sim["cause"])
        saw_partial = saw_partial or any(copy["partial"] for copy in sim["copies"])
    if causes != {"shrink", "offgrid"}:
        return False
    if not saw_partial:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(20, 26)
    W = random.randint(20, 28)
    count = random.choice([2, 3])

    vertical_direction = random.choice(["U", "D"])
    horizontal_direction = random.choice(["L", "R"])
    if random.random() < 0.5:
        vertical_term, horizontal_term = "shrink", "clip"
    else:
        vertical_term, horizontal_term = "clip", "shrink"

    specs = [
        {"direction": vertical_direction, "termination": vertical_term},
        {"direction": horizontal_direction, "termination": horizontal_term},
    ]
    if count == 3:
        remaining = [d for d in ("U", "D", "L", "R") if d not in {vertical_direction, horizontal_direction}]
        specs.append(
            {
                "direction": random.choice(remaining),
                "termination": random.choice(["shrink", "clip"]),
            }
        )
    random.shuffle(specs)

    for spec in specs:
        spec["h"], spec["w"] = _sample_dimensions(spec["direction"], spec["termination"])
        spec["tail_len"] = _choose_tail_len(spec["direction"], spec["h"], spec["w"])
        spec["start_color"] = random.choice(BODY_COLORS)

    placed: list[dict] = []
    forbidden_input: set[tuple[int, int]] = set()
    forbidden_output: set[tuple[int, int]] = set()

    # Place edge-constrained objects first.
    specs.sort(key=lambda spec: 0 if spec["termination"] == "clip" else 1)

    for spec in specs:
        candidates = _candidate_positions(spec, (H, W), forbidden_input, forbidden_output)
        if not candidates:
            return None
        chosen = random.choice(candidates[: min(40, len(candidates))])
        merged = dict(spec)
        merged.update(chosen)
        placed.append(merged)
        forbidden_input |= _dilate(chosen["input_cells"], (H, W))
        forbidden_output |= _dilate(chosen["output_cells"], (H, W))

    grid = _render_grid((H, W), placed)
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tail-directed shrinking stamp input")
