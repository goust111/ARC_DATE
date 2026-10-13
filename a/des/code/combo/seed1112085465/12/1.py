from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400
MIN_GRID = 18
MAX_GRID = 30


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _color_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            comps.append({"color": int(color), "cells": comp})
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    seq = list(cells)
    min_r = min(r for r, _ in seq)
    min_c = min(c for _, c in seq)
    return frozenset((int(r - min_r), int(c - min_c)) for r, c in seq)


def _rotate_shape(cells: frozenset[tuple[int, int]], k: int) -> frozenset[tuple[int, int]]:
    pts = list(cells)
    for _ in range(int(k) % 4):
        pts = [(c, -r) for r, c in pts]
    return _normalize(pts)


def _rotations(cells: frozenset[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    seen: set[frozenset[tuple[int, int]]] = set()
    outs: list[frozenset[tuple[int, int]]] = []
    for k in range(4):
        rot = _rotate_shape(cells, k)
        if rot not in seen:
            seen.add(rot)
            outs.append(rot)
    return outs


def _extract_workpiece(comp: dict) -> dict | None:
    cells = comp["cells"]
    color = int(comp["color"])
    r0, c0, r1, c1 = _bbox(cells)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    local = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        local[r - r0, c - c0] = True

    missing = ~local
    if not bool(missing.any()):
        return None

    missing_comps = _components(missing)
    if len(missing_comps) != 1:
        return None

    notch_local = missing_comps[0]
    if len(notch_local) < 2:
        return None

    sides = set()
    if any(r == 0 for r, _ in notch_local):
        sides.add("top")
    if any(r == h - 1 for r, _ in notch_local):
        sides.add("bottom")
    if any(c == 0 for _, c in notch_local):
        sides.add("left")
    if any(c == w - 1 for _, c in notch_local):
        sides.add("right")
    if len(sides) != 1:
        return None

    side = next(iter(sides))
    notch_rows = [r for r, _ in notch_local]
    notch_cols = [c for _, c in notch_local]
    if side in {"top", "bottom"}:
        if min(notch_cols) == 0 or max(notch_cols) == w - 1:
            return None
    else:
        if min(notch_rows) == 0 or max(notch_rows) == h - 1:
            return None

    return {
        "kind": "workpiece",
        "color": color,
        "cells": [(int(r), int(c)) for r, c in cells],
        "cell_set": set((int(r), int(c)) for r, c in cells),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "notch_cells": [(int(r0 + r), int(c0 + c)) for r, c in notch_local],
        "notch_shape": _normalize(notch_local),
        "side": side,
    }


def _extract_tool(comp: dict) -> dict:
    cells = [(int(r), int(c)) for r, c in comp["cells"]]
    r0, c0, _r1, _c1 = _bbox(cells)
    local = [(r - r0, c - c0) for r, c in cells]
    shape = _normalize(local)
    return {
        "kind": "tool",
        "color": int(comp["color"]),
        "cells": cells,
        "shape": shape,
        "rotations": _rotations(shape),
    }


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    workpieces: list[dict] = []
    tools: list[dict] = []
    for comp in _color_components(grid):
        workpiece = _extract_workpiece(comp)
        if workpiece is not None:
            workpieces.append(workpiece)
        else:
            tools.append(_extract_tool(comp))
    assert workpieces, "expected at least one workpiece"
    assert tools, "expected at least one tool"
    return workpieces, tools


def _match_tools(workpieces: list[dict], tools: list[dict]) -> dict[int, int]:
    candidates: dict[int, list[int]] = {}
    for wi, workpiece in enumerate(workpieces):
        options = [ti for ti, tool in enumerate(tools) if workpiece["notch_shape"] in tool["rotations"]]
        assert options, "every workpiece notch must have a matching tool"
        candidates[wi] = options

    order = sorted(range(len(workpieces)), key=lambda wi: len(candidates[wi]))
    best: dict[int, int] | None = None

    def backtrack(pos: int, used: set[int], current: dict[int, int]) -> None:
        nonlocal best
        if best is not None:
            return
        if pos == len(order):
            best = dict(current)
            return
        wi = order[pos]
        for ti in candidates[wi]:
            if ti in used:
                continue
            used.add(ti)
            current[wi] = ti
            backtrack(pos + 1, used, current)
            current.pop(wi)
            used.remove(ti)

    backtrack(0, set(), {})
    assert best is not None, "tool matching must be one-to-one"
    assert len(best) == len(workpieces)
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each tool to fit a matching boundary notch, erase the original tool,
    fill the notch with the workpiece color, and mark the inserted border with
    the tool's original color.
    """
    g = np.asarray(grid, dtype=int)
    workpieces, tools = _parse_scene(g)
    matches = _match_tools(workpieces, tools)

    out = g.copy()
    matched_tools = set(matches.values())
    assert len(matched_tools) == len(workpieces)

    for ti in matched_tools:
        tool = tools[ti]
        for r, c in tool["cells"]:
            out[r, c] = BG

    for wi, ti in matches.items():
        workpiece = workpieces[wi]
        tool = tools[ti]
        notch_set = set(workpiece["notch_cells"])
        for r, c in workpiece["notch_cells"]:
            out[r, c] = workpiece["color"]
        for r, c in workpiece["notch_cells"]:
            if any((r + dr, c + dc) in workpiece["cell_set"] for dr, dc in DIRS4):
                out[r, c] = tool["color"]

    return out


def _paint(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    target[mask] = patch[mask]


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    h, w = patch.shape
    H, W = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    return not bool(np.any(canvas[rr0:rr1, cc0:cc1] != BG))


def _rect_patch(h: int, w: int, color: int) -> np.ndarray:
    return np.full((int(h), int(w)), int(color), dtype=int)


def _make_workpiece(
    outer_h: int,
    outer_w: int,
    notch_h: int,
    notch_w: int,
    side: str,
    color: int,
) -> np.ndarray:
    patch = np.full((int(outer_h), int(outer_w)), int(color), dtype=int)
    if side == "top":
        c0 = random.randint(1, outer_w - notch_w - 1)
        patch[:notch_h, c0 : c0 + notch_w] = BG
    elif side == "bottom":
        c0 = random.randint(1, outer_w - notch_w - 1)
        patch[outer_h - notch_h :, c0 : c0 + notch_w] = BG
    elif side == "left":
        r0 = random.randint(1, outer_h - notch_h - 1)
        patch[r0 : r0 + notch_h, :notch_w] = BG
    elif side == "right":
        r0 = random.randint(1, outer_h - notch_h - 1)
        patch[r0 : r0 + notch_h, outer_w - notch_w :] = BG
    else:
        raise AssertionError(side)
    return patch


def _size_pool() -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for a in range(2, 6):
        for b in range(a + 1, 7):
            sizes.append((a, b))
    random.shuffle(sizes)
    return sizes


def _build_scene() -> np.ndarray | None:
    n_objects = random.randint(2, 4)
    colors = random.sample(range(1, 10), 2 * n_objects)
    workpiece_colors = colors[:n_objects]
    tool_colors = colors[n_objects:]
    sides = [random.choice(("top", "bottom", "left", "right")) for _ in range(n_objects)]
    size_choices = _size_pool()[:n_objects]

    pieces: list[np.ndarray] = []
    for idx in range(n_objects):
        a, b = size_choices[idx]
        notch_h, notch_w = (a, b) if random.random() < 0.5 else (b, a)
        side = sides[idx]
        min_h = max(6, notch_h + 2 + (1 if side in {"left", "right"} else 0))
        min_w = max(6, notch_w + 2 + (1 if side in {"top", "bottom"} else 0))
        outer_h = random.randint(min_h, min(12, min_h + 4))
        outer_w = random.randint(min_w, min(12, min_w + 4))
        if side in {"top", "bottom"} and outer_w < notch_w + 2:
            outer_w = notch_w + 2
        if side in {"left", "right"} and outer_h < notch_h + 2:
            outer_h = notch_h + 2
        if side in {"top", "bottom"} and outer_h <= notch_h:
            outer_h = notch_h + 2
        if side in {"left", "right"} and outer_w <= notch_w:
            outer_w = notch_w + 2

        workpiece = _make_workpiece(outer_h, outer_w, notch_h, notch_w, side, workpiece_colors[idx])
        tool = _rect_patch(notch_h, notch_w, tool_colors[idx])
        tool = np.rot90(tool, random.randrange(4))
        pieces.append(workpiece)
        pieces.append(tool)

    max_h = max(p.shape[0] for p in pieces)
    max_w = max(p.shape[1] for p in pieces)
    min_side = max(MIN_GRID, max_h + 2, max_w + 2, 8 + 3 * n_objects)
    H = random.randint(min_side, MAX_GRID)
    W = random.randint(min_side, MAX_GRID)
    canvas = np.full((H, W), BG, dtype=int)

    order = list(range(len(pieces)))
    random.shuffle(order)
    for piece_index in order:
        patch = pieces[piece_index]
        placed = False
        for _ in range(250):
            r0 = random.randint(0, H - patch.shape[0])
            c0 = random.randint(0, W - patch.shape[1])
            if not _can_place(canvas, patch, r0, c0, padding=1):
                continue
            _paint(canvas, patch, r0, c0)
            placed = True
            break
        if not placed:
            return None

    workpieces, tools = _parse_scene(canvas)
    if len(workpieces) != n_objects or len(tools) != n_objects:
        return None
    matches = _match_tools(workpieces, tools)
    if len(set(matches.values())) != n_objects:
        return None
    _ = main(canvas)
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
