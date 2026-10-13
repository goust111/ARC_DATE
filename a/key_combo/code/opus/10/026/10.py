from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
RECT = 2
MARK = 8
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste_rect(out: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    H, W = out.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(H, int(r0 + h))
    cc1 = min(W, int(c0 + w))
    if rr0 < rr1 and cc0 < cc1:
        out[rr0:rr1, cc0:cc1] = int(color)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    H, W = shape
    return 0 <= int(r) < H and 0 <= int(c) < W


def _enumerate_copies(H: int, W: int, r0: int, c0: int, h: int, w: int) -> list[dict]:
    infos = [
        {
            "direction": "origin",
            "step": 0,
            "lattice": (0, 0),
            "top_left": (r0, c0),
            "partial": False,
            "mark_local": (0, 0),
            "mark_abs": (r0, c0),
            "mark_visible": True,
        }
    ]
    for sign, direction in ((-1, "left"), (1, "right")):
        step = 1
        while True:
            sr = r0 + step * h
            sc = c0 + sign * step * w
            if sr >= H or sc >= W or sc + w <= 0:
                break
            mr = step % h
            mc = step % w
            ar = sr + mr
            ac = sc + mc
            infos.append(
                {
                    "direction": direction,
                    "step": step,
                    "lattice": (step, sign * step),
                    "top_left": (sr, sc),
                    "partial": (sr + h > H) or (sc < 0) or (sc + w > W),
                    "mark_local": (mr, mc),
                    "mark_abs": (ar, ac),
                    "mark_visible": _in_bounds((H, W), ar, ac),
                }
            )
            step += 1
    return infos


def _render_output(H: int, W: int, r0: int, c0: int, h: int, w: int) -> tuple[np.ndarray, list[dict]]:
    out = np.full((H, W), BG, dtype=int)
    infos = _enumerate_copies(H, W, r0, c0, h, w)
    for info in infos:
        sr, sc = info["top_left"]
        _paste_rect(out, sr, sc, h, w, RECT)
        if info["mark_visible"]:
            ar, ac = info["mark_abs"]
            out[ar, ac] = MARK
    return out, infos


def _count_components(mask: np.ndarray) -> int:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    total = 0
    for r in range(H):
        for c in range(W):
            if seen[r, c] or not mask[r, c]:
                continue
            total += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return total


def _analyze_candidate(H: int, W: int, r0: int, c0: int, h: int, w: int) -> dict:
    out, infos = _render_output(H, W, r0, c0, h, w)
    left = [info for info in infos if info["direction"] == "left"]
    right = [info for info in infos if info["direction"] == "right"]
    visible_marks = [info for info in infos if info["mark_visible"]]
    unique_marks = {tuple(info["mark_local"]) for info in visible_marks}
    labels = {
        "clean": not any(info["partial"] for info in infos[1:]),
        "left_clip": any(info["direction"] == "left" and info["partial"] for info in infos),
        "right_clip": any(info["direction"] == "right" and info["partial"] for info in infos),
        "hidden_mark": any(info["partial"] and not info["mark_visible"] for info in infos[1:]),
        "bottom_clip": any(info["top_left"][0] + h > H for info in infos[1:]),
    }
    return {
        "H": H,
        "W": W,
        "r0": r0,
        "c0": c0,
        "h": h,
        "w": w,
        "infos": infos,
        "out": out,
        "left_count": len(left),
        "right_count": len(right),
        "visible_mark_count": len(visible_marks),
        "unique_mark_count": len(unique_marks),
        "labels": labels,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the marked rectangle down both diagonal directions and re-mark each copy by its lattice coordinate modulo the rectangle size.

    Input structure:
    - The grid has black background and exactly one non-black solid rectangle.
    - The rectangle is red except for one teal pixel at its internal top-left corner.
    - The teal pixel marks the original copy's lattice coordinate (0,0), and the full rectangle stays inside the input grid.

    Rule:
    - Read the bounding box of all non-black cells to recover the rectangle height h and width w.
    - Keep the original rectangle as lattice coordinate (0,0), then place copies at offsets (h,-w), (2h,-2w), ... and (h,w), (2h,2w), ... until a copy is fully outside the grid; clip any partially outside copy.
    - Every visible copy is drawn as a solid red rectangle, then its lattice coordinate (i,j) is converted to a local mark position (|i| mod h, |j| mod w); if that local pixel lies inside the visible clipped area, recolor it teal.

    Key insight:
    - The rectangle size defines the diagonal stride, while each copy's teal marker is not fixed: it encodes that copy's global lattice coordinate after reducing it back into local pixel coordinates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    assert bg == BG, "expected black background"
    assert set(map(int, np.unique(g))).issubset({BG, RECT, MARK})

    fg_mask = g != BG
    box = _bbox(fg_mask)
    assert box is not None, "expected one foreground rectangle"
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    rect = g[r0 : r1 + 1, c0 : c1 + 1]

    assert np.all(rect != BG), "foreground must fill its bounding box"
    assert int(fg_mask.sum()) == h * w, "expected a solid rectangle"
    assert np.count_nonzero(g == MARK) == 1, "expected exactly one teal marker"
    assert int(g[r0, c0]) == MARK, "marker must be at the rectangle's top-left corner"
    assert np.all(rect[(np.indices(rect.shape) != 0).any(axis=0)] == RECT), "all other rectangle cells must be red"

    out, _infos = _render_output(g.shape[0], g.shape[1], r0, c0, h, w)
    return out


GRID_RANGE = (18, 26)
DIM_CHOICES = [(3, 4), (4, 3), (3, 3), (4, 4)]
SCENARIOS = (
    ("clean", 1),
    ("left_clip", 2),
    ("right_clip", 2),
    ("hidden_mark", 3),
    ("bottom_clip", 2),
)


def _is_semantic_candidate(info: dict) -> bool:
    c_mid = info["c0"] + (info["w"] - 1) / 2
    H = info["H"]
    W = info["W"]
    out = info["out"]
    expected_components = 1 + info["left_count"] + info["right_count"]
    mark_total = info["visible_mark_count"]

    if info["left_count"] < 2 or info["right_count"] < 2:
        return False
    if info["unique_mark_count"] < 3:
        return False
    if not (info["r0"] <= max(4, H // 3) and W * 0.25 <= c_mid <= W * 0.75):
        return False
    if _count_components(out != BG) != expected_components:
        return False
    if int(np.count_nonzero(out == MARK)) != mark_total:
        return False
    if info["labels"]["hidden_mark"] and mark_total >= expected_components:
        return False
    return True


def _choose_candidate() -> dict | None:
    H = random.randint(*GRID_RANGE)
    W = random.randint(*GRID_RANGE)
    h, w = random.choice(DIM_CHOICES)

    target = random.choices([name for name, _weight in SCENARIOS], weights=[weight for _name, weight in SCENARIOS], k=1)[0]
    candidates = []
    targeted = []

    for r0 in range(0, H - h + 1):
        for c0 in range(0, W - w + 1):
            info = _analyze_candidate(H, W, r0, c0, h, w)
            if not _is_semantic_candidate(info):
                continue
            candidates.append(info)
            if info["labels"][target]:
                targeted.append(info)

    pool = targeted or candidates
    if not pool:
        return None
    return random.choice(pool)


def _verify_input(inp: np.ndarray, expected: np.ndarray, info: dict) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.count_nonzero(inp == MARK) != 1:
        return False
    if int(np.count_nonzero(expected == MARK)) != info["visible_mark_count"]:
        return False
    if _count_components(inp != BG) != 1:
        return False
    if _count_components(expected != BG) != 1 + info["left_count"] + info["right_count"]:
        return False
    return np.array_equal(main(inp), expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        info = _choose_candidate()
        if info is None:
            continue
        grid = np.full((info["H"], info["W"]), BG, dtype=int)
        grid[info["r0"] : info["r0"] + info["h"], info["c0"] : info["c0"] + info["w"]] = RECT
        grid[info["r0"], info["c0"]] = MARK
        if _verify_input(grid, info["out"], info):
            return grid
    raise RuntimeError("failed to generate a valid input")
