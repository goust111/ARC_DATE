from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
LEGEND_COLOR = 7
LINE_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 200
ARM_MIN = 2
ARM_MAX = 4


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _extract_legend(grid: np.ndarray) -> tuple[int, int, dict[str, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    for color0 in np.unique(g):
        color = int(color0)
        if color == BG:
            continue
        entries: list[tuple[int, int, np.ndarray]] = []
        for comp in _components(g == color):
            rs = comp[:, 0]
            cs = comp[:, 1]
            r0, r1 = int(rs.min()), int(rs.max())
            c0, c1 = int(cs.min()), int(cs.max())
            if r1 == h - 1 and r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3:
                entries.append((c0, r0, g[r0 : r1 + 1, c0 : c1 + 1].copy()))
        if len(entries) == 3:
            entries.sort(key=lambda item: item[0])
            tops = {item[1] for item in entries}
            assert len(tops) == 1, "legend entries must share one row band"
            legend_top = entries[0][1]
            return color, legend_top, {
                "cross": entries[0][2],
                "tee": entries[1][2],
                "ell": entries[2][2],
            }
    raise AssertionError("failed to locate bottom legend")


def _find_intersections(mask: np.ndarray) -> list[tuple[int, int, str]]:
    h, w = mask.shape
    found: list[tuple[int, int, str]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            left = c > 0 and bool(mask[r, c - 1])
            right = c + 1 < w and bool(mask[r, c + 1])
            up = r > 0 and bool(mask[r - 1, c])
            down = r + 1 < h and bool(mask[r + 1, c])
            has_h = left or right
            has_v = up or down
            if not (has_h and has_v):
                continue
            horiz_pass = left and right
            vert_pass = up and down
            if horiz_pass and vert_pass:
                kind = "cross"
            elif horiz_pass or vert_pass:
                kind = "tee"
            else:
                kind = "ell"
            found.append((r, c, kind))
    return found


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_color, legend_top, legend = _extract_legend(g)
    scene = g[:legend_top, :]
    intersections = _find_intersections(scene != BG)

    out = g.copy()
    for r, c, kind in intersections:
        assert 0 < r < legend_top - 1 and 0 < c < g.shape[1] - 1
        out[r - 1 : r + 2, c - 1 : c + 2] = legend[kind]
    return out


def _rotate(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(patch, k=turns).copy()


def _legend_patches() -> dict[str, np.ndarray]:
    cross = np.array(
        [
            [0, LEGEND_COLOR, 0],
            [LEGEND_COLOR, LEGEND_COLOR, LEGEND_COLOR],
            [0, LEGEND_COLOR, 0],
        ],
        dtype=int,
    )
    tee_base = np.array(
        [
            [LEGEND_COLOR, LEGEND_COLOR, LEGEND_COLOR],
            [0, LEGEND_COLOR, 0],
            [0, LEGEND_COLOR, 0],
        ],
        dtype=int,
    )
    ell_base = np.array(
        [
            [LEGEND_COLOR, 0, 0],
            [LEGEND_COLOR, 0, 0],
            [LEGEND_COLOR, LEGEND_COLOR, LEGEND_COLOR],
        ],
        dtype=int,
    )
    return {
        "cross": cross,
        "tee": _rotate(tee_base, random.randint(0, 3)),
        "ell": _rotate(ell_base, random.randint(0, 3)),
    }


def _sample_gadget(kind: str) -> dict[str, object]:
    if kind == "cross":
        left = random.randint(ARM_MIN, ARM_MAX)
        right = random.randint(ARM_MIN, ARM_MAX)
        up = random.randint(ARM_MIN, ARM_MAX)
        down = random.randint(ARM_MIN, ARM_MAX)
    elif kind == "tee":
        if random.random() < 0.5:
            left = random.randint(ARM_MIN, ARM_MAX)
            right = random.randint(ARM_MIN, ARM_MAX)
            if random.random() < 0.5:
                up = random.randint(ARM_MIN, ARM_MAX)
                down = 0
            else:
                up = 0
                down = random.randint(ARM_MIN, ARM_MAX)
        else:
            up = random.randint(ARM_MIN, ARM_MAX)
            down = random.randint(ARM_MIN, ARM_MAX)
            if random.random() < 0.5:
                left = random.randint(ARM_MIN, ARM_MAX)
                right = 0
            else:
                left = 0
                right = random.randint(ARM_MIN, ARM_MAX)
    else:
        assert kind == "ell"
        left = random.randint(ARM_MIN, ARM_MAX) if random.random() < 0.5 else 0
        right = 0 if left else random.randint(ARM_MIN, ARM_MAX)
        up = random.randint(ARM_MIN, ARM_MAX) if random.random() < 0.5 else 0
        down = 0 if up else random.randint(ARM_MIN, ARM_MAX)

    h_color, v_color = random.sample(LINE_COLORS, 2)
    return {
        "kind": kind,
        "left": left,
        "right": right,
        "up": up,
        "down": down,
        "h_color": h_color,
        "v_color": v_color,
        "order": random.choice(("hv", "vh")),
    }


def _padded_shape(spec: dict[str, object]) -> tuple[int, int]:
    up = int(spec["up"])
    down = int(spec["down"])
    left = int(spec["left"])
    right = int(spec["right"])
    return up + down + 3, left + right + 3


def _place_box(occupied: np.ndarray, shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = occupied.shape
    ph, pw = shape
    if ph > h or pw > w:
        return None
    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - pw)
        if occupied[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        occupied[r0 : r0 + ph, c0 : c0 + pw] = True
        return r0, c0
    return None


def _draw_gadget(grid: np.ndarray, spec: dict[str, object], top: int, left0: int) -> tuple[int, int]:
    left = int(spec["left"])
    right = int(spec["right"])
    up = int(spec["up"])
    down = int(spec["down"])
    row = top + up + 1
    col = left0 + left + 1

    def draw_h() -> None:
        grid[row, col - left : col + right + 1] = int(spec["h_color"])

    def draw_v() -> None:
        grid[row - up : row + down + 1, col] = int(spec["v_color"])

    if spec["order"] == "hv":
        draw_h()
        draw_v()
    else:
        draw_v()
        draw_h()
    return row, col


def _place_legend(grid: np.ndarray, patches: dict[str, np.ndarray]) -> None:
    h, w = grid.shape
    gap1 = random.randint(1, 3)
    gap2 = random.randint(1, 3)
    total = 9 + gap1 + gap2
    start = random.randint(0, w - total)
    row = h - 3
    col = start
    for idx, kind in enumerate(("cross", "tee", "ell")):
        grid[row : row + 3, col : col + 3] = patches[kind]
        if idx == 0:
            col += 3 + gap1
        elif idx == 1:
            col += 3 + gap2


def _stamp_area(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    return r - 1, c - 1, r + 1, c + 1


def _gen_attempt() -> np.ndarray | None:
    kinds = ["cross", "tee", "ell"]
    kinds.extend(random.choice(("cross", "tee", "ell")) for _ in range(random.randint(0, 2)))
    random.shuffle(kinds)

    scene_h = random.randint(18, 24)
    width = random.randint(20, 30)
    full_h = scene_h + 4
    grid = np.zeros((full_h, width), dtype=int)
    occupied = np.zeros((scene_h, width), dtype=bool)
    patches = _legend_patches()

    centers: list[tuple[int, int, str]] = []
    for kind in kinds:
        spec = _sample_gadget(kind)
        pos = _place_box(occupied, _padded_shape(spec))
        if pos is None:
            return None
        center = _draw_gadget(grid, spec, pos[0], pos[1])
        centers.append((center[0], center[1], kind))

    _place_legend(grid, patches)

    if np.any(grid[scene_h, :] != BG):
        return None

    try:
        legend_color, legend_top, parsed = _extract_legend(grid)
    except AssertionError:
        return None
    if legend_color != LEGEND_COLOR or legend_top != scene_h + 1:
        return None

    found = _find_intersections(grid[:legend_top, :] != BG)
    if len(found) != len(centers):
        return None
    if sorted((r, c, k) for r, c, k in found) != sorted(centers):
        return None

    out = main(grid)
    covered = np.zeros_like(grid, dtype=bool)
    for r, c, kind in centers:
        if not np.array_equal(out[r - 1 : r + 2, c - 1 : c + 2], parsed[kind]):
            return None
        sr0, sc0, sr1, sc1 = _stamp_area((r, c))
        covered[sr0 : sr1 + 1, sc0 : sc1 + 1] = True

    keep_mask = ~covered
    keep_mask[legend_top:, :] = True
    if not np.array_equal(out[keep_mask], grid[keep_mask]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
