import random
from collections import Counter

import numpy as np


BG = 8
MAX_GEN_ATTEMPTS = 400
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_info(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    color = int(grid[comp[0]])
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    if set(comp) != border:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    marker_cells = np.argwhere((interior != _get_bg(grid)) & (interior != color))
    if marker_cells.shape[0] > 1:
        return None

    marker = None
    if marker_cells.shape[0] == 1:
        mr, mc = map(int, marker_cells[0])
        marker = (mr, mc)

    return {
        "bbox": (r0, c0, r1, c1),
        "color": color,
        "interior_h": r1 - r0 - 1,
        "interior_w": c1 - c0 - 1,
        "marker": marker,
    }


def _map_coord(pos: int, src_len: int, dst_len: int) -> int:
    if src_len <= 1 or dst_len <= 1:
        return 0
    return int(round(pos * (dst_len - 1) / (src_len - 1)))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    frames: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        for comp in _components(g == color):
            info = _frame_info(g, comp)
            if info is not None:
                frames.append(info)

    assert len(frames) >= 3, "expected multiple small frames plus one large frame"
    areas = [info["interior_h"] * info["interior_w"] for info in frames]
    big_idx = int(np.argmax(areas))
    assert areas.count(areas[big_idx]) == 1, "largest frame must be unique"
    big = frames[big_idx]
    small = [info for i, info in enumerate(frames) if i != big_idx and info["marker"] is not None]
    assert len(small) >= 2, "expected at least two marked small frames"

    out = g.copy()
    br0, bc0, br1, bc1 = big["bbox"]
    for info in small:
        sr, sc = info["marker"]
        tr = _map_coord(sr, info["interior_h"], big["interior_h"])
        tc = _map_coord(sc, info["interior_w"], big["interior_w"])
        out[br0 + 1 + tr, bc0 + 1 + tc] = int(info["color"])
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1, c1 = r0 + h - 1, c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _free_with_padding(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    h0, w0 = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0, r0 + h + pad)
    cc1 = min(w0, c0 + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _mark_occ(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _sample_marker_pos(rng: random.Random, ih: int, iw: int) -> tuple[int, int]:
    preferred = [
        (0, 0),
        (0, iw - 1),
        (ih - 1, 0),
        (ih - 1, iw - 1),
        (ih // 2, iw // 2),
        (0, iw // 2),
        (ih // 2, 0),
        (ih // 2, iw - 1),
        (ih - 1, iw // 2),
    ]
    preferred = [(r, c) for r, c in preferred if 0 <= r < ih and 0 <= c < iw]
    if preferred and rng.random() < 0.8:
        return preferred[rng.randrange(len(preferred))]
    return rng.randrange(ih), rng.randrange(iw)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(24, 34)
    w = rng.randint(24, 34)
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    palette = [c for c in range(10) if c != BG]
    rng.shuffle(palette)

    big_h = rng.randint(12, min(18, h - 4))
    big_w = rng.randint(12, min(18, w - 4))
    big_r0 = rng.randint(1, h - big_h - 1)
    big_c0 = rng.randint(1, w - big_w - 1)
    big_color = palette.pop()
    _draw_frame(grid, big_r0, big_c0, big_h, big_w, big_color)
    _mark_occ(occ, big_r0, big_c0, big_h, big_w)
    big_ih, big_iw = big_h - 2, big_w - 2

    n_small = rng.randint(2, 5)
    frame_specs: list[tuple[int, int, int, int, int, int, int]] = []
    used_targets: set[tuple[int, int]] = set()
    for _ in range(n_small):
        placed = False
        for _retry in range(200):
            fh = rng.randint(4, 9)
            fw = rng.randint(4, 9)
            if (fh - 2) * (fw - 2) >= big_ih * big_iw:
                continue
            r0 = rng.randint(1, h - fh - 1)
            c0 = rng.randint(1, w - fw - 1)
            if not _free_with_padding(occ, r0, c0, fh, fw, pad=1):
                continue
            color_choices = [c for c in palette if c != big_color]
            if not color_choices:
                return None
            color = color_choices.pop()
            palette.remove(color)
            ih, iw = fh - 2, fw - 2
            marker_r, marker_c = _sample_marker_pos(rng, ih, iw)
            target = (_map_coord(marker_r, ih, big_ih), _map_coord(marker_c, iw, big_iw))
            if target in used_targets:
                palette.append(color)
                continue
            used_targets.add(target)
            marker_color_pool = [c for c in range(10) if c not in (BG, color)]
            marker_color = rng.choice(marker_color_pool)
            _draw_frame(grid, r0, c0, fh, fw, color)
            grid[r0 + 1 + marker_r, c0 + 1 + marker_c] = marker_color
            _mark_occ(occ, r0, c0, fh, fw)
            frame_specs.append((r0, c0, fh, fw, color, marker_r, marker_c))
            placed = True
            break
        if not placed:
            return None

    out = main(grid)
    new_marks = np.argwhere(out != grid)
    if new_marks.shape[0] != len(frame_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
