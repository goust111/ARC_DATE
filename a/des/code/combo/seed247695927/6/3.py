from __future__ import annotations

import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def is_solid_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def paint_rect(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def find_structure(grid: np.ndarray) -> tuple[dict, dict, list[dict], list[dict], int]:
    bg = get_bg(grid)
    rects = [comp for comp in find_components(grid, bg) if is_solid_rect(comp)]
    candidates = []

    by_color: dict[int, list[dict]] = {}
    for comp in rects:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    for color, comps in by_color.items():
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a, b = comps[i], comps[j]
                ar0, ac0, ar1, ac1 = a["bbox"]
                br0, bc0, br1, bc1 = b["bbox"]
                ah, aw = ar1 - ar0 + 1, ac1 - ac0 + 1
                bh, bw = br1 - br0 + 1, bc1 - bc0 + 1
                if (ah, aw) != (bh, bw):
                    continue
                if ar0 < br0 and ac0 < bc0:
                    src, dst = a, b
                elif br0 < ar0 and bc0 < ac0:
                    src, dst = b, a
                else:
                    continue

                sr0, sc0, sr1, sc1 = src["bbox"]
                dr0, dc0, dr1, dc1 = dst["bbox"]
                if not (sr0 > 0 and sc0 > 0 and dr1 + 1 < grid.shape[0] and dc1 + 1 < grid.shape[1]):
                    continue

                right_controls = []
                below_controls = []
                for comp in rects:
                    if comp is src or comp is dst or int(comp["color"]) == color:
                        continue
                    r0, c0, r1, c1 = comp["bbox"]
                    if r0 == sr0 and r1 == sr1 and c0 == c1 and sc1 < c0 < dc0:
                        right_controls.append(comp)
                    if c0 == sc0 and c1 == sc1 and r0 == r1 and sr1 < r0 < dr0:
                        below_controls.append(comp)

                if len(right_controls) != 2 or len(below_controls) != 2:
                    continue

                right_controls.sort(key=lambda comp: comp["bbox"][1])
                below_controls.sort(key=lambda comp: comp["bbox"][0])
                right_offsets = [comp["bbox"][1] - sc1 for comp in right_controls]
                below_offsets = [comp["bbox"][0] - sr1 for comp in below_controls]
                if right_offsets != below_offsets or right_offsets[0] != 1:
                    continue
                step = right_offsets[1] - right_offsets[0]
                if step <= 0:
                    continue
                period = 2 * step
                if (dc0 - 1 - sc1 - right_offsets[0]) % period != 0:
                    continue
                if (dr0 - 1 - sr1 - below_offsets[0]) % period != 0:
                    continue

                candidates.append((src, dst, right_controls, below_controls, bg))

    if len(candidates) != 1:
        raise AssertionError("expected exactly one valid decorated-anchor structure")
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    src, dst, right_controls, below_controls, bg = find_structure(g)

    sr0, sc0, sr1, sc1 = src["bbox"]
    dr0, dc0, dr1, dc1 = dst["bbox"]
    anchor_color = int(src["color"])
    right_offsets = [int(comp["bbox"][1] - sc1) for comp in right_controls]
    below_offsets = [int(comp["bbox"][0] - sr1) for comp in below_controls]
    right_colors = [int(comp["color"]) for comp in right_controls]
    below_colors = [int(comp["color"]) for comp in below_controls]
    period = 2 * (right_offsets[1] - right_offsets[0])

    out = g.copy()

    paint_rect(out, sr0, dc0, sr1, dc1, anchor_color)
    paint_rect(out, dr0, sc0, dr1, sc1, anchor_color)

    first_c = sc1 + right_offsets[0]
    last_c = dc0 - 1
    for c in range(first_c, last_c + 1):
        offset = c - sc1
        if (offset - right_offsets[0]) % period == 0:
            color = right_colors[0]
        elif (offset - right_offsets[1]) % period == 0:
            color = right_colors[1]
        else:
            continue
        paint_rect(out, sr0, c, sr1, c, color)
        paint_rect(out, dr0, c, dr1, c, color)
    out[sr0 - 1, first_c : last_c + 1] = right_colors[0]
    out[dr1 + 1, first_c : last_c + 1] = right_colors[0]

    first_r = sr1 + below_offsets[0]
    last_r = dr0 - 1
    left_border = sc0 - 1
    right_border = dc1 + 1
    for r in range(first_r, last_r + 1):
        out[r, left_border] = below_colors[0]
        out[r, right_border] = below_colors[0]
        offset = r - sr1
        if (offset - below_offsets[0]) % period == 0:
            out[r, left_border : sc1 + 1] = below_colors[0]
            out[r, dc0 : right_border + 1] = below_colors[0]
        elif (offset - below_offsets[1]) % period == 0:
            out[r, sc0 : sc1 + 1] = below_colors[1]
            out[r, dc0 : dc1 + 1] = below_colors[1]

    return out


MAX_GENERATION_ATTEMPTS = 300


def build_output(
    rows: int,
    cols: int,
    bg: int,
    anchor_color: int,
    right_colors: tuple[int, int],
    below_colors: tuple[int, int],
    sr0: int,
    sc0: int,
    ah: int,
    aw: int,
    step: int,
    n_h: int,
    n_v: int,
) -> tuple[np.ndarray, tuple[int, int, int, int, int, int, int, int]]:
    out = np.full((rows, cols), int(bg), dtype=int)
    sr1, sc1 = sr0 + ah - 1, sc0 + aw - 1
    period = 2 * step
    dr0 = sr1 + 2 + period * n_v
    dc0 = sc1 + 2 + period * n_h
    dr1, dc1 = dr0 + ah - 1, dc0 + aw - 1

    for r0, c0 in ((sr0, sc0), (sr0, dc0), (dr0, sc0), (dr0, dc0)):
        paint_rect(out, r0, c0, r0 + ah - 1, c0 + aw - 1, anchor_color)

    first_c = sc1 + 1
    last_c = dc0 - 1
    for c in range(first_c, last_c + 1):
        offset = c - sc1
        if (offset - 1) % period == 0:
            color = right_colors[0]
        elif (offset - (1 + step)) % period == 0:
            color = right_colors[1]
        else:
            continue
        paint_rect(out, sr0, c, sr1, c, color)
        paint_rect(out, dr0, c, dr1, c, color)
    out[sr0 - 1, first_c : last_c + 1] = right_colors[0]
    out[dr1 + 1, first_c : last_c + 1] = right_colors[0]

    first_r = sr1 + 1
    last_r = dr0 - 1
    left_border = sc0 - 1
    right_border = dc1 + 1
    for r in range(first_r, last_r + 1):
        out[r, left_border] = below_colors[0]
        out[r, right_border] = below_colors[0]
        offset = r - sr1
        if (offset - 1) % period == 0:
            out[r, left_border : sc1 + 1] = below_colors[0]
            out[r, dc0 : right_border + 1] = below_colors[0]
        elif (offset - (1 + step)) % period == 0:
            out[r, sc0 : sc1 + 1] = below_colors[1]
            out[r, dc0 : dc1 + 1] = below_colors[1]

    return out, (sr0, sc0, sr1, sc1, dr0, dc0, dr1, dc1)


def derive_input(out: np.ndarray, bg: int, box: tuple[int, int, int, int, int, int, int, int], step: int) -> np.ndarray:
    sr0, sc0, sr1, sc1, dr0, dc0, dr1, dc1 = box
    keep = np.zeros(out.shape, dtype=bool)
    keep[sr0 : sr1 + 1, sc0 : sc1 + 1] = True
    keep[dr0 : dr1 + 1, dc0 : dc1 + 1] = True
    keep[sr0 : sr1 + 1, sc1 + 1] = True
    keep[sr0 : sr1 + 1, sc1 + 1 + step] = True
    keep[sr1 + 1, sc0 : sc1 + 1] = True
    keep[sr1 + 1 + step, sc0 : sc1 + 1] = True

    inp = np.full_like(out, int(bg))
    inp[keep] = out[keep]
    return inp


def gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    anchor_color, right1, right2, below1, below2 = palette[:5]

    ah = random.randint(1, 4)
    aw = random.randint(1, 5)
    step = random.randint(1, 3)
    n_h = random.randint(2, 5)
    n_v = random.randint(2, 5)
    period = 2 * step

    structure_h = ah * 2 + 1 + period * n_v + 2
    structure_w = aw * 2 + 1 + period * n_h + 2
    total_h = structure_h + random.randint(0, 4) + random.randint(0, 4)
    total_w = structure_w + random.randint(0, 4) + random.randint(0, 4)
    if total_h > 30 or total_w > 30:
        return None

    sr0 = random.randint(1, total_h - structure_h + 1)
    sc0 = random.randint(1, total_w - structure_w + 1)

    out, box = build_output(
        total_h,
        total_w,
        bg,
        anchor_color,
        (right1, right2),
        (below1, below2),
        sr0,
        sc0,
        ah,
        aw,
        step,
        n_h,
        n_v,
    )
    inp = derive_input(out, bg, box, step)

    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
