from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8]
LEGEND_ENTRIES = 4
LEGEND_WIDTH = 3 * LEGEND_ENTRIES + 1
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _flood_from_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bool(mask[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bool(mask[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _find_legend(grid: np.ndarray) -> tuple[int, int, dict[int, int]]:
    h, w = grid.shape
    best: tuple[int, int, dict[int, int]] | None = None
    for r in range(h):
        row = grid[r]
        cols = np.flatnonzero(row != BLACK)
        if cols.size == 0:
            continue
        c0 = int(cols[0])
        c1 = int(cols[-1])
        segment = row[c0 : c1 + 1]
        if (len(segment) - 1) % 3 != 0:
            continue
        count = (len(segment) - 1) // 3
        if count < LEGEND_ENTRIES:
            continue
        mapping: dict[int, int] = {}
        ok = True
        for idx in range(count):
            if int(segment[3 * idx]) != GREY or int(segment[3 * idx + 2]) != MAROON:
                ok = False
                break
            color = int(segment[3 * idx + 1])
            if color in {BLACK, GREY, MAROON}:
                ok = False
                break
            mapping[idx + 1] = color
        if int(segment[-1]) != GREY:
            ok = False
        if ok and (best is None or r > best[0]):
            best = (int(r), int(c0), mapping)
    assert best is not None, "expected a legend row"
    return best


def _count_component_holes(scene: np.ndarray, coords: list[tuple[int, int]]) -> int:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    outside = _flood_from_boundary(~mask)
    holes = (~mask) & (~outside)
    return len(_components(holes))


def _outer_contour(scene: np.ndarray, coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = scene.shape
    exterior = _flood_from_boundary(scene == BLACK)
    contour: set[tuple[int, int]] = set()
    for r, c in coords:
        for nr, nc in _neighbors4(r, c, h, w):
            if bool(exterior[nr, nc]):
                contour.add((int(r), int(c)))
                break
        else:
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                contour.add((int(r), int(c)))
    return contour


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each holed monochrome sprite's outer contour using the legend entry indexed by hole count.

    Input structure:
    - The background is black.
    - The upper area contains several separated monochrome connected sprites; each sprite is a solid-looking shape with 1 to 4 internal single-pixel black holes.
    - The bottom legend row is split by grey separators into left-to-right cells; each cell contains a target color pixel followed by a maroon pixel.

    Rule:
    - Read the legend cells from left to right so that cell 1, 2, 3, 4 map hole counts to target contour colors.
    - For each sprite, count how many enclosed black hole components it contains.
    - Recolor only the sprite's outer contour pixels, namely sprite pixels that touch the exterior black background in 4-neighborhood, with the legend color for that hole count; keep inner pixels unchanged and keep holes black.

    Key insight:
    The sprite's original color is irrelevant; the real key is the number of enclosed holes, which indexes a visible lookup table that supplies the outline color.
    """
    g = np.asarray(grid, dtype=int)
    legend_row, _legend_start, mapping = _find_legend(g)
    scene = g[:legend_row, :]

    out = g.copy()
    comps = _components(scene != BLACK)
    for coords in comps:
        colors = {int(scene[r, c]) for r, c in coords}
        assert len(colors) == 1, "sprites must be monochrome"
        hole_count = _count_component_holes(scene, coords)
        assert hole_count in mapping, f"unsupported hole count {hole_count}"
        target_color = mapping[hole_count]
        for r, c in _outer_contour(scene, coords):
            out[r, c] = target_color
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components(mask)) == 1


def _hole_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    outside = _flood_from_boundary(~mask)
    return _components((~mask) & (~outside))


def _random_notch(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    for _ in range(20):
        trial = mask.copy()
        side = random.choice(["top", "bottom", "left", "right"])
        if side in {"top", "bottom"}:
            depth = random.randint(1, min(2, h - 3))
            width = random.randint(1, w - 2)
            c0 = random.randint(0, w - width)
            if side == "top":
                trial[:depth, c0 : c0 + width] = False
            else:
                trial[h - depth :, c0 : c0 + width] = False
        else:
            depth = random.randint(1, min(2, w - 3))
            height = random.randint(1, h - 2)
            r0 = random.randint(0, h - height)
            if side == "left":
                trial[r0 : r0 + height, :depth] = False
            else:
                trial[r0 : r0 + height, w - depth :] = False
        if int(trial.sum()) >= 12 and _mask_connected(trial):
            return trial
    return mask


def _interior_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not bool(mask[r, c]):
                continue
            if bool(mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]):
                cells.append((int(r), int(c)))
    random.shuffle(cells)
    return cells


def _random_transform(mask: np.ndarray) -> np.ndarray:
    k = random.randrange(4)
    out = np.rot90(mask, k)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _make_sprite_mask(hole_count: int) -> np.ndarray:
    for _ in range(150):
        lo = 6 if hole_count >= 3 else 5
        h = random.randint(lo, 7)
        w = random.randint(lo, 7)
        mask = np.ones((h, w), dtype=bool)
        for _ in range(random.randint(0, 4)):
            mask = _random_notch(mask)
        candidates = _interior_candidates(mask)
        if len(candidates) < hole_count:
            continue

        for _ in range(120):
            holes: list[tuple[int, int]] = []
            for cell in candidates:
                if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in holes):
                    holes.append(cell)
                    if len(holes) == hole_count:
                        break
            if len(holes) != hole_count:
                continue
            trial = mask.copy()
            for r, c in holes:
                trial[r, c] = False
            if not _mask_connected(trial):
                continue
            hole_comps = _hole_components(trial)
            if len(hole_comps) != hole_count:
                continue
            if any(len(comp) != 1 for comp in hole_comps):
                continue
            return _random_transform(trial)
    raise RuntimeError(f"failed to build sprite with {hole_count} holes")


def _row_plan(sprite_count: int) -> list[int]:
    if sprite_count == 4:
        plan = [2, 2]
    elif sprite_count == 5:
        plan = random.choice([[2, 3], [1, 2, 2]])
    elif sprite_count == 6:
        plan = random.choice([[2, 2, 2], [1, 2, 3]])
    else:
        raise AssertionError(f"unsupported sprite count {sprite_count}")
    random.shuffle(plan)
    return plan


def _choose_sprite_colors(counts: list[int], palette: dict[int, int]) -> list[int]:
    colors: list[int] = []
    unused = PALETTE_COLORS.copy()
    random.shuffle(unused)
    for hole_count in counts:
        choices = [c for c in PALETTE_COLORS if c != palette[hole_count]]
        preferred = [c for c in unused if c in choices]
        color = random.choice(preferred if preferred else choices)
        colors.append(color)
        if color in unused:
            unused.remove(color)
    return colors


def _build_legend(palette: dict[int, int]) -> np.ndarray:
    legend = np.full(LEGEND_WIDTH, GREY, dtype=int)
    for idx in range(LEGEND_ENTRIES):
        legend[3 * idx + 1] = palette[idx + 1]
        legend[3 * idx + 2] = MAROON
    return legend


def _expected_output(
    grid: np.ndarray,
    sprites: list[dict[str, object]],
    palette: dict[int, int],
    legend_row: int,
) -> np.ndarray:
    out = grid.copy()
    scene = grid[:legend_row, :]
    exterior = _flood_from_boundary(scene == BLACK)
    h, w = scene.shape
    for sprite in sprites:
        patch = sprite["patch"]
        r0 = int(sprite["r"])
        c0 = int(sprite["c"])
        hole_count = int(sprite["holes"])
        target = int(palette[hole_count])
        assert isinstance(patch, np.ndarray)
        for rr, cc in np.argwhere(patch != BLACK):
            r = r0 + int(rr)
            c = c0 + int(cc)
            if (
                r == 0
                or r == h - 1
                or c == 0
                or c == w - 1
                or any(bool(exterior[nr, nc]) for nr, nc in _neighbors4(r, c, h, w))
            ):
                out[r, c] = target
    return out


def _generate_attempt() -> np.ndarray | None:
    counts = [1, 2, 3, 4]
    extra = random.randint(0, 2)
    counts.extend(random.choices([1, 2, 3, 4], k=extra))
    random.shuffle(counts)

    palette_colors = random.sample(PALETTE_COLORS, LEGEND_ENTRIES)
    palette = {idx + 1: color for idx, color in enumerate(palette_colors)}
    sprite_colors = _choose_sprite_colors(counts, palette)

    sprites: list[dict[str, object]] = []
    for hole_count, color in zip(counts, sprite_colors, strict=True):
        mask = _make_sprite_mask(hole_count)
        patch = np.where(mask, int(color), BLACK).astype(int)
        sprites.append({"holes": int(hole_count), "color": int(color), "patch": patch})

    random.shuffle(sprites)
    row_sizes = _row_plan(len(sprites))
    rows: list[list[dict[str, object]]] = []
    start = 0
    for size in row_sizes:
        row = sprites[start : start + size]
        random.shuffle(row)
        rows.append(row)
        start += size

    row_widths: list[int] = []
    row_heights: list[int] = []
    row_gaps: list[list[int]] = []
    for row in rows:
        widths = [int(sprite["patch"].shape[1]) for sprite in row]  # type: ignore[index]
        heights = [int(sprite["patch"].shape[0]) for sprite in row]  # type: ignore[index]
        gaps = [random.randint(1, 3) for _ in range(max(0, len(row) - 1))]
        row_widths.append(sum(widths) + sum(gaps))
        row_heights.append(max(heights))
        row_gaps.append(gaps)

    base_width = max(LEGEND_WIDTH + 2, max(row_widths) + 2)
    if base_width > 30:
        return None
    scene_w = random.randint(base_width, min(30, base_width + 5))

    v_gaps = [random.randint(1, 2) for _ in range(max(0, len(rows) - 1))]
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    scene_h_min = top_margin + bottom_margin + sum(row_heights) + sum(v_gaps)
    if scene_h_min > 28:
        return None
    extra_h = random.randint(0, 28 - scene_h_min)
    slack = [top_margin, bottom_margin, *v_gaps]
    for _ in range(extra_h):
        slack[random.randrange(len(slack))] += 1
    top_margin = slack[0]
    bottom_margin = slack[1]
    v_gaps = slack[2:]
    scene_h = top_margin + bottom_margin + sum(row_heights) + sum(v_gaps)

    grid = np.full((scene_h + 2, scene_w), BLACK, dtype=int)
    current_r = top_margin
    placed: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        row_h = row_heights[row_index]
        row_w = row_widths[row_index]
        start_c = random.randint(1, scene_w - row_w - 1)
        cursor_c = start_c
        for item_index, sprite in enumerate(row):
            patch = sprite["patch"]
            assert isinstance(patch, np.ndarray)
            ph, pw = patch.shape
            offset_r = random.randint(0, row_h - ph)
            r0 = current_r + offset_r
            c0 = cursor_c
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            placed.append(
                {
                    "holes": int(sprite["holes"]),
                    "color": int(sprite["color"]),
                    "patch": patch,
                    "r": int(r0),
                    "c": int(c0),
                }
            )
            cursor_c += pw
            if item_index < len(row) - 1:
                cursor_c += row_gaps[row_index][item_index]
        current_r += row_h
        if row_index < len(rows) - 1:
            current_r += v_gaps[row_index]

    legend = _build_legend(palette)
    legend_start = random.randint(1, scene_w - LEGEND_WIDTH - 1)
    grid[-1, legend_start : legend_start + LEGEND_WIDTH] = legend

    expected = _expected_output(grid, placed, palette, legend_row=scene_h + 1)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if len(_components(grid[: scene_h + 1, :] != BLACK)) < len(placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
