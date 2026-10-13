from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_arrow(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    norm = {(r - r0, c - c0) for r, c in coords}

    for shaft_len in range(3, 6):
        if h == 3 and w == shaft_len + 1:
            right = {(1, c) for c in range(shaft_len + 1)} | {(0, shaft_len), (2, shaft_len)}
            left = {(1, c) for c in range(shaft_len + 1)} | {(0, 0), (2, 0)}
            if norm == right:
                tip = (r0 + 1, c1)
                return {"dir": (0, 1), "tip": tip, "shaft_len": shaft_len}
            if norm == left:
                tip = (r0 + 1, c0)
                return {"dir": (0, -1), "tip": tip, "shaft_len": shaft_len}
        if w == 3 and h == shaft_len + 1:
            down = {(r, 1) for r in range(shaft_len + 1)} | {(shaft_len, 0), (shaft_len, 2)}
            up = {(r, 1) for r in range(shaft_len + 1)} | {(0, 0), (0, 2)}
            if norm == down:
                tip = (r1, c0 + 1)
                return {"dir": (1, 0), "tip": tip, "shaft_len": shaft_len}
            if norm == up:
                tip = (r0, c0 + 1)
                return {"dir": (-1, 0), "tip": tip, "shaft_len": shaft_len}
    return None


def _analyze(grid: np.ndarray) -> tuple[int, list[dict[str, object]], list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components4(g != bg)

    sprites: list[dict[str, object]] = []
    arrows: list[dict[str, object]] = []
    for coords in comps:
        colors = {int(g[r, c]) for r, c in coords}
        if len(colors) >= 2:
            r0, c0, r1, c1 = _bbox(coords)
            sprites.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
                }
            )
        else:
            arrow = _parse_arrow(coords)
            assert arrow is not None, "monochrome component must be an arrow"
            arrows.append(arrow)

    assert 2 <= len(sprites) <= 4
    assert arrows
    sprites.sort(key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))

    cell_to_sprite: dict[tuple[int, int], int] = {}
    for sid, sprite in enumerate(sprites):
        for cell in sprite["coords"]:
            cell_to_sprite[cell] = sid

    counts = [0] * len(sprites)
    h, w = g.shape
    for arrow in arrows:
        dr, dc = arrow["dir"]
        r, c = arrow["tip"]
        r += dr
        c += dc
        target = None
        while 0 <= r < h and 0 <= c < w:
            sid = cell_to_sprite.get((r, c))
            if sid is not None:
                target = sid
                break
            r += dr
            c += dc
        assert target is not None, "every arrow must point to a sprite"
        counts[target] += 1

    for count in counts:
        assert 1 <= count <= 3
    return bg, sprites, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count how many arrows point to each multi-color sprite, scale each sprite by that
    count, and pack the scaled sprites left-to-right with a one-column gap.
    """
    bg, sprites, counts = _analyze(np.asarray(grid, dtype=int))

    scaled = [
        np.repeat(np.repeat(np.asarray(sprite["patch"], dtype=int), count, axis=0), count, axis=1)
        for sprite, count in zip(sprites, counts)
    ]
    out_h = max(patch.shape[0] for patch in scaled)
    out_w = sum(patch.shape[1] for patch in scaled) + len(scaled) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for i, patch in enumerate(scaled):
        h, w = patch.shape
        out[:h, col : col + w] = patch
        col += w
        if i != len(scaled) - 1:
            col += 1
    return out


BG = 0
SPRITE_PALETTE = [1, 2, 3, 4, 5, 6]
ARROW_PALETTE = [7, 8, 9]
SIDE_NAMES = ("left", "right", "above", "below")
MAX_GENERATION_ATTEMPTS = 300


def _make_patch(h: int, w: int) -> np.ndarray:
    modes = ["rows", "cols", "checker", "diag", "blocks", "random"]
    for _ in range(100):
        color_count = random.randint(2, 3)
        colors = random.sample(SPRITE_PALETTE, color_count)
        patch = np.zeros((h, w), dtype=int)
        mode = random.choice(modes)

        if mode == "rows":
            for r in range(h):
                patch[r, :] = colors[r % color_count]
        elif mode == "cols":
            for c in range(w):
                patch[:, c] = colors[c % color_count]
        elif mode == "checker":
            for r in range(h):
                for c in range(w):
                    patch[r, c] = colors[(r + c) % color_count]
        elif mode == "diag":
            for r in range(h):
                for c in range(w):
                    patch[r, c] = colors[(r + 2 * c) % color_count]
        elif mode == "blocks":
            split_r = random.randint(1, h - 1)
            split_c = random.randint(1, w - 1)
            patch[:, :] = colors[0]
            patch[:split_r, :split_c] = colors[1]
            if color_count == 3:
                patch[split_r:, split_c:] = colors[2]
            else:
                patch[split_r:, split_c:] = colors[1]
        else:
            patch[:, :] = random.choice(colors)
            for r in range(h):
                for c in range(w):
                    patch[r, c] = random.choice(colors)

        if len(set(int(v) for v in patch.ravel())) == color_count:
            return patch
    raise RuntimeError("failed to build sprite patch")


def _arrow_cells(side: str, sprite_bbox: tuple[int, int, int, int], row_or_col: int, shaft_len: int, gap: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = sprite_bbox
    if side == "left":
        row = row_or_col
        tip_c = c0 - gap - 1
        return [(row, tip_c - d) for d in range(shaft_len + 1)] + [(row - 1, tip_c), (row + 1, tip_c)]
    if side == "right":
        row = row_or_col
        tip_c = c1 + gap + 1
        return [(row, tip_c + d) for d in range(shaft_len + 1)] + [(row - 1, tip_c), (row + 1, tip_c)]
    if side == "above":
        col = row_or_col
        tip_r = r0 - gap - 1
        return [(tip_r - d, col) for d in range(shaft_len + 1)] + [(tip_r, col - 1), (tip_r, col + 1)]
    col = row_or_col
    tip_r = r1 + gap + 1
    return [(tip_r + d, col) for d in range(shaft_len + 1)] + [(tip_r, col - 1), (tip_r, col + 1)]


def _render_expected(patches: list[np.ndarray], counts: list[int]) -> np.ndarray:
    scaled = [np.repeat(np.repeat(patch, count, axis=0), count, axis=1) for patch, count in zip(patches, counts)]
    out_h = max(patch.shape[0] for patch in scaled)
    out_w = sum(patch.shape[1] for patch in scaled) + len(scaled) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for i, patch in enumerate(scaled):
        h, w = patch.shape
        out[:h, col : col + w] = patch
        col += w
        if i != len(scaled) - 1:
            col += 1
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    sprite_count = random.randint(2, 4)
    sprite_specs: list[dict[str, object]] = []
    signatures: set[tuple[tuple[int, int], tuple[int, ...]]] = set()

    for _ in range(sprite_count):
        for _retry in range(100):
            h = random.randint(2, 4)
            w = random.randint(2, 4)
            patch = _make_patch(h, w)
            sig = (patch.shape, tuple(int(v) for v in patch.ravel()))
            if sig in signatures:
                continue
            signatures.add(sig)
            count = random.randint(1, 3)
            sides = random.sample(list(SIDE_NAMES), count)
            arrows: dict[str, dict[str, int]] = {}
            for side in sides:
                arrows[side] = {
                    "shaft_len": random.randint(3, 5),
                    "gap": random.randint(1, 3),
                    "color": random.choice(ARROW_PALETTE),
                }
            pad_left = 1 + (arrows["left"]["shaft_len"] + arrows["left"]["gap"] + 1 if "left" in arrows else 0)
            pad_right = 1 + (arrows["right"]["shaft_len"] + arrows["right"]["gap"] + 1 if "right" in arrows else 0)
            pad_top = 1 + (arrows["above"]["shaft_len"] + arrows["above"]["gap"] + 1 if "above" in arrows else 0)
            pad_bottom = 1 + (arrows["below"]["shaft_len"] + arrows["below"]["gap"] + 1 if "below" in arrows else 0)
            sprite_specs.append(
                {
                    "patch": patch,
                    "count": count,
                    "arrows": arrows,
                    "pad_left": pad_left,
                    "pad_right": pad_right,
                    "pad_top": pad_top,
                    "pad_bottom": pad_bottom,
                    "reserved_w": pad_left + w + pad_right,
                    "reserved_h": pad_top + h + pad_bottom,
                }
            )
            break
        else:
            return None

    x = random.randint(0, 2)
    for spec in sprite_specs:
        spec["reserved_x"] = x
        spec["reserved_y"] = random.randint(0, 6)
        patch = np.asarray(spec["patch"], dtype=int)
        spec["sprite_r0"] = int(spec["reserved_y"]) + int(spec["pad_top"])
        spec["sprite_c0"] = int(spec["reserved_x"]) + int(spec["pad_left"])
        spec["sprite_r1"] = int(spec["sprite_r0"]) + patch.shape[0] - 1
        spec["sprite_c1"] = int(spec["sprite_c0"]) + patch.shape[1] - 1
        x += int(spec["reserved_w"]) + random.randint(2, 5)

    height = max(int(spec["reserved_y"]) + int(spec["reserved_h"]) for spec in sprite_specs) + random.randint(0, 3)
    width = x + random.randint(0, 3)
    grid = np.full((height, width), BG, dtype=int)

    for spec in sprite_specs:
        patch = np.asarray(spec["patch"], dtype=int)
        r0 = int(spec["sprite_r0"])
        c0 = int(spec["sprite_c0"])
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w] = patch

    for spec in sprite_specs:
        r0 = int(spec["sprite_r0"])
        c0 = int(spec["sprite_c0"])
        patch = np.asarray(spec["patch"], dtype=int)
        h, w = patch.shape
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
        arrows = spec["arrows"]
        for side, arrow_spec in arrows.items():
            if side in ("left", "right"):
                row = random.randint(bbox[0], bbox[2])
                cells = _arrow_cells(side, bbox, row, int(arrow_spec["shaft_len"]), int(arrow_spec["gap"]))
            else:
                col = random.randint(bbox[1], bbox[3])
                cells = _arrow_cells(side, bbox, col, int(arrow_spec["shaft_len"]), int(arrow_spec["gap"]))
            for rr, cc in cells:
                if not (0 <= rr < height and 0 <= cc < width):
                    return None
                if grid[rr, cc] != BG:
                    return None
            for rr, cc in cells:
                grid[rr, cc] = int(arrow_spec["color"])

    expected = _render_expected([np.asarray(spec["patch"], dtype=int) for spec in sprite_specs], [int(spec["count"]) for spec in sprite_specs])

    try:
        actual = main(grid)
        _bg, _sprites, counts = _analyze(grid)
    except AssertionError:
        return None

    if counts != [int(spec["count"]) for spec in sprite_specs]:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid ARC input")
