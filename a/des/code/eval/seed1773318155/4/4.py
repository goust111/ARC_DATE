import random
from typing import Iterable

import numpy as np


TYPE_LR = "lr"
TYPE_UD = "ud"
TYPE_ROT = "rot"
TYPE_ASYM = "asym"
DIRS = ("up", "down", "left", "right")
ALL_COLORS = tuple(range(10))
GEN_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 200
CONTENT_ATTEMPTS = 200
SIZE_CHOICES = (3, 4, 5)
ROT_SIZE_CHOICES = (3, 5)
OPTIONAL_EXTRA_TYPES = (TYPE_LR, TYPE_UD, TYPE_ROT, TYPE_ASYM)
_EXTRA_TYPE_INDEX = 0


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _symmetry_flags(content: np.ndarray) -> dict[str, bool]:
    return {
        TYPE_LR: bool(np.array_equal(content, np.fliplr(content))),
        TYPE_UD: bool(np.array_equal(content, np.flipud(content))),
        TYPE_ROT: bool(np.array_equal(content, np.rot90(content, 2))),
    }


def _classify_content(content: np.ndarray) -> str:
    active = [name for name, flag in _symmetry_flags(content).items() if flag]
    assert len(active) <= 1, "content symmetry must be unique"
    return active[0] if active else TYPE_ASYM


def _marker_offsets(direction: str, h: int, w: int) -> list[tuple[int, int]]:
    if direction == "up":
        return [(-2, dc) for dc in range(w)]
    if direction == "down":
        return [(h + 1, dc) for dc in range(w)]
    if direction == "left":
        return [(dr, -2) for dr in range(h)]
    if direction == "right":
        return [(dr, w + 1) for dr in range(h)]
    raise AssertionError(f"unknown direction: {direction}")


def _placements(content: np.ndarray, kind: str, direction: str, r: int, c: int) -> list[tuple[np.ndarray, int, int]]:
    h, w = map(int, content.shape)
    items = [(content, int(r), int(c))]
    if kind == TYPE_LR:
        for step in range(1, 4):
            if direction == "up":
                items.append((content, int(r - step * h), int(c)))
            elif direction == "down":
                items.append((content, int(r + step * h), int(c)))
            elif direction == "left":
                items.append((content, int(r), int(c - step * w)))
            else:
                items.append((content, int(r), int(c + step * w)))
        return items
    if kind == TYPE_UD:
        if direction in ("left", "right"):
            mirrored = np.fliplr(content)
            offset_r = int(r)
            offset_c = int(c - w) if direction == "left" else int(c + w)
        else:
            mirrored = np.flipud(content)
            offset_r = int(r - h) if direction == "up" else int(r + h)
            offset_c = int(c)
        items.append((mirrored, offset_r, offset_c))
        return items
    if kind == TYPE_ROT:
        rotated = np.rot90(content, -1)
        rh, rw = map(int, rotated.shape)
        if direction == "left":
            offset_r = int(r + (h - rh) // 2)
            offset_c = int(c - rw)
        elif direction == "right":
            offset_r = int(r + (h - rh) // 2)
            offset_c = int(c + w)
        elif direction == "up":
            offset_r = int(r - rh)
            offset_c = int(c + (w - rw) // 2)
        else:
            offset_r = int(r + h)
            offset_c = int(c + (w - rw) // 2)
        items.append((rotated, offset_r, offset_c))
        return items
    return items


def _paste_non_bg(out: np.ndarray, arr: np.ndarray, r: int, c: int, bg: int) -> None:
    h, w = map(int, arr.shape)
    assert 0 <= r and 0 <= c and r + h <= out.shape[0] and c + w <= out.shape[1]
    mask = arr != bg
    window = out[r : r + h, c : c + w]
    assert np.all(window[mask] == bg), "placements must not overlap"
    window[mask] = arr[mask]


def _side_hits(grid: np.ndarray, bg: int, r0: int, c0: int, r1: int, c1: int) -> dict[str, list[tuple[int, int]]]:
    h, w = grid.shape
    side_cells = {
        "up": [(r0 - 1, cc) for cc in range(c0, c1 + 1)],
        "down": [(r1 + 1, cc) for cc in range(c0, c1 + 1)],
        "left": [(rr, c0 - 1) for rr in range(r0, r1 + 1)],
        "right": [(rr, c1 + 1) for rr in range(r0, r1 + 1)],
    }
    hits: dict[str, list[tuple[int, int]]] = {}
    for side, coords in side_cells.items():
        found = []
        for rr, cc in coords:
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != bg:
                found.append((int(rr), int(cc)))
        hits[side] = found
    return hits


def _find_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sprites: list[dict] = []
    covered = np.zeros((h, w), dtype=bool)
    for oh in range(5, 8):
        for ow in range(5, 8):
            for r0 in range(0, h - oh + 1):
                r1 = int(r0 + oh - 1)
                for c0 in range(0, w - ow + 1):
                    c1 = int(c0 + ow - 1)
                    border_color = int(g[r0, c0])
                    if border_color == bg:
                        continue
                    ring = np.zeros((oh, ow), dtype=bool)
                    ring[0, :] = True
                    ring[-1, :] = True
                    ring[:, 0] = True
                    ring[:, -1] = True
                    patch = g[r0 : r1 + 1, c0 : c1 + 1]
                    if not np.all(patch[ring] == border_color):
                        continue
                    inner = patch[1:-1, 1:-1]
                    if inner.shape[0] < 3 or inner.shape[0] > 5 or inner.shape[1] < 3 or inner.shape[1] > 5:
                        continue
                    if np.any(inner == border_color):
                        continue
                    if not np.any(inner != bg):
                        continue
                    hits = _side_hits(g, bg, r0, c0, r1, c1)
                    non_empty_sides = {side: pts for side, pts in hits.items() if pts}
                    if len(non_empty_sides) != 1:
                        continue
                    direction, marker_cells = next(iter(non_empty_sides.items()))
                    if len(marker_cells) != 1:
                        continue
                    marker_r, marker_c = marker_cells[0]
                    marker_color = int(g[marker_r, marker_c])
                    if marker_color in {bg, border_color}:
                        continue
                    if marker_color in set(map(int, np.unique(inner))):
                        continue
                    ok = True
                    for rr in range(r0 - 1, r1 + 2):
                        for cc in range(c0 - 1, c1 + 2):
                            if not (0 <= rr < h and 0 <= cc < w):
                                continue
                            if r0 <= rr <= r1 and c0 <= cc <= c1:
                                continue
                            if (rr, cc) == (marker_r, marker_c):
                                continue
                            if int(g[rr, cc]) != bg:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue
                    sprite_cover = np.zeros((h, w), dtype=bool)
                    sprite_cover[r0 : r1 + 1, c0 : c1 + 1] = True
                    sprite_cover[marker_r, marker_c] = True
                    if np.any(covered & sprite_cover):
                        continue
                    covered |= sprite_cover
                    sprites.append(
                        {
                            "border_color": border_color,
                            "marker_color": marker_color,
                            "direction": direction,
                            "outer_bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "marker": (int(marker_r), int(marker_c)),
                            "content_r": int(r0 + 1),
                            "content_c": int(c0 + 1),
                            "content": inner.copy(),
                        }
                    )
    non_bg = g != bg
    assert sprites, "at least one framed sprite is required"
    assert np.all(covered[non_bg]), "all foreground cells must belong to detected sprites"
    sprites.sort(key=lambda item: (item["content_r"], item["content_c"]))
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect framed sprites with a one-pixel directional marker, classify the
    interior by symmetry, erase frame/marker, and grow the content.

    - left-right symmetric content: copy the content three more times in the
      marker direction.
    - up-down symmetric content: place one mirror copy in the marker direction.
    - 180-degree rotational symmetry: place one clockwise 90-degree rotated
      copy in the marker direction, center-aligned on the orthogonal axis.
    - asymmetric content: keep only the original content.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    sprites = _find_sprites(g, bg)

    out = np.full_like(g, bg)
    for sprite in sprites:
        content = sprite["content"]
        _paste_non_bg(out, content, sprite["content_r"], sprite["content_c"], bg)

    for sprite in sprites:
        content = sprite["content"]
        kind = _classify_content(content)
        placements = _placements(content, kind, sprite["direction"], sprite["content_r"], sprite["content_c"])
        for arr, r, c in placements[1:]:
            _paste_non_bg(out, arr, r, c, bg)
    return out


def _sample_content_palette(rng: random.Random, bg: int, size: int) -> list[int]:
    colors = [color for color in ALL_COLORS if color != bg]
    return rng.sample(colors, size)


def _random_array(rng: random.Random, shape: tuple[int, int], palette: list[int]) -> np.ndarray:
    h, w = shape
    return np.array([[palette[rng.randrange(len(palette))] for _ in range(w)] for _ in range(h)], dtype=int)


def _generate_content(kind: str, bg: int, rng: random.Random) -> np.ndarray:
    for _ in range(CONTENT_ATTEMPTS):
        if kind == TYPE_ROT:
            h = rng.choice(ROT_SIZE_CHOICES)
            w = rng.choice(ROT_SIZE_CHOICES)
        else:
            h = rng.choice(SIZE_CHOICES)
            w = rng.choice(SIZE_CHOICES)
        palette_size = rng.randint(2, 4)
        palette = _sample_content_palette(rng, bg, palette_size)
        if kind == TYPE_LR:
            arr = np.empty((h, w), dtype=int)
            for rr in range(h):
                for cc in range(w):
                    arr[rr, cc] = palette[rng.randrange(len(palette))] if cc <= w - 1 - cc else arr[rr, w - 1 - cc]
        elif kind == TYPE_UD:
            arr = np.empty((h, w), dtype=int)
            for rr in range(h):
                for cc in range(w):
                    arr[rr, cc] = palette[rng.randrange(len(palette))] if rr <= h - 1 - rr else arr[h - 1 - rr, cc]
        elif kind == TYPE_ROT:
            arr = np.empty((h, w), dtype=int)
            filled = np.zeros((h, w), dtype=bool)
            for rr in range(h):
                for cc in range(w):
                    if filled[rr, cc]:
                        continue
                    mate = (h - 1 - rr, w - 1 - cc)
                    color = palette[rng.randrange(len(palette))]
                    arr[rr, cc] = color
                    arr[mate] = color
                    filled[rr, cc] = True
                    filled[mate] = True
        else:
            arr = _random_array(rng, (h, w), palette)
        if len(np.unique(arr)) < 2:
            continue
        active = [name for name, flag in _symmetry_flags(arr).items() if flag]
        if len(active) <= 1 and (active[0] if active else TYPE_ASYM) == kind:
            return arr
    raise RuntimeError(f"failed to generate content for {kind}")


def _input_cells(content: np.ndarray, direction: str, marker_offset: tuple[int, int], r: int, c: int) -> set[tuple[int, int]]:
    h, w = map(int, content.shape)
    cells = {(rr, cc) for rr in range(r - 1, r + h + 1) for cc in range(c - 1, c + w + 1)}
    mr, mc = marker_offset
    cells.add((int(r + mr), int(c + mc)))
    return cells


def _output_cells(content: np.ndarray, kind: str, direction: str, r: int, c: int, bg: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for arr, rr, cc in _placements(content, kind, direction, r, c):
        ah, aw = map(int, arr.shape)
        for dr in range(ah):
            for dc in range(aw):
                if int(arr[dr, dc]) != bg:
                    cells.add((int(rr + dr), int(cc + dc)))
    return cells


def _pad_cells(cells: Iterable[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for rr, cc in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    padded.add((nr, nc))
    return padded


def _build_spec(kind: str, bg: int, rng: random.Random, shape: tuple[int, int], blocked: set[tuple[int, int]]) -> dict | None:
    h, w = shape
    content = _generate_content(kind, bg, rng)
    direction = rng.choice(DIRS)
    marker_offset = rng.choice(_marker_offsets(direction, int(content.shape[0]), int(content.shape[1])))
    offsets = _input_cells(content, direction, marker_offset, 0, 0) | _output_cells(content, kind, direction, 0, 0, bg)
    min_r = min(rr for rr, _ in offsets)
    max_r = max(rr for rr, _ in offsets)
    min_c = min(cc for _, cc in offsets)
    max_c = max(cc for _, cc in offsets)
    row_lo = int(-min_r)
    row_hi = int(h - 1 - max_r)
    col_lo = int(-min_c)
    col_hi = int(w - 1 - max_c)
    if row_lo > row_hi or col_lo > col_hi:
        return None
    content_colors = set(map(int, np.unique(content)))
    candidates = [color for color in ALL_COLORS if color not in content_colors and color != bg]
    if len(candidates) < 2:
        return None
    border_color, marker_color = rng.sample(candidates, 2)
    for _ in range(PLACEMENT_ATTEMPTS):
        r = rng.randint(row_lo, row_hi)
        c = rng.randint(col_lo, col_hi)
        interaction = _input_cells(content, direction, marker_offset, r, c) | _output_cells(content, kind, direction, r, c, bg)
        padded = _pad_cells(interaction, h, w)
        if padded & blocked:
            continue
        mr, mc = marker_offset
        return {
            "kind": kind,
            "direction": direction,
            "content": content,
            "content_r": int(r),
            "content_c": int(c),
            "border_color": int(border_color),
            "marker_color": int(marker_color),
            "marker": (int(r + mr), int(c + mc)),
            "blocked": padded,
        }
    return None


def _render_input(specs: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for spec in specs:
        content = spec["content"]
        r = spec["content_r"]
        c = spec["content_c"]
        h, w = map(int, content.shape)
        grid[r - 1 : r + h + 1, c - 1 : c + w + 1] = spec["border_color"]
        grid[r : r + h, c : c + w] = content
        mr, mc = spec["marker"]
        grid[mr, mc] = spec["marker_color"]
    return grid


def _render_output(specs: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for spec in specs:
        for arr, r, c in _placements(spec["content"], spec["kind"], spec["direction"], spec["content_r"], spec["content_c"]):
            _paste_non_bg(out, arr, r, c, bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    global _EXTRA_TYPE_INDEX
    rng = random.Random()
    h = rng.randint(26, 32)
    w = rng.randint(26, 32)
    bg = rng.randrange(10)
    types = [TYPE_LR, TYPE_UD, TYPE_ROT, TYPE_ASYM]
    if rng.random() < 0.5:
        extra = OPTIONAL_EXTRA_TYPES[_EXTRA_TYPE_INDEX % len(OPTIONAL_EXTRA_TYPES)]
        _EXTRA_TYPE_INDEX += 1
        types.append(extra)
    rng.shuffle(types)

    blocked: set[tuple[int, int]] = set()
    specs: list[dict] = []
    for kind in types:
        spec = None
        for _ in range(PLACEMENT_ATTEMPTS):
            spec = _build_spec(kind, bg, rng, (h, w), blocked)
            if spec is not None:
                break
        if spec is None:
            return None
        blocked |= spec["blocked"]
        specs.append(spec)

    grid = _render_input(specs, (h, w), bg)
    expected = _render_output(specs, (h, w), bg)
    if not np.array_equal(main(grid), expected):
        return None
    sprites = _find_sprites(grid, bg)
    if len(sprites) != len(specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
