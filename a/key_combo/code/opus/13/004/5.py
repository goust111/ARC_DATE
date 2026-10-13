from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PLATFORM = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, colors: list[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for color in colors:
        for r in range(h):
            for c in range(w):
                if seen[r, c] or int(grid[r, c]) != color:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while q:
                    rr, cc = q.popleft()
                    coords.append((rr, cc))
                    for nr, nc in _neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                comps.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": _bbox_from_coords(coords),
                    }
                )
    return comps


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_mask(sprite: np.ndarray) -> np.ndarray:
    fg = sprite != BG
    h, w = fg.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not fg[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not fg[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not fg[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return (~fg) & (~exterior)


def _platform_slots(row: np.ndarray) -> list[tuple[int, int, int]]:
    row = np.asarray(row, dtype=int)
    assert set(map(int, row)) <= {BG, PLATFORM}
    runs: list[tuple[int, int]] = []
    c = 0
    w = int(row.shape[0])
    while c < w:
        assert int(row[c]) == PLATFORM
        c0 = c
        while c < w and int(row[c]) == PLATFORM:
            c += 1
        runs.append((c0, c - 1))
        if c >= w:
            break
        while c < w and int(row[c]) == BG:
            c += 1
    assert runs and runs[0][0] == 0 and runs[-1][1] == w - 1
    slots: list[tuple[int, int, int]] = []
    for (_, left_end), (right_start, _) in zip(runs, runs[1:]):
        start = left_end + 1
        end = right_start - 1
        assert start <= end
        slots.append((start, end, end - start + 1))
    return slots


def _orient_to_width(sprite: np.ndarray, holes: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    for k in range(4):
        rs = np.rot90(sprite, k=k)
        rh = np.rot90(holes, k=k)
        if int(rs.shape[1]) == int(width):
            return rs, rh
    raise AssertionError(f"no orientation matches slot width {width}")


def _paint_patch(dst: np.ndarray, patch: np.ndarray, holes: np.ndarray, r0: int, c0: int) -> None:
    placed = patch.copy()
    placed[holes] = PLATFORM
    mask = (patch != BG) | holes
    view = dst[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = placed[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """Match hollow objects to platform slots by enclosed area, rotate them into place, and fill their holes with platform color.

    Input structure:
    - Black is the background and the bottom row is a solid gray platform.
    - The row above the platform alternates gray separator runs and black gaps; the gaps are the target slots.
    - Above the shelf are several disconnected monochrome objects of distinct colors; some enclose black holes and some are solid distractors.

    Rule:
    - Find every 4-connected foreground object above the shelf and measure how many background cells it fully encloses.
    - Discard objects with zero enclosed area, crop the hollow ones to their bounding boxes, and match each hollow object to the unique slot whose width equals that enclosed area.
    - Rotate a cropped hollow object if needed so its width matches the slot width, bottom-align it into the slot, and recolor the object's enclosed hole cells to gray so they merge into the platform.

    Key insight:
    - The matching key is not the outer size of the object but the area of its enclosed hole, so topology determines both which objects survive and where they are placed.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 4 and w >= 5
    assert np.all(g[h - 1, :] == PLATFORM)
    shelf = h - 2
    slots = _platform_slots(g[shelf, :])

    colors = sorted(set(map(int, g.ravel())) - {BG, PLATFORM})
    comps = _find_components(g, colors)

    hollow_objects: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        assert r1 < shelf
        sprite = _crop_bbox(g, comp["bbox"])
        holes = _hole_mask(sprite)
        area = int(holes.sum())
        if area == 0:
            continue
        hollow_objects.append(
            {
                "color": int(comp["color"]),
                "sprite": sprite,
                "holes": holes,
                "hole_area": area,
            }
        )

    out = g.copy()
    out[(out != BG) & (out != PLATFORM)] = BG

    slot_by_width = {width: (start, end) for start, end, width in slots}
    obj_by_area = {int(obj["hole_area"]): obj for obj in hollow_objects}
    assert len(slot_by_width) == len(slots)
    assert len(obj_by_area) == len(hollow_objects)
    assert set(slot_by_width) == set(obj_by_area)

    for width, (start, end) in sorted(slot_by_width.items()):
        obj = obj_by_area[width]
        sprite, holes = _orient_to_width(obj["sprite"], obj["holes"], width)
        ph, pw = sprite.shape
        assert pw == width
        top = shelf - ph + 1
        left = start + (width - pw) // 2
        assert top >= 0
        _paint_patch(out, sprite, holes, top, left)

    return out


def _sprite_from_rows(rows: list[str], color: int) -> np.ndarray:
    arr = np.zeros((len(rows), len(rows[0])), dtype=int)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "1":
                arr[r, c] = color
    return arr


def _hollow_library(width: int, color: int) -> list[np.ndarray]:
    if width == 3:
        return [
            _sprite_from_rows(
                [
                    "111",
                    "101",
                    "101",
                    "101",
                    "111",
                ],
                color,
            )
        ]
    if width == 4:
        return [
            _sprite_from_rows(
                [
                    "1111",
                    "1001",
                    "1001",
                    "1111",
                ],
                color,
            )
        ]
    if width == 5:
        return [
            _sprite_from_rows(
                [
                    "11111",
                    "10001",
                    "10101",
                    "11111",
                ],
                color,
            ),
            _sprite_from_rows(
                [
                    "11111",
                    "10101",
                    "10001",
                    "11111",
                ],
                color,
            ),
            _sprite_from_rows(
                [
                    "11111",
                    "11001",
                    "10001",
                    "11111",
                ],
                color,
            ),
            _sprite_from_rows(
                [
                    "11111",
                    "10011",
                    "10001",
                    "11111",
                ],
                color,
            ),
        ]
    raise ValueError(f"unsupported hollow width {width}")


def _solid_blob(color: int) -> np.ndarray:
    for _ in range(200):
        hh, ww = random.choice(
            [(2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)]
        )
        target = random.randint(6, min(12, hh * ww))
        cells = {(random.randrange(hh), random.randrange(ww))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < hh and 0 <= nc < ww and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        arr = np.zeros((hh, ww), dtype=int)
        for r, c in cells:
            arr[r, c] = color
        holes = _hole_mask(arr)
        if np.count_nonzero(arr) >= 6 and not holes.any():
            return arr
    raise RuntimeError("failed to build solid distractor")


def _place_patch(canvas: np.ndarray, patch: np.ndarray, blocked: np.ndarray, max_row: int) -> bool:
    ph, pw = patch.shape
    candidates = []
    for r in range(0, max_row - ph + 1):
        for c in range(0, canvas.shape[1] - pw + 1):
            if blocked[r : r + ph, c : c + pw].any():
                continue
            candidates.append((r, c))
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    mask = patch != BG
    view = canvas[r0 : r0 + ph, c0 : c0 + pw]
    view[mask] = patch[mask]
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(canvas.shape[0], r0 + ph + 1)
    cc1 = min(canvas.shape[1], c0 + pw + 1)
    blocked[rr0:rr1, cc0:cc1] = True
    return True


def _build_platform(widths: list[int], separator_widths: list[int]) -> tuple[np.ndarray, int]:
    total_w = sum(widths) + sum(separator_widths)
    height = random.randint(14, 18)
    grid = np.zeros((height, total_w), dtype=int)
    shelf = height - 2
    grid[height - 1, :] = PLATFORM
    c = 0
    for i, width in enumerate(widths):
        sw = separator_widths[i]
        grid[shelf, c : c + sw] = PLATFORM
        c += sw
        c += width
    grid[shelf, c : c + separator_widths[-1]] = PLATFORM
    return grid, shelf


def _render_output(base: np.ndarray, shelf: int, widths: list[int], hollow_sprites: list[np.ndarray]) -> np.ndarray:
    out = base.copy()
    slots = _platform_slots(out[shelf, :])
    slot_by_width = {width: (start, end) for start, end, width in slots}
    for width, sprite in zip(widths, hollow_sprites):
        holes = _hole_mask(sprite)
        oriented, oriented_holes = _orient_to_width(sprite, holes, width)
        start, end = slot_by_width[width]
        top = shelf - oriented.shape[0] + 1
        left = start + (width - oriented.shape[1]) // 2
        _paint_patch(out, oriented, oriented_holes, top, left)
    return out


def _derive_input(base: np.ndarray, shelf: int, sources: list[np.ndarray]) -> np.ndarray | None:
    inp = base.copy()
    blocked = np.zeros_like(inp, dtype=bool)
    blocked[shelf:, :] = True
    order = list(range(len(sources)))
    random.shuffle(order)
    order.sort(key=lambda idx: sources[idx].shape[0] * sources[idx].shape[1], reverse=True)
    for idx in order:
        if not _place_patch(inp, sources[idx], blocked, shelf - 1):
            return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    slot_count = random.choice((2, 3))
    widths = sorted(random.sample((3, 4, 5), slot_count))
    total_objects = random.randint(max(4, slot_count + 1), 6)
    solid_count = total_objects - slot_count
    colors = random.sample(PALETTE, total_objects)
    hollow_colors = colors[:slot_count]
    solid_colors = colors[slot_count:]

    separator_widths = [random.randint(1, 3) for _ in range(slot_count + 1)]
    base, shelf = _build_platform(widths, separator_widths)

    hollow_outputs: list[np.ndarray] = []
    for width, color in zip(widths, hollow_colors):
        hollow_outputs.append(random.choice(_hollow_library(width, color)))

    output = _render_output(base, shelf, widths, hollow_outputs)

    sources: list[np.ndarray] = []
    nonsquare_indices: list[int] = []
    for i, (width, sprite) in enumerate(zip(widths, hollow_outputs)):
        source = sprite.copy()
        if sprite.shape[0] != sprite.shape[1]:
            nonsquare_indices.append(i)
            if random.random() < 0.5:
                source = np.rot90(source)
        sources.append(source)
    if nonsquare_indices and all(sources[i].shape == hollow_outputs[i].shape for i in nonsquare_indices):
        picked = random.choice(nonsquare_indices)
        sources[picked] = np.rot90(sources[picked])

    for color in solid_colors:
        sources.append(_solid_blob(color))

    inp = _derive_input(base, shelf, sources)
    if inp is None:
        return None

    comps = _find_components(inp, sorted(set(map(int, inp.ravel())) - {BG, PLATFORM}))
    if not (4 <= len(comps) <= 6):
        return None
    hollow_areas = sorted(
        int(_hole_mask(_crop_bbox(inp, comp["bbox"])).sum())
        for comp in comps
        if int(_hole_mask(_crop_bbox(inp, comp["bbox"])).sum()) > 0
    )
    if hollow_areas != sorted(widths):
        return None
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed")
