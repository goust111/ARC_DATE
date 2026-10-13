from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, get_bg, paste, random_free_location


BG = 0
WALL = 8
MARK = 2
SPRITE_COLORS = (1, 3, 4, 6)
LAYOUTS = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2))
GEN_RETRIES = 400
SPRITE_RETRIES = 200


def _bboxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _wall_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere((grid == WALL) | (grid == MARK))
    assert coords.size > 0, "expected a template frame"
    return bbox_coords(coords)


def _extract_cavities(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict]]:
    frame_bbox = _wall_bbox(grid)
    frame = crop_bbox(grid, frame_bbox)
    fh, fw = frame.shape

    cavities: list[dict] = []
    for coords in components_mask(frame == BG, connectivity=4):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == fh - 1) or np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == fw - 1):
            continue
        lr0, lc0, lr1, lc1 = bbox_coords(coords)
        h = int(lr1 - lr0 + 1)
        w = int(lc1 - lc0 + 1)
        assert int(coords.shape[0]) == h * w, "cavity must be a solid rectangle"
        assert np.all(frame[lr0 : lr1 + 1, lc0 : lc1 + 1] == BG)
        assert lr0 > 0, "cavity must have a top wall segment"
        top_wall = frame[lr0 - 1, lc0 : lc1 + 1]
        cavities.append(
            {
                "bbox": (
                    int(frame_bbox[0] + lr0),
                    int(frame_bbox[1] + lc0),
                    int(frame_bbox[0] + lr1),
                    int(frame_bbox[1] + lc1),
                ),
                "size": (h, w),
                "marked": bool(np.any(top_wall == MARK)),
            }
        )

    assert cavities, "expected at least one cavity"
    cavities.sort(key=lambda item: item["bbox"])
    return frame_bbox, cavities


def _extract_sprites(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    bg = int(get_bg(grid))
    sprites: list[dict] = []

    for color in colors_present(grid, exclude={bg, WALL, MARK}):
        for coords in components_mask(grid == int(color), connectivity=4):
            bbox = bbox_coords(coords)
            assert not _bboxes_intersect(bbox, frame_bbox), "sprite must be outside the frame bbox"
            patch = crop_bbox(grid, bbox)
            local = patch == int(color)
            assert np.array_equal(patch[local], np.full(int(local.sum()), int(color)))
            sprites.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "size": tuple(map(int, patch.shape)),
                    "patch": patch,
                    "mask": local,
                    "area": int(local.sum()),
                }
            )

    assert sprites, "expected external sprites"
    sprites.sort(key=lambda item: item["bbox"])
    return sprites


def _compose_solution(grid: np.ndarray, *, force_flip: bool | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frame_bbox, cavities = _extract_cavities(g)
    sprites = _extract_sprites(g, frame_bbox)

    cavity_by_size = {tuple(item["size"]): item for item in cavities}
    sprite_by_size = {tuple(item["size"]): item for item in sprites}
    assert len(cavity_by_size) == len(cavities), "cavity sizes must be unique"
    assert len(sprite_by_size) == len(sprites), "sprite sizes must be unique"
    assert set(cavity_by_size) == set(sprite_by_size), "sprite/cavity sizes must match exactly"

    out = crop_bbox(g, frame_bbox)
    fr0, fc0, _fr1, _fc1 = frame_bbox
    for size, cavity in cavity_by_size.items():
        sprite = sprite_by_size[size]
        use_flip = cavity["marked"] if force_flip is None else bool(force_flip)
        patch = np.flipud(sprite["patch"]) if use_flip else sprite["patch"]
        mask = patch != bg
        cr0, cc0, _cr1, _cc1 = cavity["bbox"]
        paste(out, patch, cr0 - fr0, cc0 - fc0, mask=mask, overwrite=True)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outside sprite to the same-sized cavity, then optionally vertical-flip it according to a top-wall marker before inlaying it.

    Input structure:
    - Black background with one large teal rectangular frame whose teal inner walls split it into 2 to 4 black rectangular cavities.
    - Some cavity top wall segments contain a red marker pixel replacing teal; each cavity is checked independently.
    - Outside the frame bbox, there is one monochrome 4-connected sprite for each cavity, and each sprite bbox size uniquely matches one cavity size.

    Rule:
    - Find the frame bbox using teal and red wall pixels, then find every enclosed black rectangular cavity inside it.
    - Find every outside monochrome sprite using 4-connectivity, and pair sprites to cavities by identical bbox height and width.
    - For each cavity, inspect only the wall row immediately above that cavity: if it contains red, vertically mirror the matched sprite within its bbox; otherwise keep it unchanged.
    - Paste the possibly mirrored sprite into the cavity with top-left alignment, and return the tight crop of the whole frame bbox.

    Key insight:
    - Size matching identifies which sprite belongs to which cavity, but the hidden red pixel inside each cavity's own top wall decides that cavity's local flip state, so mirror decisions are not global.
    """
    return _compose_solution(grid)


def _build_template(rows: int, cols: int, row_heights: list[int], col_widths: list[int], marks: list[bool]) -> tuple[np.ndarray, list[dict]]:
    total_h = 1 + sum(row_heights) + rows
    total_w = 1 + sum(col_widths) + cols
    template = np.full((total_h, total_w), WALL, dtype=int)

    cavities: list[dict] = []
    idx = 0
    r = 1
    for h in row_heights:
        c = 1
        for w in col_widths:
            template[r : r + h, c : c + w] = BG
            marked = bool(marks[idx])
            if marked:
                marker_col = random.randint(c, c + w - 1)
                template[r - 1, marker_col] = MARK
            cavities.append(
                {
                    "bbox": (r, c, r + h - 1, c + w - 1),
                    "size": (h, w),
                    "marked": marked,
                }
            )
            idx += 1
            c += w + 1
        r += h + 1

    return template, cavities


def _connected(mask: np.ndarray) -> bool:
    return len(components_mask(mask, connectivity=4)) == 1


def _random_sprite_patch(h: int, w: int, color: int) -> np.ndarray:
    for _ in range(SPRITE_RETRIES):
        area = random.randint(max(3, (h * w + 1) // 2), h * w - 1)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = {start}
        while int(mask.sum()) < area:
            candidates = set()
            for r, c in frontier:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                        candidates.add((rr, cc))
            if not candidates:
                break
            cell = random.choice(tuple(candidates))
            mask[cell] = True
            frontier.add(cell)

        if int(mask.sum()) != area:
            continue
        if not _connected(mask):
            continue
        if bbox_coords(np.argwhere(mask)) != (0, 0, h - 1, w - 1):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.ones((h, w), dtype=bool)):
            continue

        patch = np.zeros((h, w), dtype=int)
        patch[mask] = int(color)
        return patch

    raise RuntimeError(f"failed to build a non-symmetric sprite for size {(h, w)}")


def _choose_distinct_sizes(rows: int, cols: int) -> tuple[list[int], list[int]]:
    for _ in range(200):
        row_heights = random.sample(range(2, 6), rows)
        col_widths = random.sample(range(2, 6), cols)
        sizes = [(h, w) for h in row_heights for w in col_widths]
        if len(set(sizes)) != len(sizes):
            continue
        if max(h for h, _w in sizes) < 2:
            continue
        return row_heights, col_widths
    raise RuntimeError("failed to choose unique cavity sizes")


def _choose_marks(count: int) -> list[bool]:
    while True:
        marks = [bool(random.getrandbits(1)) for _ in range(count)]
        if any(marks) and not all(marks):
            return marks


def _render_generated_scene(template: np.ndarray, sprites: list[np.ndarray]) -> np.ndarray:
    min_h = max(18, template.shape[0] + 6, max(sprite.shape[0] for sprite in sprites) + 2)
    min_w = max(18, template.shape[1] + 6, max(sprite.shape[1] for sprite in sprites) + 2)
    canvas_h = random.randint(min_h, 30)
    canvas_w = random.randint(min_w, 30)

    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    occupied = np.zeros_like(grid)

    frame_box = np.full(template.shape, 9, dtype=int)
    fr0, fc0 = random_free_location(occupied, frame_box, transparent=0, border=1, padding=2)
    paste(occupied, frame_box, fr0, fc0, transparent=0)
    paste(grid, template, fr0, fc0, transparent=BG)

    for patch in sprites:
        box = np.full(patch.shape, 9, dtype=int)
        sr0, sc0 = random_free_location(occupied, box, transparent=0, border=1, padding=1)
        paste(occupied, box, sr0, sc0, transparent=0)
        paste(grid, patch, sr0, sc0, transparent=BG)

    return grid


def _validate_generated(grid: np.ndarray) -> None:
    frame_bbox, cavities = _extract_cavities(grid)
    sprites = _extract_sprites(grid, frame_bbox)

    assert 2 <= len(cavities) <= 4
    assert len(cavities) == len(sprites)
    assert any(cavity["marked"] for cavity in cavities)
    assert any(not cavity["marked"] for cavity in cavities)
    assert len({tuple(cavity["size"]) for cavity in cavities}) == len(cavities)
    assert len({tuple(sprite["size"]) for sprite in sprites}) == len(sprites)
    assert len({int(sprite["color"]) for sprite in sprites}) == len(sprites)

    for sprite in sprites:
        h, w = sprite["size"]
        assert h >= 2
        assert not np.array_equal(sprite["patch"], np.flipud(sprite["patch"]))
        assert _connected(sprite["mask"])
        assert bbox_coords(np.argwhere(sprite["mask"])) == (0, 0, h - 1, w - 1)

    actual = main(grid)
    never_flip = _compose_solution(grid, force_flip=False)
    always_flip = _compose_solution(grid, force_flip=True)

    assert not np.array_equal(actual, never_flip), "mark detection must matter"
    assert not np.array_equal(actual, always_flip), "unmarked cavities must also matter"
    assert actual.shape == crop_bbox(grid, frame_bbox).shape


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rows, cols = random.choice(LAYOUTS)
        row_heights, col_widths = _choose_distinct_sizes(rows, cols)
        cavity_count = rows * cols
        marks = _choose_marks(cavity_count)
        template, cavities = _build_template(rows, cols, row_heights, col_widths, marks)

        colors = random.sample(SPRITE_COLORS, cavity_count)
        sprites = []
        for cavity, color in zip(cavities, colors):
            h, w = cavity["size"]
            sprites.append(_random_sprite_patch(h, w, color))

        try:
            grid = _render_generated_scene(template, sprites)
            _validate_generated(grid)
            return grid
        except (AssertionError, RuntimeError, ValueError):
            continue

    raise RuntimeError("failed to generate a valid puzzle input")
