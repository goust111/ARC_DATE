from __future__ import annotations

import numpy as np


CORNER_TYPES = ("tl", "tr", "bl", "br")
MISSING_TO_TARGET = {
    "tl": (1, 1),
    "tr": (1, -3),
    "bl": (-3, 1),
    "br": (-3, -3),
}
MISSING_CELLS = {
    "tl": (0, 0),
    "tr": (0, 2),
    "bl": (2, 0),
    "br": (2, 2),
}

# --------------- solver ---------------


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    comps: list[dict] = []

    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        mask = grid == color
        seen = np.zeros((h, w), dtype=bool)
        starts = np.argwhere(mask)
        for sr, sc in starts:
            sr = int(sr)
            sc = int(sc)
            if seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _component_submask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return sub


def _is_frame_component(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False

    sub = _component_submask(comp)
    expected = np.zeros_like(sub, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(sub, expected))


def _sprite_missing_corner(comp: dict) -> str | None:
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1) != 3 or (c1 - c0 + 1) != 3:
        return None
    if int(comp["area"]) != 8:
        return None

    sub = _component_submask(comp)
    missing = np.argwhere(~sub)
    if missing.shape[0] != 1:
        return None

    mr, mc = map(int, missing[0])
    for kind, cell in MISSING_CELLS.items():
        if (mr, mc) == cell:
            return kind
    return None


def _target_origin(frame_bbox: tuple[int, int, int, int], kind: str) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    dr, dc = MISSING_TO_TARGET[kind]
    base_r = r0 if dr > 0 else r1
    base_c = c0 if dc > 0 else c1
    return base_r + dr, base_c + dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each 3x3 sprite into the frame corner indicated by its missing corner,
    then repair the missing cell to form a solid 3x3 block.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    comps = _components_by_color(g, bg)

    frame_candidates = [comp for comp in comps if _is_frame_component(comp)]
    assert frame_candidates, "expected a rectangular frame"
    if len(frame_candidates) > 1:
        areas = [(comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1) for comp in frame_candidates]
        max_area = max(areas)
        frame_candidates = [comp for comp, area in zip(frame_candidates, areas) if area == max_area]
    assert len(frame_candidates) == 1, "frame must be uniquely identifiable"
    frame = frame_candidates[0]

    sprites: list[tuple[dict, str]] = []
    for comp in comps:
        if comp is frame:
            continue
        kind = _sprite_missing_corner(comp)
        if kind is not None:
            sprites.append((comp, kind))

    assert sprites, "expected at least one missing-corner sprite"
    kinds = [kind for _comp, kind in sprites]
    assert len(set(kinds)) == len(kinds), "each target corner must be used at most once"

    out = g.copy()
    for comp, _kind in sprites:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    frame_bbox = frame["bbox"]
    for comp, kind in sprites:
        rr, cc = _target_origin(frame_bbox, kind)
        out[rr : rr + 3, cc : cc + 3] = int(comp["color"])

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
FRAME_H_MIN = 8
FRAME_W_MIN = 8
FRAME_MARGIN = 1
SPRITE_SIZE = 3
SPRITE_GAP = 1
N_SPRITES_RANGE = (1, 5)
MAX_ATTEMPTS = 300


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sprite_patch(color: int, bg: int, kind: str) -> np.ndarray:
    patch = np.full((SPRITE_SIZE, SPRITE_SIZE), color, dtype=int)
    mr, mc = MISSING_CELLS[kind]
    patch[mr, mc] = bg
    return patch


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int
) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _sample_sprite_boxes(
    rng: np.random.Generator,
    height: int,
    width: int,
    frame_bbox: tuple[int, int, int, int],
    n_sprites: int,
) -> list[tuple[int, int, int, int]] | None:
    placed: list[tuple[int, int, int, int]] = []
    for _ in range(n_sprites):
        candidates: list[tuple[int, int, int, int]] = []
        for r0 in range(height - SPRITE_SIZE + 1):
            for c0 in range(width - SPRITE_SIZE + 1):
                box = (r0, c0, r0 + SPRITE_SIZE - 1, c0 + SPRITE_SIZE - 1)
                if _boxes_too_close(box, frame_bbox, SPRITE_GAP):
                    continue
                if any(_boxes_too_close(box, other, SPRITE_GAP) for other in placed):
                    continue
                candidates.append(box)
        if not candidates:
            return None
        choice = candidates[int(rng.integers(0, len(candidates)))]
        placed.append(choice)
    return placed


def _build_expected_output(
    height: int,
    width: int,
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
    sprite_specs: list[tuple[str, int]],
) -> np.ndarray:
    out = np.full((height, width), bg, dtype=int)
    _draw_frame(out, frame_bbox, frame_color)
    for kind, color in sprite_specs:
        rr, cc = _target_origin(frame_bbox, kind)
        out[rr : rr + SPRITE_SIZE, cc : cc + SPRITE_SIZE] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    height = int(rng.integers(*GRID_H_RANGE))
    width = int(rng.integers(*GRID_W_RANGE))
    if min(height, width) < FRAME_H_MIN + 2 * FRAME_MARGIN:
        return None

    n_sprites = int(rng.integers(*N_SPRITES_RANGE))
    kinds_idx = rng.choice(len(CORNER_TYPES), size=n_sprites, replace=False)
    kinds = [CORNER_TYPES[int(i)] for i in kinds_idx]

    palette = rng.choice(np.arange(10), size=2 + n_sprites, replace=False)
    bg = int(palette[0])
    frame_color = int(palette[1])
    sprite_colors = [int(c) for c in palette[2:]]

    max_frame_h = min(height - 2 * FRAME_MARGIN, height - 4)
    max_frame_w = min(width - 2 * FRAME_MARGIN, width - 4)
    if max_frame_h < FRAME_H_MIN or max_frame_w < FRAME_W_MIN:
        return None

    frame_h = int(rng.integers(FRAME_H_MIN, max_frame_h + 1))
    frame_w = int(rng.integers(FRAME_W_MIN, max_frame_w + 1))
    frame_r0 = int(rng.integers(FRAME_MARGIN, height - frame_h - FRAME_MARGIN + 1))
    frame_c0 = int(rng.integers(FRAME_MARGIN, width - frame_w - FRAME_MARGIN + 1))
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    boxes = _sample_sprite_boxes(rng, height, width, frame_bbox, n_sprites)
    if boxes is None:
        return None

    inp = np.full((height, width), bg, dtype=int)
    _draw_frame(inp, frame_bbox, frame_color)
    for box, kind, color in zip(boxes, kinds, sprite_colors):
        r0, c0, _r1, _c1 = box
        inp[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE] = _sprite_patch(color, bg, kind)

    out = _build_expected_output(
        height,
        width,
        bg,
        frame_color,
        frame_bbox,
        list(zip(kinds, sprite_colors)),
    )

    vals, counts = np.unique(inp, return_counts=True)
    if int(vals[counts.argmax()]) != bg:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
