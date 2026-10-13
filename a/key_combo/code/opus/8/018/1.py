import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

CORNER_TO_TRANSFORM = {
    (0, 0): "identity",
    (0, 1): "hflip",
    (1, 0): "vflip",
    (1, 1): "hvflip",
}

GEN_RETRIES = 300


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _apply_transform(mask: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return mask.copy()
    if transform == "hflip":
        return mask[:, ::-1].copy()
    if transform == "vflip":
        return mask[::-1, :].copy()
    if transform == "hvflip":
        return mask[::-1, ::-1].copy()
    raise ValueError(f"unknown transform {transform}")


def _all_transforms_unique(mask: np.ndarray) -> bool:
    variants = {
        _apply_transform(mask, "identity").tobytes(),
        _apply_transform(mask, "hflip").tobytes(),
        _apply_transform(mask, "vflip").tobytes(),
        _apply_transform(mask, "hvflip").tobytes(),
    }
    return len(variants) == 4


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    red_components = _find_components(g, RED)
    assert len(red_components) == 1, "expected exactly one red source sprite"
    sprite_box = red_components[0]["bbox"]
    sr0, sc0, sr1, sc1 = sprite_box
    sprite_mask = g[sr0 : sr1 + 1, sc0 : sc1 + 1] == RED
    assert 3 <= int(sprite_mask.sum()) <= 6, "sprite must use 3..6 red cells"
    assert not np.array_equal(sprite_mask, sprite_mask[:, ::-1]), "sprite must not be horizontally symmetric"
    assert not np.array_equal(sprite_mask, sprite_mask[::-1, :]), "sprite must not be vertically symmetric"
    assert _all_transforms_unique(sprite_mask), "the four reflections must be distinct"

    inner_h, inner_w = sprite_mask.shape
    grey_components = _find_components(g, GREY)
    assert 4 <= len(grey_components) <= 6, "expected 4..6 grey frames"

    frames = []
    all_blue = set(map(tuple, np.argwhere(g == BLUE)))
    used_blue = set()
    for component in grey_components:
        r0, c0, r1, c1 = component["bbox"]
        outer_h = r1 - r0 + 1
        outer_w = c1 - c0 + 1
        assert outer_h == inner_h + 2 and outer_w == inner_w + 2, "frame interior size must match sprite bbox"
        assert component["area"] == outer_h * outer_w - inner_h * inner_w, "grey component must be a 1-cell ring"

        frame = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(frame[0, :] == GREY)
        assert np.all(frame[-1, :] == GREY)
        assert np.all(frame[:, 0] == GREY)
        assert np.all(frame[:, -1] == GREY)
        inner = frame[1:-1, 1:-1]
        assert inner.shape == sprite_mask.shape
        assert not np.any(inner == GREY), "frame interior must not contain grey"

        blue_positions = np.argwhere(inner == BLUE)
        assert blue_positions.shape == (1, 2), "each frame must contain exactly one blue indicator"
        br, bc = map(int, blue_positions[0])
        corner_key = (0 if br == 0 else 1, 0 if bc == 0 else 1)
        assert (br, bc) in {
            (0, 0),
            (0, inner_w - 1),
            (inner_h - 1, 0),
            (inner_h - 1, inner_w - 1),
        }, "indicator must sit at an interior corner"

        used_blue.add((r0 + 1 + br, c0 + 1 + bc))
        frames.append(
            {
                "outer_bbox": (r0, c0, r1, c1),
                "inner_top_left": (r0 + 1, c0 + 1),
                "transform": CORNER_TO_TRANSFORM[corner_key],
                "corner_key": corner_key,
            }
        )

    assert used_blue == all_blue, "all blue pixels must be frame indicators"
    return sprite_mask, sprite_box, frames


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the red sprite into each grey frame according to the frame's blue corner indicator.

    Input structure:
    - The background is black.
    - One red 4-connected sprite sits in its own bounding box, and its bbox defines the stamp size.
    - Elsewhere, 4 to 6 grey rectangular rings have interiors exactly matching that bbox size, each containing one blue pixel at one interior corner.

    Rule:
    - Extract the red sprite mask from its bbox.
    - For each frame, read which interior corner contains the blue pixel: upper-left means no change, upper-right means horizontal flip, lower-left means vertical flip, and lower-right means both flips.
    - Stamp the transformed red mask into the frame interior aligned to the interior's upper-left cell, overwriting the blue pixel only when the transformed mask covers it.

    Key insight:
    - The blue corner pixel is an explicit orientation code for the reflection to apply before stamping the sprite into that frame.
    """
    g = np.asarray(grid, dtype=int)
    sprite_mask, _sprite_box, frames = _parse_input(g)

    out = g.copy()
    for frame in frames:
        transformed = _apply_transform(sprite_mask, frame["transform"])
        r0, c0 = frame["inner_top_left"]
        h, w = transformed.shape
        target = out[r0 : r0 + h, c0 : c0 + w]
        target[transformed] = RED
    return out


def _sample_sprite_mask(rng: random.Random) -> np.ndarray:
    dims = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3)]
    for _ in range(500):
        h, w = rng.choice(dims)
        max_area = min(6, h * w - 1)
        if max_area < 3:
            continue
        area = rng.randint(3, max_area)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in list(cells):
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if np.array_equal(mask, mask[:, ::-1]) or np.array_equal(mask, mask[::-1, :]):
            continue
        if not _all_transforms_unique(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a non-degenerate sprite")


def _rect_clear(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    return not blocked[r0 : r0 + h, c0 : c0 + w].any()


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int) -> None:
    hh, ww = blocked.shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(hh, r0 + h + margin)
    cc1 = min(ww, c0 + w + margin)
    blocked[rr0:rr1, cc0:cc1] = True


def _frame_indicator_position(corner_key: tuple[int, int], inner_h: int, inner_w: int) -> tuple[int, int]:
    return (0 if corner_key[0] == 0 else inner_h - 1, 0 if corner_key[1] == 0 else inner_w - 1)


def _render_expected(
    inp: np.ndarray,
    sprite_mask: np.ndarray,
    frame_specs: list[dict],
) -> np.ndarray:
    out = inp.copy()
    for spec in frame_specs:
        transformed = _apply_transform(sprite_mask, spec["transform"])
        r0, c0 = spec["inner_top_left"]
        h, w = transformed.shape
        out[r0 : r0 + h, c0 : c0 + w][transformed] = RED
    return out


def _validate_generated_case(
    inp: np.ndarray,
    expected: np.ndarray,
    sprite_mask: np.ndarray,
    frame_specs: list[dict],
) -> bool:
    if len(frame_specs) < 4:
        return False
    if {spec["corner_key"] for spec in frame_specs} != set(CORNER_TO_TRANSFORM):
        return False
    if not _all_transforms_unique(sprite_mask):
        return False

    try:
        parsed_mask, sprite_box, parsed_frames = _parse_input(inp)
    except AssertionError:
        return False

    if not np.array_equal(parsed_mask, sprite_mask):
        return False
    if len(parsed_frames) != len(frame_specs):
        return False

    out = main(inp)
    if not np.array_equal(out, expected):
        return False

    sr0, sc0, sr1, sc1 = sprite_box
    if not np.array_equal(out[sr0 : sr1 + 1, sc0 : sc1 + 1] == RED, sprite_mask):
        return False

    transformed_bytes = set()
    for spec in frame_specs:
        transformed = _apply_transform(sprite_mask, spec["transform"])
        transformed_bytes.add(transformed.tobytes())
        r0, c0 = spec["inner_top_left"]
        inner_before = inp[r0 : r0 + transformed.shape[0], c0 : c0 + transformed.shape[1]]
        inner_after = out[r0 : r0 + transformed.shape[0], c0 : c0 + transformed.shape[1]]
        if not np.all(inner_after[transformed] == RED):
            return False
        if not np.array_equal(inner_after[~transformed], inner_before[~transformed]):
            return False
    return len(transformed_bytes) == 4


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        sprite_mask = _sample_sprite_mask(rng)
        inner_h, inner_w = sprite_mask.shape
        outer_h, outer_w = inner_h + 2, inner_w + 2
        frame_count = rng.randint(4, 6)

        min_h = max(16, outer_h * 2 + 7)
        min_w = max(16, outer_w * 2 + 7)
        h = rng.randint(min_h, min(26, max(min_h, 22)))
        w = rng.randint(min_w, min(26, max(min_w, 22)))

        grid = np.full((h, w), BLACK, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)

        sprite_r = rng.randint(1, min(3, h - inner_h - 2))
        sprite_c = rng.randint(1, min(3, w - inner_w - 2))
        grid[sprite_r : sprite_r + inner_h, sprite_c : sprite_c + inner_w][sprite_mask] = RED
        _mark_blocked(blocked, sprite_r, sprite_c, inner_h, inner_w, margin=2)

        corner_keys = list(CORNER_TO_TRANSFORM)
        while len(corner_keys) < frame_count:
            corner_keys.append(rng.choice(list(CORNER_TO_TRANSFORM)))
        rng.shuffle(corner_keys)

        frame_specs = []
        placement_failed = False
        for corner_key in corner_keys:
            candidates = []
            for r0 in range(1, h - outer_h):
                for c0 in range(1, w - outer_w):
                    if _rect_clear(blocked, r0, c0, outer_h, outer_w):
                        candidates.append((r0, c0))
            if not candidates:
                placement_failed = True
                break
            r0, c0 = rng.choice(candidates)
            grid[r0 : r0 + outer_h, c0 : c0 + outer_w] = GREY
            grid[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = BLACK
            br, bc = _frame_indicator_position(corner_key, inner_h, inner_w)
            grid[r0 + 1 + br, c0 + 1 + bc] = BLUE
            _mark_blocked(blocked, r0, c0, outer_h, outer_w, margin=1)
            frame_specs.append(
                {
                    "corner_key": corner_key,
                    "transform": CORNER_TO_TRANSFORM[corner_key],
                    "inner_top_left": (r0 + 1, c0 + 1),
                }
            )
        if placement_failed:
            continue

        expected = _render_expected(grid, sprite_mask, frame_specs)
        if _validate_generated_case(grid, expected, sprite_mask, frame_specs):
            return grid
    raise RuntimeError("failed to generate a valid mirror-stamp ARC input")
