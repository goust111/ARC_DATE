import numpy as np


BACKGROUND = 0
GREY = 5
CONTROL_COLORS = (1, 2, 3, 4, 7)
SPRITE_PATCHES = (
    np.array([[GREY, GREY], [GREY, GREY]], dtype=int),
    np.array([[BACKGROUND, GREY], [GREY, GREY]], dtype=int),
    np.array([[GREY, BACKGROUND], [GREY, GREY]], dtype=int),
    np.array([[GREY, GREY], [BACKGROUND, GREY]], dtype=int),
    np.array([[GREY, GREY], [GREY, BACKGROUND]], dtype=int),
)
MAX_GENERATION_ATTEMPTS = 600


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sprite_descriptors(grid: np.ndarray) -> tuple[dict[int, int], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 4 and width >= 4, "grid is too small"
    assert np.all((g[1:, :] == BACKGROUND) | (g[1:, :] == GREY)), "body must contain only background and grey sprites"

    marker_by_col = {int(c): int(g[0, c]) for c in range(width) if int(g[0, c]) not in (BACKGROUND, GREY)}
    assert 3 <= len(marker_by_col) <= 5, "expected 3-5 control pixels in the top row"

    sprites = []
    for coords in _components(g == GREY):
        r0, c0, r1, c1 = _bbox_coords(coords)
        assert r1 - r0 == 1 and c1 - c0 == 1, "each sprite must occupy a 2x2 bounding box"
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        mask = patch == GREY
        assert int(mask.sum()) in (3, 4), "each sprite must contain 3 or 4 grey pixels"
        assert np.all((patch == BACKGROUND) | (patch == GREY)), "sprite patch must contain only grey and background"

        mirror_c0 = width - patch.shape[1] - c0
        mirror_c1 = mirror_c0 + patch.shape[1] - 1
        assert c1 < mirror_c0, "original and mirrored sprite positions must not overlap"

        left_hits = [color for col, color in marker_by_col.items() if c0 <= col <= c1]
        right_hits = [color for col, color in marker_by_col.items() if mirror_c0 <= col <= mirror_c1]
        assert len(left_hits) == 1, "original sprite span must contain exactly one control pixel"
        assert len(right_hits) == 1, "mirrored sprite span must contain exactly one control pixel"

        sprites.append(
            {
                "r0": r0,
                "c0": c0,
                "mirror_c0": int(mirror_c0),
                "mask": mask,
                "left_color": int(left_hits[0]),
                "right_color": int(right_hits[0]),
            }
        )

    sprites.sort(key=lambda item: (int(item["r0"]), int(item["c0"])))
    assert 2 <= len(sprites) <= 4, "expected 2-4 grey sprites"
    return marker_by_col, sprites


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _marker_by_col, sprites = _sprite_descriptors(g)
    out = np.full_like(g, BACKGROUND)
    out[0, :] = g[0, :]

    for sprite in sprites:
        r0 = int(sprite["r0"])
        c0 = int(sprite["c0"])
        mirror_c0 = int(sprite["mirror_c0"])
        mask = np.asarray(sprite["mask"], dtype=bool)
        out[r0 : r0 + 2, c0 : c0 + 2][mask] = int(sprite["left_color"])
        out[r0 : r0 + 2, mirror_c0 : mirror_c0 + 2][mask] = int(sprite["right_color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each grey 2x2 sprite to its horizontal mirror position, then recolor the original and copy by top-row column markers.

    Input structure:
    - Black background with 3-5 colored singleton control pixels in the top row.
    - Rows below contain 2-4 grey 2x2 sprites on the left side; each sprite may have one background cell inside its 2x2 box.
    - For every sprite, both its original 2-column span and its mirrored 2-column span contain exactly one top-row control pixel.

    Rule:
    - Keep the top-row control pixels and the black background unchanged.
    - For each grey sprite, compute the mirrored box start column as `new_c0 = grid_width - sprite_width - old_c0` and place an unflipped copy there while keeping the original.
    - Recolor the grey pixels in the original box with the unique control-pixel color whose top-row column lies inside that box's column span, and recolor the mirrored copy the same way using its own mirrored column span.

    Key insight:
    - The mirrored copy keeps the same local shape, but its new horizontal position changes which top-row control pixel covers it, so position determines color.
    """
    return _expected_output(grid)


def _valid_column_records(width: int, marker_cols: list[int]) -> list[dict[str, int]]:
    records = []
    max_c0 = (width - 4) // 2
    for c0 in range(max_c0 + 1):
        left_hits = [c for c in marker_cols if c0 <= c <= c0 + 1]
        mirror_c0 = width - 2 - c0
        right_hits = [c for c in marker_cols if mirror_c0 <= c <= mirror_c0 + 1]
        if len(left_hits) == 1 and len(right_hits) == 1:
            records.append(
                {
                    "c0": int(c0),
                    "mirror_c0": int(mirror_c0),
                    "left_marker_col": int(left_hits[0]),
                    "right_marker_col": int(right_hits[0]),
                }
            )
    return records


def _has_color_diversity(chosen: list[dict[str, int]], marker_count: int) -> bool:
    left_cols = {item["left_marker_col"] for item in chosen}
    right_cols = {item["right_marker_col"] for item in chosen}
    if len(left_cols) >= 2 and len(right_cols) >= 2:
        return True
    if marker_count == 3 and len(left_cols | right_cols) >= 3 and (len(left_cols) >= 2 or len(right_cols) >= 2):
        return True
    return False


def _sample_layout(rng: np.random.Generator, height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    row_slots = list(range(2, height - 1, 2))
    if len(row_slots) < 2:
        return None

    marker_count = int(rng.integers(3, min(5, width) + 1))
    marker_cols = sorted(int(x) for x in rng.choice(width, size=marker_count, replace=False))
    marker_colors = [int(x) for x in rng.choice(CONTROL_COLORS, size=marker_count, replace=False)]
    marker_by_col = dict(zip(marker_cols, marker_colors))

    records = _valid_column_records(width, marker_cols)
    max_sprites = min(4, len(records), len(row_slots))
    if max_sprites < 2:
        return None

    sprite_count = int(rng.integers(2, max_sprites + 1))
    for _ in range(80):
        chosen_idx = rng.choice(len(records), size=sprite_count, replace=False)
        chosen = [records[int(i)] for i in chosen_idx]
        if not _has_color_diversity(chosen, marker_count):
            continue

        rows = sorted(int(x) for x in rng.choice(row_slots, size=sprite_count, replace=False))
        rng.shuffle(chosen)

        patch_choices = [SPRITE_PATCHES[0].copy(), SPRITE_PATCHES[int(rng.integers(1, len(SPRITE_PATCHES)))].copy()]
        while len(patch_choices) < sprite_count:
            patch_choices.append(SPRITE_PATCHES[int(rng.integers(0, len(SPRITE_PATCHES)))].copy())
        rng.shuffle(patch_choices)

        inp = np.full((height, width), BACKGROUND, dtype=int)
        for col, color in marker_by_col.items():
            inp[0, col] = int(color)

        expected = np.full_like(inp, BACKGROUND)
        expected[0, :] = inp[0, :]

        for r0, record, patch in zip(rows, chosen, patch_choices):
            c0 = int(record["c0"])
            mirror_c0 = int(record["mirror_c0"])
            mask = patch == GREY
            inp[r0 : r0 + 2, c0 : c0 + 2] = patch
            expected[r0 : r0 + 2, c0 : c0 + 2][mask] = marker_by_col[record["left_marker_col"]]
            expected[r0 : r0 + 2, mirror_c0 : mirror_c0 + 2][mask] = marker_by_col[record["right_marker_col"]]

        if not _validate_example(inp, expected):
            continue
        return inp, expected

    return None


def _validate_example(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        marker_by_col, sprites = _sprite_descriptors(inp)
        if not np.array_equal(expected, _expected_output(inp)):
            return False
        if not np.array_equal(main(inp), expected):
            return False
        if not np.array_equal(inp[0], expected[0]):
            return False
        if not np.all(inp[1] == BACKGROUND):
            return False

        left_colors = set()
        right_colors = set()
        areas = set()
        rebuilt = np.full_like(inp, BACKGROUND)
        rebuilt[0, :] = inp[0, :]

        for sprite in sprites:
            r0 = int(sprite["r0"])
            c0 = int(sprite["c0"])
            mirror_c0 = int(sprite["mirror_c0"])
            mask = np.asarray(sprite["mask"], dtype=bool)
            left_color = int(sprite["left_color"])
            right_color = int(sprite["right_color"])

            if np.any(inp[r0 : r0 + 2, mirror_c0 : mirror_c0 + 2] != BACKGROUND):
                return False
            if not np.all(expected[r0 : r0 + 2, c0 : c0 + 2][mask] == left_color):
                return False
            if not np.all(expected[r0 : r0 + 2, mirror_c0 : mirror_c0 + 2][mask] == right_color):
                return False
            if not np.all(expected[r0 : r0 + 2, c0 : c0 + 2][~mask] == BACKGROUND):
                return False
            if not np.all(expected[r0 : r0 + 2, mirror_c0 : mirror_c0 + 2][~mask] == BACKGROUND):
                return False

            rebuilt[r0 : r0 + 2, c0 : c0 + 2][mask] = left_color
            rebuilt[r0 : r0 + 2, mirror_c0 : mirror_c0 + 2][mask] = right_color
            left_colors.add(left_color)
            right_colors.add(right_color)
            areas.add(int(mask.sum()))

        if len(marker_by_col) < 3 or len(marker_by_col) > 5:
            return False
        if not _has_color_diversity(
            [
                {
                    "left_marker_col": next(col for col, color in marker_by_col.items() if color == int(sprite["left_color"])),
                    "right_marker_col": next(col for col, color in marker_by_col.items() if color == int(sprite["right_color"])),
                }
                for sprite in sprites
            ],
            len(marker_by_col),
        ):
            return False
        if areas != {3, 4}:
            return False
        if not np.array_equal(rebuilt, expected):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = int(rng.integers(10, 15))
        width = int(rng.integers(10, 15))
        result = _sample_layout(rng, height, width)
        if result is None:
            continue
        inp, _expected = result
        return inp
    raise RuntimeError("failed to generate a valid input for puzzle 7")
