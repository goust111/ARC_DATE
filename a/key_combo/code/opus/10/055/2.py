from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
MARKER = 6

CANDIDATES = {
    "A": {"minor": 1, "major": 2},
    "B": {"minor": 4, "major": 7},
}

BLOCK_SIZE_CHOICES = (3, 4)
BLOCK_ROWS_RANGE = (3, 5)
BLOCK_COLS_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 400


def _is_separator_line(line: np.ndarray) -> bool:
    return set(map(int, line)).issubset({BLACK, MARKER})


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    assert separators, "expected separator lines"
    bounds = [-1, *separators, int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in intervals]
    assert min(sizes) > 0, "separator lines must leave non-empty blocks"
    assert len(set(sizes)) == 1, "blocks must have a uniform size"
    return intervals


def _partition_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    row_separators = [r for r in range(grid.shape[0]) if _is_separator_line(grid[r, :])]
    col_separators = [c for c in range(grid.shape[1]) if _is_separator_line(grid[:, c])]
    row_intervals = _regular_intervals(row_separators, grid.shape[0])
    col_intervals = _regular_intervals(col_separators, grid.shape[1])
    row_sizes = {r1 - r0 for r0, r1 in row_intervals}
    col_sizes = {c1 - c0 for c0, c1 in col_intervals}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "block size must be unique"
    block_h = next(iter(row_sizes))
    block_w = next(iter(col_sizes))
    assert block_h == block_w, "blocks must be square"
    return row_intervals, col_intervals, int(block_h)


def _known_values(block: np.ndarray) -> np.ndarray:
    return block[(block != BLACK) & (block != MARKER)]


def _classify_block(block: np.ndarray) -> tuple[str, str]:
    values = _known_values(block)
    assert values.size > 0, "every block must expose at least one non-marker cell"
    colors = set(map(int, values))

    for name, spec in CANDIDATES.items():
        pair = {spec["minor"], spec["major"]}
        if colors.issubset(pair):
            if colors == {spec["minor"]}:
                return name, "blank"
            return name, "copy"
    raise AssertionError("block does not match either candidate palette")


def _analyze_blocks(grid: np.ndarray) -> tuple[list[dict], int]:
    row_intervals, col_intervals, block_size = _partition_grid(grid)
    infos: list[dict] = []
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            block = grid[r0:r1, c0:c1].copy()
            candidate, role = _classify_block(block)
            infos.append(
                {
                    "candidate": candidate,
                    "role": role,
                    "block": block,
                    "row": (r0, r1),
                    "col": (c0, c1),
                }
            )
    return infos, block_size


def _marker_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.flat if int(v) != BLACK)
    assert counts, "expected non-black colors"
    least = min(counts.values())
    colors = [color for color, count in counts.items() if count == least]
    assert colors == [MARKER], "marker color must be the unique least frequent non-black color"
    return MARKER


def _has_unique_least_marker(grid: np.ndarray) -> bool:
    counts = Counter(int(v) for v in grid.flat if int(v) != BLACK)
    if not counts or MARKER not in counts:
        return False
    least = min(counts.values())
    return counts[MARKER] == least and [color for color, count in counts.items() if count == least] == [MARKER]


def _recover_sprite(copy_blocks: list[np.ndarray], candidate: str) -> np.ndarray:
    assert copy_blocks, "expected at least one copy block"
    clean_blocks = [block for block in copy_blocks if MARKER not in block]
    if clean_blocks:
        sprite = clean_blocks[0].copy()
        for block in clean_blocks[1:]:
            assert np.array_equal(block, sprite), "all clean copies must agree"
        return sprite

    spec = CANDIDATES[candidate]
    sprite = np.full(copy_blocks[0].shape, -1, dtype=int)
    for r in range(sprite.shape[0]):
        for c in range(sprite.shape[1]):
            visible = [int(block[r, c]) for block in copy_blocks if int(block[r, c]) != MARKER]
            assert visible, "damaged copies must collectively reveal every sprite cell"
            colors = set(visible)
            assert colors.issubset({spec["minor"], spec["major"]}), "revealed pixels must use the candidate palette"
            assert len(colors) == 1, "all visible votes at a cell must agree"
            sprite[r, c] = visible[0]

    return sprite


def _matches_sprite(block: np.ndarray, sprite: np.ndarray) -> bool:
    known = block != MARKER
    return bool(np.all(block[known] == sprite[known]))


def _render_selected_output(infos: list[dict], block_size: int, target: str, sprite: np.ndarray) -> np.ndarray:
    rows = sorted({info["row"] for info in infos})
    cols = sorted({info["col"] for info in infos})
    height = rows[-1][1]
    width = cols[-1][1]
    out = np.zeros((height, width), dtype=int)
    minor = CANDIDATES[target]["minor"]
    uniform = np.full((block_size, block_size), minor, dtype=int)

    for info in infos:
        r0, r1 = info["row"]
        c0, c1 = info["col"]
        if info["candidate"] != target:
            continue
        if info["role"] == "copy":
            assert _matches_sprite(info["block"], sprite), "target copies must match the recovered sprite"
            out[r0:r1, c0:c1] = uniform
        else:
            values = _known_values(info["block"])
            assert np.all(values == minor), "blank blocks must use the candidate's minority color"
            out[r0:r1, c0:c1] = sprite
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the marker-dense sprite family and decode the block grid by inverting that family with its blank code.

    Input structure:
    - The grid is split by black separator rows and columns into equal square blocks.
    - Blocks belong to candidate A (colors 1/2), candidate B (colors 4/7), or a uniform minority-color blank block for one candidate.
    - Magenta marker pixels (6) are globally rare and replace some cells in sprite-copy blocks; separator lines may also contain marker pixels.

    Rule:
    - Infer marker color 6 as the unique least frequent non-black color.
    - Classify each block by palette while treating marker pixels as wildcards, then count marker pixels across A-copy blocks and B-copy blocks.
    - The candidate with the larger marker total is the target; recover its full sprite from a clean copy or by per-cell majority over non-marker votes.
    - Replace target sprite-copy blocks with a uniform minority-color block, replace target blank blocks with the recovered sprite, and clear every non-target block plus all separator lines to black.

    Key insight:
    - Marker color has a dual role: it both chooses which sprite family matters and acts as wildcard damage while that chosen family is decoded.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _marker_color(g)

    infos, block_size = _analyze_blocks(g)
    copy_blocks = {"A": [], "B": []}
    marker_totals = {"A": 0, "B": 0}

    for info in infos:
        if info["role"] == "copy":
            copy_blocks[info["candidate"]].append(info["block"])
            marker_totals[info["candidate"]] += int(np.count_nonzero(info["block"] == MARKER))

    assert copy_blocks["A"] and copy_blocks["B"], "both candidates must have sprite copies"
    assert marker_totals["A"] != marker_totals["B"], "marker totals must choose a unique target"

    target = "A" if marker_totals["A"] > marker_totals["B"] else "B"
    sprite = _recover_sprite(copy_blocks[target], target)
    colors = set(map(int, sprite.flat))
    assert colors == {CANDIDATES[target]["minor"], CANDIDATES[target]["major"]}, "sprite must use both candidate colors"

    return _render_selected_output(infos, block_size, target, sprite)


def _sprite_mask(sprite: np.ndarray, color: int) -> np.ndarray:
    return sprite == int(color)


def _generate_sprite(block_size: int, minor: int, major: int, avoid_mask: np.ndarray | None = None) -> np.ndarray:
    area = block_size * block_size
    max_minor = min(6, (area - 1) // 2)
    for _ in range(300):
        minor_count = random.randint(3, max_minor)
        idxs = random.sample(range(area), minor_count)
        rows = {idx // block_size for idx in idxs}
        cols = {idx % block_size for idx in idxs}
        if len(rows) < 2 or len(cols) < 2:
            continue
        sprite = np.full((block_size, block_size), major, dtype=int)
        for idx in idxs:
            sprite[idx // block_size, idx % block_size] = minor
        mask = _sprite_mask(sprite, minor)
        if avoid_mask is not None and np.array_equal(mask, avoid_mask):
            continue
        return sprite
    raise RuntimeError("failed to generate a sprite")


def _sample_category_counts(total_blocks: int, all_target_damaged: bool) -> dict[str, int]:
    counts = {
        "target_copy": 2 if all_target_damaged else 1,
        "target_blank": 1,
        "other_copy": 1,
        "other_blank": 1,
    }
    remaining = total_blocks - sum(counts.values())
    assert remaining >= 0
    keys = list(counts)
    for _ in range(remaining):
        counts[random.choice(keys)] += 1
    return counts


def _all_positions(shape: tuple[int, int]) -> list[int]:
    return list(range(shape[0] * shape[1]))


def _sample_mark_sets(
    sprite: np.ndarray,
    count: int,
    *,
    all_damaged: bool,
    require_clean: bool,
    heavy: bool,
    require_coverage: bool,
) -> list[set[int]]:
    area_positions = _all_positions(sprite.shape)
    counts = Counter(map(int, sprite.flat))
    major = int(max(counts, key=counts.get))
    major_positions = {idx for idx, value in enumerate(sprite.flat) if int(value) == major}
    max_marks = min(4 if sprite.size >= 16 else 3, sprite.size - 1)

    for _ in range(600):
        mark_sets: list[set[int]] = []
        for _copy_idx in range(count):
            if heavy:
                lo = 1 if all_damaged else 0
                hi = max_marks
            else:
                lo = 0
                hi = min(2, max_marks)
            k = random.randint(lo, hi)
            chosen = set(random.sample(area_positions, k))
            if not (major_positions - chosen):
                break
            mark_sets.append(chosen)
        if len(mark_sets) != count:
            continue
        if all_damaged and any(len(marks) == 0 for marks in mark_sets):
            continue
        if require_clean and not any(len(marks) == 0 for marks in mark_sets):
            continue
        if require_coverage:
            visible = set()
            for marks in mark_sets:
                visible.update(set(area_positions) - marks)
            if len(visible) != sprite.size:
                continue
        return mark_sets
    raise RuntimeError("failed to place marker damage")


def _apply_marks(sprite: np.ndarray, mark_set: set[int]) -> np.ndarray:
    block = sprite.copy()
    side = sprite.shape[1]
    for idx in mark_set:
        block[idx // side, idx % side] = MARKER
    return block


def _block_origin(br: int, bc: int, block_size: int) -> tuple[int, int]:
    return br * (block_size + 1), bc * (block_size + 1)


def _separator_cells(height: int, width: int, row_separators: list[int], col_separators: list[int]) -> list[tuple[int, int]]:
    cells = {(r, c) for r in row_separators for c in range(width)}
    cells.update({(r, c) for c in col_separators for r in range(height)})
    return list(cells)


def _render_case(
    block_rows: int,
    block_cols: int,
    block_size: int,
    categories: list[str],
    sprites: dict[str, np.ndarray],
    marked_copies: dict[str, list[np.ndarray]],
    add_separator_marks: bool,
) -> tuple[np.ndarray, np.ndarray]:
    height = block_rows * block_size + (block_rows - 1)
    width = block_cols * block_size + (block_cols - 1)
    grid = np.zeros((height, width), dtype=int)
    output = np.zeros((height, width), dtype=int)

    row_separators = [r * (block_size + 1) - 1 for r in range(1, block_rows)]
    col_separators = [c * (block_size + 1) - 1 for c in range(1, block_cols)]

    copy_ptr = {"A": 0, "B": 0}
    target = "A" if sum(np.count_nonzero(block == MARKER) for block in marked_copies["A"]) > sum(np.count_nonzero(block == MARKER) for block in marked_copies["B"]) else "B"
    target_uniform = np.full((block_size, block_size), CANDIDATES[target]["minor"], dtype=int)

    idx = 0
    for br in range(block_rows):
        for bc in range(block_cols):
            r0, c0 = _block_origin(br, bc, block_size)
            category = categories[idx]
            idx += 1
            candidate, role = category.split("_")
            if role == "copy":
                patch = marked_copies[candidate][copy_ptr[candidate]]
                copy_ptr[candidate] += 1
            else:
                patch = np.full((block_size, block_size), CANDIDATES[candidate]["minor"], dtype=int)
            grid[r0 : r0 + block_size, c0 : c0 + block_size] = patch

            if candidate == target:
                output[r0 : r0 + block_size, c0 : c0 + block_size] = (
                    target_uniform if role == "copy" else sprites[target]
                )

    if add_separator_marks:
        cells = _separator_cells(height, width, row_separators, col_separators)
        for r, c in random.sample(cells, random.randint(1, min(4, len(cells)))):
            grid[r, c] = MARKER

    return grid, output


def _expected_from_meta(meta: dict) -> np.ndarray:
    height = meta["grid_shape"][0]
    width = meta["grid_shape"][1]
    out = np.zeros((height, width), dtype=int)
    target = meta["target"]
    sprite = meta["sprites"][target]
    uniform = np.full(sprite.shape, CANDIDATES[target]["minor"], dtype=int)
    idx = 0
    for br in range(meta["block_rows"]):
        for bc in range(meta["block_cols"]):
            r0, c0 = _block_origin(br, bc, meta["block_size"])
            category = meta["categories"][idx]
            idx += 1
            candidate, role = category.split("_")
            if candidate != target:
                continue
            out[r0 : r0 + meta["block_size"], c0 : c0 + meta["block_size"]] = uniform if role == "copy" else sprite
    return out


def _alternative_output(meta: dict) -> np.ndarray:
    alt_target = "B" if meta["target"] == "A" else "A"
    alt = dict(meta)
    alt["target"] = alt_target
    return _expected_from_meta(alt)


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    if not _has_unique_least_marker(grid):
        return False
    expected = _expected_from_meta(meta)
    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(expected, _alternative_output(meta)):
        return False

    infos, _block_size = _analyze_blocks(grid)
    copy_counts = Counter((info["candidate"], info["role"]) for info in infos)
    if copy_counts[("A", "copy")] < 1 or copy_counts[("B", "copy")] < 1:
        return False
    if copy_counts[("A", "blank")] < 1 or copy_counts[("B", "blank")] < 1:
        return False

    marker_totals = {
        name: sum(int(np.count_nonzero(info["block"] == MARKER)) for info in infos if info["candidate"] == name and info["role"] == "copy")
        for name in ("A", "B")
    }
    if marker_totals[meta["target"]] <= marker_totals["B" if meta["target"] == "A" else "A"]:
        return False

    target_blocks = [info["block"] for info in infos if info["candidate"] == meta["target"] and info["role"] == "copy"]
    clean_target = any(MARKER not in block for block in target_blocks)
    if meta["all_target_damaged"] == clean_target:
        return False

    if meta["all_target_damaged"]:
        visible = np.zeros(meta["sprites"][meta["target"]].shape, dtype=bool)
        for block in target_blocks:
            visible |= block != MARKER
        if not visible.all():
            return False
    else:
        if not clean_target:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        block_size = random.choice(BLOCK_SIZE_CHOICES)
        block_rows = random.randint(*BLOCK_ROWS_RANGE)
        block_cols = random.randint(*BLOCK_COLS_RANGE)
        total_blocks = block_rows * block_cols

        all_target_damaged = bool(random.getrandbits(1))
        target = random.choice(["A", "B"])
        other = "B" if target == "A" else "A"
        counts = _sample_category_counts(total_blocks, all_target_damaged)

        sprite_a = _generate_sprite(block_size, CANDIDATES["A"]["minor"], CANDIDATES["A"]["major"])
        sprite_b = _generate_sprite(
            block_size,
            CANDIDATES["B"]["minor"],
            CANDIDATES["B"]["major"],
            avoid_mask=_sprite_mask(sprite_a, CANDIDATES["A"]["minor"]),
        )
        sprites = {"A": sprite_a, "B": sprite_b}

        try:
            target_marks = _sample_mark_sets(
                sprites[target],
                counts["target_copy"],
                all_damaged=all_target_damaged,
                require_clean=not all_target_damaged,
                heavy=True,
                require_coverage=all_target_damaged,
            )
            other_marks = _sample_mark_sets(
                sprites[other],
                counts["other_copy"],
                all_damaged=False,
                require_clean=all_target_damaged,
                heavy=False,
                require_coverage=False,
            )
        except RuntimeError:
            continue

        target_total = sum(len(marks) for marks in target_marks)
        other_total = sum(len(marks) for marks in other_marks)
        if target_total <= other_total:
            continue

        marked_copies = {
            target: [_apply_marks(sprites[target], marks) for marks in target_marks],
            other: [_apply_marks(sprites[other], marks) for marks in other_marks],
        }

        categories = (
            [f"{target}_copy"] * counts["target_copy"]
            + [f"{target}_blank"] * counts["target_blank"]
            + [f"{other}_copy"] * counts["other_copy"]
            + [f"{other}_blank"] * counts["other_blank"]
        )
        random.shuffle(categories)

        add_separator_marks = random.random() < 0.35
        grid, _output = _render_case(
            block_rows,
            block_cols,
            block_size,
            categories,
            sprites,
            marked_copies,
            add_separator_marks,
        )
        if not _has_unique_least_marker(grid):
            continue

        meta = {
            "block_rows": block_rows,
            "block_cols": block_cols,
            "block_size": block_size,
            "categories": categories,
            "sprites": sprites,
            "target": target,
            "all_target_damaged": all_target_damaged,
            "grid_shape": grid.shape,
        }
        if _validate_case(grid, meta):
            return grid

    raise RuntimeError("failed to generate a valid input")
